"""
Sparse Mixture-of-Experts: a router picks the top-k experts per token out
of N, only those run, outputs are combined by router weight. Includes the
load-balancing auxiliary loss (else routing collapses onto a few experts)
and an optional always-on "shared expert" (the DeepSeekMoE refinement).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A small FFN -- identical shape to a standard Transformer FFN, just
    smaller, since many of these run per layer instead of one big one."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class SparseMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, num_shared_experts=0):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        # DeepSeekMoE-style shared experts: always run, on every token, no routing.
        self.shared_experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_shared_experts)])

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (num_tokens, d_model)
        num_tokens = x_flat.shape[0]

        router_logits = self.router(x_flat)  # (num_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        top_weights, top_experts = router_probs.topk(self.top_k, dim=-1)  # (num_tokens, top_k)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)  # renormalize over chosen k

        output = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            token_mask, k_slot = (top_experts == expert_id).nonzero(as_tuple=True)
            if token_mask.numel() == 0:
                continue  # this expert got no tokens this batch -- skip entirely, the whole point of sparsity
            tokens_for_expert = x_flat[token_mask]
            expert_out = self.experts[expert_id](tokens_for_expert)
            weight = top_weights[token_mask, k_slot].unsqueeze(-1)
            output[token_mask] += weight * expert_out

        for shared_expert in self.shared_experts:
            output = output + shared_expert(x_flat)  # unconditional, every token

        aux_loss = self._load_balancing_loss(router_probs, top_experts)
        return output.view(batch, seq_len, d_model), aux_loss

    def _load_balancing_loss(self, router_probs, top_experts):
        """Switch-Transformer-style auxiliary loss: penalizes the router
        for concentrating tokens on a few experts. Minimized when both the
        fraction of tokens routed to an expert AND the router's average
        probability mass on that expert are uniform across experts."""
        num_tokens = router_probs.shape[0]
        chosen_onehot = F.one_hot(top_experts, num_classes=self.num_experts).float()  # (tokens, k, experts)
        fraction_routed = chosen_onehot.sum(dim=(0, 1)) / (num_tokens * self.top_k)
        avg_router_prob = router_probs.mean(dim=0)
        return self.num_experts * (fraction_routed * avg_router_prob).sum()


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model, d_ff = 2, 6, 16, 32
    num_experts, top_k = 4, 2

    x = torch.randn(batch, seq_len, d_model)

    moe = SparseMoE(d_model, d_ff, num_experts, top_k)
    out, aux_loss = moe(x)
    print("MoE output shape:", out.shape)
    print("load-balancing aux loss:", aux_loss.item())

    total_params = sum(p.numel() for p in moe.experts.parameters())
    active_params_per_token = total_params * top_k / num_experts
    print(f"\ntotal expert params: {total_params} "
          f"(all {num_experts} experts, only used across many different tokens)")
    print(f"active expert params per token: {active_params_per_token:.0f} "
          f"(only top_k={top_k} of {num_experts} experts run per token)")

    # DeepSeekMoE-style: routed experts + always-on shared experts.
    moe_shared = SparseMoE(d_model, d_ff, num_experts, top_k, num_shared_experts=1)
    out_shared, _ = moe_shared(x)
    print(f"\nwith 1 shared expert (always-on for every token): output shape {out_shared.shape}")

    # Sanity check: an expert that receives zero tokens should contribute nothing.
    router_probs = F.softmax(moe.router(x.view(-1, d_model)), dim=-1)
    _, top_experts = router_probs.topk(top_k, dim=-1)
    used_experts = set(top_experts.unique().tolist())
    print(f"\nexperts actually used this batch: {sorted(used_experts)} out of {list(range(num_experts))}")
