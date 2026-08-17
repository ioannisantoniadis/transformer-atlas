"""
DeepSeek-V2 block: Multi-Head Latent Attention (compressed KV cache +
decoupled RoPE) combined with DeepSeekMoE (many fine-grained routed
experts + always-on shared experts). Both pieces are simplified,
self-contained versions of ../multi-head-latent-attention/ and
../mixture-of-experts/ -- read those for the full explanation of each.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


def rope_frequencies(dim, base=10000.0):
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))


def apply_rope(x, theta):
    """x: (..., seq_len, dim) -- seq_len must be the second-to-last axis."""
    seq_len, dim = x.shape[-2], x.shape[-1]
    positions = torch.arange(seq_len).float()
    angles = positions[:, None] * theta[None, :]  # (seq_len, dim // 2)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    x_pairs = x.view(*x.shape[:-1], dim // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    rotated = torch.stack([-x2, x1], dim=-1).view(*x.shape)
    return x * cos + rotated * sin


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, d_latent, d_rope):
        super().__init__()
        self.num_heads, self.d_head, self.d_rope = num_heads, d_head, d_rope
        self.w_down = nn.Linear(d_model, d_latent, bias=False)         # -> what gets cached
        self.w_up_k = nn.Linear(d_latent, num_heads * d_head, bias=False)
        self.w_up_v = nn.Linear(d_latent, num_heads * d_head, bias=False)
        self.w_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.w_q_rope = nn.Linear(d_model, num_heads * d_rope, bias=False)
        self.w_k_rope = nn.Linear(d_model, d_rope, bias=False)          # shared across heads
        self.w_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.theta = rope_frequencies(d_rope)

    def forward(self, x):
        seq_len = x.shape[0]
        c_kv = self.w_down(x)  # (seq_len, d_latent) -- the entire cache footprint per token
        k_content = self.w_up_k(c_kv).view(seq_len, self.num_heads, self.d_head)
        v = self.w_up_v(c_kv).view(seq_len, self.num_heads, self.d_head)
        q_content = self.w_q(x).view(seq_len, self.num_heads, self.d_head)

        # apply_rope expects seq_len second-to-last -- move heads out of the way, apply, move back.
        q_rope = self.w_q_rope(x).view(seq_len, self.num_heads, self.d_rope).transpose(0, 1)
        q_rope = apply_rope(q_rope, self.theta).transpose(0, 1)
        k_rope = apply_rope(self.w_k_rope(x), self.theta).unsqueeze(1).expand(-1, self.num_heads, -1)

        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope], dim=-1)

        scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(self.d_head + self.d_rope)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        weights = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        out = torch.einsum("hqk,khd->qhd", weights, v).reshape(seq_len, -1)
        return self.w_o(out)


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class DeepSeekMoE(nn.Module):
    """Fine-grained routed experts (many, small) + always-on shared experts."""

    def __init__(self, d_model, d_ff_expert, num_routed_experts, top_k, num_shared_experts):
        super().__init__()
        self.top_k = top_k
        self.num_routed_experts = num_routed_experts
        self.router = nn.Linear(d_model, num_routed_experts, bias=False)
        self.routed_experts = nn.ModuleList([Expert(d_model, d_ff_expert) for _ in range(num_routed_experts)])
        self.shared_experts = nn.ModuleList([Expert(d_model, d_ff_expert) for _ in range(num_shared_experts)])

    def forward(self, x):
        router_probs = F.softmax(self.router(x), dim=-1)
        top_weights, top_experts = router_probs.topk(self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for expert_id in range(self.num_routed_experts):
            token_idx, k_slot = (top_experts == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            out[token_idx] += top_weights[token_idx, k_slot].unsqueeze(-1) * self.routed_experts[expert_id](x[token_idx])

        for shared in self.shared_experts:
            out = out + shared(x)  # unconditional, every token
        return out


class DeepSeekV2Block(nn.Module):
    def __init__(self, d_model, num_heads, d_head, d_latent, d_rope,
                 d_ff_expert, num_routed_experts, top_k, num_shared_experts):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, num_heads, d_head, d_latent, d_rope)
        self.moe_norm = RMSNorm(d_model)
        self.moe = DeepSeekMoE(d_model, d_ff_expert, num_routed_experts, top_k, num_shared_experts)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.moe_norm(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    seq_len, d_model = 10, 32
    num_heads, d_head, d_latent, d_rope = 8, 8, 12, 4          # MLA: small latent + small rope slice
    d_ff_expert, num_routed_experts, top_k, num_shared_experts = 16, 16, 2, 1  # DeepSeekMoE: many small experts

    block = DeepSeekV2Block(d_model, num_heads, d_head, d_latent, d_rope,
                             d_ff_expert, num_routed_experts, top_k, num_shared_experts)
    x = torch.randn(seq_len, d_model)
    out = block(x)
    print("block output shape:", out.shape)

    mla_cache = d_latent + d_rope
    mha_cache = 2 * num_heads * d_head
    print(f"\nMLA cache scalars/token: {mla_cache} vs standard MHA: {mha_cache} "
          f"({mha_cache / mla_cache:.1f}x smaller)")

    total_expert_params = sum(p.numel() for p in block.moe.routed_experts.parameters())
    active_expert_params = total_expert_params * top_k / num_routed_experts
    shared_params = sum(p.numel() for p in block.moe.shared_experts.parameters())
    print(f"routed experts: {num_routed_experts} total, {top_k} active/token "
          f"({active_expert_params:.0f} of {total_expert_params} params) "
          f"+ {num_shared_experts} shared expert(s) ({shared_params} params, always active)")
