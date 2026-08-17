"""
Mixtral: a Llama-style block (RMSNorm + RoPE + GQA) with its SwiGLU FFN
sublayer replaced by a sparse top-2 Mixture-of-Experts layer of SwiGLU
experts. Everything except the FFN sublayer is identical to ../llama/ --
this file demonstrates that MoE is a drop-in swap for one sublayer, not a
change to the rest of the block.
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


def rope_frequencies(head_dim, base=10000.0):
    exponents = torch.arange(0, head_dim, 2).float() / head_dim
    return 1.0 / (base ** exponents)


def apply_rope(x, theta):
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    positions = torch.arange(seq_len, device=x.device).float()
    angles = positions[:, None] * theta[None, :].to(x.device)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    x_pairs = x.view(*x.shape[:-1], head_dim // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    rotated = torch.stack([-x2, x1], dim=-1).view(*x.shape)
    return x * cos + rotated * sin


class GroupedQueryAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.w_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)
        self.theta = rope_frequencies(self.d_k)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q = self.w_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        q, k = apply_rope(q, self.theta), apply_rope(k, self.theta)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        weights = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out)


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class SparseMoEFFN(nn.Module):
    """Top-k routed SwiGLU experts -- the sublayer that replaces a single
    dense SwiGLU FFN. See ../mixture-of-experts/ for the general mechanism."""

    def __init__(self, d_model, d_ff, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        router_probs = F.softmax(self.router(x_flat), dim=-1)
        top_weights, top_experts = router_probs.topk(self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            token_idx, k_slot = (top_experts == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[expert_id](x_flat[token_idx])
            output[token_idx] += top_weights[token_idx, k_slot].unsqueeze(-1) * expert_out
        return output.view(batch, seq_len, d_model)


class MixtralBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff, num_experts, top_k):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttentionWithRoPE(d_model, num_heads, num_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.moe = SparseMoEFFN(d_model, d_ff, num_experts, top_k)  # <-- the only difference from LlamaBlock

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.moe(self.ffn_norm(x))
        return x


class Mixtral(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_kv_heads, num_layers, d_ff, num_experts, top_k):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [MixtralBlock(d_model, num_heads, num_kv_heads, d_ff, num_experts, top_k) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        x = self.token_emb(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, d_model, num_heads, num_kv_heads, num_layers = 100, 32, 8, 2, 3
    d_ff, num_experts, top_k = 64, 8, 2  # 8 experts, top-2 routing -- Mixtral's actual configuration shape

    model = Mixtral(vocab_size, d_model, num_heads, num_kv_heads, num_layers, d_ff, num_experts, top_k)
    batch, seq_len = 2, 9
    token_ids = torch.randint(0, vocab_size, (batch, seq_len))
    logits = model(token_ids)
    print("logits shape:", logits.shape)

    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for p in model.blocks[0].moe.experts.parameters())
    active_expert_params = expert_params * top_k / num_experts
    print(f"total model params: {total_params:,}")
    print(f"expert params in one MoE layer: {expert_params:,} "
          f"(all {num_experts} experts, stored) vs "
          f"{active_expert_params:,.0f} active per token (top_k={top_k})")
