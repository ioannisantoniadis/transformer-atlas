"""
LLaMA-style decoder block: RMSNorm + RoPE + Grouped-Query Attention +
SwiGLU, pre-norm, composed into a full causal LM. This file is
self-contained (each building block is small enough to re-implement here
rather than import from its own topic folder) but every piece matches
../rmsnorm-and-swiglu/, ../rotary-position-embedding/, and
../multi-query-and-grouped-query-attention/ -- read those for the "why"
behind each piece.
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
    """x: (batch, heads, seq_len, head_dim)."""
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    positions = torch.arange(seq_len, device=x.device).float()
    angles = positions[:, None] * theta[None, :].to(x.device)  # (seq_len, head_dim // 2)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)

    x_pairs = x.view(*x.shape[:-1], head_dim // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    rotated = torch.stack([-x2, x1], dim=-1).view(*x.shape)
    return x * cos + rotated * sin


class GroupedQueryAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
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

        q = apply_rope(q, self.theta)
        k = apply_rope(k, self.theta)  # RoPE applied before KV-head sharing, per token position
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class LlamaBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttentionWithRoPE(d_model, num_heads, num_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))   # pre-norm: normalize BEFORE the sublayer
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Llama(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_kv_heads, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)  # no positional embedding table -- RoPE lives in attention
        self.blocks = nn.ModuleList([LlamaBlock(d_model, num_heads, num_kv_heads) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        x = self.token_emb(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, d_model, num_heads, num_kv_heads, num_layers = 100, 32, 8, 2, 4  # GQA: 8 query heads, 2 KV heads

    model = Llama(vocab_size, d_model, num_heads, num_kv_heads, num_layers)
    batch, seq_len = 2, 10
    token_ids = torch.randint(0, vocab_size, (batch, seq_len))

    logits = model(token_ids)
    print("logits shape:", logits.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total_params:,}")

    # Confirm GQA is actually active: K/V projections should be smaller
    # than Q's, by exactly the num_heads / num_kv_heads ratio.
    attn = model.blocks[0].attn
    q_params = attn.w_q.weight.numel()
    k_params = attn.w_k.weight.numel()
    print(f"w_q params: {q_params}, w_k params: {k_params}, ratio: {q_params / k_params:.1f} "
          f"(expected {num_heads / num_kv_heads:.1f} = num_heads / num_kv_heads)")
