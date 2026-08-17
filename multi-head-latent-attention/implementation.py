"""
Multi-Head Latent Attention (MLA, DeepSeek-V2): compress K/V into a shared
low-rank latent per token instead of caching full per-head K/V, reconstruct
per-head K/V from it via up-projections, and use decoupled RoPE (a small
uncompressed positional slice appended to the compressed content) since
RoPE's position-dependent rotation is incompatible with compressing K.
Demonstrates the cache-size win and the query-absorption trick.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from importlib import import_module
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rotary-position-embedding"))
_rope = import_module("implementation")
rope_frequencies, apply_rope = _rope.rope_frequencies, _rope.apply_rope


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, d_latent, d_rope):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_rope = d_rope

        # Compress K/V jointly into one small shared latent per token — this is what gets cached.
        self.w_down = nn.Linear(d_model, d_latent, bias=False)
        self.w_up_k = nn.Linear(d_latent, num_heads * d_head, bias=False)  # per-head K reconstruction
        self.w_up_v = nn.Linear(d_latent, num_heads * d_head, bias=False)  # per-head V reconstruction

        # Query content path (kept full-rank here for simplicity; DeepSeek-V2
        # also low-rank-compresses queries, but that's for activation memory
        # during training, not the KV-cache argument this file demonstrates).
        self.w_q = nn.Linear(d_model, num_heads * d_head, bias=False)

        # Decoupled RoPE path: small, uncompressed, applied directly.
        self.w_q_rope = nn.Linear(d_model, num_heads * d_rope, bias=False)
        self.w_k_rope = nn.Linear(d_model, d_rope, bias=False)  # shared across heads

        self.w_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.rope_theta = rope_frequencies(d_rope)

    def forward(self, x, positions):
        seq_len = x.shape[0]

        c_kv = self.w_down(x)  # (seq_len, d_latent) -- this is the entire KV cache footprint
        k_content = self.w_up_k(c_kv).view(seq_len, self.num_heads, self.d_head)
        v = self.w_up_v(c_kv).view(seq_len, self.num_heads, self.d_head)

        q_content = self.w_q(x).view(seq_len, self.num_heads, self.d_head)

        # apply_rope expects (..., seq_len, dim) -- move heads out of the way, apply, move back.
        q_rope = self.w_q_rope(x).view(seq_len, self.num_heads, self.d_rope).transpose(0, 1)
        q_rope = apply_rope(q_rope, positions, self.rope_theta).transpose(0, 1)

        k_rope = self.w_k_rope(x)  # (seq_len, d_rope), shared across heads
        k_rope = apply_rope(k_rope, positions, self.rope_theta)
        k_rope = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1)  # broadcast shared rope-key to all heads

        q = torch.cat([q_content, q_rope], dim=-1)  # (seq_len, heads, d_head + d_rope)
        k = torch.cat([k_content, k_rope], dim=-1)

        scores = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(self.d_head + self.d_rope)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = torch.einsum("hqk,khd->qhd", weights, v).reshape(seq_len, -1)
        return self.w_o(out), c_kv, k_rope[:, 0, :]  # return cache contents too


if __name__ == "__main__":
    torch.manual_seed(0)
    seq_len, d_model, num_heads, d_head = 10, 64, 8, 16
    d_latent, d_rope = 12, 4  # deliberately small latent to show compression

    mla = MultiHeadLatentAttention(d_model, num_heads, d_head, d_latent, d_rope)
    x = torch.randn(seq_len, d_model)
    positions = torch.arange(seq_len)

    out, c_kv_cache, k_rope_cache = mla(x, positions)
    print("output shape:", out.shape)

    # Cache size comparison, per token per layer.
    mla_cache = d_latent + d_rope
    mha_cache = 2 * num_heads * d_head        # full multi-head: K and V per head
    gqa2_cache = 2 * 2 * d_head               # GQA with 2 KV-head groups
    mqa_cache = 2 * 1 * d_head                # MQA: 1 shared KV head
    print(f"\ncache scalars/token/layer:")
    print(f"  standard MHA (8 heads): {mha_cache}")
    print(f"  GQA (2 groups):         {gqa2_cache}")
    print(f"  MQA (1 group):          {mqa_cache}")
    print(f"  MLA (latent={d_latent}, rope={d_rope}): {mla_cache}")

    # Absorption trick: q . k_content == (q @ W_up_K^T) . c_kv, for a single
    # head -- i.e. you can score directly against the cached latent without
    # ever reconstructing k_content.
    head = 0
    w_up_k_head = mla.w_up_k.weight[head * d_head:(head + 1) * d_head, :]  # (d_head, d_latent)
    q_content = mla.w_q(x).view(seq_len, num_heads, d_head)[:, head, :]
    c_kv = mla.w_down(x)

    direct_scores = q_content @ w_up_k_head @ c_kv.T                 # reconstruct k, then dot
    absorbed_scores = (q_content @ w_up_k_head) @ c_kv.T             # fold W_up_K into q first
    print(f"\nabsorption trick matches direct reconstruction: "
          f"{torch.allclose(direct_scores, absorbed_scores, atol=1e-4)}")
