"""
Multi-Query (MQA) and Grouped-Query Attention (GQA): share key/value heads
across groups of query heads to shrink the KV cache. num_kv_heads ==
num_heads recovers standard multi-head attention; num_kv_heads == 1
recovers MQA; anything in between is GQA.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads must divide evenly into num_kv_heads groups"
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, num_heads * self.d_k)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k)   # far fewer params than w_q
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_o = nn.Linear(num_heads * self.d_k, d_model)

    def forward(self, x, causal=True):
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Broadcast each KV head across its group of query heads.
        k = k.repeat_interleave(self.group_size, dim=1)  # (batch, num_heads, seq_len, d_k)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out)

    def kv_cache_size_per_token(self):
        """Number of scalars cached per token per layer (K + V)."""
        return 2 * self.num_kv_heads * self.d_k


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model, num_heads = 2, 10, 32, 8

    x = torch.randn(batch, seq_len, d_model)

    configs = {"MHA (num_kv_heads=8)": 8, "GQA (num_kv_heads=2)": 2, "MQA (num_kv_heads=1)": 1}
    for name, num_kv_heads in configs.items():
        attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        out = attn(x)
        kv_params = sum(p.numel() for p in [attn.w_k.weight, attn.w_v.weight])
        print(f"{name:22s} output {tuple(out.shape)}  "
              f"K+V proj params: {kv_params:5d}  "
              f"KV cache scalars/token/layer: {attn.kv_cache_size_per_token()}")
