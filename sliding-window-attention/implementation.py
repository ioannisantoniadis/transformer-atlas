"""
Sliding window attention: each position attends only to the last W
positions (plus itself), instead of the full causal history. Demonstrates
the banded mask, the multi-layer effective-receptive-field argument, and
the bounded rolling KV cache this enables at inference.
"""

import math

import torch
import torch.nn.functional as F


def sliding_window_mask(seq_len, window_size):
    """(seq_len, seq_len) boolean mask: position i attends to
    max(0, i - window_size + 1) .. i."""
    positions = torch.arange(seq_len)
    i, j = positions.unsqueeze(1), positions.unsqueeze(0)
    causal = j <= i
    within_window = j > (i - window_size)
    return causal & within_window


def attention_with_mask(q, k, v, mask):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class RollingKVCache:
    """Fixed-capacity cache holding only the last `window_size` K/V entries
    — unlike a full causal cache, this never grows past window_size."""

    def __init__(self, window_size, d_k):
        self.window_size = window_size
        self.k = torch.empty(0, d_k)
        self.v = torch.empty(0, d_k)

    def append(self, k_new, v_new):
        self.k = torch.cat([self.k, k_new], dim=0)[-self.window_size:]
        self.v = torch.cat([self.v, v_new], dim=0)[-self.window_size:]
        return self.k, self.v


if __name__ == "__main__":
    torch.manual_seed(0)
    seq_len, d_k, window_size = 12, 8, 4

    mask = sliding_window_mask(seq_len, window_size)
    print("sliding window mask (1=attend, 0=masked):")
    for row in mask.int().tolist():
        print(" ", "".join(str(v) for v in row))

    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    out, weights = attention_with_mask(q, k, v, mask)
    print("\noutput shape:", out.shape)
    nonzero_per_row = (weights > 0).sum(dim=-1)
    print("attended-positions per row (should cap at window_size once past it):", nonzero_per_row.tolist())

    # Multi-layer effective receptive field: with window_size=4 and 3
    # layers, information from position 0 can reach position up to
    # roughly 3 * (window_size - 1) later, not just window_size - 1.
    num_layers = 3
    effective_reach = num_layers * (window_size - 1)
    print(f"\nsingle-layer reach: {window_size - 1} positions back")
    print(f"effective reach after {num_layers} stacked layers: up to {effective_reach} positions back")

    # Rolling KV cache: append tokens one at a time, cache never exceeds window_size.
    cache = RollingKVCache(window_size, d_k)
    for t in range(seq_len):
        k_cache, v_cache = cache.append(k[t:t+1], v[t:t+1])
    print(f"\nfinal rolling cache size: {k_cache.shape[0]} (bounded at window_size={window_size}, "
          f"vs {seq_len} for a full causal cache)")
