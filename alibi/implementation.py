"""
ALiBi: instead of adding positional embeddings to the input, bias raw
attention scores by a fixed, head-specific penalty proportional to
distance. No extra parameters, no rotation -- and because the penalty is
just linear in (i - j), it's defined identically at any distance, which is
what gives ALiBi its length-extrapolation property.
"""

import math

import torch
import torch.nn.functional as F


def alibi_slopes(num_heads):
    """Fixed (not learned) per-head slopes, a geometric sequence starting
    at 2^(-8/num_heads) -- the formula from the paper. Different heads get
    very different decay rates 'for free'."""
    start = 2 ** (-8 / num_heads)
    return torch.tensor([start ** (i + 1) for i in range(num_heads)])


def alibi_bias(seq_len, slopes):
    """(num_heads, seq_len, seq_len) bias matrix: -slope * (i - j) for j <= i."""
    positions = torch.arange(seq_len)
    distance = positions.unsqueeze(1) - positions.unsqueeze(0)  # i - j, (seq_len, seq_len)
    distance = distance.clamp(min=0).float()  # only care about j <= i (causal); ignore j > i here
    return -slopes.view(-1, 1, 1) * distance.unsqueeze(0)  # (num_heads, seq_len, seq_len)


def attention_with_alibi(q, k, v, slopes):
    """q, k, v: (num_heads, seq_len, d_k)."""
    num_heads, seq_len, d_k = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    scores = scores + alibi_bias(seq_len, slopes)

    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


if __name__ == "__main__":
    torch.manual_seed(0)
    num_heads, seq_len, d_k = 4, 10, 8

    slopes = alibi_slopes(num_heads)
    print("per-head slopes (steepest to gentlest decay):", [f"{s:.4f}" for s in slopes.tolist()])

    q = torch.randn(num_heads, seq_len, d_k)
    k = torch.randn(num_heads, seq_len, d_k)
    v = torch.randn(num_heads, seq_len, d_k)

    out, weights = attention_with_alibi(q, k, v, slopes)
    print("\noutput shape:", out.shape)

    # Show the recency bias directly: for the last query, attention weight
    # should (on average, ignoring content) trend down as distance grows,
    # more sharply for the steep-slope head than the gentle-slope head.
    last_query = seq_len - 1
    steep_head, gentle_head = 0, num_heads - 1
    print(f"\nattention weights for the last query position (head 0, steepest decay):")
    print(" ", [f"{w:.3f}" for w in weights[steep_head, last_query].tolist()])
    print(f"attention weights for the last query position (head {gentle_head}, gentlest decay):")
    print(" ", [f"{w:.3f}" for w in weights[gentle_head, last_query].tolist()])

    # Extrapolation property: the bias formula is defined identically at
    # any sequence length -- no table lookup, nothing "runs out."
    for test_len in (seq_len, seq_len * 5, seq_len * 50):
        bias = alibi_bias(test_len, slopes)
        farthest_penalty = bias[0, -1, 0].item()  # head 0, last query vs first key
        print(f"\nseq_len={test_len:4d}: bias(head0, farthest pair) = {farthest_penalty:.2f} "
              f"(same formula, any length -- nothing to run out of)")
