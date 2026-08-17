"""
FlashAttention's algorithm (tiled attention with online/streaming softmax),
implemented in pure PyTorch for readability. This reproduces the *exact
math* of attention with the same numerical result as the naive version —
it does NOT reproduce the memory-bandwidth win, which requires a fused
CUDA/Triton kernel keeping blocks in on-chip SRAM. Here it's for
understanding the algorithm, not for speed.
"""

import math

import torch


def naive_attention(q, k, v, causal=False):
    """Standard attention: materializes the full (seq_q, seq_k) score matrix."""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if causal:
        seq_q, seq_k = scores.shape[-2:]
        mask = torch.tril(torch.ones(seq_q, seq_k, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def flash_attention(q, k, v, block_size=4, causal=False):
    """Tiled attention with online softmax. Never materializes the full
    (seq_q, seq_k) matrix at once — only one (block_q, block_k) tile."""
    seq_len, d_k = q.shape
    scale = 1.0 / math.sqrt(d_k)

    out = torch.zeros_like(q)
    running_max = torch.full((seq_len,), float("-inf"))     # per-query running max score
    running_sum = torch.zeros(seq_len)                       # per-query running softmax denominator

    for k_start in range(0, seq_len, block_size):
        k_end = min(k_start + block_size, seq_len)
        k_block = k[k_start:k_end]                           # (block_k, d_k)
        v_block = v[k_start:k_end]

        scores = q @ k_block.transpose(-2, -1) * scale        # (seq_len, block_k) — small tile, not full matrix

        if causal:
            q_positions = torch.arange(seq_len).unsqueeze(1)
            k_positions = torch.arange(k_start, k_end).unsqueeze(0)
            mask = q_positions >= k_positions
            scores = scores.masked_fill(~mask, float("-inf"))

        block_max = scores.max(dim=-1).values                 # (seq_len,)
        new_max = torch.maximum(running_max, block_max)

        # Rescale the running output/sum to the new max, then add this block's contribution.
        correction = torch.exp(running_max - new_max)
        correction = torch.nan_to_num(correction, nan=0.0)    # handles the first block, where running_max=-inf

        exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
        exp_scores = torch.nan_to_num(exp_scores, nan=0.0)    # fully-masked rows (causal, no valid keys yet)

        running_sum = running_sum * correction + exp_scores.sum(dim=-1)
        out = out * correction.unsqueeze(-1) + exp_scores @ v_block
        running_max = new_max

    return out / running_sum.unsqueeze(-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    seq_len, d_k = 17, 8  # odd length on purpose, to exercise a ragged last block

    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)

    for causal in (False, True):
        naive_out = naive_attention(q, k, v, causal=causal)
        flash_out = flash_attention(q, k, v, block_size=4, causal=causal)
        max_diff = (naive_out - flash_out).abs().max().item()
        print(f"causal={causal!s:5s} max abs diff (naive vs tiled-online-softmax): {max_diff:.2e}")
        assert torch.allclose(naive_out, flash_out, atol=1e-5), "tiled attention should exactly match naive attention"

    print("tiled/online-softmax attention matches naive attention exactly, as expected.")
