"""
Star Attention (NVIDIA): a two-phase distributed attention scheme for long-
context inference. Phase 1 (context encoding) computes each context
block's K/V using only that block plus a shared "anchor" block -- fully
parallel, no cross-host communication, but an approximation of full causal
attention. Phase 2 (query encoding/generation) merges each block's LOCAL,
UNNORMALIZED partial softmax stats into the exact global attention output
via an online-softmax merge -- this step is exact, not approximate.

This file simulates multiple "hosts" as a list of tensors on one machine.
"""

import math

import torch


def phase1_local_mask(seq_len, anchor_len, block_size):
    """Boolean mask for encoding: token i (in some block) attends to the
    anchor block (positions < anchor_len) plus its own block, causally.
    Anchor-block tokens themselves just get ordinary causal attention."""
    positions = torch.arange(seq_len)
    block_id = torch.where(positions < anchor_len, torch.zeros_like(positions),
                            (positions - anchor_len) // block_size + 1)

    i, j = positions.unsqueeze(1), positions.unsqueeze(0)
    causal = j <= i
    same_block_or_anchor = (block_id.unsqueeze(1) == block_id.unsqueeze(0)) | (j < anchor_len)
    return causal & same_block_or_anchor


def local_partial_softmax(q, k, v):
    """One host's contribution: unnormalized numerator, denominator, and
    max score for its local block only -- the only things sent over the
    wire to the merging host (small vectors, not the full K/V)."""
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # (block_k,)
    local_max = scores.max()
    exp_scores = torch.exp(scores - local_max)
    numerator = exp_scores @ v          # (d_v,)
    denominator = exp_scores.sum()      # scalar
    return numerator, denominator, local_max


def merge_partials(partials):
    """Online-softmax merge across hosts' partial results -- exact, not
    approximate: reproduces full softmax attention over the union of all
    hosts' keys/values, communicating only (numerator, denominator, max)
    per host instead of the full local K/V."""
    global_max = torch.stack([p[2] for p in partials]).max()

    out = 0.0
    denom = 0.0
    for numerator, denominator, local_max in partials:
        correction = torch.exp(local_max - global_max)
        out = out + numerator * correction
        denom = denom + denominator * correction
    return out / denom


if __name__ == "__main__":
    torch.manual_seed(0)
    d_k = 8
    anchor_len, block_size, num_blocks = 3, 4, 4
    seq_len = anchor_len + block_size * num_blocks

    x_k = torch.randn(seq_len, d_k)
    x_v = torch.randn(seq_len, d_k)

    # --- Phase 1: block-local encoding mask ---
    mask = phase1_local_mask(seq_len, anchor_len, block_size)
    print("phase-1 attend mask (rows=query token, cols=key token; anchor=first "
          f"{anchor_len} cols, blocks of {block_size} after):")
    for row in mask.int().tolist():
        print(" ", "".join(str(v) for v in row))

    # --- Phase 2: distributed query attention, simulated across "hosts" ---
    query = torch.randn(d_k)
    anchor_k, anchor_v = x_k[:anchor_len], x_v[:anchor_len]

    # Each non-anchor block is one "host". Anchor is included as its own group.
    groups_k = [anchor_k] + [x_k[anchor_len + b * block_size: anchor_len + (b + 1) * block_size]
                              for b in range(num_blocks)]
    groups_v = [anchor_v] + [x_v[anchor_len + b * block_size: anchor_len + (b + 1) * block_size]
                              for b in range(num_blocks)]

    partials = [local_partial_softmax(query, gk, gv) for gk, gv in zip(groups_k, groups_v)]
    merged_output = merge_partials(partials)

    # Ground truth: naive full softmax attention over the concatenation of
    # every group's keys/values -- what phase 2 claims to reproduce exactly.
    full_k = torch.cat(groups_k, dim=0)
    full_v = torch.cat(groups_v, dim=0)
    scores = (query @ full_k.T) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    naive_output = weights @ full_v

    print(f"\nphase-2 distributed merge output: {merged_output[:4].tolist()}")
    print(f"naive full-attention output:       {naive_output[:4].tolist()}")
    print(f"phase-2 merge is exact: {torch.allclose(merged_output, naive_output, atol=1e-5)}")
    print(f"\ncommunication per generation step: {len(partials)} small (numerator, denominator, max) "
          f"triples, vs {seq_len} full key/value vectors for naive distributed full attention")
