"""
CSA + HCA: DeepSeek-V4's three-tier hybrid attention, stacked on top of
MLA-style latent compression (see ../multi-head-latent-attention/). Three
branches run per query -- (1) exact local sliding-window attention, (2)
Compressed Sparse Attention: pool every m tokens' K/V into one summary
entry, then attend over only the top-k most relevant summaries, and (3)
Heavily Compressed Attention: pool far more aggressively and attend
densely over the short compressed cache -- combined with a learned
per-query gate. Verifies the three branches against a small toy sequence,
then measures how much smaller the effective KV footprint and attended-
pair count are than full O(n^2) dense attention, including at the
million-token scale the technique targets (via direct counting, since a
literal 1M x 1M dense attention isn't something to materialize here).
"""

import math
import torch
import torch.nn.functional as F


def sliding_window_attention(q, k, v, window):
    """Exact softmax attention restricted to the last `window` keys per
    query (causal). q,k,v: (seq_len, d)."""
    n, d = q.shape
    scores = q @ k.transpose(0, 1) / math.sqrt(d)
    idx = torch.arange(n)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
    local = (idx.unsqueeze(1) - idx.unsqueeze(0)) < window
    scores = scores.masked_fill(~(causal & local), float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def compress_kv(k, v, block_size):
    """Pool every `block_size` tokens' K/V into one summary entry (mean
    pooling). Returns (k_compressed, v_compressed, last_covered_pos) where
    last_covered_pos[b] is the last original sequence position folded into
    compressed block b -- used so a query can only attend to a compressed
    block that is entirely in its causal past."""
    n, d = k.shape
    pad = (-n) % block_size
    if pad:
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
    num_blocks = k.shape[0] // block_size
    k_c = k.view(num_blocks, block_size, d).mean(dim=1)
    v_c = v.view(num_blocks, block_size, d).mean(dim=1)
    last_covered_pos = torch.arange(num_blocks) * block_size + (block_size - 1)
    return k_c, v_c, last_covered_pos


def compressed_sparse_attention(q, k, v, block_size, top_k):
    """CSA: pool K/V into blocks, score every query against every
    compressed block, keep only the causally-valid top-k blocks per query,
    attend over just those."""
    n, d = q.shape
    k_c, v_c, last_covered_pos = compress_kv(k, v, block_size)
    scores = q @ k_c.transpose(0, 1) / math.sqrt(d)  # (n, num_blocks)

    query_pos = torch.arange(n).unsqueeze(1)
    causal = last_covered_pos.unsqueeze(0) <= query_pos  # block fully in the past
    scores = scores.masked_fill(~causal, float("-inf"))

    k_eff = min(top_k, k_c.shape[0])
    top_scores, top_idx = scores.topk(k_eff, dim=-1)  # (n, k_eff)
    weights = torch.softmax(top_scores, dim=-1)
    gathered_v = v_c[top_idx]  # (n, k_eff, d)
    out = torch.einsum("nk,nkd->nd", weights, gathered_v)
    # queries with zero valid blocks (very start of sequence) get no signal from this branch
    out = torch.where(torch.isfinite(top_scores).any(dim=-1, keepdim=True), out, torch.zeros_like(out))
    return out, k_c.shape[0], k_eff


def heavily_compressed_attention(q, k, v, block_size):
    """HCA: pool much more aggressively, then attend densely (no top-k --
    the compressed cache is already short) over every causally-valid block."""
    n, d = q.shape
    k_c, v_c, last_covered_pos = compress_kv(k, v, block_size)
    scores = q @ k_c.transpose(0, 1) / math.sqrt(d)  # (n, num_blocks)
    query_pos = torch.arange(n).unsqueeze(1)
    causal = last_covered_pos.unsqueeze(0) <= query_pos
    scores = scores.masked_fill(~causal, float("-inf"))
    has_any = torch.isfinite(scores).any(dim=-1, keepdim=True)
    weights = torch.softmax(scores, dim=-1)
    weights = torch.where(has_any, weights, torch.zeros_like(weights))
    return weights @ v_c, k_c.shape[0]


def dense_causal_attention(q, k, v):
    n, d = q.shape
    scores = q @ k.transpose(0, 1) / math.sqrt(d)
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class GatedCSAHCA(torch.nn.Module):
    """Combines the three branches with a learned per-query softmax gate,
    the way DeepSeek-V4 blends window/CSA/HCA contributions."""

    def __init__(self, d, window, csa_block, csa_top_k, hca_block):
        super().__init__()
        self.window, self.csa_block, self.csa_top_k, self.hca_block = window, csa_block, csa_top_k, hca_block
        self.gate = torch.nn.Linear(d, 3)

    def forward(self, q, k, v):
        window_out = sliding_window_attention(q, k, v, self.window)
        csa_out, _, _ = compressed_sparse_attention(q, k, v, self.csa_block, self.csa_top_k)
        hca_out, _ = heavily_compressed_attention(q, k, v, self.hca_block)
        weights = torch.softmax(self.gate(q), dim=-1)  # (n, 3)
        branches = torch.stack([window_out, csa_out, hca_out], dim=1)  # (n, 3, d)
        return torch.einsum("nb,nbd->nd", weights, branches)


if __name__ == "__main__":
    torch.manual_seed(0)
    n, d = 64, 16
    q, k, v = torch.randn(n, d), torch.randn(n, d), torch.randn(n, d)

    # --- Correctness / shape check on a toy sequence ---
    model = GatedCSAHCA(d, window=8, csa_block=4, csa_top_k=4, hca_block=16)
    out = model(q, k, v)
    print(f"toy sequence: n={n}, d={d}")
    print(f"gated CSA+HCA output shape: {tuple(out.shape)} (expected ({n}, {d}))")

    dense_out = dense_causal_attention(q, k, v)
    print(f"dense causal attention output shape (for comparison): {tuple(dense_out.shape)}")
    print(f"outputs differ from dense attention (expected -- different, approximate mechanism): "
          f"{not torch.allclose(out, dense_out, atol=1e-2)}")

    # --- How many (query, key) pairs does each branch actually score, vs O(n^2) dense? ---
    window, csa_block, csa_top_k, hca_block = 8, 4, 4, 16
    window_pairs = sum(min(i + 1, window) for i in range(n))
    csa_num_blocks = math.ceil(n / csa_block)
    csa_score_pairs = n * csa_num_blocks           # scoring every query against every compressed block
    csa_attend_pairs = n * min(csa_top_k, csa_num_blocks)  # then attending over just the top-k
    hca_num_blocks = math.ceil(n / hca_block)
    hca_pairs = n * hca_num_blocks
    total_pairs = window_pairs + csa_score_pairs + hca_pairs
    dense_pairs = n * (n + 1) // 2  # causal dense attention

    print(f"\n(query, key) pairs scored, n={n}:")
    print(f"  dense causal attention:        {dense_pairs}")
    print(f"  window branch (w={window}):          {window_pairs}")
    print(f"  CSA branch (m={csa_block}, top-k={csa_top_k}): {csa_score_pairs} scored, {csa_attend_pairs} attended")
    print(f"  HCA branch (M={hca_block}):          {hca_pairs}")
    print(f"  total (window + CSA scoring + HCA):  {total_pairs}  "
          f"({100 * total_pairs / dense_pairs:.1f}% of dense)")

    # --- KV cache footprint: entries that must be stored/compared against, vs dense ---
    print(f"\nKV cache footprint (entries), n={n}:")
    print(f"  dense: {n} raw K/V pairs")
    print(f"  CSA compressed cache: {csa_num_blocks} entries (pool ratio {csa_block}x)")
    print(f"  HCA compressed cache: {hca_num_blocks} entries (pool ratio {hca_block}x)")

    # --- Same arithmetic at the million-token scale the technique targets.
    #     This is direct counting, not a literal forward pass (a 1M x 1M
    #     dense attention isn't something to materialize on CPU) -- it shows
    #     how the mechanism's cost *scales*, illustrative rather than a
    #     reproduction of DeepSeek's reported ~27% FLOPs / ~10% KV cache
    #     numbers, which depend on their specific tuned hyperparameters. ---
    n_long = 1_000_000
    window_l, csa_block_l, csa_top_k_l, hca_block_l = 2048, 16, 64, 128

    dense_pairs_l = n_long * (n_long + 1) // 2
    window_pairs_l = n_long * window_l  # dominates over the small edge effect at seq start
    csa_num_blocks_l = math.ceil(n_long / csa_block_l)
    csa_score_pairs_l = n_long * csa_num_blocks_l
    hca_num_blocks_l = math.ceil(n_long / hca_block_l)
    hca_pairs_l = n_long * hca_num_blocks_l
    total_pairs_l = window_pairs_l + csa_score_pairs_l + hca_pairs_l

    dense_kv_l = n_long
    compressed_kv_l = csa_num_blocks_l + hca_num_blocks_l

    print(f"\nat n={n_long:,} (illustrative, direct counting -- not a literal forward pass):")
    print(f"  attended/scored (query,key) pairs: {total_pairs_l:,} vs dense {dense_pairs_l:,} "
          f"({100 * total_pairs_l / dense_pairs_l:.4f}% of dense)")
    print(f"  KV cache entries (CSA + HCA compressed caches): {compressed_kv_l:,} vs dense {dense_kv_l:,} "
          f"({100 * compressed_kv_l / dense_kv_l:.2f}% of dense)")
