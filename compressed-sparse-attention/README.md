# Compressed Sparse Attention + Heavily Compressed Attention (CSA/HCA)

**Lab:** DeepSeek AI · **Year:** 2026 · **Paper:** [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)

## The problem

[MLA](../multi-head-latent-attention/) compresses what gets *cached* per
token (K/V down to one small latent vector) but every token's compressed
entry is still cached and still a candidate for every query to attend to —
the attention computation itself is still O(n) per query, O(n²) total.
[Sliding-window attention](../sliding-window-attention/) and
[Longformer-style sparsity](../longformer-and-sparse-attention/) cut that
by restricting *which* positions each query looks at, but a fixed local
window or fixed sparsity pattern is a blunt instrument at truly long
context (DeepSeek-V4 targets 1M-token contexts): a window misses relevant
far-away tokens entirely, and picking individual far-away tokens to attend
to (rather than summarizing regions of them) still costs one score
computation per candidate token, which doesn't shrink as context grows.

## The idea

Don't choose between "local dense" and "compress everything" — use three
tiers simultaneously, each suited to a different distance range, and blend
their outputs:

1. **Local window** — exact softmax attention over just the last `w`
   tokens (same mechanism as [sliding-window attention](../sliding-window-attention/)).
   Fine-grained, no compression, cheap because `w` is small and fixed.
2. **Compressed Sparse Attention (CSA)** — pool every `m` tokens' K/V into
   one summary entry (mean-pooling in this implementation; DeepSeek uses a
   learned pooling), then run **sparse top-k selection** over those
   summaries: score the query against every compressed entry, keep only
   the top-k most relevant, and attend over just those. This is
   "compress, *then* select" — cheaper than scoring every raw token
   (there are `n/m` summaries, not `n` tokens) and cheaper still than
   attending over every summary (only `k` of them are used per query).
3. **Heavily Compressed Attention (HCA)** — pool much more aggressively
   (DeepSeek reports ~128 tokens → 1 entry, an order of magnitude coarser
   than CSA's pooling), then attend **densely** — no top-k needed, because
   at that compression ratio the whole compressed cache for a 1M-token
   context is already short enough to attend over directly. This is the
   tier that gives the far-past a coarse, cheap "gist" every query can see
   without any per-query selection cost at all.

```
query position i
  │
  ├─► window branch:  dense softmax attention over tokens [i-w, i]
  │
  ├─► CSA branch:     pool K/V in blocks of m  ->  top-k of the n/m summaries -> attend
  │
  └─► HCA branch:     pool K/V in blocks of M (M >> m)  ->  attend densely over all n/M summaries

  output_i = gate_i · [window_i ; CSA_i ; HCA_i]     (learned per-query mixing weights)
```

Each branch answers a different question: "what's right next to me"
(window), "which distant *regions* actually matter, at moderate
resolution" (CSA), and "what's the coarse shape of everything else"
(HCA). None of the three branches costs O(n²): the window is O(n·w), CSA
is O(n·(n/m)) to score summaries plus O(n·k) to attend, and HCA is
O(n·(n/M)) — and because `m` and (especially) `M` grow with context length
in practice, the compressed branches stay cheap even as `n` reaches into
the millions.

## How it's actually used

CSA/HCA is the production attention stack of **DeepSeek-V4** (both
V4-Pro, 1.6T params/49B activated, and V4-Flash, 284B params/13B
activated), layered on top of MLA-style latent compression — MLA
compresses *what's stored per token*, CSA/HCA compresses *how many of
those stored entries a query has to look at*. The two ideas are
complementary, not competing: DeepSeek-V4 does both. At 1M-token context,
DeepSeek reports roughly 27% of per-token inference FLOPs and roughly 10%
of the KV cache footprint of DeepSeek-V3.2's attention stack, though those
exact figures depend on the specific `w`, `m`, `M`, and `k` DeepSeek
tuned — the demo below shows the *mechanism* and its general scaling
behavior, not a reproduction of those exact numbers.

## Tradeoffs

More moving parts than a single attention mechanism: three branches, each
with its own hyperparameter (`w`, `m`+`k`, `M`), plus a gate to learn how
to weight them per query — all extra surface area GQA or plain MLA don't
have. Compression is lossy: both CSA's and (especially) HCA's pooled
summaries discard within-block detail permanently, so a query needing a
*specific* fact buried inside a heavily-compressed region may not recover
it as precisely as full attention would (mitigated, not eliminated, by the
window branch handling the *nearby* precision-sensitive case and CSA's
finer pooling ratio handling the *mid-range* case). And unlike
FlashAttention, this isn't a free exactness-preserving rewrite of the same
computation — it's a genuinely different, approximate attention pattern
that has to be learned end-to-end, not retrofitted onto a trained model.

## References

- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) (DeepSeek-AI, 2026)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024) — the [MLA](../multi-head-latent-attention/) compression this stacks on top of
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) (DeepSeek-AI, 2025) — the compress-then-select sparse attention design CSA's selection stage builds on
