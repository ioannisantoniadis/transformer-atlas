# Sliding Window Attention

**Lab:** Mistral AI (popularized at LLM scale); the masking idea predates it (Longformer, AI2, 2020) · **Year:** 2023 · **Paper:** [Mistral 7B](https://arxiv.org/abs/2310.06825)
**Family:** Transformer

## The problem

Causal attention (see [`transformer`](../transformer/), [`gpt`](../gpt/))
still lets every position attend to *all* previous positions — cost is
O(n²) in sequence length, and at generation time the
[KV cache](../kv-caching-and-paged-attention/) grows linearly with context,
without bound.

## The idea

Restrict each position to attend only to the last `W` tokens (its "window"),
instead of the full causal history:

```
Full causal:            Sliding window (W=4):
position i sees 0..i    position i sees max(0, i-W+1)..i

j: 0 1 2 3 4 5 6 7       j: 0 1 2 3 4 5 6 7
i=0 ■                    i=0 ■
i=1 ■ ■                  i=1 ■ ■
i=2 ■ ■ ■                i=2 ■ ■ ■
i=3 ■ ■ ■ ■              i=3 ■ ■ ■ ■
i=4 ■ ■ ■ ■ ■            i=4   ■ ■ ■ ■
i=5 ■ ■ ■ ■ ■ ■          i=5     ■ ■ ■ ■
i=6 ■ ■ ■ ■ ■ ■ ■        i=6       ■ ■ ■ ■
i=7 ■ ■ ■ ■ ■ ■ ■ ■      i=7         ■ ■ ■ ■
```

Naively this looks like it caps how far information can travel — but
stacked across `L` layers, information can still propagate up to
`L × W` tokens away, the same argument CNNs use for stacked local
convolutions building a large receptive field. A 32-layer model with
`W=4096` has an effective receptive field far beyond 4096 tokens by its
final layer, even though any single layer only looks 4096 tokens back.

Implementation-wise, it's a masking change: instead of the causal
lower-triangular mask, use a banded mask that's also zero more than `W`
steps in the past. This has a direct payoff for inference too: with a
sliding window, the KV cache only ever needs to hold the last `W` tokens'
K/V (a rolling buffer), not the entire history — bounded memory regardless
of how long the generation runs.

## How it's actually used

Mistral 7B shipped this as its headline efficiency feature (`W=4096`),
combined with [GQA](../multi-query-and-grouped-query-attention/) and
[RoPE](../rotary-position-embedding/). It's a strong default when a model
is expected to mostly need *local* context — most tokens' relevant
evidence is nearby, and sliding-window trades away the (rarer) cases where
a token 50K positions back is directly relevant. Some models hedge by
mixing: most layers use sliding-window, a few layers keep full attention
(a pattern also used for efficient long-context designs elsewhere in the
field). Contrast with [Star Attention](../star-attention/), which restricts
attention differently — splitting *context* into independently-processed
blocks with a shared global anchor, rather than restricting each *query*
to a trailing window.

## Tradeoffs

A token outside the window is invisible to that layer directly — the
model must rely on multi-layer propagation to connect distant information,
which is a real (if often small) capability loss versus full attention,
especially for tasks needing precise long-range recall (e.g. "what was the
exact number mentioned 10,000 tokens ago"). In exchange: strictly bounded
per-step compute and, more importantly, bounded KV-cache memory —
independent of total generation length.

## References

- [Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) (Beltagy et al., 2020) — earlier local-attention-window formulation
- [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) (Child et al., 2019, OpenAI) — earlier sparse/local attention patterns
