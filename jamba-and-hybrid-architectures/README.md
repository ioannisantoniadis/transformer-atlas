# Jamba & Hybrid Architectures

**Lab:** AI21 Labs · **Year:** 2024 · **Paper:** [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887), [Jamba-1.5](https://arxiv.org/abs/2408.12570)
**Family:** Hybrid

## The problem

The two branches of this map make opposite trades. Attention
(this repo's Transformer branch) keeps every token individually
addressable via a KV cache — precise, but [`kv-caching-and-paged-attention`](../kv-caching-and-paged-attention/)
already covers why that cache is the dominant memory-bandwidth cost at
long context: it grows linearly with sequence length, unavoidably.
[`mamba`](../mamba/)'s state is fixed-size and
cheap regardless of context length, but it's a *compressed* summary — the
same fixed-capacity ceiling every entry in the state-space branch shares,
which shows up as *reduced precision* on tasks that need to recall one
specific fact from deep in a long context. Neither branch alone gets both
properties at once.

## The idea

Don't choose a branch — merge them inside one model. Interleave a small
minority of full-attention layers among many state-space layers, so the
(few) attention layers handle precise retrieval when it's actually needed
and the (many) state-space layers handle cheap long-range context
compression the rest of the time. Jamba's released configuration:
**1 attention layer for every 7 Mamba layers**, with Mixture-of-Experts
(a cross-cutting technique — see [`mixture-of-experts`](../mixture-of-experts/))
applied to every other layer's feedforward block, independently of which
sequence-mixing mechanism that layer uses:

```
1 block = 8 layers: [Mamba, Mamba, Mamba, Attention, Mamba, Mamba, Mamba, Mamba]
                                        ^
                              1-in-8 layers keep a real KV cache;
                              the other 7 keep a fixed-size state
```

Because only 1 layer in 8 needs a KV cache at all, the memory cost scales
with `(attention layers) × O(context length)` instead of
`(all layers) × O(context length)` — a large constant-factor reduction in
exactly the cost [`kv-caching-and-paged-attention`](../kv-caching-and-paged-attention/)
identifies as the bottleneck, while keeping just enough attention layers
to catch what the state-space layers alone would compress away.

## How it's actually used

At a reported 256K-token context, Jamba's attention cache needs roughly
4GB versus roughly 32GB for an all-attention model of comparable size —
this repo's implementation below works through that specific memory-vs-
context-length comparison for an all-attention, all-Mamba, and 1:7-hybrid
model directly, plus a toy retrieval test showing the hybrid recovers a
specific planted fact about as reliably as all-attention while paying
close to all-Mamba's memory cost. This interleaving pattern — a small
attention minority, a state-space majority, plus independent MoE routing
— has become the standard shape for the "hybrid" branch generally
(Zamba, Griffin/RecurrentGemma follow the same idea with different
ratios and specific mechanisms), not just this one model family.

## Tradeoffs

A hybrid is strictly more complex to implement and serve than either pure
branch — two different sequence-mixing kernels in one model, two
different caching strategies to manage at inference time. The interleave
ratio itself is a real hyperparameter with no universal answer: too few
attention layers and precise-retrieval tasks degrade toward pure-Mamba
behavior; too many and the memory savings that motivated the hybrid in
the first place shrink back toward pure-attention cost.

## References

- [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) (AI21 Labs, 2024)
- [Jamba-1.5: Hybrid Transformer-Mamba Models at Scale](https://arxiv.org/abs/2408.12570) (AI21 Labs, 2024)
