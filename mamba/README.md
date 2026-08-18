# Mamba (Selective State Spaces)

**Lab:** Albert Gu (CMU), Tri Dao (Princeton) · **Year:** 2023 · **Paper:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
**Family:** State-Space

## The problem

[`s4-and-structured-state-spaces`](../s4-and-structured-state-spaces/)'s
`A`, `B`, `C` are fixed for the whole sequence — the same dynamics apply
to every token regardless of content. That's exactly what makes S4's
parallel-convolution trick work, but it also means the model has no way
to *decide*, per token, "this one matters, write it into the state
strongly" versus "this one is filler, barely touch the state." Content-
independent dynamics are a real ceiling on tasks that need selective
attention to specific tokens — the same class of task where softmax
attention's per-token, content-dependent weights naturally win.

## The idea

Make `B`, `C`, and the discretization step `Δ` **functions of the input**
`x_t` instead of fixed parameters — "selectivity." The model now decides,
per token, how much of it to write into the state and how fast to forget
what's already there, much closer to attention's content-dependent
behavior:

```
S4 (fixed):        Ā, B̄, C same for every t
Mamba (selective):  Δ_t, B_t, C_t = functions of x_t   -- input-dependent per step
```

The cost: input-dependent dynamics break S4's global-convolution trick
(the kernel is no longer fixed, so there's no single kernel to FFT). Mamba
recovers training-time parallelism a different way — a hardware-aware
**parallel scan** (a parallel-prefix-sum-style algorithm) that computes
the selective recurrence in `O(log n)` parallel depth instead of `n`
sequential steps, without ever materializing the full state history in
slow memory. This scan-based algorithm was itself a real engineering
contribution, not just the selectivity idea — [`mamba-2`](../mamba-2/)
picks up from here, showing a further-constrained special case of this
same selective recurrence can skip the scan entirely and run as a matmul.

## How it's actually used

Mamba is the backbone of a growing set of production and near-production
LLMs (Codestral Mamba, Falcon Mamba) as a pure-SSM alternative to
attention-only Transformers, and — more commonly in frontier labs' actual
shipped models — as one ingredient in a hybrid stack rather than the
whole architecture; see [`jamba-and-hybrid-architectures`](../jamba-and-hybrid-architectures/)
for how AI21 Labs interleaves it with a minority of attention layers to
get most of attention's precise retrieval at a fraction of its memory
cost. It's also the direct ancestor of two other lineages in this map:
[`mamba-2`](../mamba-2/)'s structured state-space duality result, and --
arrived at independently from the attention side -- the delta-rule
lineage in [`gated-deltanet-and-kda`](../gated-deltanet-and-kda/).

## Tradeoffs

Selectivity gets back much of what fixed dynamics gave up, but the state
is still a fixed-size summary, not an ever-growing addressable cache —
the same fundamental capacity ceiling every entry in this branch and in
[`linear-attention`](../linear-attention/) shares. Precise recall of one
specific fact from deep in a very long context is still attention's
strength; Mamba's advantage is everything about long-range context being
*cheap*, not about it being *exact*. The selective scan also isn't free
to implement well -- it needs a custom hardware-aware kernel, unlike S4's
single fixed convolution.

## References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu, Dao, 2023)
- [`mamba-2`](../mamba-2/) — the structured state-space duality result built directly on this
