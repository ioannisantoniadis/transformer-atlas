# Gemma

**Lab:** Google DeepMind · **Year:** 2024 · **Paper:** [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295), [Gemma 2](https://arxiv.org/abs/2408.00118)

## The problem

Gemma is Google DeepMind's open-weight distillation of techniques from
their (closed) Gemini line. Like [Qwen](../qwen/), it's mostly the by-now-
standard [LLaMA](../llama/)-style template — the interesting part is a
different set of small, deliberate deviations, this time aimed
specifically at **training stability and inference efficiency at Google's
scale**, rather than at a single quality metric.

## The idea

Four specific choices distinguish Gemma's block from the LLaMA template:

**1. Tied, scaled embeddings.** The input embedding matrix and the final
output (unembedding) projection share the same weights (`W_embed` and
`W_head` are the same matrix, transposed) — halving the parameter cost of
what is, at Gemma's ~256K-token vocabulary, a very large matrix. Sharing
these weights couples the scale of the embedding lookup to the scale
needed for good output logits, which don't naturally match — Gemma
compensates by multiplying the embedding output by `sqrt(d_model)` right
after the lookup, rescaling it into a range better suited to the residual
stream it's about to enter.

**2. GeGLU instead of SwiGLU.** Same gated-FFN shape as
[SwiGLU](../rmsnorm-and-swiglu/) (`(gate(x W_gate) ⊙ (x W_up)) W_down`),
but the gating activation is **GELU** instead of **SiLU/Swish** — a small,
easily-missed substitution within the same GLU family the
[GLU Variants](https://arxiv.org/abs/2002.05202) paper originally compared
side by side.

**3. Logit soft-capping (Gemma 2).** Both attention logits (before
softmax) and the final output logits (before the softmax that produces
next-token probabilities) are passed through a bounded, smooth cap:

```
capped_logits = cap * tanh(logits / cap)
```

This keeps logits from growing unboundedly large during training — a
known contributor to training instability at scale — while staying smooth
and differentiable everywhere, unlike a hard clip.

**4. Alternating local/global attention (Gemma 2).** Instead of every
layer using the same attention pattern, Gemma 2 **alternates**, layer by
layer, between [sliding-window attention](../sliding-window-attention/)
(cheap, local) and full causal attention (expensive, global):

```
layer 1: sliding window     layer 2: full attention
layer 3: sliding window     layer 4: full attention
...
```

This is a direct, explicit trade on the efficiency/capability spectrum
this repo's attention-variant entries live on: get most of sliding-
window's efficiency, most of the time, while still giving the model
regular, periodic access to true full-sequence attention — rather than
committing entirely to one pattern for the whole model the way
[Mistral](../sliding-window-attention/) or a standard
[LLaMA](../llama/) model does.

## How it's actually used

Gemma is a useful contrast case to [Qwen](../qwen/) in this repo's map:
both take the same LLaMA-derived starting point and diverge in small,
independently-justified ways, but toward different goals — Qwen's QKV
bias targets extrapolation/stability in their specific training regime;
Gemma's soft-capping and local/global alternation target training
stability and inference cost at Google's scale, plus a parameter-saving
weight-tying choice suited to its unusually large vocabulary.

## Tradeoffs

Weight tying saves real parameters at large vocabulary sizes but couples
two roles (input representation, output prediction) that don't have to
want the same weights — the `sqrt(d_model)` rescaling is a fix for a
problem this choice itself introduces. Logit soft-capping trades a small
amount of representational sharpness (extremely confident logits get
compressed) for materially better training stability. Alternating
local/global attention is a direct middle ground on the same
efficiency/capability axis every attention-variant entry in this repo
sits on, rather than a new axis of its own.

## References

- [Gemma: Open Models Based on Gemini Research and Technology](https://arxiv.org/abs/2403.08295) (Gemma Team, Google DeepMind, 2024)
- [Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) (Gemma Team, Google DeepMind, 2024)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020) — where GeGLU and SwiGLU are compared side by side
