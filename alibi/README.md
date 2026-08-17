# ALiBi (Attention with Linear Biases)

**Lab:** University of Washington / Facebook AI Research (Press, Smith, Lewis) · **Year:** 2021 · **Paper:** [Train Short, Test Long](https://arxiv.org/abs/2108.12409)

## The problem

Like [RoPE](../rotary-position-embedding/), ALiBi is a response to the two
weaknesses of the original [Transformer](../transformer/)'s sinusoidal (or
GPT's learned) positional encoding: it's added once at the input, and it
generalizes poorly to sequence lengths longer than what the model was
trained on. ALiBi takes the most minimal possible approach to fixing both.

## The idea

Add **no** positional information to the embeddings at all. Instead, bias
the raw attention scores directly, penalizing a query/key pair in
proportion to how far apart they are:

```
score_ij = q_i · k_j / sqrt(d_k)  -  m * (i - j)         (for j <= i, causal)
```

`m` is a fixed (not learned), head-specific slope, and `(i - j)` is just
the distance between positions — no sinusoids, no rotation, no embedding
table. Closer keys get a smaller penalty, distant keys get a bigger one;
softmax then naturally down-weights far-away tokens more, similar in
spirit to a soft, learnable-free version of
[sliding-window attention](../sliding-window-attention/) baked directly
into the score rather than a hard mask.

Different heads get different slopes `m` (a fixed geometric sequence, e.g.
`1/2, 1/4, 1/8, ...` for 8 heads), so some heads decay sharply (behave
almost like a local window) while others decay gently (retain more
long-range sensitivity) — the model gets a spread of effective "attention
horizons" for free, set by a fixed formula rather than learned or manually
chosen per head.

```
        distance |i-j|:   0    1    2    3    4    5
   head with large m:   0.0  -0.5  -1.0  -1.5  -2.0  -2.5   (steep decay, ~local)
   head with small m:   0.0  -0.06 -0.13 -0.19 -0.25 -0.31   (gentle decay, ~global)
```

**Why this extrapolates well.** Because the bias is a simple linear
function of raw distance `(i - j)`, it's defined identically for *any*
distance, seen in training or not — position 50,000 minus position 1
computes the same way as position 5 minus position 1. There's no
embedding table to run out of entries in and no rotation frequency scheme
whose behavior was only validated up to a specific length. The original
paper's headline result: a model trained on sequences of length 1024
retains strong performance evaluated on sequences of length 2048+, while
sinusoidal and learned-embedding baselines degrade sharply past their
training length.

## How it's actually used

ALiBi shipped in BLOOM (BigScience) and influenced later length-
extrapolation work, but it lost ground to RoPE as the default choice in
most subsequent frontier open-weight models (LLaMA, Mistral, DeepSeek all
use RoPE) — RoPE tends to give better absolute quality at trained lengths,
even though ALiBi's extrapolation story is arguably cleaner. It remains
the simplest baseline to reach for when extrapolation robustness matters
more than squeezing out the last bit of in-distribution quality, and is
worth understanding as the "minimalist" point in the positional-encoding
design space, opposite RoPE's "rotate everything" approach.

## Tradeoffs

Extremely cheap (no extra parameters, no rotation, just an addition to
the score matrix) and strong length extrapolation. In exchange, it gives
the model a strong built-in *recency* prior — the penalty always makes
farther tokens strictly less attractive regardless of content — which is
a reasonable default for language but can work against tasks where a
distant token is highly relevant and a nearby one isn't (RoPE, by
contrast, only affects *phase*, not a monotonic penalty, so it doesn't
bias attention toward recency in the same structural way).

## References

- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) (Press, Smith & Lewis, 2021)
- [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) (BigScience, 2022) — large-scale ALiBi adopter
