# Speculative Decoding

**Lab:** Google Research; DeepMind (concurrently) · **Year:** 2023 · **Paper:** [Leviathan et al.](https://arxiv.org/abs/2211.17192), [Chen et al.](https://arxiv.org/abs/2302.01318)

## The problem

Autoregressive generation (see [`gpt`](../gpt/),
[`kv-caching-and-paged-attention`](../kv-caching-and-paged-attention/)) is
fundamentally serial: producing token `t+1` requires a full forward pass
through the entire model, conditioned on token `t`, which requires token
`t-1`, and so on. Even with a KV cache eliminating redundant recomputation,
each *new* token still costs one full forward pass through every layer of
a potentially huge model — and GPUs are usually underutilized doing this,
because a single new token's forward pass doesn't have enough parallel
work to fill the hardware. Generation latency is dominated by the number
of *sequential* large-model forward passes, not by raw FLOPs.

## The idea

Use a **small, fast draft model** to propose several tokens ahead
speculatively, then verify all of them in a **single** forward pass of the
large target model (processing a short sequence in parallel is exactly
what GPUs are good at) — accepting the draft tokens that the target model
would plausibly have generated itself, and falling back to the target
model's own choice the moment they diverge.

```
draft model (small, cheap):  proposes tokens  x, y, z   (3 fast sequential steps)
target model (large, slow):  verifies x, y, z + next    (1 parallel forward pass)

if target agrees with x, y  → accept both, for the cost of ONE target pass
if target disagrees at z    → keep x, y; resample z from the target's own distribution
```

**The correctness trick: rejection sampling.** Naively "accept the draft
token if it's the target's argmax" would bias generation toward the
draft's preferences. Instead, for each proposed token `x` with draft
probability `q(x)` and target probability `p(x)`:

```
accept x with probability  min(1, p(x) / q(x))

if rejected: resample from the "leftover" distribution
             p'(x) ∝ max(0, p(x) - q(x))
```

This is the key mathematical result both papers prove: samples produced
by this accept/reject procedure are **distributed exactly as if you had
sampled from the target model alone**, token for token — speculative
decoding is a strictly lossless speedup technique, not an approximation,
despite doing most of the work with a smaller, weaker model. (This is a
meaningfully different guarantee from most of the rest of this repo's
techniques — GQA, sliding-window, MLA, etc. all trade some model quality
for efficiency; speculative decoding trades *only* wall-clock time,
because a fresh, uncorrelated sample from `p` is substituted the instant
the draft diverges.)

## How it's actually used

The draft model is typically a much smaller version of the same model
family (or a distilled model trained to mimic it), so its proposals agree
with the target often enough to be worth verifying in bulk. Expected
speedup depends entirely on the **acceptance rate** — how often the draft
and target agree — which itself depends on task, temperature, and how
well-matched the draft model is to the target. This is now standard in
production LLM serving stacks (alongside
[KV caching and PagedAttention](../kv-caching-and-paged-attention/)) as
one of the few techniques that reduces *latency* specifically, as opposed
to throughput or memory.

## Tradeoffs

Requires maintaining and running a second (draft) model, and the speedup
is entirely conditional on the draft agreeing with the target often
enough — a poorly-matched draft model can make things *slower*, not
faster, since every rejection costs the draft's wasted proposal work on
top of the target's verification pass. When acceptance rates are high
(similar model family, lower-entropy generation tasks), the wall-clock
win can be substantial with zero quality cost — the rare "free lunch" in
this repo's map, precisely because the guarantee is exact-distribution
matching, not an approximation.

## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan, Kalman & Matias, Google Research, 2023)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (Chen et al., DeepMind, 2023)
