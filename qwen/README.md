# Qwen

**Lab:** Alibaba · **Year:** 2023–2024 · **Paper:** [Qwen Technical Report](https://arxiv.org/abs/2309.16609), [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671)
**Family:** Transformer

## The problem

By the time Qwen shipped, the [LLaMA](../llama/)-style block (RMSNorm +
RoPE + GQA + SwiGLU) was already the dominant open-weight template. Qwen's
architecture is, deliberately, mostly *that template* — its interest for
this repo is less "a new mechanism" and more "a case study in which small,
specific deviations a lab chooses to keep, and why," which is exactly the
kind of decision that tends to get lost once a recipe becomes the default
everyone copies without re-deriving.

## The idea

Qwen (v1) is a [LLaMA](../llama/)-style block with one notable, deliberate
departure: **it keeps bias terms on the attention Q/K/V projections**,
where LLaMA and most contemporaries had already dropped all linear-layer
biases (a common simplification once RMSNorm made bias terms feel
increasingly redundant elsewhere in the network).

```
   Llama-style attention proj:      Qwen-style attention proj:
   q = x @ W_q                      q = x @ W_q + b_q
   k = x @ W_k                      k = x @ W_k + b_k
   v = x @ W_v                      v = x @ W_v + b_v
```

The Qwen team's stated motivation was empirical: they found QKV bias gave
a measurable boost to extrapolation and training stability in their
setup, a small but deliberate exception to "no biases anywhere," which by
that point had become close to unquestioned folklore in the LLaMA-derived
lineage. It's a useful reminder that "everyone drops biases now" is a
convention, not a law — worth re-testing rather than copying blindly, and
Qwen is the concrete example, in this repo's map, of a lab doing exactly
that and keeping the result that worked for them.

**Qwen2** brought the architecture the rest of the way into the
mainstream LLaMA-2-era template: adding
[GQA](../multi-query-and-grouped-query-attention/) (dropped for the
smallest model sizes, kept for larger ones) alongside the retained QKV
bias, plus an extended context length via
[RoPE-scaling-style](../yarn-and-rope-scaling/) techniques for the longer-
context model variants.

## How it's actually used

Reading this file after [`llama`](../llama/) should feel almost like a
diff, not a rewrite — that's the point. This is the entry in the repo's
map that best illustrates how much the field's "default block" has
converged: most differences between contemporary open-weight labs' base
architectures, Qwen included, now show up as small, specific, individually
justified deviations from a shared template, not as fundamentally
different designs.

## Tradeoffs

QKV bias adds a small number of extra parameters (one bias vector per
projection, negligible relative to the weight matrices) and a small
extra compute cost (an addition, not a multiply) for, per Qwen's reported
results, a genuine quality/stability benefit in their training setup —
not a universal law that every model should re-add bias terms, but a
reminder that "the standard recipe" is a starting point subject to
empirical revision by whoever is actually running the ablations.

## References

- [Qwen Technical Report](https://arxiv.org/abs/2309.16609) (Bai et al., Alibaba, 2023)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) (Yang et al., Alibaba, 2024)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023) — the template Qwen modifies
