# Quantization for Inference (GPTQ / AWQ)

**Lab:** GPTQ — IST Austria; AWQ — MIT / Song Han's lab · **Year:** 2022 / 2023 · **Paper:** [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978)
**Family:** Cross-cutting — applies to any backbone (Transformer, State-Space, or Hybrid).

## The problem

At inference, [KV caching](../kv-caching-and-paged-attention/) makes
generation memory-bandwidth-bound, not compute-bound (see that entry): the
GPU spends more time *moving* weights and cached activations than
*multiplying* them. A 70B-parameter model stored in 16-bit floats needs
140GB just for weights, all of which has to stream through memory
bandwidth for every forward pass. Shrinking the number of *bits per
parameter* directly shrinks that bottleneck — quantization is the lever
for that, applied specifically for serving an already-trained model, not
for training.

## The idea

Store each weight using far fewer bits (commonly 4 or 8, down from 16)
than it was trained with, via a per-group **scale** (and often a
**zero-point**) that maps a small integer range back to the original
floating-point range:

```
quantize:    w_int  = round((w_float - zero_point) / scale)   -- clipped to, e.g., [-8, 7] for 4-bit
dequantize:  w_approx = w_int * scale + zero_point
```

The scale/zero-point are typically computed **per output channel** (or
per small group of weights) rather than once for the whole matrix, since
a single global scale would badly under-represent both very small and
very large weights in the same tensor. This alone (round-to-nearest,
per-channel scale) is the baseline; GPTQ and AWQ each improve on it in a
different way:

**GPTQ (2022).** Round-to-nearest quantizes each weight independently,
ignoring that rounding one weight changes what the *optimal* rounding of
its neighbors would have been (since they interact through the layer's
output). GPTQ quantizes weights one at a time and, after each one, updates
the *remaining* unquantized weights to compensate for the error just
introduced — an efficient, layer-wise adaptation of a second-order
optimal-brain-quantization idea, using calibration data to know which
directions in weight space matter most for preserving the layer's output.

**AWQ (Activation-aware Weight Quantization, 2023).** A different
observation: not all weight channels matter equally — channels that
tend to multiply against **large-magnitude activations** contribute more
to the layer's output and are more damaging to quantize coarsely. AWQ
uses a small amount of calibration data to identify these salient
channels, then **rescales** them (multiply the weight channel up, divide
the corresponding activation down by the same factor — a no-op
mathematically, done in full precision) so they land in a numeric range
the low-bit quantization grid represents more precisely, before
quantizing everything uniformly.

```
   per-channel scale s_c chosen so that:
   quantize(w_c * s_c) is more accurate for salient channel c,
   and multiplying the matching activation by 1/s_c cancels it out exactly
```

## How it's actually used

Quantization is applied **after** training (post-training quantization,
PTQ) to a model that was trained in full or mixed precision — a separate
step from anything covered in this repo's architecture entries, run once
before deployment. 4-bit weight quantization (via GPTQ, AWQ, or similar)
is now a standard option in most open-weight model serving toolchains,
letting a model that needs 2 high-end GPUs in 16-bit run on 1 in 4-bit,
at a typically small, measurable quality cost.

## Tradeoffs

Lower bit-width means real quality loss — the question is how much, and
both GPTQ and AWQ exist specifically to push that cost down at a given
bit-width versus naive round-to-nearest. The gains are substantial and
close to unconditional for memory-bandwidth-bound serving: 4-bit weights
mean ~4x less data to move per forward pass versus 16-bit, directly
addressing the bottleneck [KV caching](../kv-caching-and-paged-attention/)
identifies. Quantizing activations (not just weights) can compound the
memory win further but is generally harder to do without more quality
loss, since activation distributions are less well-behaved and vary with
input at inference time.

## References

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (Frantar et al., 2022)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (Lin et al., 2023)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022) — earlier, related 8-bit quantization approach
