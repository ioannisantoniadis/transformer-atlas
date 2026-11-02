# Nemotron-Labs-Diffusion

**Lab:** NVIDIA · **Year:** 2026 · **Paper:** [Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding](https://arxiv.org/abs/2607.05722)
**Family:** Transformer

## The problem

Two ways to make an LLM emit more than one token per forward pass already
exist, and each has a catch:

- [Speculative decoding](../speculative-decoding/) keeps the target model's
  outputs but needs a **second, separate draft model**, which must be
  trained, hosted, and kept matched to the target.
- **Masked-diffusion LMs** denoise a whole block of masked tokens in
  parallel, so one model drafts many tokens at once, but they have
  historically trailed autoregressive (AR) models on quality, and their
  bidirectional attention doesn't fit the AR serving stack (KV cache,
  causal kernels).

Nemotron-Labs-Diffusion's answer is to train a single decoder that is
*both*, then choose at inference time which way to run it.

## The idea

Take a pretrained AR model (the released models start from Ministral3 base
checkpoints) and continue training it on two losses at once:

```
L(θ) = L_AR(θ) + α · L_diff(θ)          α = 0.3

L_AR   = −Σ_i log p_θ(x_i | x_<i)                         next-token prediction
L_diff = −E_t [ (1/t) Σ_b log p_θ(x^b | x̃_t^b, x^<b) ]    block diffusion
```

`L_diff` is the [Block Diffusion](https://arxiv.org/abs/2503.09573)
objective: split the sequence into contiguous blocks, corrupt only the
current block `b` (each token replaced by a mask with probability `t`),
and predict the clean block from the corrupted block `x̃_t^b` plus the
**clean** prefix `x^<b`. Both losses come out of one forward pass over a
dual-stream input, a noisy copy and a clean copy of the sequence side by
side, with this attention mask:

```
                  keys:  clean stream            noisy stream
queries:
  clean stream           strictly causal          ✗ never
  noisy stream, block b  clean blocks < b only    same block b only (bidirectional)
```

The paper's change to prior block-diffusion training is the strictly
causal clean stream. It makes the clean-stream outputs a valid AR model,
so `L_AR` can be read off the same pass without leaking labels. The
diffusion pathway still sees its block bidirectionally.

The same weights then decode in three modes:

| Mode | Per step | Tokens per step |
|---|---|---|
| **AR** | one causal forward, sample `x_i ~ p_θ(· \| x_<i)` | 1 |
| **Diffusion** | denoise a block of masks; commit positions whose confidence clears a threshold; repeat until the block is full | ≥ 1 |
| **Self-speculation** | forward 1 (diffusion pathway): append `k` masks, denoise them in parallel as a draft. Forward 2 (AR pathway): one causal pass over the draft. Accept the longest prefix where draft = AR argmax, plus the AR token at the first mismatch | 1 to `k+1` per 2 forwards |

```
prefix ──► [M M M M] ──diffusion pass──► [a b c d]  draft
prefix ──► [a b c d] ──causal AR pass──► [a b x e]  AR's own next-token choices
                                           ✓ ✓ ✗
keep: a b, then x from the AR pass  →  3 tokens for 2 forwards
```

Self-speculation is [speculative decoding](../speculative-decoding/) with
the draft model folded into the target. Acceptance is greedy exact match
against the AR argmax, not the rejection-sampling rule from that entry,
so the guarantee is "same tokens as greedy AR decoding" rather than "same
distribution as sampling." The paper also describes a quadratic variant
that drafts and verifies in a single forward with a structured mask.

`implementation.py` trains a two-layer toy decoder with this loss and mask
on a small synthetic language. It checks that the clean stream never sees a
future token, and that self-speculation produces exactly the greedy AR
output while committing several tokens per draft-and-verify cycle.

## How it's actually used

Released as open weights at 3B, 8B and 14B (base and instruct, plus an 8B
vision-language variant). The training recipe in the paper is 1T tokens of
pure-AR continued pretraining (`α = 0`), then 300B tokens of joint training
at `α = 0.3`, then SFT on 45B tokens, also at `α = 0.3`. Evaluations use
block size 32. For self-speculation, the drafter gets a small LoRA adapter
(about 0.4% of parameters, on the attention output projection) trained only
on the accepted prefix and the first rejected position.

Reported numbers for the 8B instruct model, averaged over 10 tasks: AR mode
63.6, diffusion mode 63.2 at 2.57 tokens per forward, and linear
self-speculation 62.8 at 5.99 tokens per forward, against Qwen3-8B AR at
62.75. The abstract summarizes this as "6× more tokens per forward than
Qwen3-8B with comparable accuracy," and about 4× throughput on SPEED-Bench
with SGLang on a GB200. The body reports 3.3× over AR with the default
kernel and 3.97× with an optimized one. Note that tokens per forward counts
the acceptance length; linear self-speculation pays two forwards per cycle.

The intended deployment split: self-speculation or diffusion when latency
matters at low concurrency, plain AR at high concurrency, where the GPU is
already saturated and drafting buys little.

## Tradeoffs

- **It's a pretraining-time choice.** Unlike the entries in section 9 of
  [`MAP.md`](../MAP.md), none of this can be bolted onto an existing AR
  checkpoint at serving time. The model has to be trained with the joint
  objective, which is why it sits with the full model architectures.
- **Greedy-match verification.** Self-speculation reproduces greedy AR
  outputs, not the sampled AR distribution. In practice the paper still
  sees small accuracy gaps against AR mode, which it attributes to kernel
  mismatches between one-token decoding and multi-token prefill.
- **Two forwards per cycle.** Linear self-speculation only commits a
  contiguous accepted prefix and needs a draft pass and a verify pass, so
  real speedup is roughly half the headline tokens-per-forward figure.
- **Diffusion mode is sensitive to block length.** The paper finds larger
  blocks hurt accuracy and that confidence-threshold unmasking is "far
  from optimal" against an oracle ceiling. Diffusion training is also
  higher-variance than AR training because of the random masking.
- **Adoption.** As of late 2026 it's one lab's model family; no other
  model has been found shipping the same joint-objective recipe.

## References

- [Nemotron-Labs-Diffusion](https://arxiv.org/abs/2607.05722) (Fu et al., 2026) — primary paper
- [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) (Arriola et al., 2025) — the block-diffusion objective and mask it builds on
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2023) — the two-model scheme self-speculation replaces
