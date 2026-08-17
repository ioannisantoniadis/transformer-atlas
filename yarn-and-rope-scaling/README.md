# YaRN and RoPE Scaling

**Lab:** Position Interpolation — Meta AI; NTK-aware scaling — community (u/bloc97 et al., unpublished blog origins); YaRN — Nous Research / EleutherAI / others · **Year:** 2023 · **Paper:** [Position Interpolation](https://arxiv.org/abs/2306.15595), [YaRN](https://arxiv.org/abs/2309.00071)

## The problem

[RoPE](../rotary-position-embedding/) generalizes to unseen positions
better than learned absolute embeddings, but not perfectly: a model
trained with sequences up to length `L_train` still degrades noticeably
past that length, because attention scores start depending on rotation
angles the model never saw during training. Frontier labs routinely want
to take an already-trained model and extend its usable context far beyond
`L_train` (e.g. 4K → 128K) without retraining from scratch. This entry
covers three increasingly refined ways to stretch RoPE past its training
length, each fixing a shortcoming of the last.

## The idea

**1. Position Interpolation (PI).** The simplest fix: linearly compress
positions before feeding them to RoPE, so the *largest* position at the
new target length maps to the *largest* position seen in training.

```
m_scaled = m * (L_train / L_target)
```

Cheap, and it works, but it compresses *every* frequency equally — including
the high-frequency (fast-rotating) dimension pairs that distinguish
*nearby* tokens from each other. Squeezing those hurts the model's ability
to tell adjacent tokens apart, a real quality cost even within the
original training length.

**2. NTK-aware scaling.** Instead of scaling positions, scale RoPE's
`base` (the constant controlling how fast each frequency rotates — see
[`rotary-position-embedding`](../rotary-position-embedding/)):

```
base_scaled = base * (L_target / L_train) ^ (d / (d - 2))
```

This stretches low frequencies (long-wavelength dimension pairs, which
encode coarse, long-range position) much more than high frequencies
(short-wavelength pairs, encoding fine, local position) — leaving nearby-
token discrimination close to untouched while still extending how far the
coarse position signal reaches. Named for its resemblance to Neural
Tangent Kernel theory about how networks learn different frequencies.

**3. YaRN (Yet another RoPE extensioN method).** Observes that neither
extreme is quite right: very high frequencies (short wavelength relative
to context) should be left alone (extrapolate — the model has seen many
full periods of that rotation already, more of the same is fine),
very low frequencies (long wavelength relative to context) should be
interpolated (PI-style — the model has seen less than one full period of
that rotation over `L_train`, so it needs compressing), and frequencies in
between get a smooth blend. YaRN computes a per-dimension **ramp
function** `γ_i ∈ [0, 1]` based on each dimension pair's wavelength
relative to the target length, and mixes:

```
angle_i = γ_i * (original angle) + (1 - γ_i) * (PI-interpolated angle)
```

YaRN also adds a small **attention temperature** adjustment (scaling
attention logits slightly) to compensate for the change in the
distribution of angles hitting softmax — an empirical correction the
paper found necessary for best results, on top of the frequency-ramp idea.

```
   wavelength vs target context length:
   short (high-freq dims)  ─► γ≈1 ─► keep original angle (extrapolate)
   long  (low-freq dims)   ─► γ≈0 ─► use PI-interpolated angle
   in between              ─► smooth ramp between the two
```

## How it's actually used

This is a **post-hoc, often training-free (or light-finetune) context
extension technique** — applied to an already-trained RoPE model to serve
it at longer context than it was trained on, not a change made from
scratch. YaRN specifically became a common choice for community and
production long-context extensions of LLaMA-family models; NTK-aware
scaling alone is a common lighter-weight fallback when a full YaRN
implementation isn't available.

## Tradeoffs

All three techniques trade some quality (typically small, growing with
how aggressively you extend past `L_train`) for usable context length far
beyond training, without the cost of retraining from scratch. PI is
simplest but costs the most quality at a given extension factor; NTK-aware
scaling is a strict improvement on PI at similar implementation
complexity; YaRN costs the most implementation complexity (a
per-dimension ramp and a temperature correction, both with a few
hyperparameters) but is generally the most quality-preserving of the
three at large extension factors, per its own reported results.

## References

- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) (Chen et al., Meta AI, 2023)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) (Peng et al., 2023)
- [NTK-Aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) — the original community writeup NTK-aware scaling comes from
