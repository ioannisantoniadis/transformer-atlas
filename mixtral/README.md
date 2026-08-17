# Mixtral

**Lab:** Mistral AI · **Year:** 2024 · **Paper:** [Mixtral of Experts](https://arxiv.org/abs/2401.04088)

## The problem

[LLaMA](../llama/)'s block (RMSNorm + RoPE + GQA + SwiGLU) is a strong
dense-model template, but dense models tie total parameters to per-token
compute: to get more capacity, every token pays for it. Mistral AI wanted
LLaMA-level (in fact, Mistral-7B-level, their own dense model's) building
blocks with substantially more total capacity, without a proportional
inference cost increase.

## The idea

Take the [LLaMA](../llama/)-style block essentially unchanged, and replace
just the SwiGLU feedforward sublayer with a sparse
[Mixture-of-Experts](../mixture-of-experts/) layer — 8 experts per layer,
top-2 routing (each token's FFN computation is handled by 2 of the 8
experts, chosen per token by a router):

```
   Llama block:                    Mixtral block:
   RMSNorm                         RMSNorm
     │                                │
   GQA + RoPE                      GQA + RoPE      ← unchanged from Llama
     │                                │
   + residual                      + residual
   RMSNorm                         RMSNorm
     │                                │
   SwiGLU FFN                      MoE layer        ← the only swap:
     │                             (8 experts,        8x SwiGLU experts
   + residual                       top-2 routed)     + router
                                      │
                                    + residual
```

Everything else — attention mechanism, normalization, positional
encoding — is identical to a dense Mistral/LLaMA-style model. This is a
clean illustration of the point made in
[`mixture-of-experts`](../mixture-of-experts/): MoE is a swap for the FFN
sublayer specifically, orthogonal to whatever attention mechanism the rest
of the block uses.

Mixtral 8x7B has 8 experts of ~7B-parameter-equivalent size each per MoE
layer, ~47B total parameters, but only activates ~13B parameters' worth of
compute per token (attention + router, always active, plus 2 of 8
experts) — Mistral AI's reported comparison was quality competitive with
significantly larger dense models (Llama 2 70B) at a fraction of the
inference compute.

## How it's actually used

Mixtral is the reference example, in this repo, of "take a known-good
dense architecture, swap the FFN for MoE, change nothing else" — the
simplest possible way to add MoE to an existing recipe. Contrast with
[`deepseek-v2`](../deepseek-v2/), which changes both the attention
mechanism (to [MLA](../multi-head-latent-attention/)) *and* uses a more
elaborate MoE variant (DeepSeekMoE's fine-grained + shared experts) —
Mixtral is the more conservative, LLaMA-adjacent design point in this
repo's map.

## Tradeoffs

Same tradeoffs as [`mixture-of-experts`](../mixture-of-experts/) generally
(higher memory footprint for the full parameter set, routing/load-balance
complexity) applied to an otherwise-familiar dense recipe — Mixtral is
useful precisely because it isolates the MoE tradeoff without also
changing the attention mechanism, making it easy to attribute
quality/efficiency differences to the MoE swap specifically.

## References

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Jiang et al., Mistral AI, 2024)
- [Mistral 7B](https://arxiv.org/abs/2310.06825) (Jiang et al., 2023) — the dense model Mixtral's block is built from
