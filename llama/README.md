# LLaMA

**Lab:** Meta AI · **Year:** 2023 · **Paper:** [LLaMA](https://arxiv.org/abs/2302.13971), [LLaMA 2](https://arxiv.org/abs/2307.09288), [LLaMA 3](https://arxiv.org/abs/2407.21783)

## The problem

By 2023, [GPT](../gpt/)'s decoder-only recipe (causal attention + FFN
blocks, autoregressive LM objective) was well established, but the
specific building-block choices from 2019 weren't the best available
anymore. LLaMA's contribution isn't a new mechanism — it's the
composition that became the de facto template for essentially every
open-weight decoder-only model that followed.

## The idea

Take the [GPT](../gpt/)-style decoder-only skeleton and swap in, block for
block, the improvements covered elsewhere in this repo:

| Component | GPT-2/3 (2019/2020) | LLaMA |
|---|---|---|
| Normalization | LayerNorm | [RMSNorm](../rmsnorm-and-swiglu/) |
| Feedforward | ReLU/GELU MLP | [SwiGLU](../rmsnorm-and-swiglu/) |
| Positional encoding | Learned absolute | [RoPE](../rotary-position-embedding/) |
| Attention | Full multi-head | [GQA](../multi-query-and-grouped-query-attention/) (from LLaMA 2 onward) |
| Norm placement | Post-norm | Pre-norm (norm before each sublayer, not after) |

```
   token ids
       │
  token embedding                (no separate positional embedding table --
       │                          position is injected inside attention via RoPE)
   ┌───▼─────────────────┐
   │ RMSNorm              │
   │      │                │
   │ GQA + RoPE            │  x N layers
   │      │                │
   │  + residual            │
   │ RMSNorm                │
   │      │                │
   │ SwiGLU FFN              │
   │      │                │
   │  + residual              │
   └───┬─────────────────┘
       │
   RMSNorm (final)
       │
   linear → vocab logits
```

None of these individual swaps is novel by the time LLaMA ships — RMSNorm
(2019), SwiGLU (2020), and RoPE (2021) all predate it. LLaMA's actual
contribution was demonstrating this *specific combination*, trained
carefully on a large, clean, well-documented data mix, produces
state-of-the-art quality at a given parameter count — and doing it with
open weights, which made this exact composition the shared starting point
for the next several years of open-weight model releases (Mistral, Qwen,
Yi, and many others follow the same block structure).

## How it's actually used

This is the block every other "full architecture" entry in this repo
modifies further: [`mixtral`](../mixtral/) replaces the SwiGLU FFN with an
[MoE](../mixture-of-experts/) layer; [`deepseek-v2`](../deepseek-v2/)
replaces GQA with [MLA](../multi-head-latent-attention/) and the FFN with
DeepSeekMoE. If you understand this file, you understand roughly 80% of
every open-weight decoder-only LLM released since 2023.

## Tradeoffs

None specific to this composition beyond the tradeoffs of its individual
pieces (see their own README files) — the point of this entry is that
these particular choices compose cleanly and each one is close to a
strict improvement over what it replaced, at negligible extra complexity.

## References

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) (Touvron et al., 2023)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (Meta AI, 2024)
