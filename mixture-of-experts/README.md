# Mixture of Experts (MoE)

**Lab:** Google (sparse gating, Switch Transformer); DeepSeek AI (fine-grained/shared-expert variant) · **Year:** 2017 / 2021 / 2024 · **Paper:** [Shazeer et al.](https://arxiv.org/abs/1701.06538), [Switch Transformer](https://arxiv.org/abs/2101.03961), [DeepSeekMoE](https://arxiv.org/abs/2401.06066)

## The problem

In a standard Transformer block (see [`transformer`](../transformer/),
[`rmsnorm-and-swiglu`](../rmsnorm-and-swiglu/)), every token passes
through the **same** feedforward network. Making the model more capable by
making that FFN bigger means every token pays the compute cost of the
whole bigger FFN — parameter count and compute cost are locked together.

## The idea

Replace the single FFN with **many** smaller FFNs ("experts") plus a
lightweight **router** that picks, per token, which small number of
experts actually process it:

```
       token x
          │
    ┌─────▼─────┐
    │  Router    │  → scores over all experts
    │ (Linear)   │
    └─────┬─────┘
          │ top-k
   ┌──────┼──────┬─────────┐
   ▼      ▼      ▼         ▼
expert1 expert2 expert3 ... expertN     (only top-k actually run)
   │      │      │
   └──────┴──────┘
     weighted sum (by router scores) → output
```

The router computes a score per expert (`softmax(x W_router)`), picks the
top-`k` (commonly `k=1` or `k=2`), runs only those experts, and combines
their outputs weighted by the router's scores for the chosen experts. This
decouples **total parameters** (sum over all experts — can be huge) from
**active compute per token** (only `k` experts' worth — stays small).
A model can have, say, 8× the parameters of a dense model while running
each token through only ~2× the compute of one expert.

**Load balancing.** Left alone, a router tends to collapse onto favoring a
few experts (a rich-get-richer dynamic during training), leaving others
undertrained and wasted. Every practical MoE adds an **auxiliary
load-balancing loss** that penalizes uneven routing — encouraging the
router to spread tokens roughly evenly across experts so capacity is
actually used.

**Two influential refinements this repo also groups under this topic:**

- **Switch Transformer (Google, 2021)** simplified routing to `k=1`
  (route each token to exactly *one* expert) — showing you don't need
  top-2+ for MoE to work well, which substantially simplifies
  implementation and communication cost in distributed training.
- **DeepSeekMoE (2024)** made two changes: *finer-grained experts* (many
  more, smaller experts instead of fewer, larger ones, giving the router
  more precise combinations to choose from) and *shared experts* (a small
  number of experts that process **every** token unconditionally,
  alongside the routed ones) — meant to let common/general knowledge live
  in the always-on shared experts, freeing the routed experts to
  specialize more cleanly instead of each having to also encode generic
  patterns.

## How it's actually used

MoE FFN layers replace the dense FFN in a subset or all of a model's
blocks — attention layers are typically left dense. [`mixtral`](../mixtral/)
(Mistral AI) and [`deepseek-v2`](../deepseek-v2/) (DeepSeek AI, which uses
the DeepSeekMoE refinement specifically) are the two full architectures in
this repo built around it. MoE is now a standard lever frontier labs use
to scale total model capacity without scaling inference cost proportionally.

## Tradeoffs

More total parameters to store (memory/disk cost scales with *all*
experts, not just active ones) and materially more implementation
complexity — efficient training and serving need expert-parallelism
(spreading experts across devices) and careful load balancing, both real
engineering burdens beyond a dense model. In exchange: much better
capability-per-FLOP at inference, which is why every major lab shipping a
frontier-class model now has at least one MoE variant in its lineup.

## References

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017)
- [Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) (Dai et al., 2024)
