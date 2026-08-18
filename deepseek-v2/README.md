# DeepSeek-V2

**Lab:** DeepSeek AI · **Year:** 2024 · **Paper:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434), [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
**Family:** Transformer

## The problem

[Mixtral](../mixtral/) showed one way to add MoE capacity to a
LLaMA-style block: keep attention (GQA) unchanged, swap only the FFN.
DeepSeek-V2 pushes on both halves of the block at once — attention *and*
feedforward — specifically optimizing for cheap, high-throughput serving
at very large scale, which was the stated design goal (the paper's title
literally leads with "economical").

## The idea

Combine the two most involved entries in this repo into one block:

```
   DeepSeek-V2 block:
   RMSNorm
     │
   Multi-Head Latent Attention   ← replaces GQA: KV cache compressed into
     │                              a small shared latent + decoupled RoPE
                                     (see multi-head-latent-attention/)
   + residual
   RMSNorm
     │
   DeepSeekMoE                   ← replaces a plain top-k MoE layer:
     │                              many fine-grained experts + always-on
                                     shared experts (see mixture-of-experts/)
   + residual
```

**Attention: [Multi-Head Latent Attention](../multi-head-latent-attention/).**
Instead of GQA's "fewer, shared KV heads," MLA compresses every token's
K/V into one small low-rank latent vector, reconstructed per-head via
learned up-projections, with a small decoupled RoPE slice handling
position. The result: a KV cache far smaller than even aggressive GQA,
without GQA's head-capacity tradeoff.

**Feedforward: [DeepSeekMoE](../mixture-of-experts/).** Instead of
Mixtral's "8 experts, pick top 2," DeepSeekMoE uses *many more, smaller*
experts (finer-grained routing choices) plus a handful of experts that run
on **every** token unconditionally (shared experts) alongside the routed
ones — the idea being that generic, always-useful computation lives in the
shared experts, letting the routed experts specialize more sharply instead
of each needing to also encode common patterns.

Both changes point the same direction: **spend more design effort making
inference cheap at large scale**, rather than just making the model
bigger. This is consistent with DeepSeek-V2/V3's broader reputation — very
large total parameter counts (DeepSeek-V3: 671B total, ~37B active per
token) made tractable specifically because of MLA's cache compression and
DeepSeekMoE's fine-grained sparsity.

## How it's actually used

This is the most "everything turned on" entry in this repo's full-
architecture section — useful precisely because reading it after
[`llama`](../llama/) and [`mixtral`](../mixtral/) shows how independent
these two swaps (attention mechanism, FFN sparsity strategy) really are:
nothing about MLA depends on DeepSeekMoE or vice versa, they're just both
present in this one model family.

## Tradeoffs

Compounds the tradeoffs of both components: MLA's extra architectural
complexity (compression/decompression path, decoupled RoPE) plus
DeepSeekMoE's routing/load-balancing complexity across many more experts
than a typical MoE. DeepSeek-AI's published results argue this
complexity buys a genuinely better efficiency frontier — very large total
capacity at inference cost comparable to much smaller dense models — which
is the whole bet behind treating "make attention and FFN both
cache/compute-efficient" as worth the added engineering.

## References

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024)
- [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) (Dai et al., 2024)
