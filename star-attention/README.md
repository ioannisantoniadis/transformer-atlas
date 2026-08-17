# Star Attention

**Lab:** NVIDIA · **Year:** 2024 · **Paper:** [Star Attention: Efficient LLM Inference over Long Sequences](https://arxiv.org/abs/2411.17116)

## The problem

[Sliding window](../sliding-window-attention/) and [linear
attention](../linear-attention/) change the attention *computation* to
make one machine cheaper. Star Attention targets a different bottleneck:
running inference over a **very long context** (hundreds of thousands of
tokens) when the context is processed across **multiple machines/devices**.
Splitting a long sequence across hosts for parallel processing is
attractive, but naively it means every host needs every other host's keys
and values to compute correct global attention — an expensive
all-to-all communication pattern that gets worse as context grows.

## The idea

A two-phase scheme that trades a controlled amount of accuracy for a large
cut in cross-host communication:

**Phase 1 — context encoding (distributed, block-local).** Split the long
context into blocks and distribute them across hosts. Every host also gets
a copy of a small shared **anchor block** (typically the first block of the
context) prepended to its own block. Each host computes attention *only*
within `anchor ∪ its own block` — never looking at any other host's block
directly. This is embarrassingly parallel: no cross-host communication
during this phase at all. The anchor block acts as a shared point of
reference so every block's local representations are computed with at
least *some* global context, rather than being totally isolated.

```
Host 1: [anchor][block 1]     ─┐
Host 2: [anchor][block 2]      │  computed independently,
Host 3: [anchor][block 3]      │  no communication between hosts
Host 4: [anchor][block 4]     ─┘
```

**Phase 2 — query encoding & generation (distributed, then merged).** A
query token needs attention scores against the *entire* context, not just
one block. Each host computes its **local, unnormalized** partial
softmax numerator and denominator against its own block's keys/values
(the same accumulation flash-attention uses internally — see
[`flash-attention`](../flash-attention/)) and sends just those small
partial results to one designated host, which merges them with the
standard online-softmax merge rule into the correct global attention
output. This step is **mathematically exact** given the phase-1 KV — the
approximation in the whole scheme lives entirely in phase 1 (blocks never
saw each other's content while building their local KV), not in how the
partial results get combined.

```
query
  │
  ├──► Host 1: partial (numerator₁, denominator₁) against block 1's KV ─┐
  ├──► Host 2: partial (numerator₂, denominator₂) against block 2's KV  │
  ├──► Host 3: partial (numerator₃, denominator₃) against block 3's KV  ├─► merge (online-softmax rule) ─► exact output
  └──► Host 4: partial (numerator₄, denominator₄) against block 4's KV ─┘
```

Communication per generation step is now O(number of blocks) small
vectors, not O(context length) — the thing that made distributing a long
context across hosts expensive in the first place.

## How it's actually used

This is an **inference-serving technique**, not a change to model weights
or training — any already-trained decoder-only model can be served with
Star Attention, the same way any model can be served with
[PagedAttention](../kv-caching-and-paged-attention/). It targets the
specific regime of very long contexts served across multiple hosts/devices,
where communication (not compute) becomes the bottleneck; it's less
relevant for short contexts or single-device serving.

## Tradeoffs

Phase 1's block-local encoding is an approximation of full causal
attention — a token's phase-1 representation reflects the anchor block
plus its own block, not the entire preceding context, so quality can
degrade relative to exact full-context attention (the paper reports this
gap is small in practice for typical long-context tasks, growing with more
aggressive block splitting). In exchange, phase 1 is fully parallel with
zero cross-host communication, and phase 2's cross-host communication is
reduced to a small constant-size exchange per host rather than sharing
full KV — a large, direct win when scaling context length across many
devices. Contrast with [sliding-window attention](../sliding-window-attention/),
which restricts *which positions a query can see* on a single device;
Star Attention restricts *which blocks talk to which host* to cut
distributed communication, and is compatible with running full attention
within each host's local scope.

## References

- [Star Attention: Efficient LLM Inference over Long Sequences](https://arxiv.org/abs/2411.17116) (Acharya et al., NVIDIA, 2024)
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu et al., 2023) — related distributed-attention idea, see `ring-attention` in [`MAP.md`](../MAP.md)
