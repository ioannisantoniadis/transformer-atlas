# Ring Attention

**Lab:** UC Berkeley (Liu, Zaharia, Abbeel) · **Year:** 2023 · **Paper:** [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
**Family:** Transformer

## The problem

[FlashAttention](../flash-attention/) removes the memory bottleneck of
attention *within a single device* by tiling the computation so the full
`n × n` score matrix never has to be materialized in memory. But at some
context length, even the `Q`, `K`, `V` tensors themselves — let alone
activations — stop fitting on one device. To go further, the sequence has
to be **sharded across multiple devices**. Naively, that means every
device needs every other device's K/V to compute correct attention: an
all-to-all communication pattern whose cost grows with the number of
devices and the sequence length, easily becoming the bottleneck instead
of compute.

## The idea

Arrange devices in a **ring**. Each device starts by holding one block of
`Q`, `K`, `V` for its slice of the sequence. Attention proceeds in rounds:
at each round, every device computes attention between its (fixed) `Q`
block and the `K`/`V` block currently in front of it, accumulating the
result with an online-softmax merge (the same running-max / running-sum
trick used inside [FlashAttention](../flash-attention/) — but now
accumulated *across devices*, not just across tiles on one device). Then
every device passes its current `K`/`V` block to its neighbor around the
ring and receives a new one — like a bucket brigade.

```
        ┌────────────────────────────┐
        │                            │
  device 0 ──K/V──► device 1 ──K/V──► device 2 ──K/V──► device 3 ──K/V──►(back to 0)

  round 1: each device attends its Q block against its OWN K/V block
  round 2: each device attends its Q block against its NEIGHBOR's K/V block (just received)
  round 3: ... continues around the ring
  round N: every device's Q block has now seen every device's K/V block
```

After `N` rounds (`N` = number of devices), every `Q` block has attended
to every `K`/`V` block exactly once, and each device's online-softmax
accumulator holds the *exact* full-attention output for its queries —
mathematically identical to computing full attention on one giant device,
just distributed. Crucially, the K/V block being communicated at each
round is small (one block, not the whole sequence) and — because the
compute for the current round and the communication of the *next* round's
block can happen concurrently — the communication can be largely hidden
behind computation, rather than adding to the critical path.

This is architecturally the multi-device generalization of the same
insight [FlashAttention](../flash-attention/) uses within one device
(never materialize the full matrix, accumulate blockwise with online
softmax) and the same online-softmax-merge primitive
[Star Attention](../star-attention/) uses to combine partial results —
Ring Attention is what you get by applying that primitive to *every*
block, in a communication pattern designed to overlap with compute, rather
than restricting which blocks talk to which host the way Star Attention
does for a different (lower-communication-budget) inference regime.

## How it's actually used

Ring Attention is primarily a **training-time** (and to a lesser extent
long-context inference) technique for scaling context length by adding
more devices, keeping each device's memory footprint roughly constant
regardless of total sequence length — the "near-infinite context" framing
in the title. It's the mechanism behind some of the longest-context
training runs reported by research groups experimenting with million-
token-plus contexts, and is a building block other long-context serving
systems (including some Star-Attention-style setups) build on top of or
compare against.

## Tradeoffs

Requires `N` communication rounds per attention layer (one per device in
the ring) — a real cost if compute-per-round is too small to hide it
behind block transfer time, which matters more as you add more devices to
the ring. In exchange: per-device memory becomes independent of total
sequence length (each device only ever holds its own block plus one
incoming block), which is precisely the constraint that makes scaling
context length by adding devices tractable in the first place.

## References

- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (Liu, Zaharia & Abbeel, 2023)
- [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/abs/2305.19370) (Liu & Abbeel, 2023) — the single-device blockwise groundwork Ring Attention extends across devices
