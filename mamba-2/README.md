# Mamba-2 (Structured State Space Duality)

**Lab:** Tri Dao (Princeton), Albert Gu (CMU) · **Year:** 2024 · **Paper:** [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
**Family:** State-Space

## The problem

[`mamba`](../mamba/)'s selective scan recovers training-time parallelism
for input-dependent dynamics, but a scan is still a sequential-dependency
algorithm at heart — it doesn't map onto GPU tensor cores the way a plain
matrix multiply does. Modern accelerators are built to be extremely fast
at matmuls specifically; an architecture whose core operation is a scan,
however parallel-friendly, is leaving a lot of that hardware's actual
throughput unused compared to attention's matmul-heavy computation.

## The idea

Constrain the state matrix to a **scalar times identity** — a special,
*less general* case of Mamba's full per-channel diagonal `A` — and prove
that under this constraint, the selective recurrence becomes *exactly* a
form of causal linear attention: a masked matrix multiply with a specific
(1-semiseparable) triangular decay mask, not just an analogy:

```
scalar-A SSM recurrence  ≡  causal attention with a structured decay mask
        (sequential)               (matmul -- runs on tensor cores)
```

This is the **structured state-space duality (SSD)** result — the same
computation, two algorithmic realizations, exactly the pattern
[`s4-and-structured-state-spaces`](../s4-and-structured-state-spaces/)'s
recurrence/convolution duality already established, one level up. Because
the matmul realization exists, training can run on tensor cores instead
of a scan kernel, a 2-8x speedup over Mamba's original algorithm. It's
also the same phenomenon [`gated-deltanet-and-kda`](../gated-deltanet-and-kda/)
arrives at from the opposite direction — starting from linear attention's
kernel view and arriving at a state-space-shaped recurrence with the
delta rule, rather than starting from a state-space model and arriving at
attention. This repo's implementation proves the SSD equivalence
directly: the sequential scalar-SSM recurrence and its equivalent masked
matrix multiply produce the same output, exactly.

## How it's actually used

SSD is the reformulation that made the state-space branch fast enough on
modern GPUs to be a serious production option at all, not just an
asymptotic (`O(n)` vs `O(n²)`) argument on paper. It's also the most
direct, explicit statement in the literature that attention and
state-space recurrences are two views of the same underlying computation
— the paper is literally titled "Transformers are SSMs" — which is the
fact this map's whole State-Space branch and its cross-links back to
[`linear-attention`](../linear-attention/) and
[`gated-deltanet-and-kda`](../gated-deltanet-and-kda/) are built on.

## Tradeoffs

Scalar-times-identity `A` is strictly less expressive than Mamba's full
per-channel diagonal `A` -- a real capability trade made in exchange for
the matmul speedup, not a free lunch. And SSD changes *how fast* the
state-space branch runs, not *how much* it can remember: the fixed-size
state is still the same fundamental capacity ceiling every entry in this
branch shares.

## References

- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) (Dao, Gu, 2024) — ICML 2024
- [`mamba`](../mamba/) — the selective state-space model this result generalizes and re-derives
