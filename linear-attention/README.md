# Linear Attention

**Lab:** Idiap Research Institute / EPFL (Linear Transformers); Google (Performer) · **Year:** 2020 · **Paper:** [Transformers are RNNs](https://arxiv.org/abs/2006.16236), [Performer](https://arxiv.org/abs/2009.14794)

## The problem

Softmax attention is O(n²) in both compute and memory because `softmax`
is applied to the full `QK^T` similarity matrix — you need every pairwise
score simultaneously to normalize them into a distribution. This is the
same root cause [FlashAttention](../flash-attention/) works around (by
never materializing the full matrix, while still computing the same
softmax result) and [sliding-window](../sliding-window-attention/) works
around (by restricting *which* pairs are computed at all). Linear
attention takes a third approach: change the similarity function itself so
the O(n²) structure isn't needed in the first place.

## The idea

Softmax attention computes `softmax(QK^T)V`. The trick: replace the
`exp(q·k)` similarity implicit in softmax with a similarity built from a
**feature map** `φ` applied to `q` and `k` separately:

```
Attention(Q,K,V)_i = Σ_j sim(q_i, k_j) v_j / Σ_j sim(q_i, k_j)
softmax attention:   sim(q,k) = exp(q · k)              — can't factor apart
linear attention:    sim(q,k) = φ(q) · φ(k)             — factors apart!
```

Because `φ(q_i) · φ(k_j)` factors into separate functions of `i` and `j`,
you can regroup the summation using the associative property of matrix
multiplication:

```
Σ_j (φ(q_i)·φ(k_j)) v_j  =  φ(q_i) · Σ_j φ(k_j) v_j^T
```

`Σ_j φ(k_j) v_j^T` is a single `(d_k × d_v)` matrix that doesn't depend on
`i` — compute it **once**, summed over all `j`, then every query just does
one multiplication against it. This turns attention into **O(n)** instead
of O(n²): one pass to accumulate the K/V summary, one pass to query it. It
also reveals the causal (autoregressive) form as an honest-to-god
**recurrent state update** — hence the "Transformers are RNNs" framing:

```
causal linear attention as a recurrence:
  S_0 = 0                              # (d_k x d_v) state matrix
  S_i = S_{i-1} + φ(k_i) v_i^T         # O(1) update per new token
  out_i = φ(q_i) · S_i / normalizer_i
```

This is exactly the shape that motivates a lot of recent "efficient
sequence model" work (state-space models like Mamba, RWKV, etc. — outside
this repo's decoder-only-Transformer scope, but the O(n) recurrent-state
framing is the same idea).

`φ` needs to produce non-negative outputs (so the "attention weights"
stay non-negative, like softmax's). Common choices: `elu(x) + 1`
(used in the original Linear Transformers paper — simple, cheap, what this
folder implements) or random Fourier features approximating the softmax
kernel (Performer's approach — more faithful to true softmax attention,
more complex).

## How it's actually used

Rarely as a drop-in replacement in frontier decoder-only LLMs today — pure
linear attention tends to underperform softmax attention on quality,
especially for tasks needing precise retrieval over context (the fixed-size
state `S` is a lossy summary; softmax attention keeps every token
individually addressable). Its real influence has been as the conceptual
seed for the broader "linear-attention-like" family the field explored
next: gated variants, hybrid architectures that mix a few full-attention
layers with many linear/recurrent layers, and non-Transformer sequence
models (state-space models) that took the recurrent-state idea further.
Understanding this file is the fastest path to understanding *why* that
whole family exists.

## Tradeoffs

O(n) compute/memory instead of O(n²) — a large win for long sequences —
paid for with a fixed-size state that must summarize the entire past
losslessly-in-principle but not in practice: the `(d_k × d_v)` state
matrix has a hard capacity limit, unlike softmax attention's implicit
"remember every token, weighted." This is the central quality/efficiency
tradeoff the whole linear/recurrent-attention research line still
navigates.

## References

- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) (Katharopoulos et al., 2020)
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) (Choromanski et al., 2020)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu & Dao, 2023) — where this line of thinking leads next (outside this repo's scope, but the connection is direct)
