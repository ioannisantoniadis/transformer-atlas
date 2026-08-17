# FlashAttention

**Lab:** Stanford (Tri Dao et al.) · **Year:** 2022 (v1), 2023 (v2), 2024 (v3) · **Paper:** [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691)

## The problem

This is the odd entry in this repo: it changes **zero** of the math in
[scaled dot-product attention](../transformer/) — same inputs, same
outputs, same gradients. What it changes is how the computation is
scheduled on a GPU.

Naive attention materializes the full `n × n` score matrix
(`QK^T`, then softmax, then multiply by `V`) in GPU high-bandwidth memory
(HBM). For long sequences this matrix is huge (n=8192 → 67M floats *per
head*), and — this is the key insight — GPUs are usually **memory-bandwidth
bound** on this operation, not compute-bound: the GPU spends more time
moving that matrix between slow HBM and fast on-chip SRAM than it spends
doing the actual multiplications.

## The idea

Never materialize the full `n × n` matrix. Split `Q`, `K`, `V` into blocks
that fit in fast on-chip SRAM, and compute attention **block by block**,
maintaining a running (online) softmax that gets corrected as new blocks
arrive — a streaming reformulation of softmax that doesn't need every
score up front.

```
for each block of K, V (loaded into SRAM):
    compute partial scores for this block
    update running max, running sum, running weighted output
    (rescale previous partial output as the running max shifts)
# after all blocks: final output is exact, not approximate
```

This is an *exact* algorithm — the output is numerically the same
attention as the naive version (up to floating point order-of-operations),
not an approximation. The speedup comes purely from doing O(n²/block_size)
small block computations that stay in fast memory, instead of one large
computation that round-trips through slow memory. FlashAttention-2
improves parallelization (across sequence length, not just batch/heads)
and reduces non-matmul overhead; FlashAttention-3 targets newer GPU
hardware (H100) specifically.

The demo in `implementation.py` below is a **from-scratch tiled/online-
softmax implementation in pure PyTorch**, written to make the algorithm
legible. It reproduces the *algorithm*, verified numerically against naive
attention — it does **not** reproduce the actual speed/memory win, which
requires a fused CUDA/Triton kernel (that's what the `flash-attn` PyPI
package and `torch.nn.functional.scaled_dot_product_attention`'s flash
backend provide) so that intermediate blocks genuinely never touch HBM.

## How it's actually used

FlashAttention (or an equivalent fused-kernel attention) is close to a
default in any serious training or inference stack today — it's not a
model-architecture choice at all (models don't "opt in" to it the way they
opt into [GQA](../multi-query-and-grouped-query-attention/) or
[RoPE](../rotary-position-embedding/)); it's an implementation detail that
any attention variant using standard softmax attention benefits from.
`F.scaled_dot_product_attention` in PyTorch dispatches to a flash-style
kernel automatically when available.

## Tradeoffs

None on the quality axis (it's exact). The cost is engineering complexity —
a real implementation needs hand-tuned block sizes per GPU architecture,
careful handling of the backward pass's recomputation strategy, and (until
recently) separate kernels per hardware generation. It also doesn't help
compute-bound regimes (very small sequences, or heavily batched small
matmuls) as much, since the bottleneck it targets is specifically memory
bandwidth on long sequences.

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Self-attention Does Not Need O(n^2) Memory](https://arxiv.org/abs/2112.05682) (Rabe & Staats, 2021) — the online-softmax idea FlashAttention builds on
