# Multi-Head Latent Attention (MLA)

**Lab:** DeepSeek AI · **Year:** 2024 · **Paper:** [DeepSeek-V2](https://arxiv.org/abs/2405.04434)

## The problem

[GQA/MQA](../multi-query-and-grouped-query-attention/) shrink the KV cache
by sharing K/V heads across query heads — but that's a fairly blunt
instrument: quality degrades as you push the number of KV heads down,
because you're literally deleting representational capacity (fewer
distinct heads). DeepSeek-V2 wanted GQA/MQA-or-better cache compression
*without* that capacity loss.

## The idea

Instead of sharing whole K/V heads, **compress** each token's K and V into
a single small low-rank **latent vector** `c_kv` (shared across all heads),
then reconstruct full-size, per-head K and V from it on demand via learned
up-projection matrices:

```
c_kv = x W_down                    # (d_model) → (d_latent), d_latent << d_model — this is what gets cached
k_i  = c_kv W_up_K,i               # reconstructed per-head K, head i
v_i  = c_kv W_up_V,i               # reconstructed per-head V, head i
```

The cache only ever needs to store `c_kv` (plus a small positional piece,
below) — a single small vector per token, not one K/V pair per head. This
gets closer to MQA-level cache size while keeping every head able to
express its own distinct K/V, because the *up-projection* still gives each
head a different view of the same compressed information; nothing is
literally shared/duplicated the way MQA shares one raw K/V head.

**The absorption trick.** At inference, you never need to actually
reconstruct `k_i` at all. Since `q_i · k_i = q_i · (c_kv W_up_K,i) =
(q_i W_up_K,i^T) · c_kv`, you can fold `W_up_K,i` into the query side once
per query and then dot the result directly against the cached `c_kv` —
same math, but the per-head up-projection of K never has to be
materialized at attention time.

**Why RoPE complicates this.** [RoPE](../rotary-position-embedding/)
rotates `q` and `k` by an angle that depends on their *position*, which sits
*between* `c_kv` and the up-projection in a way that breaks the
associativity the absorption trick relies on — you can't fold a
position-dependent rotation into a fixed, precomputed matrix. DeepSeek-V2's
fix is **decoupled RoPE**: split each head's query/key into two pieces — a
larger "content" piece (no RoPE, goes through the compress/absorb path
above) and a small extra "rope" piece (RoPE applied directly, shared
across heads for the key side) — then concatenate before the dot product.
Only the small rope piece needs the full, uncompressed treatment; the bulk
of the representation still benefits from compression.

```
   token x
     │
     ├──────────────► W_down ──► c_kv  (cached — small!)
     │                              │
     │                     per-head W_up_K, W_up_V
     │                              │
     │                    k_content_i, v_i (reconstructed)
     │
     └──► W_KR (shared) ──► RoPE ──► k_rope  (cached — small, uncompressed)

   attention score_i = [q_content_i ; q_rope_i] · [k_content_i ; k_rope]
```

## How it's actually used

MLA is the headline attention mechanism of DeepSeek-V2 and DeepSeek-V3,
paired with [DeepSeekMoE](../mixture-of-experts/) in the feedforward
sublayer (see [`deepseek-v2`](../deepseek-v2/) for the full block). The
DeepSeek-V2 paper reports a KV cache substantially smaller than even a
2.25-group GQA configuration at equivalent quality — the motivating result
for treating this as a genuine advance over GQA rather than a variant of
it.

## Tradeoffs

More architectural complexity than GQA (a compression/decompression path
plus a decoupled positional path, versus just fewer K/V heads) and an
extra hyperparameter (`d_latent`) to tune. The payoff is cache compression
without GQA's head-capacity tradeoff — in DeepSeek's reported results, MLA
matches or beats full multi-head attention quality while cutting cache
size dramatically, which is a better Pareto point than GQA offers at
comparable compression.

## References

- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) (DeepSeek-AI, 2024)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (DeepSeek-AI, 2024) — MLA at larger scale
