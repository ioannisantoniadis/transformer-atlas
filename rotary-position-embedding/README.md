# Rotary Position Embedding (RoPE)

**Lab:** Zhuiyi Technology (adopted by EleutherAI's GPT-NeoX, then everyone) · **Year:** 2021 · **Paper:** [RoFormer](https://arxiv.org/abs/2104.09864)

## The problem

The original [Transformer](../transformer/) *adds* a positional signal to
the input embeddings once, up front (sinusoidal or, in [GPT](../gpt/),
learned). Two issues:

1. It only encodes **absolute** position. What attention scores actually
   care about is usually **relative** position (token A is 3 places before
   token B) — absolute encodings make the model work harder to recover
   that.
2. Learned absolute position embeddings (GPT-2/3 style) are capped at the
   max sequence length seen in training — you can't cleanly run the model
   on a longer sequence, because there's no embedding for position 4097 if
   you only trained on positions 0..4095.

## The idea

Instead of adding a position vector to the embedding, **rotate** the query
and key vectors by an angle proportional to their position, before
computing attention. Pair up each vector's dimensions `(x_{2i}, x_{2i+1})`
and treat each pair as a 2D coordinate; rotate it by `m * θ_i`, where `m`
is the token's position and `θ_i` is a frequency that varies by dimension
pair (low pairs rotate fast, high pairs rotate slowly — same idea as the
original sinusoidal frequencies).

```
RoPE(x, m)_{2i, 2i+1} = R(m * θ_i) · (x_{2i}, x_{2i+1})

R(φ) = [ cos φ   -sin φ ]
       [ sin φ    cos φ ]
```

Apply this to `q` at position `m` and `k` at position `n` before the dot
product. Because rotation preserves dot products up to the *difference* in
angle, `RoPE(q, m) · RoPE(k, n)` ends up depending only on `m - n`, the
**relative** position — even though each vector was rotated using its own
*absolute* position. That's the trick: absolute-looking rotations produce
relative-only attention scores.

```
   q at position m ──rotate by m·θ──▶ q'
   k at position n ──rotate by n·θ──▶ k'

   q' · k'  depends only on (m - n)
```

No extra parameters, no positional embedding table, and — because the
rotation is a well-defined function of position for *any* `m`, not a
lookup into a fixed-size table — it extrapolates more gracefully to
sequence lengths beyond what was trained on (though not perfectly; see
`yarn-and-rope-scaling` in [`MAP.md`](../MAP.md) for how frontier models
extend RoPE to much longer contexts than trained).

## How it's actually used

RoPE is applied to `q` and `k` only (never `v`), inside every attention
layer, not once at the input. It's now close to a default choice: LLaMA,
Mistral, DeepSeek-V2, Qwen, and most other modern open-weight decoder-only
models use it. It composes cleanly with
[GQA/MQA](../multi-query-and-grouped-query-attention/) (rotate before
sharing KV heads) and is a specific design consideration in
[MLA](../multi-head-latent-attention/), which has to special-case RoPE
because it's incompatible with MLA's low-rank KV compression in its naive
form.

## Tradeoffs

Slightly more compute per attention call (a rotation per q/k vector per
layer, versus a one-time add at the input) in exchange for better relative-
position modeling and better length generalization. `head_dim` must be
even (dimensions are rotated in pairs). Pure RoPE still degrades on
sequences much longer than training length without additional scaling
tricks (NTK-aware scaling, YaRN, position interpolation).

## References

- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021)
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) (Chen et al., 2023)
- [Understanding RoPE](https://blog.eleuther.ai/rotary-embeddings/) (EleutherAI blog)
