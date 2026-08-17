# RMSNorm and SwiGLU

**Lab:** RMSNorm — Edinburgh/Sussex (Zhang & Sennrich); SwiGLU — Google · **Year:** 2019 / 2020 · **Paper:** [RMSNorm](https://arxiv.org/abs/1910.07467), [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)

Two small, independent swaps that nearly every post-2022 open-weight model
(LLaMA, Mistral, DeepSeek, Qwen, ...) makes to the original
[Transformer](../transformer/) block. Neither changes the architecture's
shape — both are drop-in replacements for a norm and an activation — which
is exactly why they spread so fast: cheap to adopt, consistently better.

## RMSNorm — replacing LayerNorm

**The problem.** LayerNorm normalizes a vector by subtracting its mean and
dividing by its standard deviation, then applies a learned scale and
shift:

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

The re-centering (subtracting the mean) turns out to contribute little to
LayerNorm's benefit — the useful part is *re-scaling*.

**The idea.** Drop the mean-subtraction and the bias term; normalize only
by the root-mean-square:

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
```

This is cheaper (no mean reduction, fewer learned parameters — no `beta`)
and empirically matches LayerNorm's quality in transformer LMs.

## SwiGLU — replacing the ReLU/GELU feedforward

**The problem.** The original FFN is `Linear → activation → Linear`, a
single gate-free nonlinearity applied to a single projection of the input.

**The idea.** Use a **Gated Linear Unit**: project the input two ways, run
one projection through an activation to act as a gate, and multiply it
elementwise into the other:

```
FFN_SwiGLU(x) = (Swish(x W_gate) ⊙ (x W_up)) W_down
Swish(x) = x * sigmoid(x)          # aka SiLU
```

Three weight matrices instead of two, but each is typically sized down
(often `d_ff = 8/3 * d_model` instead of `4 * d_model`) so total parameter
count stays comparable. The gating lets the network learn to pass through
or suppress each feature dynamically, rather than always applying the same
fixed nonlinearity — the [GLU Variants](https://arxiv.org/abs/2002.05202)
paper found this beats plain ReLU/GELU FFNs across tasks, with no
principled explanation offered beyond "it just works better" (a running
theme in the paper's conclusion, quoted almost verbatim in follow-up
model reports).

```
        x
      ┌─┴─┐
   W_gate  W_up
      │      │
   Swish     │
      └──⊙───┘
          │
        W_down
          │
        output
```

## How it's actually used

Both appear together in the [`llama`](../llama/) block (and
[`mixtral`](../mixtral/), [`deepseek-v2`](../deepseek-v2/), which build on
the same block). RMSNorm is used in **pre-norm** position (normalize
before attention/FFN, not after — see the residual-stream discussion in
[`transformer`](../transformer/)), applied twice per block: once before
attention, once before the FFN.

## Tradeoffs

RMSNorm gives up mean-centering, which theoretically could matter for some
distributions — in practice this hasn't shown up as a real cost, and the
compute savings (and simplicity) win. SwiGLU's third weight matrix means
~50% more FFN parameters at a given `d_ff`, offset by shrinking `d_ff`
itself; net effect is roughly parameter-neutral with a quality gain,
which is why it's now close to a default rather than a niche choice.

## References

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023) — popularized both together
