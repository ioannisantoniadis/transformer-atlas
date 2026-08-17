# Manifold-Constrained Hyper-Connections (mHC)

**Lab:** DeepSeek AI · **Year:** 2025 · **Paper:** [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)

## The problem

Every technique elsewhere in this repo modifies attention, positional
encoding, normalization, or the feedforward/routing sublayer — all of them
additively layered on top of the same thing: a single residual stream that
carries `x -> x + f(x)` unchanged from the first block to the last. That
residual stream is what gives transformers their trainability at depth (it's
why a 96-layer network doesn't just vanish/explode-gradient itself into
uselessness): each block is a small perturbation of an identity mapping, so
gradients flow straight through the `+` unchanged.

Hyper-connections (the line of work mHC extends) noticed that "one residual
stream" is an arbitrary choice — nothing stops you from carrying several
parallel streams and letting each layer read a *learned mixture* of them
instead of a fixed sum. That's strictly more expressive. But it's also more
dangerous: the plain version lets each layer's mixing matrix be arbitrary
(row-normalized via softmax so each output stream is *some* weighted
combination of inputs), and nothing keeps the *composition* of many such
matrices, across many layers, anywhere near the identity. In practice this
is exactly what breaks: at small scale it trains fine, and at 3B/9B/27B
scale (where DeepSeek validated this) plain hyper-connections become
unstable — the whole point of the residual stream (near-identity signal
propagation) quietly erodes as depth grows.

## The idea

Keep the multi-stream generalization, but constrain *how* streams get
mixed between layers so the identity-mapping guarantee survives.

Instead of one stream `x`, carry `H` parallel streams `X = [x_1, ..., x_H]`
(stacked as an `H`-row matrix). Between layers, mix them with a matrix `M`:

```
X_out = M @ X_in         (mixing across the H streams, per position)
X_out[h] += f_h(X_in)    (each layer's block writes into one or more streams)
```

Plain hyper-connections make `M` **row-stochastic**: each output stream is
a softmax-weighted combination of input streams (rows sum to 1). That's
enough to keep any *individual* mixing step bounded on average, but a
product of row-stochastic matrices is not itself guaranteed to stay
well-behaved — its spectral (operator) norm can exceed 1, so composing `L`
of them across `L` layers can amplify or collapse the stream norms
exponentially in depth. This mirrors exactly why the original identity
residual (a literal, unconstrained-scale identity mapping) is so hard to
improve on: relax the constraint even a little, without an equally strong
replacement guarantee, and deep networks stop training predictably.

mHC's fix: constrain `M` to be **doubly stochastic** — rows *and* columns
each sum to 1, i.e. a point on the Birkhoff polytope. It gets there by
running **Sinkhorn-Knopp** iterations (alternately renormalizing rows,
then columns, of a positive matrix) on the raw mixing logits until the
matrix converges to (approximately) doubly stochastic:

```
M_0 = exp(mixing_logits)              # entrywise positive
M_{t+1} = row-normalize(M_t)          # rows now sum to 1
M_{t+1} = col-normalize(M_{t+1})      # columns now sum to 1 (rows drift slightly, repeat)
... repeat a fixed number of iterations ...
```

Why this restores the identity guarantee: by the **Birkhoff-von Neumann
theorem**, every doubly stochastic matrix is a convex combination of
permutation matrices, and every permutation matrix has spectral norm
exactly 1 (it's an orthogonal matrix — it just relabels streams, it never
amplifies or shrinks them). Because the operator norm is a convex function,
any convex combination of norm-1 matrices also has operator norm ≤ 1. So
every single mHC mixing step is a *non-expansive* map, and — critically —
so is the **product of many of them across depth**, since the product of
doubly stochastic matrices is again doubly stochastic. Plain row-stochastic
mixing has no such guarantee: it can compound into a badly-conditioned
transform over enough layers, which is exactly the instability DeepSeek
reports at scale.

## How it's actually used

mHC is a production component of **DeepSeek-V4** (both V4-Pro and
V4-Flash), sitting at the residual-stream level underneath the rest of the
block — it composes with everything else in this repo (attention variant,
normalization, MoE routing) rather than replacing any of it. DeepSeek
reports it validated at 3B/9B/27B scales with roughly 6-7% compute/memory
overhead (extra streams and the Sinkhorn iterations aren't free) in
exchange for training stability that plain hyper-connections lose at
scale. It's paired with the Muon optimizer in DeepSeek's stack, though the
two are independent choices — mHC doesn't require a specific optimizer.

## Tradeoffs

More streams (`H`) means more activation memory carried through every
layer, and the Sinkhorn-Knopp projection is extra sequential compute per
layer (a handful of row/column normalization passes over an `H x H`
matrix — small since `H` is a small constant, but not zero, hence the
~6-7% overhead). It also adds a new hyperparameter surface (`H`, number of
Sinkhorn iterations) that a single fixed residual stream never had. In
return you get a strictly more expressive inter-layer connectivity pattern
than a fixed sum, with a mathematical (not just empirical) guarantee that
composing it across depth stays close to an identity-like mapping — the
property that made the original residual connection work in the first
place.

## References

- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (DeepSeek AI, 2025)
- [Hyper-Connections](https://arxiv.org/abs/2409.19606) (Zhu et al., ByteDance, 2024) — the unconstrained multi-stream generalization mHC builds on and stabilizes
- [Sinkhorn Distances: Lightspeed Computation of Optimal Transport](https://arxiv.org/abs/1306.0895) (Cuturi, 2013) — the Sinkhorn-Knopp iteration mHC uses to project onto doubly-stochastic matrices
