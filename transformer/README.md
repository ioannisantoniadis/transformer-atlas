# The Transformer

**Lab:** Google Brain · **Year:** 2017 · **Paper:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
**Family:** Transformer

## The problem

Before this paper, sequence modeling meant RNNs/LSTMs (sequential, one
token at a time — can't parallelize across the sequence, and gradients
struggle over long distances) or CNNs (parallel, but need many layers to
connect far-apart tokens). Attention mechanisms already existed as an
add-on to RNNs (Bahdanau 2014). The claim of this paper was that attention
alone, with no recurrence and no convolution, is enough — and it's far more
parallelizable.

## The idea

**Scaled dot-product attention** lets every position look at every other
position directly, weighting them by relevance:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

- `Q` (queries), `K` (keys), `V` (values) are linear projections of the
  input. `QK^T` is a similarity score between every pair of positions.
- Dividing by `sqrt(d_k)` keeps the dot products from growing large with
  dimension, which would push softmax into a saturated, near-one-hot regime
  with vanishing gradients.
- `softmax` turns scores into a probability distribution over positions;
  the output is a weighted average of `V`.

**Multi-head attention** runs this several times in parallel with
different learned projections, then concatenates the results:

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

Each head can specialize (one head might track syntactic adjacency,
another long-range coreference) — a single attention computation over the
full dimension can't easily represent multiple independent notions of
"relevant" at once.

**Everything else in the block exists to make attention trainable at
depth:**
- *Residual connections* (`x + Sublayer(x)`) so gradients have a direct
  path through many layers.
- *LayerNorm* to stabilize the scale of activations layer to layer.
- *Position-wise feedforward* (`Linear → ReLU → Linear`, applied
  identically to every position) — attention mixes information *across*
  positions, the FFN transforms *within* each position; you need both.
- *Sinusoidal positional encoding* — attention itself is permutation-
  invariant (it has no notion of order), so position has to be injected
  separately. The original paper adds fixed sinusoids of different
  frequencies to the input embeddings, chosen so relative positions are a
  linear function the model can learn to exploit.

```
   Input tokens
        │
        ▼
  + Positional Encoding
        │
   ┌────▼────┐
   │ Self-    │  x N layers
   │ Attention│
   └────┬────┘
        │ (+ residual, LayerNorm)
   ┌────▼────┐
   │  FFN     │
   └────┬────┘
        │ (+ residual, LayerNorm)
        ▼
     Output
```

The original paper is an **encoder-decoder**: an encoder stack with
bidirectional self-attention, a decoder stack with *causal* (masked)
self-attention plus cross-attention into the encoder, for sequence-to-
sequence tasks like translation. Almost every modern LLM (GPT, LLaMA,
Mistral, DeepSeek, ...) keeps only the decoder half, with causal self-
attention and no cross-attention — see [`gpt`](../gpt/).

## How it's actually used

Nobody ships the 2017 architecture unmodified anymore — but every later
entry in this repo is a *diff* against this file: swap sinusoidal PE for
[RoPE](../rotary-position-embedding/), swap LayerNorm+ReLU-FFN for
[RMSNorm+SwiGLU](../rmsnorm-and-swiglu/), swap full multi-head attention for
[GQA](../multi-query-and-grouped-query-attention/) or
[MLA](../multi-head-latent-attention/), etc. Understanding this file is the
prerequisite for all of them.

## Tradeoffs

Self-attention is O(n²) in sequence length in both compute and memory (the
`QK^T` matrix is n×n) — the central tension the rest of this repo's
attention-variant entries exist to relieve. In exchange, it gives O(1)
path length between any two positions (vs O(n) for RNNs) and full
parallelism across the sequence during training.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar) — the canonical visual walkthrough
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) (Harvard NLP) — line-by-line paper-to-code
