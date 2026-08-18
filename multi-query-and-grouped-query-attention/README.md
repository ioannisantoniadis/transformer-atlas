# Multi-Query and Grouped-Query Attention (MQA / GQA)

**Lab:** Google · **Year:** 2019 / 2023 · **Paper:** [MQA (Shazeer)](https://arxiv.org/abs/1911.02150), [GQA (Ainslie et al.)](https://arxiv.org/abs/2305.13245)
**Family:** Transformer

## The problem

This one is about **inference**, not training quality. At generation time,
a decoder-only model uses a [KV cache](../kv-caching-and-paged-attention/):
it stores the key/value vectors for every past token so it doesn't
recompute them at every new step. In standard multi-head attention (see
[`transformer`](../transformer/)), every one of the `h` heads has its own
K and V projections — so the cache holds `h` separate K/V pairs per token,
per layer. For long contexts and large batch sizes, this cache becomes the
dominant memory cost and the dominant memory-bandwidth bottleneck (loading
it back from memory every generation step is what's actually slow, more
than the matmuls).

## The idea

Keep multiple **query** heads (so the model retains its representational
capacity), but share a much smaller number of **key/value** heads across
groups of query heads.

- **MQA (2019):** the extreme case — all query heads share a single K/V
  head. Maximizes memory savings, but the quality hit can be noticeable.
- **GQA (2023):** an interpolation — query heads are split into `g` groups,
  each group sharing one K/V head. `g = num_heads` recovers standard MHA;
  `g = 1` recovers MQA. In practice `g` is chosen (e.g. 8 for a 32-head
  model) to sit close to MQA's memory savings while staying close to MHA's
  quality — GQA models can also be produced cheaply by "uptraining" an
  existing MHA checkpoint (mean-pooling its K/V heads into groups and
  fine-tuning briefly), which is how the GQA paper validated it without
  training from scratch.

```
MHA (h=8 query heads, 8 KV heads)     GQA (8 query, 2 KV, g=2)      MQA (8 query, 1 KV)
Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8                Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8         Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
│  │  │  │  │  │  │  │                 └┬┘ └┬┘ └┬┘ └┬┘                └──────┬──────┘
K1 K2 K3 K4 K5 K6 K7 K8                 K1   K2   K3   K4                    K1
V1 V2 V3 V4 V5 V6 V7 V8                 V1   V2   V3   V4                    V1
(8 KV pairs cached)                    (2 KV pairs cached)              (1 KV pair cached)
```

The mechanism itself is simple: instead of projecting to `h` separate K/V
heads, project to `g` K/V heads, then repeat (broadcast) each one across
`h/g` query heads before the attention dot product — everything downstream
of that is identical to standard multi-head attention.

## How it's actually used

GQA is close to universal in current open-weight decoder-only models —
LLaMA 2/3, Mistral, Mixtral, DeepSeek-V2 (as a fallback path — see
[MLA](../multi-head-latent-attention/) for DeepSeek's actual choice), Qwen.
It composes directly with [RoPE](../rotary-position-embedding/) (rotate
before the head-sharing) and is largely orthogonal to
[sliding-window](../sliding-window-attention/) or
[FlashAttention](../flash-attention/) (those change *which* positions are
attended to or *how* the computation is scheduled; GQA changes *how many
distinct K/V projections exist*).

## Tradeoffs

Directly trades representational capacity for KV-cache memory/bandwidth:
fewer distinct K/V heads means less diversity in what different heads can
attend to. GQA's whole value proposition is that this trade is nearly free
at moderate group sizes — the quality gap versus full MHA is small, while
the cache shrinks by `h/g`×, which is a large, direct win for serving cost
and achievable batch size/context length at inference.

## References

- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Shazeer, 2019) — MQA
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) (Touvron et al., 2023) — first large-scale GQA adopter
