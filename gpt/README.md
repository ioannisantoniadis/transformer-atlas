# GPT (decoder-only Transformer)

**Lab:** OpenAI · **Year:** 2018–2020 · **Paper:** [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [GPT-3](https://arxiv.org/abs/2005.14165)

## The problem

The original [Transformer](../transformer/) is an encoder-decoder built
for sequence-to-sequence tasks (translation): a bidirectional encoder
reads the source, a causal decoder generates the target while
cross-attending to it. Most of what we want from a language model —
open-ended text generation, few-shot task-following via a prompt — doesn't
have a separate "source" and "target." It's just: predict the next token,
given everything so far.

## The idea

Keep only the decoder stack, drop cross-attention (there's no encoder to
attend to), and train with a single, simple objective: causal (autoregressive)
language modeling — predict token `t+1` from tokens `1..t`.

```
   token ids
       │
       ▼
  token embedding + learned positional embedding
       │
   ┌───▼────────────┐
   │ Causal self-    │   x N layers, pre-norm
   │ attention + FFN │   (see transformer/EncoderBlock,
   └───┬────────────┘    same block, causal mask only)
       │
       ▼
   final LayerNorm
       │
       ▼
  linear → vocab logits
```

Two details distinguish this from the encoder block in
[`transformer`](../transformer/):

1. **Causal mask everywhere, always** — position `i` can only attend to
   positions `≤ i`. This is what makes autoregressive generation
   consistent between training (teacher-forced, full sequence at once) and
   inference (one token at a time).
2. **Learned, not sinusoidal, positional embeddings** — GPT-2/3 use a
   trainable embedding table indexed by position instead of the original
   paper's fixed sinusoids. (This is itself superseded in most
   contemporary models by [RoPE](../rotary-position-embedding/), which
   encodes position more efficiently and generalizes better to sequence
   lengths not seen during training.)

The other headline result, especially from GPT-3, isn't architectural at
all: **scale**. Same recipe, radically more parameters/data/compute,
produces qualitatively new behavior (few-shot in-context learning — the
model performs a task from a handful of examples in the prompt, no
gradient update). This is the empirical basis for the "scaling laws"
thinking that still drives frontier lab decisions today.

## How it's actually used

This decoder-only-causal-LM template is the shared skeleton under
essentially every model in this repo's "full architectures" section —
[`llama`](../llama/), [`mixtral`](../mixtral/), [`deepseek-v2`](../deepseek-v2/)
all still are, at the top level, "embed → N causal decoder blocks → unembed."
Everything from GPT-2 onward is a story about what gets swapped inside that
block (attention variant, positional scheme, norm, FFN, MoE routing) — this
folder is the version with nothing swapped yet.

## Tradeoffs

Losing the encoder means losing native bidirectional context — every
position's representation only ever sees the past, even at training time.
For pure generation this is the right tradeoff (it matches how the model
will be used at inference); for tasks that benefit from seeing the whole
input first (classification, some retrieval), encoder-only (BERT-style) or
encoder-decoder (T5-style) models can have an advantage, which is why
those families didn't disappear even as decoder-only came to dominate
generative LLMs.

## References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, Radford et al., 2019)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, Brown et al., 2020)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) (Jay Alammar)
