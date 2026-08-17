# transformer-atlas

A structured, hands-on map of the transformer architecture — from the 2017
original through the attention variants, positional encodings, MoE routing,
and inference tricks that big AI labs (Google, OpenAI, Meta, Mistral AI,
DeepSeek, NVIDIA, ...) shipped on top of it to get to today's frontier LLMs.

This repo exists to make one question answerable in an afternoon instead of
a month of scattered paper-reading: **"given the original Transformer, what
changed, why, who did it, and how does it actually work?"**

## Start here

- [`MAP.md`](MAP.md) — the full taxonomy: every architecture/technique in
  scope, grouped by category, with paper links and a build-status marker.
  This is the map; the folders below are the territory.
- Pick any topic folder. Each one is self-contained:
  - `README.md` — plain-language intuition, the key equation(s), a diagram,
    what problem it solves relative to the baseline, and links to the
    original paper plus one or two follow-ups worth reading.
  - `implementation.py` — a minimal, dependency-light PyTorch
    implementation, written to be *read*, not to be fast or production-safe.
    Every file runs standalone (`python implementation.py`) and prints a
    small demo/sanity-check on random or toy data so you can see shapes and
    behavior without a GPU or a dataset.

## Scope

**In scope:** the decoder-only LLM lineage (the architecture family behind
GPT, LLaMA, Mistral, DeepSeek, etc.) and the mainstream, widely-cited
modifications made to it — attention variants (MQA/GQA, FlashAttention,
sliding-window, linear attention, MLA, Star Attention, ...), positional
encodings (RoPE, ALiBi), normalization/activation swaps (RMSNorm, SwiGLU),
Mixture-of-Experts routing, and the inference-serving techniques (KV
caching, PagedAttention, speculative decoding) that determine how these
models are actually run.

**Out of scope (for now):** encoder-only models (BERT-family), pure
encoder-decoder models (T5/BART), vision/multimodal transformers, and
training/alignment methods (RLHF, DPO, instruction tuning) — these are
different enough subjects to deserve their own map rather than being
squeezed into this one. `MAP.md` documents this boundary explicitly.

## Structure

Every topic is a **flat, top-level folder** — there's deliberately no
`architectures/` vs `attention/` subdivision on disk, because many entries
don't fit one bucket cleanly (is MLA an attention variant or part of the
DeepSeek-V2 architecture? both). `MAP.md` is where the categorization and
narrative ordering live; the filesystem stays flat and searchable.

```
transformer-atlas/
├── MAP.md                    # taxonomy + reading order
├── TEMPLATE.md                # template for adding a new topic
├── transformer/                # each topic:
│   ├── README.md                #   intuition, math, diagram, references
│   └── implementation.py        #   minimal runnable PyTorch
├── rotary-position-embedding/
├── multi-query-and-grouped-query-attention/
├── flash-attention/
├── mixture-of-experts/
├── llama/
├── mixtral/
├── deepseek-v2/
├── ...
```

## Running the code

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# any topic:
python transformer/implementation.py
python multi-head-latent-attention/implementation.py
```

Everything runs on CPU with toy tensor shapes — the point is to see the
mechanism, not to train a real model.

## Contributing / extending

This is primarily a personal learning reference, but it's structured so new
topics slot in cleanly — see [`TEMPLATE.md`](TEMPLATE.md) for the format
each folder follows, and the 🔲 rows in `MAP.md` for gaps.

## License

MIT — see [LICENSE](LICENSE).
