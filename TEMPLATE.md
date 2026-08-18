# Adding a new topic

Each topic is a top-level folder: `kebab-case-name/README.md` +
`kebab-case-name/implementation.py`. Copy this structure.

## `README.md` skeleton

```markdown
# Topic Name

**Lab:** who introduced it · **Year:** YYYY · **Paper:** [Title](arxiv-link)
**Family:** Transformer / State-Space / Hybrid — which sequence-mixing
lineage this belongs to. Drives which branch it appears on in `MAP.md`'s
git-graph diagram and the interactive Lineage view. A cross-cutting
technique that applies regardless of backbone (MoE routing, KV caching,
quantization) can state that explicitly instead of picking one.

## The problem

What broke, or what was slow/expensive, in the baseline this is a response
to. Name the baseline explicitly (e.g. "vanilla multi-head attention's O(n^2)
memory over the sequence").

## The idea

Plain-language intuition first — the one-paragraph version you'd give a
colleague on a whiteboard. Then the key equation(s), with each symbol
defined. Then a small diagram (ASCII or Mermaid — both render on GitHub).

## How it's actually used

Which real models ship this, and any detail that matters for using it
correctly (e.g. "GQA needs num_kv_heads to divide num_heads"; "RoPE assumes
even head_dim").

## Tradeoffs

What you give up. Nothing in this repo is free — say what it costs
(memory, compute, quality, implementation complexity).

## References

- Primary paper (link above)
- 1-2 follow-ups or the paper that made it mainstream (e.g. the model paper
  that adopted it), if different from the original
```

## `implementation.py` conventions

- Pure PyTorch, no framework dependencies beyond `torch` (+ `numpy` if
  needed). No GPU required.
- One file, readable top to bottom — prefer clarity over reuse; it's fine
  for two topic folders to each define their own small `RMSNorm` rather than
  sharing an import.
- End with a `if __name__ == "__main__":` block that builds a toy example
  (small random tensors, sensible shapes) and prints something that shows
  the mechanism working — e.g. output shape, an attention map, a routing
  decision, a before/after comparison.
- A short module docstring at the top restating what the file demonstrates
  and its relationship to the baseline (one or two sentences, not a
  restatement of the whole README).

## Updating the map

After adding a folder, add its row to the relevant category table in
[`MAP.md`](MAP.md), and add it to the git-graph diagram (a `commit` on
the `transformer` or `state-space` branch, or a `merge` if it's a
Hybrid). If it's Transformer-family, also add it to
`docs/visual-map.html`'s `ITEMS` array so it shows up in the interactive
Pipeline/Timeline/Lineage views.
