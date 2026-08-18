# Longformer and Sparse Attention Patterns

**Lab:** Allen Institute for AI (Longformer); OpenAI (Sparse Transformer, earlier) · **Year:** 2020 / 2019 · **Paper:** [Longformer](https://arxiv.org/abs/2004.05150), [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)
**Family:** Transformer

## The problem

[Sliding-window attention](../sliding-window-attention/) fixes full
attention's O(n²) cost by restricting every query to a local window — but
a *pure* local window has a real weakness: some tasks need at least a few
tokens to see, and be seen by, the *entire* sequence (a `[CLS]`-style
summary token, a question in a QA task that needs to attend over the
whole document, a small set of special tokens coordinating global
information). A strictly local window can't do that no matter how many
layers you stack, without those global tokens paying full O(n²) attention
themselves.

## The idea

Combine two attention patterns in the same layer, applied to different
tokens:

- **Local (sliding-window) attention** for most tokens — see
  [`sliding-window-attention`](../sliding-window-attention/), unchanged.
- **Global attention** for a small, chosen set of tokens — these tokens
  attend to *every* token in the sequence, and *every* token attends back
  to them. A handful of global tokens costs only O(n) extra (each one
  scans the full sequence once), not O(n²), because the count of global
  tokens is small and fixed, not a fraction of n.

```
   Local window (regular tokens):       Global tokens (e.g. [CLS], question tokens):
   ■ ■ · · · · · · · ·                  ■ ■ ■ ■ ■ ■ ■ ■ ■ ■   (attends to everything)
   ■ ■ ■ · · · · · · ·                  ■ · · · · · · · · ·   (and everything attends back)
   · ■ ■ ■ · · · · · ·
   · · ■ ■ ■ · · · · ·
   ...
```

Any single local-window layer still can't connect two far-apart *regular*
tokens directly — but with a global token in the mix it now takes only
**two** attention layers, not many stacked local ones: layer 1, the
global token attends to (and absorbs information from) a far-away token;
layer 2, a different far-away token attends to that now-informed global
token. Compare that to the many stacked local layers pure sliding-window
attention needs to connect the same two positions (the effective-
receptive-field argument in
[`sliding-window-attention`](../sliding-window-attention/)) — a global
token acts as a shared "hub" that collapses that path length dramatically.
Which tokens get global attention is typically task-defined (`[CLS]` for
classification, all question tokens for QA) rather than learned.

**Earlier precedent: Sparse Transformer (OpenAI, 2019).** Before
Longformer's local+global scheme, the Sparse Transformer used **fixed,
structured sparse patterns** — e.g. "strided" attention (attend to every
`k`-th previous token, covering long range cheaply) combined with local
attention, or row/column factorizations of the full attention matrix. The
underlying idea is the same family: replace the dense O(n²) mask with a
*designed* sparse mask cheap enough to compute exactly, rather than
approximating dense attention or restricting to local-only.

## How it's actually used

Longformer's local+global pattern is most associated with long-document
NLP tasks (summarization, QA over long documents) from the encoder/BERT-
style modeling era rather than today's decoder-only frontier LLMs — the
specific mechanism seen less often, by name, in current frontier decoder-
only releases. Its lasting influence is conceptual: "most attention should
be local; give a small number of tokens a global role" is a recurring
design motif that shows up, in different forms, across the long-context
efficiency literature this repo maps (compare the anchor-block idea in
[Star Attention](../star-attention/), which similarly treats one small
piece of context as globally shared while restricting everything else).

## Tradeoffs

Cheaper than full attention (O(n·w + n·g) for window size `w` and `g`
global tokens, versus O(n²)) while retaining a genuine, exact global
information channel — unlike pure local windows, which rely on indirect,
multi-layer propagation for long-range connections. The cost is design
overhead: unlike sliding-window's single hyperparameter, this pattern
requires deciding *which* tokens get global treatment, which is often
task-specific rather than automatic, and doesn't generalize as cleanly to
arbitrary generative decoder-only use as sliding-window or RoPE-based
approaches do.

## References

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) (Beltagy, Peters & Cohan, 2020)
- [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) (Child et al., OpenAI, 2019)
- [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) (Zaheer et al., Google, 2020) — a closely related local+global+random sparse pattern
