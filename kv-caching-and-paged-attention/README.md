# KV Caching and PagedAttention

**Lab:** KV caching — folklore/no single paper; PagedAttention — UC Berkeley (vLLM) · **Year:** — / 2023 · **Paper:** [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## The problem

Autoregressive generation (see [`gpt`](../gpt/)) produces one token at a
time, feeding each new token back in as input for the next step. Done
naively, generating token `t+1` means re-running the *entire* forward
pass over tokens `1..t` — recomputing every attention key/value that was
already computed on the previous step. That's wasted, repeated work: an
O(n²) total cost across a generation of length n, purely from redundant
recomputation, on top of attention's own O(n²) cost per step.

## The idea: KV caching

Since causal attention (see [`transformer`](../transformer/)) means a
token's key and value vectors never depend on *future* tokens, they never
change once computed. So compute each token's K/V exactly once, **cache**
it, and at every subsequent step only compute Q/K/V for the *new* token,
attending it against the growing cache of past K/V:

```
step 1: prompt "The cat sat" -> compute K,V for all 3 tokens, cache them
step 2: generate "on" -> compute K,V for "on" only, append to cache,
                          attend new Q against cache of 4 K/V pairs
step 3: generate "the" -> compute K,V for "the" only, append,
                          attend new Q against cache of 5 K/V pairs
...
```

This turns each generation step's attention cost from O(t²) (recomputing
everything) into O(t) new work (one new token's worth), at the cost of
**memory**: the cache holds one K and one V vector per head per layer per
past token, growing linearly with generated length. This memory cost is
exactly what [GQA/MQA](../multi-query-and-grouped-query-attention/),
[MLA](../multi-head-latent-attention/), and
[sliding-window attention](../sliding-window-attention/) each attack from
a different angle — fewer/smaller cached vectors per token, or a bounded
window of tokens kept at all.

## The idea: PagedAttention

KV caching solves redundant compute but creates a new problem at the
**serving** level: a production system runs many requests concurrently,
each with its own growing KV cache of unpredictable, not-known-in-advance
final length. Naive implementations pre-allocate a large contiguous memory
block per request sized for the worst case — wasting huge amounts of GPU
memory to internal fragmentation, and blocking new requests from starting
until enough contiguous memory is free.

PagedAttention (built for the vLLM serving system) borrows the idea
straight from OS virtual memory: don't store each request's KV cache as
one contiguous block. Split it into fixed-size **pages** (blocks of, say,
16 tokens' worth of K/V), store pages anywhere in a shared physical memory
pool, and keep a small per-request **page table** mapping logical token
position to physical page location — exactly analogous to how an OS maps
virtual addresses to physical memory pages.

```
Request A's logical KV cache:  [page 0][page 1][page 2]
Request B's logical KV cache:  [page 0][page 1]

Physical GPU memory pool (pages can be non-contiguous, shared, reused):
[ A:page1 ][ B:page0 ][ free ][ A:page0 ][ B:page1 ][ A:page2 ][ free ]...
```

This nearly eliminates internal fragmentation (allocate pages on demand,
not a worst-case block up front), and — because pages are
content-addressed rather than owned outright — enables **cheap sharing**:
requests with an identical prompt prefix (e.g. a shared system prompt
across many users) can literally share the same physical pages for that
prefix, copying only when one request's generation diverges from another's
(copy-on-write, again straight from OS design).

## How it's actually used

KV caching is universal — every production autoregressive LLM serving
stack uses it; the question is never "cache or not," only how to shrink or
manage the cache (which is what most of this repo's attention-variant
entries are ultimately in service of). PagedAttention specifically is the
core mechanism behind vLLM, one of the most widely used open-source LLM
serving engines, and the same page-table idea has since been adopted or
reimplemented by several other serving frameworks.

## Tradeoffs

KV caching trades memory for compute — an unambiguous win for any
generation of more than a token or two, which is why it's never actually
optional in practice. PagedAttention adds a layer of indirection (page
table lookups) versus one flat contiguous cache, which costs a little
compute overhead per attention call, in exchange for dramatically better
GPU memory utilization and higher achievable concurrent-request throughput
in a serving system — the paper reports large throughput gains over
contiguous-allocation baselines specifically because more requests fit in
the same GPU memory at once.

## References

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al., UC Berkeley, 2023)
- [vLLM project](https://github.com/vllm-project/vllm) — the serving engine PagedAttention was built for
