"""
Two related but separate ideas:
1. KV caching: cache each token's key/value once instead of recomputing
   the whole sequence at every generation step. Verified here by checking
   cached and from-scratch generation produce identical outputs.
2. PagedAttention: a simulated page-table memory manager for KV caches,
   showing fixed-size pages + a logical->physical mapping (like OS virtual
   memory) instead of one contiguous worst-case allocation per request --
   including copy-on-write sharing of a common prompt prefix.
"""

import math

import torch
import torch.nn.functional as F


# --- Part 1: KV caching ---------------------------------------------------

class ToyCausalAttention:
    """A minimal single-head causal attention 'layer' with learned-looking
    fixed random projections, just to have something to cache K/V for."""

    def __init__(self, d_model, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w_q = torch.randn(d_model, d_model, generator=g)
        self.w_k = torch.randn(d_model, d_model, generator=g)
        self.w_v = torch.randn(d_model, d_model, generator=g)

    def project(self, x):
        return x @ self.w_q, x @ self.w_k, x @ self.w_v


def generate_naive(attn, tokens):
    """Recomputes K/V for the ENTIRE sequence at every step."""
    outputs = []
    for t in range(1, len(tokens) + 1):
        prefix = tokens[:t]
        q, k, v = attn.project(prefix)
        d_k = q.size(-1)
        scores = q @ k.T / math.sqrt(d_k)
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool))
        weights = F.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
        out = (weights @ v)[-1]  # only the new token's output is needed
        outputs.append(out)
    return torch.stack(outputs)


def generate_with_cache(attn, tokens):
    """Computes K/V for each NEW token once, appends to a growing cache."""
    d_model = tokens.shape[-1]
    k_cache = torch.empty(0, d_model)
    v_cache = torch.empty(0, d_model)
    outputs = []
    for t in range(len(tokens)):
        new_token = tokens[t:t+1]
        q_new, k_new, v_new = attn.project(new_token)
        k_cache = torch.cat([k_cache, k_new], dim=0)
        v_cache = torch.cat([v_cache, v_new], dim=0)

        d_k = q_new.size(-1)
        scores = q_new @ k_cache.T / math.sqrt(d_k)  # new query against the whole cache -- no mask needed,
        weights = F.softmax(scores, dim=-1)           # cache only ever holds past+current tokens
        outputs.append((weights @ v_cache)[0])
    return torch.stack(outputs)


# --- Part 2: PagedAttention-style memory management -----------------------

class PagedKVCache:
    """Simulates PagedAttention's memory manager: a shared pool of
    fixed-size physical pages, per-request page tables mapping logical
    block index -> physical page id, and copy-on-write sharing for a
    common prefix (e.g. a shared system prompt)."""

    def __init__(self, page_size, num_physical_pages):
        self.page_size = page_size
        self.free_pages = list(range(num_physical_pages))
        self.page_owners = {}          # physical page id -> set of request ids referencing it (for CoW)
        self.page_tables = {}          # request id -> list of physical page ids (logical order)

    def _alloc_page(self, request_id):
        page_id = self.free_pages.pop(0)
        self.page_owners[page_id] = {request_id}
        return page_id

    def start_request(self, request_id, num_prompt_tokens, shared_prefix_owner=None):
        num_pages_needed = math.ceil(num_prompt_tokens / self.page_size)
        if shared_prefix_owner is not None:
            # Copy-on-write: share the prefix owner's existing physical pages
            # instead of allocating + copying new ones.
            shared_pages = self.page_tables[shared_prefix_owner][:num_pages_needed]
            for page_id in shared_pages:
                self.page_owners[page_id].add(request_id)
            self.page_tables[request_id] = list(shared_pages)
        else:
            self.page_tables[request_id] = [self._alloc_page(request_id) for _ in range(num_pages_needed)]

    def append_token(self, request_id, token_index):
        """Allocate a new page only when the current last page is full --
        and only a fresh (non-shared) page, so writes never corrupt a page
        another request is still reading (the actual copy-on-write step)."""
        table = self.page_tables[request_id]
        needs_new_page = token_index % self.page_size == 0
        if needs_new_page:
            table.append(self._alloc_page(request_id))
        else:
            last_page = table[-1]
            if len(self.page_owners[last_page]) > 1:
                # Diverging from a shared prefix -- copy-on-write.
                self.page_owners[last_page].discard(request_id)
                table[-1] = self._alloc_page(request_id)

    def memory_used_pages(self):
        return len(self.page_owners)


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, seq_len = 8, 6
    attn = ToyCausalAttention(d_model)
    tokens = torch.randn(seq_len, d_model)

    out_naive = generate_naive(attn, tokens)
    out_cached = generate_with_cache(attn, tokens)
    print("naive (recompute-everything) output shape:", out_naive.shape)
    print("cached (incremental) output shape:", out_cached.shape)
    print("outputs match exactly:", torch.allclose(out_naive, out_cached, atol=1e-5))

    naive_recomputes = sum(range(1, seq_len + 1))  # 1 + 2 + ... + seq_len tokens re-projected
    cached_computes = seq_len                       # 1 token projected per step
    print(f"\ntoken-projections done: naive={naive_recomputes}, cached={cached_computes} "
          f"({naive_recomputes / cached_computes:.1f}x less work with caching)")

    print("\n--- PagedAttention-style memory manager ---")
    pager = PagedKVCache(page_size=4, num_physical_pages=20)

    # Two requests share a 6-token system prompt, then diverge.
    pager.start_request("req_A", num_prompt_tokens=6)
    pager.start_request("req_B", num_prompt_tokens=6, shared_prefix_owner="req_A")
    print(f"pages used after 2 requests sharing a prompt: {pager.memory_used_pages()} "
          f"(vs {len(pager.page_tables['req_A']) + math.ceil(6 / 4)} if B didn't share)")

    for i in range(6, 10):  # both generate new, DIFFERENT tokens -> triggers copy-on-write
        pager.append_token("req_A", i)
        pager.append_token("req_B", i)
    print(f"pages used after divergent generation: {pager.memory_used_pages()} "
          f"(copy-on-write kicked in once generation diverged)")
