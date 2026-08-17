"""
Linear attention: replace softmax's exp(q.k) similarity with a factorized
feature-map similarity phi(q).phi(k), which lets attention be computed in
O(n) instead of O(n^2) — and, in the causal case, as an explicit O(1)-per-
step recurrence over a fixed-size state matrix. Demonstrates both the
O(n) parallel form and the recurrent form, and checks they agree.
"""

import torch
import torch.nn.functional as F


def feature_map(x):
    """phi(x) = elu(x) + 1, kept non-negative like softmax attention weights."""
    return F.elu(x) + 1


def linear_attention_parallel(q, k, v, causal=True, eps=1e-6):
    """O(n) form via the associativity trick. q,k: (seq_len, d_k). v: (seq_len, d_v)."""
    phi_q, phi_k = feature_map(q), feature_map(k)

    if not causal:
        kv = phi_k.transpose(0, 1) @ v                       # (d_k, d_v), summed over all j — done once
        normalizer = phi_q @ phi_k.sum(dim=0)                 # (seq_len,)
        return (phi_q @ kv) / (normalizer.unsqueeze(-1) + eps)

    # Causal: cumulative sums instead of one global sum, so query i only
    # sees keys/values j <= i.
    kv_terms = torch.einsum("id,ie->ide", phi_k, v)           # (seq_len, d_k, d_v), per-position outer product
    kv_cumsum = kv_terms.cumsum(dim=0)                        # running sum of outer products up to each i
    numerator = torch.einsum("id,ide->ie", phi_q, kv_cumsum)  # (seq_len, d_v)

    k_cumsum = phi_k.cumsum(dim=0)                             # (seq_len, d_k)
    normalizer = (phi_q * k_cumsum).sum(dim=-1)                # (seq_len,)
    return numerator / (normalizer.unsqueeze(-1) + eps)


def linear_attention_recurrent(q, k, v, eps=1e-6):
    """Explicit O(1)-per-step recurrence: state S accumulates phi(k) v^T,
    each output is phi(q_i) . S_i / normalizer_i. Mathematically identical
    to the causal parallel form above, computed one token at a time."""
    seq_len, d_k = q.shape
    d_v = v.shape[-1]
    phi_q, phi_k = feature_map(q), feature_map(k)

    state = torch.zeros(d_k, d_v)          # S_0
    z = torch.zeros(d_k)                    # running sum of phi(k), for normalization
    outputs = []
    for i in range(seq_len):
        state = state + torch.outer(phi_k[i], v[i])   # S_i = S_{i-1} + phi(k_i) v_i^T   -- O(1) update
        z = z + phi_k[i]
        out_i = (phi_q[i] @ state) / (phi_q[i] @ z + eps)
        outputs.append(out_i)
    return torch.stack(outputs)


def softmax_attention_causal(q, k, v):
    import math
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    mask = torch.tril(torch.ones(q.size(0), k.size(0), dtype=torch.bool))
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


if __name__ == "__main__":
    torch.manual_seed(0)
    seq_len, d_k, d_v = 9, 6, 5

    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_v)

    out_parallel = linear_attention_parallel(q, k, v, causal=True)
    out_recurrent = linear_attention_recurrent(q, k, v)
    print("parallel-form output shape:", out_parallel.shape)
    print("recurrent-form output shape:", out_recurrent.shape)
    print("parallel and recurrent forms match:",
          torch.allclose(out_parallel, out_recurrent, atol=1e-5))

    # Not the same numbers as softmax attention (different similarity
    # function) — but same shape, same role, O(n) instead of O(n^2).
    out_softmax = softmax_attention_causal(q, k, v)
    print("\nsoftmax attention output (for comparison, not equality):", out_softmax.shape)
    print("linear vs softmax outputs differ (expected, different sim fn):",
          not torch.allclose(out_parallel, out_softmax, atol=1e-2))

    # Compute-order argument: for causal linear attention, the state
    # matrix size (d_k x d_v) is fixed regardless of seq_len -- it's what
    # gives the O(1)-per-token, O(n)-total complexity.
    print(f"\nfixed recurrent state size: {d_k} x {d_v} = {d_k * d_v} scalars, "
          f"independent of seq_len={seq_len}")
