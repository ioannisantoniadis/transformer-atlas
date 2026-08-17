"""
Delta-rule linear attention: DeltaNet's erase-then-write state update, and
its two production gating refinements (Gated DeltaNet: one scalar decay for
the whole state; KDA: one decay rate per value-channel). Demonstrates (1)
the delta rule correctly overwrites a reused key, where plain additive
linear attention blends the old and new values together, and (2) a
per-channel gate can protect a long-lived association from decay while
still forgetting noisy channels fast -- something a single scalar gate,
forced to pick one rate for the whole state, cannot do at the same time.
"""

import torch


def linear_attention_write(S, k, v):
    """Plain additive linear-attention state update: S += v k^T. Never erases."""
    return S + torch.outer(v, k)


def delta_rule_write(S, k, v, beta=1.0, alpha=1.0):
    """Gated delta-rule update: S = diag(alpha) @ S @ (I - beta k k^T) + beta v k^T.
    alpha: a scalar (Gated DeltaNet, uniform decay) or a (d_v,) vector
    (KDA, one decay rate per value-channel)."""
    d_v, d_k = S.shape
    gate = torch.diag(alpha) if torch.is_tensor(alpha) and alpha.dim() > 0 else alpha * torch.eye(d_v)
    erase = torch.eye(d_k) - beta * torch.outer(k, k)
    return gate @ S @ erase + beta * torch.outer(v, k)


def read(S, k):
    return S @ k


if __name__ == "__main__":
    torch.manual_seed(0)
    d_k, d_v = 8, 8

    # --- Part 1: overwrite correctness ---
    print("=== overwrite: same key written twice with different values ===")
    k = torch.nn.functional.normalize(torch.randn(d_k), dim=0)
    v1, v2 = torch.randn(d_v), torch.randn(d_v)

    S_naive = torch.zeros(d_v, d_k)
    S_naive = linear_attention_write(S_naive, k, v1)
    S_naive = linear_attention_write(S_naive, k, v2)
    out_naive = read(S_naive, k)

    S_delta = torch.zeros(d_v, d_k)
    S_delta = delta_rule_write(S_delta, k, v1, beta=1.0)
    S_delta = delta_rule_write(S_delta, k, v2, beta=1.0)
    out_delta = read(S_delta, k)

    naive_error = (out_naive - v2).norm().item()
    delta_error = (out_delta - v2).norm().item()
    print(f"v2 (the value that SHOULD be recalled): {[f'{x:.3f}' for x in v2[:4].tolist()]} ...")
    print(f"plain linear-attention recall (blended, wrong): {[f'{x:.3f}' for x in out_naive[:4].tolist()]} ...")
    print(f"delta-rule recall (beta=1, overwrite):          {[f'{x:.3f}' for x in out_delta[:4].tolist()]} ...")
    print(f"error vs v2 -- plain linear attention: {naive_error:.4f}, delta rule: {delta_error:.4f}")
    assert delta_error < 1e-4, "delta rule should recall v2 exactly with beta=1"
    assert naive_error > 0.5, "plain linear attention should NOT recall v2 cleanly -- it's blended with v1"
    print("delta rule overwrites cleanly; plain linear attention cannot.\n")

    # --- Part 2: a scalar gate can't protect one channel group while forgetting another ---
    print("=== gating: protect a long-lived fact, forget noisy channels, at the same time ===")
    # channels 0-3 of the VALUE hold a "sticky" fact, written once, under a
    # key confined to key-subspace dimension 0. channels 4-7 get overwritten
    # with fresh noise every step, under keys confined to key-subspace
    # dimensions 1-7 -- so noise writes are exactly orthogonal to the sticky
    # key (their erase terms cannot touch it), isolating decay as the ONLY
    # thing that can still erode the sticky fact.
    sticky_k = torch.zeros(d_k)
    sticky_k[0] = 1.0
    sticky_v = torch.zeros(d_v)
    sticky_v[:4] = torch.randn(4)

    num_steps = 40
    noise_keys = []
    for _ in range(num_steps):
        kt = torch.randn(d_k)
        kt[0] = 0.0  # orthogonal to sticky_k by construction
        noise_keys.append(torch.nn.functional.normalize(kt, dim=0))
    noise_vals = []
    for _ in range(num_steps):
        v = torch.zeros(d_v)
        v[4:] = torch.randn(4)
        noise_vals.append(v)

    def run(alpha):
        S = torch.zeros(d_v, d_k)
        S = delta_rule_write(S, sticky_k, sticky_v, beta=1.0, alpha=1.0)  # write the sticky fact, ungated
        for t in range(num_steps):
            S = delta_rule_write(S, noise_keys[t], noise_vals[t], beta=1.0, alpha=alpha)
        return S

    err_none = (read(run(alpha=1.0), sticky_k)[:4] - sticky_v[:4]).norm().item()
    err_scalar = (read(run(alpha=0.9), sticky_k)[:4] - sticky_v[:4]).norm().item()
    channel_alpha = torch.tensor([0.999, 0.999, 0.999, 0.999, 0.9, 0.9, 0.9, 0.9])
    err_channel = (read(run(alpha=channel_alpha), sticky_k)[:4] - sticky_v[:4]).norm().item()

    print(f"sticky-fact recall error after {num_steps} unrelated writes:")
    print(f"  no gating (alpha=1, unbounded state):        {err_none:.4f}")
    print(f"  uniform scalar gate, strong enough to forget: {err_scalar:.4f}")
    print(f"  per-channel gate (protect 0-3, decay 4-7):    {err_channel:.4f}")
    assert err_channel < err_scalar, (
        "a per-channel gate should protect the sticky fact far better than a "
        "uniform scalar gate that must apply the SAME decay to the noisy channels too"
    )
    print("a single scalar strong enough to forget the noisy channels also erases the sticky fact;")
    print("a per-channel gate (KDA's contribution over Gated DeltaNet) can do both at once.")
