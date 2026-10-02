"""
mHC (Manifold-Constrained Hyper-Connections): generalizes the single
residual stream into H parallel streams mixed between layers by a learned
matrix M. Plain hyper-connections (Zhu et al. 2024) leave M unconstrained --
a learned static matrix plus a small input-dependent term, s * tanh(.) + A --
so its row and column sums can drift from 1; mHC projects M onto the
Birkhoff polytope of doubly-stochastic matrices via Sinkhorn-Knopp
iterations. Demonstrates why that matters, with the mHC paper's own metric:
the "Amax gain" of the matrix composed across depth (largest absolute row
sum for the forward pass, largest absolute column sum for the backward
pass; 1 for an identity-like mapping). Unconstrained mixing compounds small
per-layer deviations into large gains; doubly-stochastic mixing keeps both
gains at 1 at any depth (exactly, once Sinkhorn has converged; the paper
uses 20 iterations, which leaves a small bounded drift), and (by
Birkhoff-von Neumann) its spectral
norm at <= 1.
"""

import torch


def sinkhorn_knopp(positive_matrix, num_iters=20, eps=1e-8):
    """Project a strictly positive matrix onto (approximately) the Birkhoff
    polytope of doubly-stochastic matrices by alternating row/column
    normalization. Converges to a doubly-stochastic matrix for any positive
    input (Sinkhorn's theorem)."""
    m = positive_matrix.clone()
    for _ in range(num_iters):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)  # rows sum to 1
        m = m / (m.sum(dim=-2, keepdim=True) + eps)  # columns sum to 1
    return m


def unconstrained_mixing(H, perturbation_scale, generator):
    """Plain hyper-connections: an identity-initialized learned matrix plus a
    tanh-bounded perturbation, with no constraint on row or column sums."""
    noise = torch.randn(H, H, generator=generator, dtype=torch.float64)
    return torch.eye(H, dtype=torch.float64) + perturbation_scale * torch.tanh(noise)


def amax_gains(composed):
    """The mHC paper's 'Amax Gain Magnitude': worst-case forward (row-sum) and
    backward (column-sum) amplification of a composed mixing matrix."""
    forward = composed.sum(dim=-1).abs().max().item()
    backward = composed.sum(dim=-2).abs().max().item()
    return forward, backward


if __name__ == "__main__":
    H = 4  # number of parallel hyper-connection streams
    depth = 60  # layers to compose mixing matrices across
    perturbation_scale = 0.15  # how far each learned matrix sits from the identity
    logit_scale = 2.5  # sharper-than-uniform mixing for mHC, like a trained model's

    gen = torch.Generator().manual_seed(0)

    # --- Sanity check: Sinkhorn output really is (approximately) doubly stochastic ---
    probe = sinkhorn_knopp(torch.exp(logit_scale * torch.randn(H, H, generator=gen)))
    print("Sinkhorn-projected matrix row sums: ", [f"{v:.4f}" for v in probe.sum(dim=-1).tolist()])
    print("Sinkhorn-projected matrix col sums: ", [f"{v:.4f}" for v in probe.sum(dim=-2).tolist()])
    assert torch.allclose(probe.sum(dim=-1), torch.ones(H), atol=1e-4)
    assert torch.allclose(probe.sum(dim=-2), torch.ones(H), atol=1e-4)

    # --- Compose `depth` mixing matrices, as depth does to a residual stream, and
    #     measure the composed transform's gains, over many random stacks. ---
    trials = 200
    plain_fwd, plain_bwd, mhc_fwd, mhc_bwd, mhc_spec = [], [], [], [], []
    for _ in range(trials):
        cp = torch.eye(H, dtype=torch.float64)
        cm = torch.eye(H, dtype=torch.float64)
        for _layer in range(depth):
            cp = unconstrained_mixing(H, perturbation_scale, gen) @ cp
            logits = logit_scale * torch.randn(H, H, generator=gen, dtype=torch.float64)
            cm = sinkhorn_knopp(torch.exp(logits)) @ cm
        f, b = amax_gains(cp)
        plain_fwd.append(f)
        plain_bwd.append(b)
        f, b = amax_gains(cm)
        mhc_fwd.append(f)
        mhc_bwd.append(b)
        mhc_spec.append(torch.linalg.matrix_norm(cm, ord=2).item())

    plain_fwd, plain_bwd = torch.tensor(plain_fwd), torch.tensor(plain_bwd)
    mhc_fwd, mhc_bwd, mhc_spec = torch.tensor(mhc_fwd), torch.tensor(mhc_bwd), torch.tensor(mhc_spec)

    print(f"\nAmax gain of the mixing matrix composed over {depth} layers ({trials} random stacks; 1 = identity-like):")
    print(f"  plain hyper-connections -- forward median {plain_fwd.median():.4g} (max {plain_fwd.max():.4g}), "
          f"backward median {plain_bwd.median():.4g} (max {plain_bwd.max():.4g})")
    print(f"  mHC (Sinkhorn)          -- forward max {mhc_fwd.max():.6f}, backward max {mhc_bwd.max():.6f}, "
          f"spectral norm max {mhc_spec.max():.6f}")

    # With the paper's 20 Sinkhorn iterations the projection is approximate, so the
    # composed gain drifts slightly from 1 (the paper reports a maximum of ~1.6 for its
    # 27B model, versus ~3000 for plain hyper-connections) -- bounded, not exploding.
    assert mhc_fwd.max() < 1.6 and mhc_bwd.max() < 1.6, "mHC gains should stay near 1"
    assert plain_fwd.median() > 1.5, "unconstrained mixing should drift well away from gain 1 at this depth"
    assert plain_fwd.median() > 2 * mhc_fwd.median(), "mHC should be far closer to identity-like than plain HC"

    # Run Sinkhorn to convergence instead, and the guarantee becomes exact: gains pinned
    # at 1 and (Birkhoff-von Neumann) spectral norm <= 1, at any depth.
    cm = torch.eye(H, dtype=torch.float64)
    for _layer in range(depth):
        logits = logit_scale * torch.randn(H, H, generator=gen, dtype=torch.float64)
        cm = sinkhorn_knopp(torch.exp(logits), num_iters=500) @ cm
    f, b = amax_gains(cm)
    spec = torch.linalg.matrix_norm(cm, ord=2).item()
    print(f"  mHC, Sinkhorn run to convergence -- forward {f:.8f}, backward {b:.8f}, spectral norm {spec:.8f}")
    assert abs(f - 1) < 1e-6 and abs(b - 1) < 1e-6 and spec <= 1 + 1e-6

    print("\nchecks passed: mHC gains stay near 1 with 20 iterations and exactly 1 at convergence "
          "(spectral norm <= 1); unconstrained mixing drifts.")
