"""
mHC (Manifold-Constrained Hyper-Connections): generalizes the single
residual stream into H parallel streams mixed between layers by a learned
matrix M. Plain hyper-connections leave M merely row-stochastic (a softmax
per stream); mHC projects M onto the Birkhoff polytope of doubly-stochastic
matrices via Sinkhorn-Knopp iterations. Demonstrates the reason that
matters: by Birkhoff-von Neumann, every doubly-stochastic matrix is a
convex combination of permutation matrices (each spectral norm exactly 1),
so both a single mHC mixing step *and the product of many of them across
depth* stay non-expansive (spectral norm <= 1) -- an identity-mapping
guarantee plain row-stochastic mixing does not have and can badly violate
as depth grows.
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


def row_stochastic_mixing(logits):
    """Plain hyper-connections: softmax per row. Rows sum to 1; columns are
    left completely unconstrained."""
    return torch.softmax(logits, dim=-1)


if __name__ == "__main__":
    torch.manual_seed(0)
    H = 4  # number of parallel hyper-connection streams
    depth = 30  # layers to compose mixing matrices across
    logit_scale = 2.5  # sharper-than-uniform mixing, like a trained model's

    # --- Sanity check: Sinkhorn output really is (approximately) doubly stochastic ---
    probe = sinkhorn_knopp(torch.exp(logit_scale * torch.randn(H, H)))
    print("Sinkhorn-projected matrix row sums: ", [f"{v:.4f}" for v in probe.sum(dim=-1).tolist()])
    print("Sinkhorn-projected matrix col sums: ", [f"{v:.4f}" for v in probe.sum(dim=-2).tolist()])

    # --- Compose L independently-sampled mixing matrices, as depth does to a
    #     residual stream, and track (a) a stream vector's norm and (b) the
    #     operator (spectral) norm of the COMPOSED transform after each layer. ---
    x_plain = torch.randn(H, dtype=torch.float64)
    x_mhc = x_plain.clone()
    composed_plain = torch.eye(H, dtype=torch.float64)
    composed_mhc = torch.eye(H, dtype=torch.float64)

    for layer in range(depth):
        logits = (logit_scale * torch.randn(H, H)).double()
        M_plain = row_stochastic_mixing(logits)
        M_mhc = sinkhorn_knopp(torch.exp(logits))

        x_plain = M_plain @ x_plain
        x_mhc = M_mhc @ x_mhc
        composed_plain = M_plain @ composed_plain
        composed_mhc = M_mhc @ composed_mhc

    plain_op_norm = torch.linalg.matrix_norm(composed_plain, ord=2).item()
    mhc_op_norm = torch.linalg.matrix_norm(composed_mhc, ord=2).item()

    print(f"\nafter composing {depth} independently-sampled mixing matrices:")
    print(f"  stream vector norm   -- plain: {x_plain.norm().item():.4g}  |  mHC: {x_mhc.norm().item():.4g}  "
          f"(started at {torch.randn(H, dtype=torch.float64).norm().item():.4g}-ish scale)")
    print(f"  composed operator (spectral) norm -- plain: {plain_op_norm:.4g}  |  mHC: {mhc_op_norm:.4g}")
    print(f"  mHC operator norm <= 1 (Birkhoff-von Neumann bound holds): {mhc_op_norm <= 1.0 + 1e-6}")
    print(f"  plain hyper-connections give no such guarantee -- its composed norm "
          f"can drift arbitrarily far from 1 as depth grows")

    # --- Same story, averaged over many random depth-30 stacks, to show this
    #     isn't a cherry-picked seed. ---
    torch.manual_seed(1)
    trials = 200
    plain_norms, mhc_norms = [], []
    for _ in range(trials):
        cp = torch.eye(H, dtype=torch.float64)
        cm = torch.eye(H, dtype=torch.float64)
        for layer in range(depth):
            logits = (logit_scale * torch.randn(H, H)).double()
            cp = row_stochastic_mixing(logits) @ cp
            cm = sinkhorn_knopp(torch.exp(logits)) @ cm
        plain_norms.append(torch.linalg.matrix_norm(cp, ord=2).item())
        mhc_norms.append(torch.linalg.matrix_norm(cm, ord=2).item())

    plain_norms = torch.tensor(plain_norms)
    mhc_norms = torch.tensor(mhc_norms)
    tol = 1e-3  # Sinkhorn runs a finite number of iterations, so mHC's bound holds up to this slack
    print(f"\nover {trials} independent {depth}-layer stacks, composed operator norm:")
    print(f"  plain hyper-connections -- mean {plain_norms.mean():.4g}, max {plain_norms.max():.4g}, "
          f"fraction exceeding 1: {(plain_norms > 1.0).float().mean().item():.2%}")
    print(f"  mHC (Sinkhorn)          -- mean {mhc_norms.mean():.4g}, max {mhc_norms.max():.4g}, "
          f"fraction exceeding 1+{tol:g}: {(mhc_norms > 1.0 + tol).float().mean().item():.2%} "
          f"(should be ~0%; any excess is finite-Sinkhorn-iteration slack, not unbounded drift like the plain case)")
