"""
Rotary Position Embedding (RoPE): rotate query/key vectors by an
angle proportional to position, so their dot product depends only on
relative position. Demonstrates the core relative-position property with a
direct numerical check.
"""

import torch


def rope_frequencies(head_dim, base=10000.0):
    """theta_i for i in 0..head_dim/2, shape (head_dim // 2,)."""
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    exponents = torch.arange(0, head_dim, 2).float() / head_dim
    return 1.0 / (base ** exponents)


def apply_rope(x, positions, theta):
    """x: (..., seq_len, head_dim). positions: (seq_len,). theta: (head_dim // 2,).
    Rotates each consecutive pair of dimensions by position * theta_i."""
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    angles = positions[:, None].float() * theta[None, :]  # (seq_len, head_dim // 2)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)

    x_pairs = x.view(*x.shape[:-1], head_dim // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    rotated_pairs = torch.stack([-x2, x1], dim=-1).view(*x.shape)  # 90-degree-rotated x

    return x * cos + rotated_pairs * sin


if __name__ == "__main__":
    torch.manual_seed(0)
    head_dim = 8
    theta = rope_frequencies(head_dim)

    seq_len = 10
    q = torch.randn(seq_len, head_dim)
    k = torch.randn(seq_len, head_dim)
    positions = torch.arange(seq_len)

    q_rot = apply_rope(q, positions, theta)
    k_rot = apply_rope(k, positions, theta)

    # Core property: dot product after rotation depends only on (m - n),
    # not on the absolute positions m, n themselves.
    m, n = 7, 2  # relative distance = 5
    score_a = (q_rot[m] * k_rot[n]).sum()

    shift = 3
    positions_shifted = positions + shift
    q_rot_shifted = apply_rope(q, positions_shifted, theta)
    k_rot_shifted = apply_rope(k, positions_shifted, theta)
    score_b = (q_rot_shifted[m] * k_rot_shifted[n]).sum()  # same relative distance = 5

    print(f"score at absolute positions ({m},{n}):         {score_a.item():.6f}")
    print(f"score at shifted positions ({m+shift},{n+shift}) same gap: {score_b.item():.6f}")
    print(f"relative-position invariance holds: {torch.allclose(score_a, score_b, atol=1e-4)}")

    # Rotation preserves vector norm (it's an orthogonal transform).
    print(f"norm preserved: {torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)}")
