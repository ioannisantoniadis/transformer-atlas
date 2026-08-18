"""
Structured state space duality (SSD): a scalar-decay SSM's sequential
recurrence and its equivalent masked-attention matrix multiply produce
exactly the same output -- the concrete claim behind "Transformers are
SSMs," and the reformulation that lets this class of model train as a
matmul (tensor cores) instead of a sequential/parallel-scan kernel.
"""

import torch


def scalar_ssm_recurrence(a, b, c, x):
    """h_t = a_t h_{t-1} + b_t x_t, y_t = c_t h_t. a, b, c, x: (T,)."""
    h = torch.tensor(0.0)
    ys = []
    for t in range(x.shape[0]):
        h = a[t] * h + b[t] * x[t]
        ys.append(c[t] * h)
    return torch.stack(ys), h


def scalar_ssm_as_attention_matrix(a, b, c, T):
    """M[t, s] = c_t * b_s * prod_{r=s+1}^{t} a_r for s<=t, else 0 -- the
    same recurrence written as one lower-triangular (masked-attention-shaped)
    matrix, per the SSD equivalence: y = M @ x."""
    M = torch.zeros(T, T)
    for t in range(T):
        for s in range(t + 1):
            decay = torch.tensor(1.0)
            for r in range(s + 1, t + 1):
                decay = decay * a[r]
            M[t, s] = c[t] * b[s] * decay
    return M


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== structured state space duality: recurrence == masked attention ===")
    T = 12
    a = torch.rand(T) * 0.4 + 0.5  # decay rates in [0.5, 0.9), time-varying (input-dependent, as in real Mamba-2)
    b = torch.randn(T)
    c = torch.randn(T)
    x = torch.randn(T)

    y_recurrence, _ = scalar_ssm_recurrence(a, b, c, x)
    M = scalar_ssm_as_attention_matrix(a, b, c, T)
    y_attention = M @ x

    max_diff = (y_recurrence - y_attention).abs().max().item()
    print(f"max difference between the sequential recurrence and the masked matmul: {max_diff:.2e}")
    assert torch.allclose(y_recurrence, y_attention, atol=1e-5), (
        "the scalar-decay SSM recurrence and its masked-attention matrix form "
        "should compute exactly the same output"
    )
    print("identical -- the same scalar-decay SSM is EITHER a sequential recurrence")
    print("OR a masked attention matmul with a structured (1-semiseparable) decay")
    print("mask, matching whichever hardware path is faster to run it on.")
