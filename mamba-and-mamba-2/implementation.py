"""
Selective (Mamba) vs fixed (S4-style) state-space dynamics, and Mamba-2's
structured state-space duality (SSD). Demonstrates (1) making the write
strength input-dependent lets a model preserve one important token's
contribution against a flood of filler that a fixed write strength cannot
avoid diluting, and (2) a scalar-decay SSM's sequential recurrence and its
equivalent masked-attention matrix multiply produce exactly the same
output -- the concrete claim behind "Transformers are SSMs."
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

    # --- Part 1: selective (input-dependent) vs fixed write strength ---
    print("=== selectivity: protecting one important token from a flood of filler ===")
    T = 25
    signal_pos = 3
    signal_value = 2.0
    decay = torch.full((T,), 0.95)  # same decay rate in both cases -- only b differs
    filler = torch.randn(T) * 0.3
    filler[signal_pos] = 0.0  # the signal step's "filler component" is 0; signal is injected separately below
    x = filler.clone()
    x[signal_pos] = signal_value

    # non-selective: every step writes with the SAME strength, signal or not
    b_fixed = torch.ones(T)
    y_fixed, h_fixed = scalar_ssm_recurrence(decay, b_fixed, torch.ones(T), x)

    # selective: write strength depends on the input -- large for the signal, small for filler
    b_selective = torch.where(x == signal_value, torch.tensor(4.0), torch.tensor(0.05))
    y_selective, h_selective = scalar_ssm_recurrence(decay, b_selective, torch.ones(T), x)

    def signal_to_noise(a_seq, b_seq, x_seq, h_final):
        signal_contribution = b_seq[signal_pos] * x_seq[signal_pos] * a_seq[signal_pos + 1 : T].prod()
        noise_contribution = h_final - signal_contribution
        return (signal_contribution.abs() / (noise_contribution.abs() + 1e-8)).item()

    snr_fixed = signal_to_noise(decay, b_fixed, x, h_fixed)
    snr_selective = signal_to_noise(decay, b_selective, x, h_selective)

    print(f"final state (fixed write strength):     {h_fixed.item():.3f} -- signal-to-noise ratio: {snr_fixed:.2f}")
    print(f"final state (selective write strength):  {h_selective.item():.3f} -- signal-to-noise ratio: {snr_selective:.2f}")
    assert snr_selective > snr_fixed, (
        "input-dependent (selective) write strength should let the important "
        "token dominate the final state far more than a fixed write strength can"
    )
    print("with a fixed write strength, the signal competes on equal terms with every")
    print("filler step and gets diluted; making the write strength depend on the input")
    print("(Mamba's selectivity) lets the model choose to protect it instead.\n")

    # --- Part 2: SSD -- sequential recurrence and masked-attention matmul are the same computation ---
    print("=== structured state space duality: recurrence == masked attention ===")
    T2 = 12
    a2 = torch.rand(T2) * 0.4 + 0.5  # decay rates in [0.5, 0.9), time-varying (input-dependent, as in real Mamba-2)
    b2 = torch.randn(T2)
    c2 = torch.randn(T2)
    x2 = torch.randn(T2)

    y_recurrence, _ = scalar_ssm_recurrence(a2, b2, c2, x2)
    M = scalar_ssm_as_attention_matrix(a2, b2, c2, T2)
    y_attention = M @ x2

    max_diff = (y_recurrence - y_attention).abs().max().item()
    print(f"max difference between the sequential recurrence and the masked matmul: {max_diff:.2e}")
    assert torch.allclose(y_recurrence, y_attention, atol=1e-5), (
        "the scalar-decay SSM recurrence and its masked-attention matrix form "
        "should compute exactly the same output"
    )
    print("identical -- the same scalar-decay SSM is EITHER a sequential recurrence")
    print("OR a masked attention matmul with a structured (1-semiseparable) decay")
    print("mask, matching whichever hardware path is faster to run it on.")
