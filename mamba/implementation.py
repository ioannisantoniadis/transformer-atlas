"""
Selective (Mamba) vs fixed (S4-style) state-space dynamics. Demonstrates
that making the write strength input-dependent lets a model preserve one
important token's contribution against a flood of filler that a fixed
write strength cannot avoid diluting -- the core "selectivity" idea.
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


if __name__ == "__main__":
    torch.manual_seed(0)

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
    print("(Mamba's selectivity) lets the model choose to protect it instead.")
