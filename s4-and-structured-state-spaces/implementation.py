"""
S4: a linear state-space recurrence with two dual computations -- a
sequential recurrence (cheap inference, fixed-size state) and an
equivalent convolution (parallel training). Demonstrates (1) the two
computations produce numerically identical output, the core duality S4
is built on, and (2) why the STRUCTURE of the state matrix A matters, not
just having a state: a uniform single decay rate wastes state capacity
(every channel carries redundant information), while several distinct
decay rates (a simplified stand-in for HiPPO's structured initialization)
let a fixed-size state reconstruct multiple past events that a
uniform-decay state of the same size cannot separate.
"""

import torch


def ssm_recurrence(A_bar, B_bar, C, x):
    """Sequential form: h_t = A_bar h_{t-1} + B_bar x_t, y_t = C . h_t."""
    d_state = A_bar.shape[0]
    h = torch.zeros(d_state)
    ys = []
    for t in range(x.shape[0]):
        h = A_bar @ h + B_bar * x[t]
        ys.append(C @ h)
    return torch.stack(ys)


def ssm_convolution_kernel(A_bar, B_bar, C, length):
    """K_k = C . A_bar^k . B_bar for k=0..length-1 -- the impulse response."""
    d_state = A_bar.shape[0]
    kernel = []
    A_power = torch.eye(d_state)
    for _ in range(length):
        kernel.append(C @ A_power @ B_bar)
        A_power = A_bar @ A_power
    return torch.stack(kernel)


def ssm_convolution(A_bar, B_bar, C, x):
    """Parallel form: y_t = sum_{k=0}^{t} K_k x_{t-k}, a causal convolution."""
    seq_len = x.shape[0]
    kernel = ssm_convolution_kernel(A_bar, B_bar, C, seq_len)
    y = torch.zeros(seq_len)
    for t in range(seq_len):
        y[t] = (kernel[: t + 1].flip(0) * x[: t + 1]).sum()
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    d_state = 4

    # --- Part 1: recurrence and convolution are the same computation ---
    print("=== recurrence vs convolution: same parameters, same output? ===")
    A_bar = torch.diag(torch.tensor([0.5, 0.7, 0.9, 0.95]))
    B_bar = torch.randn(d_state)
    C = torch.randn(d_state)
    x = torch.randn(30)

    y_recurrence = ssm_recurrence(A_bar, B_bar, C, x)
    y_convolution = ssm_convolution(A_bar, B_bar, C, x)
    max_diff = (y_recurrence - y_convolution).abs().max().item()
    print(f"max difference between the two computations: {max_diff:.2e}")
    assert torch.allclose(y_recurrence, y_convolution, atol=1e-5), (
        "recurrence and convolution should be exactly the same computation"
    )
    print("identical, up to floating point -- one set of parameters, two algorithms.\n")

    # --- Part 2: structured (multi-timescale) vs uniform (single-timescale) state ---
    print("=== reconstructing 4 past events from a 4-dim final state ===")
    total_len = 60
    impulse_times = [5, 20, 35, 50]  # 4 distinct past moments
    true_magnitudes = torch.tensor([1.3, -0.7, 2.1, -1.5])

    def run_to_end(A_bar_case):
        h = torch.zeros(d_state)
        for t in range(total_len):
            x_t = 0.0
            if t in impulse_times:
                x_t = true_magnitudes[impulse_times.index(t)].item()
            h = A_bar_case @ h + torch.ones(d_state) * x_t  # B_bar = all-ones: every channel receives the same input
        return h

    def reconstruct(A_bar_case, h_final):
        # K[c, i] = contribution of impulse i to channel c after decaying for (total_len-1-t_i) steps
        K = torch.stack([
            torch.stack([A_bar_case[c, c] ** (total_len - 1 - t) for t in impulse_times])
            for c in range(d_state)
        ])
        recon, *_ = torch.linalg.lstsq(K, h_final.unsqueeze(-1))
        return recon.squeeze(-1)

    # uniform: every channel decays at the SAME rate -- redundant, only 1 effective dimension
    A_uniform = torch.diag(torch.full((d_state,), 0.9))
    h_uniform = run_to_end(A_uniform)
    recon_uniform = reconstruct(A_uniform, h_uniform)

    # structured: several distinct decay rates -- a simplified HiPPO-style stand-in
    A_structured = torch.diag(torch.tensor([0.5, 0.8, 0.95, 0.99]))
    h_structured = run_to_end(A_structured)
    recon_structured = reconstruct(A_structured, h_structured)

    err_uniform = (recon_uniform - true_magnitudes).norm().item()
    err_structured = (recon_structured - true_magnitudes).norm().item()

    print(f"true impulse magnitudes:            {true_magnitudes.tolist()}")
    print(f"recovered from uniform-decay state:  {[f'{v:.2f}' for v in recon_uniform.tolist()]} (error {err_uniform:.3f})")
    print(f"recovered from structured state:     {[f'{v:.2f}' for v in recon_structured.tolist()]} (error {err_structured:.3f})")
    assert err_structured < err_uniform, (
        "a multi-timescale (structured) state should reconstruct distinct past "
        "events far better than a same-size but uniform-decay state"
    )
    print("same state size (4 numbers) either way -- uniform decay wastes capacity")
    print("(every channel carries the same information); distinct decay rates per")
    print("channel (S4's HiPPO-motivated structure) let the same-size state actually")
    print("separate multiple past events.")
