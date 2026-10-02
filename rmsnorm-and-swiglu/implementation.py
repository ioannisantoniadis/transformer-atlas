"""
RMSNorm (LayerNorm without mean-centering) and SwiGLU (gated FFN with
Swish/SiLU activation) — the normalization and feedforward swaps used by
LLaMA-family models in place of the original Transformer's LayerNorm and
ReLU-FFN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


class SwiGLU(nn.Module):
    """Gated FFN: Swish(x W_gate) elementwise-multiplied by (x W_up), then
    projected down. d_ff defaults to 8/3 * d_model so parameter count stays
    close to a standard 4x ReLU-FFN despite the extra weight matrix."""

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model = 2, 5, 16

    x = torch.randn(batch, seq_len, d_model) * 10 + 3  # non-zero mean, large scale

    rmsnorm = RMSNorm(d_model)
    normed = rmsnorm(x)
    print("input  mean/std per token (first token):", x[0, 0].mean().item(), x[0, 0].std().item())
    print("RMSNorm output rms (should be ~1 before gamma scaling):",
          (normed[0, 0] / rmsnorm.gamma).pow(2).mean().sqrt().item())

    layernorm = nn.LayerNorm(d_model)
    ln_params = sum(p.numel() for p in layernorm.parameters())
    rms_params = sum(p.numel() for p in rmsnorm.parameters())
    print(f"LayerNorm params: {ln_params}, RMSNorm params: {rms_params} (no beta/bias)")

    swiglu = SwiGLU(d_model)
    out = swiglu(x)
    print("SwiGLU output shape:", out.shape)

    standard_ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))
    swiglu_params = sum(p.numel() for p in swiglu.parameters())
    standard_params = sum(p.numel() for p in standard_ffn.parameters())
    print(f"standard 4x ReLU-FFN params: {standard_params}, SwiGLU (8/3x) params: {swiglu_params}")

    # --- Checks: RMSNorm rescales each token to unit RMS (before gamma) with d_model
    #     parameters (no beta); SwiGLU's 8/3 hidden width keeps parameters comparable to a 4x FFN. ---
    rms = (normed / rmsnorm.gamma).pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)
    assert rms_params == d_model and ln_params == 2 * d_model
    assert out.shape == x.shape
    assert abs(swiglu_params - standard_params) / standard_params < 0.15
    print("\nchecks passed: unit RMS; no bias in RMSNorm; SwiGLU parameter-matched to a 4x FFN")
