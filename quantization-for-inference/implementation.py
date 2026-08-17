"""
Weight-only post-training quantization: per-channel min-max quantization
to a low-bit integer grid, measuring reconstruction error and the memory
savings versus 16-bit weights. Also a minimal illustration of AWQ's core
idea -- scaling up "salient" weight channels (those multiplied by
large-magnitude activations) before quantizing, and scaling the matching
activations down to cancel, so quantization resolution is spent where it
actually affects the output.
"""

import torch


def quantize_symmetric(weight, num_bits, per_channel=True):
    """Symmetric min-max quantization. weight: (out_channels, in_channels).
    per_channel=True computes one scale per OUTPUT channel (row); False
    uses a single scale for the whole tensor."""
    qmax = 2 ** (num_bits - 1) - 1  # e.g. 7 for 4-bit signed

    if per_channel:
        scale = weight.abs().amax(dim=-1, keepdim=True) / qmax
    else:
        scale = weight.abs().max() / qmax
    scale = scale.clamp(min=1e-8)

    quantized = (weight / scale).round().clamp(-qmax, qmax)
    dequantized = quantized * scale
    return dequantized, quantized, scale


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- Part 1: bit-width vs reconstruction error, per-channel vs per-tensor ---
    out_channels, in_channels = 4, 64
    weight = torch.randn(out_channels, in_channels)
    weight[1] *= 8.0  # give one output channel a much larger natural scale, to show why per-channel matters

    print("=== quantization error by bit-width and granularity ===")
    for num_bits in (8, 4):
        for per_channel in (False, True):
            dequant, _, _ = quantize_symmetric(weight, num_bits, per_channel)
            mse = (weight - dequant).pow(2).mean().item()
            label = "per-channel" if per_channel else "per-tensor "
            print(f"  {num_bits}-bit, {label}: MSE = {mse:.6f}")

    print(f"\nmemory per parameter: 16-bit = 16 bits, 8-bit = 8 bits (2x smaller), "
          f"4-bit = 4 bits (4x smaller)")
    total_params = out_channels * in_channels
    for bits in (16, 8, 4):
        print(f"  {bits}-bit storage for this {out_channels}x{in_channels} matrix: "
              f"{total_params * bits / 8:.0f} bytes")

    # --- Part 2: AWQ-style activation-aware channel scaling ---
    print("\n=== activation-aware scaling (AWQ idea), 1 output channel x 4 input channels ===")
    w = torch.tensor([0.01, 0.02, -0.015, 5.0])      # channels 0-2: small weights; channel 3: a large outlier
    activation = torch.tensor([100.0, 80.0, 90.0, 1.0])  # channels 0-2 see LARGE activations -- they're "salient"
    exact_output = (w * activation).sum()

    # Naive: one quantization scale for the whole vector -- dominated by the outlier (channel 3),
    # so the small-but-salient channels 0-2 get almost no usable resolution.
    dequant_naive, _, _ = quantize_symmetric(w.unsqueeze(0), num_bits=4, per_channel=False)
    dequant_naive = dequant_naive.squeeze(0)
    output_naive = (dequant_naive * activation).sum()

    # AWQ-style: scale up the salient (high-activation) channels before quantizing,
    # scale the matching activations down by the same factor to cancel exactly.
    protect_scale = torch.tensor([40.0, 40.0, 40.0, 1.0])
    w_scaled = w * protect_scale
    activation_scaled = activation / protect_scale

    dequant_scaled, _, _ = quantize_symmetric(w_scaled.unsqueeze(0), num_bits=4, per_channel=False)
    dequant_scaled = dequant_scaled.squeeze(0)
    output_awq = (dequant_scaled * activation_scaled).sum()

    print(f"original weights:        {w.tolist()}")
    print(f"naive dequantized:       {dequant_naive.tolist()}")
    print(f"AWQ-scaled dequantized:  {(dequant_scaled / protect_scale).tolist()}  (after un-scaling, for comparison)")
    print(f"\nexact (unquantized) output: {exact_output:.4f}")
    print(f"naive 4-bit output:         {output_naive:.4f}  (error: {abs(output_naive - exact_output):.4f})")
    print(f"AWQ-style 4-bit output:     {output_awq:.4f}  (error: {abs(output_awq - exact_output):.4f})")
    print("\nprotecting the salient (high-activation) channels before quantizing reduces output error,")
    print("at 4-bit, for the same quantization grid size -- the core AWQ trade.")
