"""
Three ways to extend a RoPE model's usable context past its training
length: Position Interpolation (compress positions uniformly), NTK-aware
scaling (stretch RoPE's base so low frequencies compress more than high
frequencies), and YaRN (per-dimension ramp blending interpolation and
extrapolation based on wavelength, the most refined of the three).
"""

import math

import torch


def rope_frequencies(head_dim, base=10000.0):
    exponents = torch.arange(0, head_dim, 2).float() / head_dim
    return 1.0 / (base ** exponents)


def position_interpolation_angles(position, head_dim, base, train_len, target_len):
    """Compress the position uniformly across all frequencies."""
    theta = rope_frequencies(head_dim, base)
    scaled_position = position * (train_len / target_len)
    return scaled_position * theta


def ntk_aware_angles(position, head_dim, base, train_len, target_len):
    """Stretch the ROPE base instead of scaling position -- low
    (slow-rotating) frequencies get compressed much more than high ones."""
    scaled_base = base * (target_len / train_len) ** (head_dim / (head_dim - 2))
    theta = rope_frequencies(head_dim, scaled_base)
    return position * theta


def yarn_angles(position, head_dim, base, train_len, target_len, alpha=1.0, beta=32.0):
    """Per-dimension ramp: high-frequency dims (short wavelength relative
    to train_len) are extrapolated as-is; low-frequency dims (long
    wavelength) are interpolated (PI-style); dims in between get a linear
    blend. `alpha`/`beta` are the ramp's rotation-count thresholds, as in
    the YaRN paper."""
    theta = rope_frequencies(head_dim, base)
    wavelength = 2 * math.pi / theta
    num_rotations_within_training = train_len / wavelength  # how many full turns fit in train_len

    # gamma=1 -> keep original (extrapolate); gamma=0 -> PI-interpolate.
    gamma = ((num_rotations_within_training - alpha) / (beta - alpha)).clamp(0, 1)

    original_angle = position * theta
    interpolated_angle = position * (train_len / target_len) * theta
    return gamma * original_angle + (1 - gamma) * interpolated_angle, gamma


if __name__ == "__main__":
    head_dim, base = 32, 10000.0
    train_len, target_len = 2048, 16384  # 8x context extension

    query_position = target_len - 1  # the farthest, most out-of-distribution position

    theta = rope_frequencies(head_dim, base)
    dims_to_inspect = [0, head_dim // 4, head_dim // 2 - 1]  # high-freq, mid-freq, low-freq dim pairs

    naive_angles = query_position * theta
    pi_angles = position_interpolation_angles(query_position, head_dim, base, train_len, target_len)
    ntk_angles = ntk_aware_angles(query_position, head_dim, base, train_len, target_len)
    yarn_ang, gamma = yarn_angles(query_position, head_dim, base, train_len, target_len)

    # The angle this SAME dimension reached at the farthest position seen
    # during training -- the reference range each method should try to stay near.
    train_reference_angles = (train_len - 1) * theta

    print(f"query position {query_position} (target_len={target_len}, train_len={train_len}, "
          f"extension factor {target_len / train_len:.0f}x)\n")
    header = f"{'dim pair':>10} {'train-max':>10} {'naive':>10} {'PI':>10} {'NTK-aware':>10} {'YaRN':>10} {'YaRN gamma':>11}"
    print(header)
    for d in dims_to_inspect:
        print(f"{d:>10} {train_reference_angles[d]:>10.2f} {naive_angles[d]:>10.2f} "
              f"{pi_angles[d]:>10.2f} {ntk_angles[d]:>10.2f} {yarn_ang[d]:>10.2f} {gamma[d]:>11.3f}")

    print("\nnaive extrapolation drifts far past the training-time angle range at every frequency.")
    print("PI stays within range everywhere, but compresses high-freq dims the model didn't need compressed.")
    print("NTK-aware and YaRN leave high-freq dims closer to their natural (uncompressed) angle,")
    print("compressing mainly the low-freq dims where extrapolation actually hurts -- "
          "YaRN's gamma shows this ramp explicitly (near 1.0 = extrapolate, near 0.0 = interpolate).")
