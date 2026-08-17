"""
Ring Attention: simulate N "devices" arranged in a ring, each starting
with one Q/K/V block. Over N rounds, each device attends its fixed Q
block against a K/V block that rotates in from its neighbor, accumulating
an online-softmax running state (numerator, denominator, running max) --
the same accumulation FlashAttention uses across tiles on one device,
here applied across simulated devices. After N rounds every device holds
the exact causal attention output for its queries.
"""

import math

import torch


def block_update(state, q, k, v, causal_mode):
    """One round's contribution for one device: (running_out, running_sum,
    running_max) updated with a new K/V block, online-softmax style.
    causal_mode: 'skip' (future block, contributes nothing), 'full' (past
    block, no mask needed), or 'diagonal' (this device's own block, needs
    a local causal mask)."""
    running_out, running_sum, running_max = state
    if causal_mode == "skip":
        return state

    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (block_size, block_size)
    if causal_mode == "diagonal":
        block_size = q.size(0)
        local_causal = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        scores = scores.masked_fill(~local_causal, float("-inf"))

    block_max = scores.max(dim=-1).values
    new_max = torch.maximum(running_max, block_max)
    correction = torch.nan_to_num(torch.exp(running_max - new_max), nan=0.0)

    exp_scores = torch.nan_to_num(torch.exp(scores - new_max.unsqueeze(-1)), nan=0.0)
    new_sum = running_sum * correction + exp_scores.sum(dim=-1)
    new_out = running_out * correction.unsqueeze(-1) + exp_scores @ v

    return new_out, new_sum, new_max


def ring_attention_causal(q_blocks, k_blocks, v_blocks):
    """q_blocks[d], k_blocks[d], v_blocks[d]: device d's local blocks,
    each (block_size, d_k). Returns per-device outputs -- the exact causal
    attention result for each device's queries, computed via N rounds of
    receiving a rotating K/V block instead of ever holding the full K/V."""
    num_devices = len(q_blocks)
    block_size, d_k = q_blocks[0].shape

    states = [(torch.zeros(block_size, d_k), torch.zeros(block_size), torch.full((block_size,), float("-inf")))
              for _ in range(num_devices)]

    for round_offset in range(num_devices):  # round_offset=0: own block; then rotate in neighbors'
        for device in range(num_devices):
            kv_block_idx = (device - round_offset) % num_devices  # block currently "arrived" at this device
            if kv_block_idx > device:
                mode = "skip"       # a future block -- causally invisible, contributes nothing
            elif kv_block_idx == device:
                mode = "diagonal"   # this device's own block -- needs the local triangular mask
            else:
                mode = "full"       # a past block -- every position in it is causally valid
            states[device] = block_update(states[device], q_blocks[device], k_blocks[kv_block_idx],
                                           v_blocks[kv_block_idx], mode)

    return [out / running_sum.unsqueeze(-1) for out, running_sum, _ in states]


def naive_causal_attention(q, k, v):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    mask = torch.tril(torch.ones(q.size(0), k.size(0), dtype=torch.bool))
    weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
    return weights @ v


if __name__ == "__main__":
    torch.manual_seed(0)
    num_devices, block_size, d_k = 4, 5, 8
    seq_len = num_devices * block_size

    full_q = torch.randn(seq_len, d_k)
    full_k = torch.randn(seq_len, d_k)
    full_v = torch.randn(seq_len, d_k)

    q_blocks = list(full_q.split(block_size))
    k_blocks = list(full_k.split(block_size))
    v_blocks = list(full_v.split(block_size))

    ring_outputs = ring_attention_causal(q_blocks, k_blocks, v_blocks)
    ring_output = torch.cat(ring_outputs, dim=0)

    naive_output = naive_causal_attention(full_q, full_k, full_v)

    max_diff = (ring_output - naive_output).abs().max().item()
    print(f"devices={num_devices}, block_size={block_size}, total seq_len={seq_len}")
    print(f"max abs diff (ring-distributed vs naive full causal attention): {max_diff:.2e}")
    print(f"ring attention output matches naive full attention exactly: "
          f"{torch.allclose(ring_output, naive_output, atol=1e-5)}")

    print(f"\nper-device peak K/V held at once: {block_size} tokens "
          f"(never the full {seq_len}-token sequence, regardless of num_devices)")
    print(f"communication per full attention pass: {num_devices} block rotations of size {block_size}, "
          f"vs each device needing the full {seq_len}-token K/V without ring rotation")
