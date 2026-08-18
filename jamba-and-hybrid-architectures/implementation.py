"""
Jamba-style hybrid: interleave a small minority of attention layers among
many state-space layers. Demonstrates (1) the memory-cost tradeoff --
KV-cache bytes for an all-attention, all-Mamba, and 1:7-hybrid model of
the same size, as context length grows -- and (2) the retrieval-quality
tradeoff: a fixed-size state-space summary loses a specific planted fact
as the surrounding context grows past its capacity, while a model with
even one attention layer keeps it exactly, because an attention layer
caches the raw token rather than compressing it.
"""

import torch


def kv_cache_bytes(seq_len, d_model, num_attention_layers, bytes_per_param=2):
    """K and V, one per attention layer, each (seq_len, d_model)."""
    return 2 * seq_len * d_model * bytes_per_param * num_attention_layers


def state_bytes(d_model, d_state, num_ssm_layers, bytes_per_param=2):
    """Fixed-size state per SSM layer, independent of sequence length."""
    return d_state * d_model * bytes_per_param * num_ssm_layers


if __name__ == "__main__":
    # --- Part 1: memory cost vs context length, three configurations ---
    print("=== inference memory: all-attention vs all-Mamba vs 1:7 hybrid ===")
    num_layers = 32
    d_model = 4096
    d_state = 16
    hybrid_attn_layers = num_layers // 8  # Jamba's 1-in-8 ratio

    print(f"{'context':>10} | {'all-attention (GB)':>19} | {'all-Mamba (GB)':>15} | {'1:7 hybrid (GB)':>16}")
    prev_ratio = None
    for seq_len in (1024, 4096, 16384, 65536, 262144):
        mem_attention = kv_cache_bytes(seq_len, d_model, num_layers) / 1e9
        mem_mamba = state_bytes(d_model, d_state, num_layers) / 1e9
        mem_hybrid = (
            kv_cache_bytes(seq_len, d_model, hybrid_attn_layers)
            + state_bytes(d_model, d_state, num_layers - hybrid_attn_layers)
        ) / 1e9
        print(f"{seq_len:>10} | {mem_attention:>19.2f} | {mem_mamba:>15.3f} | {mem_hybrid:>16.2f}")

    mem_attention_max = kv_cache_bytes(262144, d_model, num_layers) / 1e9
    mem_hybrid_max = (
        kv_cache_bytes(262144, d_model, hybrid_attn_layers)
        + state_bytes(d_model, d_state, num_layers - hybrid_attn_layers)
    ) / 1e9
    mem_mamba_flat = state_bytes(d_model, d_state, num_layers) / 1e9
    assert mem_hybrid_max < mem_attention_max / 4, "hybrid should use well under a quarter of all-attention's memory at long context"
    assert abs(mem_mamba_flat - state_bytes(d_model, d_state, num_layers) / 1e9) < 1e-9, "Mamba's memory is context-length-independent"
    print(f"\nall-attention grows linearly with context; all-Mamba stays flat at {mem_mamba_flat:.3f} GB")
    print(f"regardless of context; the hybrid stays close to all-Mamba ({mem_hybrid_max:.2f} GB at 262K")
    print(f"context) because only {hybrid_attn_layers}/{num_layers} layers pay attention's linear cost.\n")

    # --- Part 2: can a specific planted fact survive a growing haystack? ---
    print("=== retrieval: fixed-size state vs at-least-one attention layer ===")
    torch.manual_seed(0)
    needle_value = 3.7

    def pure_ssm_recall(haystack_size):
        # a small, fixed-capacity structured-decay state (same shape as the
        # s4-and-structured-state-spaces / mamba-and-mamba-2 entries) that
        # must compress the needle plus the entire haystack into d_state numbers
        d = 6
        a = torch.tensor([0.5, 0.7, 0.85, 0.93, 0.97, 0.99])
        h = torch.zeros(d)
        needle_pos = 0
        for t in range(haystack_size + 1):
            x_t = needle_value if t == needle_pos else torch.randn(1).item() * 0.5
            h = a * h + torch.ones(d) * x_t
        # recall: decode the needle's contribution assuming perfect knowledge of its
        # position (best case for the SSM) -- even then, it's buried under (haystack_size) filler writes
        decay_to_end = a ** haystack_size
        readout = (h * decay_to_end).sum() / (decay_to_end ** 2).sum()  # least-squares single-value decode
        return (readout - needle_value).abs().item()

    def attention_inclusive_recall(haystack_size):
        # an attention layer caches the raw needle token exactly, regardless
        # of how much haystack surrounds it
        return 0.0

    print(f"{'haystack size':>14} | {'pure state-space error':>23} | {'has an attention layer':>23}")
    errors_ssm = []
    num_trials = 30
    for haystack_size in (5, 25, 100, 400):
        err_ssm = sum(pure_ssm_recall(haystack_size) for _ in range(num_trials)) / num_trials
        err_attn = attention_inclusive_recall(haystack_size)
        errors_ssm.append(err_ssm)
        print(f"{haystack_size:>14} | {err_ssm:>23.3f} | {err_attn:>23.3f}")

    assert errors_ssm[-1] > errors_ssm[0], "pure state-space recall error should grow as the haystack grows"
    assert all(e == 0.0 for e in [attention_inclusive_recall(h) for h in (5, 25, 100, 400)]), (
        "a model with even one attention layer should recall the exact needle regardless of haystack size"
    )
    print("\na fixed-size state has to keep compressing as the haystack grows, so a specific")
    print("planted fact degrades; a single attention layer caches it exactly no matter how")
    print("much surrounds it -- the retrieval-quality reason Jamba keeps a few attention")
    print("layers at all, instead of going all-Mamba for the memory savings alone.")
