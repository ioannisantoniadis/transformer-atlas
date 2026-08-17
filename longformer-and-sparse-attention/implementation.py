"""
Longformer-style attention: local sliding window for most tokens, plus a
small set of "global" tokens that attend to (and are attended by)
everyone. Demonstrates the mask's O(n) sparsity versus full O(n^2)
attention, and verifies the two-layer information-propagation claim: a
global token lets information reach a far-away token in two attention
layers, which a pure local window cannot do at all regardless of depth
(for tokens farther apart than num_layers * window_size).
"""

import math

import torch
import torch.nn.functional as F


def longformer_mask(seq_len, window_size, global_token_indices):
    """(seq_len, seq_len) bool mask: local window OR global row/column."""
    positions = torch.arange(seq_len)
    i, j = positions.unsqueeze(1), positions.unsqueeze(0)
    local = (i - j).abs() <= window_size // 2  # bidirectional window (Longformer is encoder-style)

    is_global = torch.zeros(seq_len, dtype=torch.bool)
    is_global[list(global_token_indices)] = True
    global_row = is_global.unsqueeze(1).expand(seq_len, seq_len)   # global tokens attend to everything
    global_col = is_global.unsqueeze(0).expand(seq_len, seq_len)   # everything attends to global tokens

    return local | global_row | global_col


def attention_with_mask(x, mask):
    """Single bidirectional attention layer, identity Q/K/V projections
    (so we can literally trace which tokens' VALUES reach the output)."""
    scores = x @ x.transpose(-2, -1) / math.sqrt(x.size(-1))
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ x, weights


if __name__ == "__main__":
    torch.manual_seed(0)
    seq_len, window_size, d_model = 21, 3, 6
    global_tokens = {10}  # one global "hub" token in the middle

    mask = longformer_mask(seq_len, window_size, global_tokens)
    full_entries = seq_len * seq_len
    local_only_entries = ((torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)).abs()
                           <= window_size // 2).sum().item()
    longformer_entries = mask.sum().item()
    print(f"seq_len={seq_len}, window_size={window_size}, global tokens={global_tokens}")
    print(f"attended pairs -- full attention: {full_entries}, local-only: {local_only_entries}, "
          f"local+global: {longformer_entries}")

    # --- Two-layer propagation claim, tested by perturbation ---
    # Token A (position 0) and token B (position 20) are far apart -- well
    # outside each other's window and outside the global token's window
    # too. Perturb A's input and see whether B's output changes after 1
    # layer vs 2 layers of stacking the SAME mask -- a direct causal test
    # of "does information from A reach B," not just a nonzero weight.
    torch.manual_seed(1)
    x = torch.randn(seq_len, d_model)
    token_a, token_b = 0, seq_len - 1

    def stacked_output(x, mask, num_layers, token):
        h = x
        for _ in range(num_layers):
            h, _ = attention_with_mask(h, mask)
        return h[token]

    def perturbation_effect(mask, num_layers):
        baseline = stacked_output(x, mask, num_layers, token_b)
        x_perturbed = x.clone()
        x_perturbed[token_a] += 100.0  # large, unmistakable perturbation to token A only
        perturbed = stacked_output(x_perturbed, mask, num_layers, token_b)
        return (perturbed - baseline).abs().sum().item()

    print(f"\ntoken A={token_a}, token B={token_b} (distance {token_b - token_a}, "
          f"window_size={window_size}, global token at {list(global_tokens)[0]})")
    print("effect on B's output of perturbing A (0.0 = no information reached B):")
    for num_layers in (1, 2):
        effect_global = perturbation_effect(mask, num_layers)
        print(f"  {num_layers} layer(s), local+global mask: {effect_global:.4f}")

    local_mask = longformer_mask(seq_len, window_size, global_token_indices=set())
    for num_layers in (1, 2):
        effect_local = perturbation_effect(local_mask, num_layers)
        print(f"  {num_layers} layer(s), local-only mask:   {effect_local:.4f}")
    print("\nexpected: local+global reaches nonzero at 2 layers (via the global hub); "
          "local-only stays 0.0 at both -- window=3 can't bridge a 20-position gap that fast.")
