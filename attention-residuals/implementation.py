"""
Attention Residuals (AttnRes): replaces the fixed unit-weight residual
connection x_l = f_l(x_{l-1}) + x_{l-1} with softmax attention over ALL
preceding layer outputs, x_l = f_l(x_{l-1}) + sum_j a_{l,j} x_j. Demonstrates
(1) the running representation's norm grows with depth under plain residual
accumulation but stays bounded under attention-weighted (convex-combination)
aggregation, and (2) the structural capability difference this creates: a
later layer under AttnRes CAN choose to lean on an earlier, cleaner layer
and downweight one that turned out to inject something unhelpful -- a fixed
weight of exactly 1 gives standard residual no such choice at all.
"""

import torch


def standard_residual_step(x_prev, new_content):
    """x_l = f_l(x_{l-1}) + x_{l-1} -- fixed weight of 1, only x_{l-1} reachable."""
    return x_prev + new_content


def attn_residual_step(layer_outputs, new_content, attn_weights):
    """x_l = f_l(x_{l-1}) + sum_j a_j * layer_outputs[j]. attn_weights must be
    non-negative and sum to 1 (a real softmax output would guarantee this)."""
    assert torch.allclose(attn_weights.sum(), torch.tensor(1.0), atol=1e-5)
    history = torch.stack(layer_outputs, dim=0)  # (num_layers, d)
    weighted = (attn_weights.unsqueeze(-1) * history).sum(dim=0)
    return new_content + weighted


if __name__ == "__main__":
    torch.manual_seed(0)
    d = 16

    # --- Part 1: state norm grows under plain accumulation, stays bounded under attention ---
    print("=== state norm across depth: unweighted sum vs attention-weighted average ===")
    depth = 12
    per_layer_content = [torch.randn(d) for _ in range(depth)]

    x = per_layer_content[0]
    standard_norms = [x.norm().item()]
    for l in range(1, depth):
        x = standard_residual_step(x, per_layer_content[l])
        standard_norms.append(x.norm().item())

    attn_norms = [per_layer_content[0].norm().item()]
    history = [per_layer_content[0]]
    for l in range(1, depth):
        weights = torch.softmax(torch.randn(len(history)), dim=0)  # some learned-looking distribution
        y = attn_residual_step(history, per_layer_content[l], weights)
        attn_norms.append(y.norm().item())
        history.append(y)

    print(f"layer:              {list(range(depth))}")
    print(f"standard residual:  {[f'{v:.1f}' for v in standard_norms]}")
    print(f"attention residual: {[f'{v:.1f}' for v in attn_norms]}")
    assert standard_norms[-1] > 2 * standard_norms[0], "plain residual accumulation should grow substantially over 12 layers"
    assert attn_norms[-1] < standard_norms[-1], "attention-weighted (convex combination) aggregation should stay far more bounded"
    print("plain residual accumulation grows with depth; attention-weighted aggregation does not.\n")

    # --- Part 2: can a later layer bypass an earlier layer that injected something bad? ---
    print("=== structural capability: recovering from one corrupting layer ===")
    clean_signal = torch.randn(d)
    small_update = torch.randn(d) * 0.1
    big_corruption = torch.randn(d) * 5.0  # one layer injects something unhelpful

    # layer sequence: 0=clean signal, 1=small legit update, 2=small legit update,
    # 3=BIG CORRUPTION, 4=small legit update, 5=small legit update
    contents = [clean_signal, small_update, small_update, big_corruption, small_update, small_update]

    # "reference" -- what layer 5 would look like if the corrupting layer had
    # never happened, i.e. summing/attending over every layer EXCEPT layer 3
    reference = clean_signal + small_update + small_update + small_update + small_update

    # standard residual: no choice -- every layer's contribution is permanent, unconditional
    x_std = contents[0]
    for l in range(1, len(contents)):
        x_std = standard_residual_step(x_std, contents[l])
    error_standard = (x_std - reference).norm().item()

    # AttnRes: layer 5 CAN structurally choose to downweight layer 3 (the
    # corrupting one) and rely more on the surrounding clean layers instead --
    # this is exactly the kind of weighting softmax attention can represent
    history = contents[:5]  # layers 0-4 available to attend over
    chosen_weights = torch.tensor([0.28, 0.24, 0.24, 0.02, 0.22])  # near-zero on layer 3
    x_attn = attn_residual_step(history, contents[5], chosen_weights)
    error_attn = (x_attn - reference).norm().item()

    print(f"corruption injected at layer 3, magnitude: {big_corruption.norm().item():.2f}")
    print(f"final-layer error vs 'no corruption' reference:")
    print(f"  standard residual (forced to include it, weight=1): {error_standard:.2f}")
    print(f"  AttnRes (learned weight ~0 on the corrupting layer): {error_attn:.2f}")
    assert error_attn < error_standard, (
        "AttnRes should be able to structurally avoid a corrupting earlier layer; "
        "standard residual has no free parameter to do so at all"
    )
    print("standard residual has exactly one weight (fixed at 1) and no way to avoid a bad layer;")
    print("AttnRes's learned per-layer weights can represent 'mostly ignore this one' instead.")
