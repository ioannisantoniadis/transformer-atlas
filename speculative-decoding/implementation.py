"""
Speculative decoding: a cheap draft model proposes tokens, a target model
verifies them via rejection sampling. Demonstrates (1) the accept/reject
rule produces samples EXACTLY distributed as the target model alone, even
though a different (draft) distribution proposed them, and (2) the
multi-token draft-then-verify loop and its expected speedup.
"""

import torch


def speculative_accept_reject(p_target, q_draft):
    """One token: draft proposes x ~ q_draft; accept with probability
    min(1, p_target(x)/q_draft(x)); on rejection, resample from the
    'leftover' distribution p' proportional to max(0, p_target - q_draft).
    Returns (token, was_accepted)."""
    x = torch.multinomial(q_draft, 1).item()
    accept_prob = min(1.0, (p_target[x] / q_draft[x]).item())
    if torch.rand(()).item() < accept_prob:
        return x, True

    residual = torch.clamp(p_target - q_draft, min=0.0)
    residual = residual / residual.sum()
    return torch.multinomial(residual, 1).item(), False


def multi_token_speculative_round(p_targets, q_drafts):
    """Propose+verify a whole draft block (K tokens) at once, the way a
    single target-model forward pass would verify all K positions in
    parallel. Stops at the first rejection (its resampled token becomes
    the final token for that position; later draft tokens are discarded,
    per the standard algorithm)."""
    accepted_count = 0
    final_tokens = []
    for p_target, q_draft in zip(p_targets, q_drafts):
        token, accepted = speculative_accept_reject(p_target, q_draft)
        final_tokens.append(token)
        if accepted:
            accepted_count += 1
        else:
            break
    return final_tokens, accepted_count


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size = 6

    # --- Correctness: does the accept/reject procedure reproduce p_target exactly? ---
    p_target = torch.tensor([0.05, 0.30, 0.10, 0.40, 0.05, 0.10])  # what the (expensive) target model would sample
    q_draft = torch.tensor([0.20, 0.20, 0.20, 0.15, 0.10, 0.15])   # a DIFFERENT, cheaper draft distribution

    num_trials = 200_000
    outputs = torch.tensor([speculative_accept_reject(p_target, q_draft)[0] for _ in range(num_trials)])
    empirical = torch.bincount(outputs, minlength=vocab_size).float() / num_trials

    print("target distribution p:      ", [f"{v:.3f}" for v in p_target.tolist()])
    print("draft distribution q:       ", [f"{v:.3f}" for v in q_draft.tolist()])
    print(f"empirical output ({num_trials} samples):", [f"{v:.3f}" for v in empirical.tolist()])
    total_variation = 0.5 * (empirical - p_target).abs().sum().item()
    print(f"total variation distance from target p: {total_variation:.4f} "
          f"(should shrink toward 0 as num_trials grows -- output matches the TARGET, not the draft)")

    # --- Throughput: multi-token draft-and-verify, averaged over many rounds ---
    print("\n--- multi-token speculative round ---")
    K = 4  # draft proposes this many tokens per round
    torch.manual_seed(1)
    # A draft that's quite well-matched to the target at most positions (high acceptance).
    p_targets = [torch.softmax(torch.randn(vocab_size), dim=0) for _ in range(K)]
    q_drafts = [torch.softmax(p + 0.3 * torch.randn(vocab_size), dim=0) for p in p_targets]

    num_rounds = 5000
    accepted_lengths = [multi_token_speculative_round(p_targets, q_drafts)[1] for _ in range(num_rounds)]
    avg_accepted = sum(accepted_lengths) / num_rounds

    print(f"draft block size K={K}, avg accepted tokens per round: {avg_accepted:.2f}")
    print(f"target-model forward passes per round: 1 (verifies up to K tokens in parallel)")
    print(f"tokens produced per target-model pass: {1 + avg_accepted:.2f} "
          f"(1 guaranteed token even on immediate rejection, plus accepted draft tokens)")
    print(f"vs naive autoregressive decoding: 1 token per target-model pass "
          f"-> ~{1 + avg_accepted:.2f}x fewer sequential target-model passes for the same output length")
