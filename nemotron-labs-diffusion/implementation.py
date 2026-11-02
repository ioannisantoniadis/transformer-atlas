"""
Nemotron-Labs-Diffusion: one decoder trained jointly with an autoregressive
loss and a block-diffusion loss, so the same weights can decode three ways --
left-to-right (AR), block-parallel denoising (diffusion), or drafting a block
with the diffusion pathway and checking it with the AR pathway
(self-speculation, no separate draft model). Demonstrates (1) the dual-stream
attention mask lets both losses share one forward pass without the clean
stream ever seeing a future token, and (2) linear self-speculation with the
paper's exact-match acceptance reproduces greedy AR decoding token for token
while needing fewer sequential AR steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 16
MASK = VOCAB          # extra token id for masked positions
SEQ_LEN = 32
BLOCK = 4


def dual_stream_mask(n_clean, noisy_positions, noisy_block_starts):
    """allowed[q, k] for the concatenated [clean | noisy] sequence.
    clean -> clean: strictly causal (the AR stream, so no label leakage).
    noisy -> noisy: bidirectional within the same block only.
    noisy -> clean: only the clean prefix before that block starts.
    clean -> noisy: never."""
    n_noisy = len(noisy_positions)
    n = n_clean + n_noisy
    allowed = torch.zeros(n, n, dtype=torch.bool)
    allowed[:n_clean, :n_clean] = torch.ones(n_clean, n_clean).tril().bool()
    for a in range(n_noisy):
        q = n_clean + a
        allowed[q, :noisy_block_starts[a]] = True
        for b in range(n_noisy):
            if noisy_block_starts[b] == noisy_block_starts[a]:
                allowed[q, n_clean + b] = True
    return allowed


class TinyDecoder(nn.Module):
    """A small pre-norm Transformer. Clean-stream logits at position i predict
    token i+1 (AR); noisy-stream logits at position i predict the clean token
    at i itself (diffusion)."""

    def __init__(self, d=64, heads=4, layers=2):
        super().__init__()
        self.tok = nn.Embedding(VOCAB + 1, d)
        self.pos = nn.Embedding(SEQ_LEN + BLOCK + 1, d)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)

    def forward(self, clean, noisy=None, noisy_positions=None, noisy_block_starts=None):
        n_clean = clean.shape[1]
        x = self.tok(clean) + self.pos(torch.arange(n_clean))
        if noisy is None:
            noisy_positions, noisy_block_starts = [], []
        else:
            x = torch.cat([x, self.tok(noisy) + self.pos(torch.tensor(noisy_positions))], dim=1)
        allowed = dual_stream_mask(n_clean, noisy_positions, noisy_block_starts)
        for blk in self.blocks:
            x = blk(x, src_mask=~allowed)
        logits = self.head(self.norm(x))
        return logits[:, :n_clean], logits[:, n_clean:]


def make_batch(batch_size):
    """Toy language: arithmetic progressions mod VOCAB with a per-sequence step,
    so later tokens are predictable from the first two -- a whole block can be
    drafted in parallel once the prefix pins down the step."""
    start = torch.randint(0, VOCAB, (batch_size, 1))
    step = torch.randint(1, 4, (batch_size, 1))
    return (start + step * torch.arange(SEQ_LEN)) % VOCAB


def joint_loss(model, x, alpha=0.3):
    """L = L_AR + alpha * L_diff (the paper's alpha = 0.3). Blocks start at a
    random offset so the drafter isn't tied to block-aligned positions."""
    offset = torch.randint(0, BLOCK, ()).item()
    starts = [max(0, ((i + offset) // BLOCK) * BLOCK - offset) for i in range(SEQ_LEN)]

    t = torch.rand(x.shape[0], SEQ_LEN).clamp(min=0.05)
    t = t.gather(1, torch.tensor(starts).expand(x.shape[0], -1))   # one noise level per block
    masked = torch.rand(x.shape) < t
    noisy = torch.where(masked, torch.full_like(x, MASK), x)

    ar_logits, diff_logits = model(x, noisy, list(range(SEQ_LEN)), starts)
    l_ar = F.cross_entropy(ar_logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    per_tok = F.cross_entropy(diff_logits.reshape(-1, VOCAB), x.reshape(-1), reduction="none").view(x.shape)
    l_diff = (per_tok * masked / t).sum() / x.numel()
    return l_ar + alpha * l_diff


@torch.no_grad()
def greedy_ar(model, prompt, length):
    seq = prompt.clone()
    while seq.shape[1] < length:
        ar_logits, _ = model(seq)
        seq = torch.cat([seq, ar_logits[:, -1:].argmax(-1)], dim=1)
    return seq, length - prompt.shape[1]          # one AR forward per token


@torch.no_grad()
def diffusion_decode(model, prompt, length, threshold=0.9):
    """Block by block: start from all-mask, commit every position whose top-1
    confidence clears the threshold (at least one per step), repeat."""
    seq, passes = prompt.clone(), 0
    while seq.shape[1] < length:
        n, k = seq.shape[1], min(BLOCK, length - seq.shape[1])
        block = torch.full((1, k), MASK)
        while (block == MASK).any():
            _, logits = model(seq, block, list(range(n, n + k)), [n] * k)
            passes += 1
            conf, pred = logits.softmax(-1).max(-1)
            conf = conf.masked_fill(block != MASK, -1.0)
            commit = (conf >= threshold) | (conf == conf.max())
            block = torch.where(commit, pred, block)
        seq = torch.cat([seq, block], dim=1)
    return seq, passes


@torch.no_grad()
def linear_self_speculation(model, prompt, length, k=BLOCK):
    """Pass 1 (diffusion pathway): append k masks, denoise them in parallel.
    Pass 2 (AR pathway): one causal forward over prefix + draft. Accept the
    longest prefix where draft == AR argmax, then take the AR token at the
    first mismatch (or the bonus token after a fully accepted draft)."""
    seq, cycles = prompt.clone(), 0
    while seq.shape[1] < length:
        n = seq.shape[1]
        _, draft_logits = model(seq, torch.full((1, k), MASK), list(range(n, n + k)), [n] * k)
        draft = draft_logits.argmax(-1)

        ar_logits, _ = model(torch.cat([seq, draft], dim=1))
        ar_next = ar_logits[:, n - 1:].argmax(-1)            # AR prediction for positions n .. n+k
        agree = (draft[0] == ar_next[0, :k]).long().cumprod(0)
        n_acc = int(agree.sum())
        seq = torch.cat([seq, draft[:, :n_acc], ar_next[:, n_acc:n_acc + 1]], dim=1)
        cycles += 1
    return seq[:, :length], cycles                 # two forwards per cycle


if __name__ == "__main__":
    torch.manual_seed(0)
    model = TinyDecoder()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(400):
        loss = joint_loss(model, make_batch(64))
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"joint AR + 0.3 * diffusion loss after training: {loss.item():.3f}")

    # Decode in float64 so AR-pathway logits for the same prefix are identical
    # whether computed one token at a time or over a whole draft block.
    model = model.double().eval()

    # --- (1) No label leakage: the clean (AR) stream never sees a later token,
    #     and a noisy block only sees the clean prefix before it. ---
    x = make_batch(1)
    noisy = torch.where(torch.rand(x.shape) < 0.5, torch.full_like(x, MASK), x)
    starts = [(i // BLOCK) * BLOCK for i in range(SEQ_LEN)]
    j = 13                                         # perturb one clean token, in block 3
    x2 = x.clone()
    x2[0, j] = (x2[0, j] + 1) % VOCAB
    ar1, d1 = model(x, noisy, list(range(SEQ_LEN)), starts)
    ar2, d2 = model(x2, noisy, list(range(SEQ_LEN)), starts)
    ar_unchanged = torch.equal(ar1[:, :j], ar2[:, :j])
    block_of_j_start = (j // BLOCK) * BLOCK
    noisy_unchanged = torch.equal(d1[:, :block_of_j_start + BLOCK], d2[:, :block_of_j_start + BLOCK])
    print(f"\nchanging clean token {j}: AR logits before it unchanged: {ar_unchanged}; "
          f"noisy logits for blocks 0..{j // BLOCK} unchanged: {noisy_unchanged}")

    # --- (2) Three decoding modes, same weights ---
    print("\n--- three decoding modes, one set of weights ---")
    torch.manual_seed(1)
    n_prompts, prompt_len = 20, 2
    totals = {"ar": 0, "diff": 0, "ss_cycles": 0}
    produced = 0
    all_match = True
    for _ in range(n_prompts):
        ref = make_batch(1)
        prompt = ref[:, :prompt_len]
        ar_seq, ar_passes = greedy_ar(model, prompt, SEQ_LEN)
        diff_seq, diff_passes = diffusion_decode(model, prompt, SEQ_LEN)
        ss_seq, ss_cycles = linear_self_speculation(model, prompt, SEQ_LEN)
        all_match &= torch.equal(ss_seq, ar_seq)
        totals["ar"] += ar_passes
        totals["diff"] += diff_passes
        totals["ss_cycles"] += ss_cycles
        produced += SEQ_LEN - prompt_len

    print(f"tokens generated: {produced} over {n_prompts} prompts")
    print(f"AR:              {totals['ar']} forwards  -> {produced / totals['ar']:.2f} tokens/forward")
    print(f"diffusion:       {totals['diff']} forwards  -> {produced / totals['diff']:.2f} tokens/forward")
    print(f"self-speculation: {totals['ss_cycles']} draft+verify cycles (2 forwards each) "
          f"-> {produced / totals['ss_cycles']:.2f} tokens/cycle, "
          f"{produced / (2 * totals['ss_cycles']):.2f} tokens/forward")
    print(f"self-speculation output == greedy AR output for every prompt: {all_match}")

    # --- Checks (Fu et al. 2026): the dual-stream mask leaks nothing to the AR
    #     stream; exact-match self-speculation reproduces greedy AR exactly and
    #     commits more than one token per draft+verify cycle. ---
    assert ar_unchanged and noisy_unchanged
    assert all_match
    tokens_per_cycle = produced / totals["ss_cycles"]
    assert 1.0 < tokens_per_cycle <= BLOCK + 1, tokens_per_cycle
    print("\nchecks passed: no leakage into the AR stream; self-speculation == greedy AR; "
          ">1 token per draft+verify cycle")
