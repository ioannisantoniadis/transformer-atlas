"""
GPT-style decoder-only Transformer: causal self-attention + FFN blocks,
learned positional embeddings, autoregressive next-token prediction.
Demonstrates the difference from ../transformer/ (encoder-decoder) is
mostly *subtraction*: no cross-attention, always-causal masking.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.d_k)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (batch, heads, seq_len, d_k)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        out = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.proj(out)


class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)  # learned, unlike sinusoidal PE
        self.blocks = nn.ModuleList([GPTBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):
        batch, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)  # (batch, seq_len, vocab_size)

    @torch.no_grad()
    def generate(self, token_ids, num_new_tokens, temperature=1.0):
        for _ in range(num_new_tokens):
            logits = self(token_ids)[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len = 100, 32, 4, 128, 2, 64

    model = GPT(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len)

    batch, seq_len = 2, 8
    token_ids = torch.randint(0, vocab_size, (batch, seq_len))
    logits = model(token_ids)
    print("logits shape:", logits.shape)  # (batch, seq_len, vocab_size)

    # Causal sanity check: changing a future token must not change an
    # earlier position's logits.
    token_ids_altered = token_ids.clone()
    token_ids_altered[:, -1] = (token_ids_altered[:, -1] + 1) % vocab_size
    logits_altered = model(token_ids_altered)
    earlier_positions_match = torch.allclose(logits[:, :-1], logits_altered[:, :-1], atol=1e-6)
    print(f"earlier-position logits unchanged by future token edit: {earlier_positions_match}")

    generated = model.generate(token_ids, num_new_tokens=5)
    print("generated shape:", generated.shape)  # (batch, seq_len + 5)
