"""
The original Transformer (Vaswani et al., 2017): scaled dot-product
attention, multi-head attention, sinusoidal positional encoding, and a
pre-norm encoder block built from them. Minimal and readable, not fast.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """q, k, v: (..., seq_len, d_k). mask: broadcastable to (..., seq_q, seq_k),
    True where attention is allowed. Returns (output, attn_weights)."""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., seq_q, seq_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly into heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        batch, seq_len, d_model = x.shape
        return x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x_q, x_kv, mask=None):
        batch, seq_q, _ = x_q.shape
        q = self._split_heads(self.w_q(x_q))  # (batch, heads, seq_q, d_k)
        k = self._split_heads(self.w_k(x_kv))
        v = self._split_heads(self.w_v(x_kv))

        if mask is not None:
            mask = mask.unsqueeze(1)  # broadcast over heads

        out, _ = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch, seq_q, -1)
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


def sinusoidal_positional_encoding(seq_len, d_model):
    """Returns (seq_len, d_model). Fixed, not learned."""
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class EncoderBlock(nn.Module):
    """Pre-norm variant (more common in practice than the original post-norm)."""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


def causal_mask(seq_len):
    """(seq_len, seq_len) boolean mask, True where attention is allowed."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model, num_heads, d_ff = 2, 6, 32, 4, 128

    x = torch.randn(batch, seq_len, d_model)
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    x = x + pe

    block = EncoderBlock(d_model, num_heads, d_ff)

    # Bidirectional (encoder-style): every position sees every position.
    out_full = block(x)
    print("full self-attention output:", out_full.shape)

    # Causal (decoder-style): position i only sees positions <= i.
    mask = causal_mask(seq_len).unsqueeze(0)  # broadcast over batch
    out_causal = block(x, mask=mask)
    print("causal self-attention output:", out_causal.shape)

    # Sanity check: attention weights for a causal head sum to 1 and are
    # zero above the diagonal.
    mha = MultiHeadAttention(d_model, num_heads)
    q = mha._split_heads(mha.w_q(x))
    k = mha._split_heads(mha.w_k(x))
    v = mha._split_heads(mha.w_v(x))
    _, weights = scaled_dot_product_attention(q, k, v, mask.unsqueeze(1))
    upper_triangle_mass = weights[0, 0].triu(diagonal=1).sum().item()
    print(f"row sums ~1.0: {weights[0, 0].sum(-1)[:3].tolist()}")
    print(f"mass above diagonal (should be ~0): {upper_triangle_mass:.6f}")
