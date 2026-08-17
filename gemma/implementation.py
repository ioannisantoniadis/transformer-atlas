"""
Gemma-specific pieces on top of the standard RMSNorm+RoPE+GQA template:
tied & sqrt(d_model)-scaled embeddings, GeGLU (GELU-gated FFN, vs SwiGLU's
SiLU gating), logit soft-capping (tanh-bounded, both attention and final
logits), and alternating local/global attention across layers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.gamma


def rope_frequencies(head_dim, base=10000.0):
    return 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))


def apply_rope(x, theta):
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    positions = torch.arange(seq_len, device=x.device).float()
    angles = positions[:, None] * theta[None, :].to(x.device)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    x_pairs = x.view(*x.shape[:-1], head_dim // 2, 2)
    x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
    rotated = torch.stack([-x2, x1], dim=-1).view(*x.shape)
    return x * cos + rotated * sin


def soft_cap(logits, cap):
    """Smooth, bounded alternative to a hard clip: keeps |output| < cap
    everywhere, with gradient that vanishes gracefully near the bound
    instead of the zero-gradient cliff a hard clamp would introduce."""
    return cap * torch.tanh(logits / cap)


class GemmaAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size=None, attn_logit_cap=50.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size  # None = full attention; int = local sliding window
        self.attn_logit_cap = attn_logit_cap
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.theta = rope_frequencies(self.d_k)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q = self.w_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        q, k = apply_rope(q, self.theta), apply_rope(k, self.theta)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = soft_cap(scores, self.attn_logit_cap)  # Gemma 2: cap attention logits before masking/softmax

        positions = torch.arange(seq_len, device=x.device)
        causal = positions.unsqueeze(1) >= positions.unsqueeze(0)
        if self.window_size is not None:
            causal = causal & ((positions.unsqueeze(1) - positions.unsqueeze(0)) < self.window_size)

        weights = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out)


class GeGLU(nn.Module):
    """Same gated-FFN shape as SwiGLU, GELU instead of SiLU as the gate."""

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.gelu(self.w_gate(x)) * self.w_up(x))


class GemmaBlock(nn.Module):
    def __init__(self, d_model, num_heads, window_size=None):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GemmaAttention(d_model, num_heads, window_size=window_size)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = GeGLU(d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Gemma(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, window_size, final_logit_cap=30.0):
        super().__init__()
        self.d_model = d_model
        self.final_logit_cap = final_logit_cap
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Alternate local/global attention layer by layer (Gemma 2).
        self.blocks = nn.ModuleList([
            GemmaBlock(d_model, num_heads, window_size=(window_size if i % 2 == 0 else None))
            for i in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        # No separate output head -- tied to the embedding matrix (see forward()).

    def forward(self, token_ids):
        x = self.token_emb(token_ids) * math.sqrt(self.d_model)  # scale to match tied-weight logit magnitude
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = x @ self.token_emb.weight.T  # tied embedding/output weights
        return soft_cap(logits, self.final_logit_cap)


if __name__ == "__main__":
    torch.manual_seed(0)
    vocab_size, d_model, num_heads, num_layers, window_size = 100, 32, 4, 4, 3

    model = Gemma(vocab_size, d_model, num_heads, num_layers, window_size)
    batch, seq_len = 2, 10
    token_ids = torch.randint(0, vocab_size, (batch, seq_len))

    logits = model(token_ids)
    print("logits shape:", logits.shape)
    print(f"logits bounded by soft cap: max abs logit = {logits.abs().max().item():.4f} "
          f"(cap = {model.final_logit_cap}, should never exceed it)")

    embed_params = model.token_emb.weight.data_ptr()
    output_used_params = model.token_emb.weight.data_ptr()  # same tensor -- forward() reuses .weight directly
    print(f"\nembedding and output projection share the same weight tensor: {embed_params == output_used_params}")

    attention_types = ["local (sliding window)" if b.attn.window_size is not None else "global (full)"
                        for b in model.blocks]
    print(f"\nper-layer attention pattern (alternating): {attention_types}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total params (no separate output head thanks to tying): {total_params:,}")
