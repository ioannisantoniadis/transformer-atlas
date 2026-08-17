"""
Qwen-style decoder block: identical to the ../llama/ block (RMSNorm +
RoPE + GQA + SwiGLU) with one deliberate difference -- bias terms are kept
on the Q/K/V projections, where Llama and most contemporaries drop all
linear-layer biases. Everything else is the same template; this file is
best read as a diff against ../llama/implementation.py.
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


class QwenAttention(nn.Module):
    """Grouped-query attention + RoPE, same as Llama's, except w_q/w_k/w_v
    keep a bias term (qkv_bias=True) -- the one deliberate Qwen deviation."""

    def __init__(self, d_model, num_heads, num_kv_heads, qkv_bias=True):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, num_heads * self.d_k, bias=qkv_bias)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=qkv_bias)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=qkv_bias)
        self.w_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)  # output proj stays bias-free
        self.theta = rope_frequencies(self.d_k)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q = self.w_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        q, k = apply_rope(q, self.theta), apply_rope(k, self.theta)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        weights = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class QwenBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, qkv_bias=True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = QwenAttention(d_model, num_heads, num_kv_heads, qkv_bias)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    batch, seq_len, d_model, num_heads, num_kv_heads = 2, 9, 32, 8, 2

    qwen_block = QwenBlock(d_model, num_heads, num_kv_heads, qkv_bias=True)
    llama_style_block = QwenBlock(d_model, num_heads, num_kv_heads, qkv_bias=False)

    x = torch.randn(batch, seq_len, d_model)
    out_qwen = qwen_block(x)
    out_llama_style = llama_style_block(x)
    print("Qwen-style (QKV bias) output shape:", out_qwen.shape)
    print("Llama-style (no bias) output shape:", out_llama_style.shape)

    qwen_attn_params = sum(p.numel() for p in qwen_block.attn.parameters())
    llama_attn_params = sum(p.numel() for p in llama_style_block.attn.parameters())
    print(f"\nattention sublayer params -- Qwen-style (with QKV bias): {qwen_attn_params}, "
          f"Llama-style (no bias): {llama_attn_params}, "
          f"extra bias params: {qwen_attn_params - llama_attn_params} "
          f"({num_heads * (d_model // num_heads)} for q + 2 x {num_kv_heads * (d_model // num_heads)} for k,v)")
