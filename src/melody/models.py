"""Decoder-only causal transformers for two input types: binary piano-roll frames (88 keys,
factorized Bernoulli output, the classic JSB/Nottingham polyphonic benchmark) and discrete
melody tokens (categorical output)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv", inv, persistent=False)

    def forward(self, n, device):
        t = torch.arange(n, device=device, dtype=self.inv.dtype)
        f = torch.outer(t, self.inv)
        return torch.cos(f), torch.sin(f)


def apply_rotary(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)
    return out.flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x, cos, sin):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class Transformer(nn.Module):
    def __init__(self, kind, vocab=None, d=256, layers=4, heads=4, dropout=0.1, n_keys=88):
        super().__init__()
        self.kind = kind
        self.inp = nn.Linear(n_keys, d) if kind == "frames" else nn.Embedding(vocab, d)
        self.start = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_keys if kind == "frames" else vocab)
        self.rope = RotaryEmbedding(d // heads)

    def forward(self, x):
        """frames: x [B, T, 88] float -> logits for frames 1..T given 0..T-1 (with a learned start).
        tokens: x [B, T] long -> next-token logits at each position."""
        h = self.inp(x)
        if self.kind == "frames":
            h = torch.cat([self.start.expand(h.size(0), 1, -1), h[:, :-1]], 1)
        cos, sin = self.rope(h.size(1), h.device)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.head(self.ln(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def frame_nll(model, x, mask):
    """Sum over keys of Bernoulli NLL, per frame. x, mask: [B, T, 88], [B, T]."""
    logits = model(x)
    nll = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1)
    return (nll * mask).sum() / mask.sum(), nll
