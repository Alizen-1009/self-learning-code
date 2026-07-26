import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def apply_rope(self, x):
        batch, seq_len, num_heads, head_dim = x.shape
        pos = torch.arange(seq_len, device=x.device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.outer(pos, self.rope_inv_freq)
        cos = freqs.cos().to(dtype=x.dtype).view(1, seq_len, 1, head_dim // 2)
        sin = freqs.sin().to(dtype=x.dtype).view(1, seq_len, 1, head_dim // 2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_even * cos - x_odd * sin
        x_rotated[..., 1::2] = x_even * sin + x_odd * cos
        return x_rotated

    def forward(self, x, mask=None):
        batch, seq_len, embed_dim = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        q = self.apply_rope(q)
        k = self.apply_rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q/k/v shape: (batch, num_heads, seq_len, head_dim)

        score = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(score, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)
        return self.out(y), attn
