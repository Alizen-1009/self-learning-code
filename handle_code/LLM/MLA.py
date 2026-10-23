import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_nope_head_dim=None,
        qk_rope_head_dim=None,
        v_head_dim=None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_lora_rank = q_lora_rank or embed_dim // 2
        self.kv_lora_rank = kv_lora_rank or embed_dim // 2
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        self.q_down = nn.Linear(embed_dim, self.q_lora_rank, bias=False)
        self.q_up = nn.Linear(
            self.q_lora_rank,
            num_heads * self.qk_head_dim,
            bias=False,
        )

        self.kv_down = nn.Linear(
            embed_dim,
            self.kv_lora_rank + qk_rope_head_dim,
            bias=False,
        )
        self.kv_up = nn.Linear(
            self.kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )
        self.out = nn.Linear(num_heads * v_head_dim, embed_dim)

        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, qk_rope_head_dim, 2).float() / qk_rope_head_dim)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def apply_rope(self, x):
        _, seq_len, _, rope_dim = x.shape
        pos = torch.arange(seq_len, device=x.device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.outer(pos, self.rope_inv_freq)
        cos = freqs.cos().to(dtype=x.dtype).view(1, seq_len, 1, rope_dim // 2)
        sin = freqs.sin().to(dtype=x.dtype).view(1, seq_len, 1, rope_dim // 2)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_even * cos - x_odd * sin
        x_rotated[..., 1::2] = x_even * sin + x_odd * cos
        return x_rotated

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        q = self.q_up(self.q_down(x))
        q = q.view(batch, seq_len, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        kv = self.kv_down(x)
        kv_latent, k_rope = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_up(kv_latent)
        kv = kv.view(
            batch,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, v = torch.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )

        q_rope = self.apply_rope(q_rope)
        k_rope = k_rope.view(batch, seq_len, 1, self.qk_rope_head_dim)
        k_rope = self.apply_rope(k_rope)

        q_nope = q_nope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        v = v.transpose(1, 2)
        # q_nope/k_nope/v shape: (batch, num_heads, seq_len, dim)
        # q_rope shape: (batch, num_heads, seq_len, rope_dim)
        # k_rope shape: (batch, 1, seq_len, rope_dim), broadcast across heads

        score_nope = q_nope @ k_nope.transpose(-2, -1)
        score_rope = q_rope @ k_rope.transpose(-2, -1)
        score = (score_nope + score_rope) / math.sqrt(self.qk_head_dim)

        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(score, dim=-1)
        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(
            batch,
            seq_len,
            self.num_heads * self.v_head_dim,
        )
        return self.out(y), attn
