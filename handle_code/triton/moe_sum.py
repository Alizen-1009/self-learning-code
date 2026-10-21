"""down [T, K, H] × scores [T, K]，沿 top-k 求和得到 [T, H]。
输入须为 token/top-k 顺序，down 尚未加权；仅用于 GPU 前向推理。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_sum_kernel(
    Down, Scores, Out,
    H: tl.constexpr, K: tl.constexpr,
    stride_dt: tl.constexpr, stride_dk: tl.constexpr, stride_dh: tl.constexpr,
    stride_st: tl.constexpr, stride_sk: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # 一个 program 负责一个 token 的一段 hidden，独占该段输出。
    token = tl.program_id(0)
    ks = tl.arange(0, BLOCK_K)
    hs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)

    # [BLOCK_K, BLOCK_H]：hidden 相邻，有利于连续布局的合并访存。
    values = tl.load(
        Down + token * stride_dt
        + ks[:, None] * stride_dk + hs[None, :] * stride_dh,
        mask=(ks[:, None] < K) & (hs[None, :] < H),
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(
        Scores + token * stride_st + ks * stride_sk,
        mask=ks < K, other=0.0,
    ).to(tl.float32)

    weighted = values * weights[:, None]
    result = tl.sum(weighted, axis=0)  # 沿 top-k 轴归约，得到 [BLOCK_H]
    tl.store(Out + token * H + hs, result, mask=hs < H)



def moe_sum(down: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    tokens, topk, hidden = down.shape
    out = torch.empty((tokens, hidden), device=down.device, dtype=down.dtype)

    block_k = triton.next_power_of_2(topk)
    block_h = min(256, max(1, 1024 // block_k))
    with torch.cuda.device(down.device):
        _moe_sum_kernel[(tokens, triton.cdiv(hidden, block_h))](
            down, scores, out, hidden, topk,
            *down.stride(), *scores.stride(),
            BLOCK_K=block_k, BLOCK_H=block_h,
            num_warps=4,
        )
    return out