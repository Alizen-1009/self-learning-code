from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_M = 64
BLOCK_N = 64


@triton.jit
def _flash_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    lse_ptr,
    stride_q_batch,
    stride_q_m,
    stride_q_d,
    stride_k_batch,
    stride_k_n,
    stride_k_d,
    stride_v_batch,
    stride_v_n,
    stride_v_d,
    stride_out_batch,
    stride_out_m,
    stride_out_d,
    stride_lse_batch,
    stride_lse_m,
    n_queries,
    n_keys,
    scale,
    is_causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_VALUE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_head = tl.arange(0, D_HEAD)
    offs_d_value = tl.arange(0, D_VALUE)

    q_ptrs = (
        q_ptr
        + pid_b * stride_q_batch
        + offs_m[:, None] * stride_q_m
        + offs_d_head[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=offs_m[:, None] < n_queries, other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D_VALUE), dtype=tl.float32)

    hi = n_keys
    if is_causal:
        hi = tl.minimum(n_keys, (pid_m + 1) * BLOCK_M)

    for start_n in tl.range(0, hi, BLOCK_N):
        k_ptrs = (
            k_ptr
            + pid_b * stride_k_batch
            + (start_n + offs_n)[:, None] * stride_k_n
            + offs_d_head[None, :] * stride_k_d
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_v_batch
            + (start_n + offs_n)[:, None] * stride_v_n
            + offs_d_value[None, :] * stride_v_d
        )

        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < n_keys, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < n_keys, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * scale
        qk = tl.where((start_n + offs_n)[None, :] < n_keys, qk, -1.0e6)

        if is_causal:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk, -1.0e6)

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)

        acc = acc * alpha[:, None]
        acc = acc + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)

    out_ptrs = (
        out_ptr
        + pid_b * stride_out_batch
        + offs_m[:, None] * stride_out_m
        + offs_d_value[None, :] * stride_out_d
    )
    lse_ptrs = lse_ptr + pid_b * stride_lse_batch + offs_m * stride_lse_m

    tl.store(out_ptrs, acc, mask=offs_m[:, None] < n_queries)
    tl.store(lse_ptrs, lse, mask=offs_m < n_queries)


def _flash_attention_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_flat = q.reshape(-1, q.shape[-2], q.shape[-1]).contiguous()
    k_flat = k.reshape(-1, k.shape[-2], k.shape[-1]).contiguous()
    v_flat = v.reshape(-1, v.shape[-2], v.shape[-1]).contiguous()

    batch_size = q_flat.shape[0]
    n_queries = q_flat.shape[1]
    n_keys = k_flat.shape[1]
    d_head = q_flat.shape[2]
    d_value = v_flat.shape[2]

    out_flat = torch.empty((batch_size, n_queries, d_value), device=q.device, dtype=q.dtype)
    lse_flat = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(n_queries, BLOCK_M), batch_size)
    _flash_attention_forward_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        out_flat,
        lse_flat,
        q_flat.stride(0),
        q_flat.stride(1),
        q_flat.stride(2),
        k_flat.stride(0),
        k_flat.stride(1),
        k_flat.stride(2),
        v_flat.stride(0),
        v_flat.stride(1),
        v_flat.stride(2),
        out_flat.stride(0),
        out_flat.stride(1),
        out_flat.stride(2),
        lse_flat.stride(0),
        lse_flat.stride(1),
        n_queries,
        n_keys,
        q.shape[-1] ** -0.5,
        is_causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D_HEAD=d_head,
        D_VALUE=d_value,
        num_warps=4,
        num_stages=2,
    )
    return (
        out_flat.reshape(*q.shape[:-2], n_queries, d_value),
        lse_flat.reshape(*q.shape[:-2], n_queries),
    )


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        _ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        out, _ = _flash_attention_forward_triton(q, k, v, is_causal=is_causal)
        return out


def get_flashattention_autograd_function_triton() -> type[FlashAttentionTriton]:
    return FlashAttentionTriton
