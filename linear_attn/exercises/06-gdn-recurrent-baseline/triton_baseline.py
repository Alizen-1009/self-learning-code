from __future__ import annotations

import torch
import triton
import triton.language as tl

D = 128


@triton.jit
def _recurrent_gdn_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    alpha_ptr,
    beta_ptr,
    state_in_ptr,
    output_ptr,
    state_out_ptr,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    bh = tl.program_id(0)
    v_tile = tl.program_id(1)
    batch = bh // H
    head = bh % H

    k_offsets = tl.arange(0, WIDTH)
    v_offsets = v_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < WIDTH

    state_offsets = (
        ((batch * H + head) * WIDTH + v_offsets[:, None]) * WIDTH
        + k_offsets[None, :]
    )
    state = tl.load(
        state_in_ptr + state_offsets,
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    for t in tl.range(0, T):
        qk_offsets = ((batch * T + t) * H + head) * WIDTH + k_offsets
        value_offsets = ((batch * T + t) * H + head) * WIDTH + v_offsets
        scalar_offset = (batch * T + t) * H + head

        query = tl.load(q_ptr + qk_offsets).to(tl.float32)
        key = tl.load(k_ptr + qk_offsets).to(tl.float32)
        value = tl.load(v_ptr + value_offsets, mask=v_mask, other=0.0).to(tl.float32)
        alpha = tl.load(alpha_ptr + scalar_offset).to(tl.float32)
        beta = tl.load(beta_ptr + scalar_offset).to(tl.float32)

        state *= alpha
        prediction = tl.sum(state * key[None, :], axis=1)
        residual = (value - prediction) * beta
        state += residual[:, None] * key[None, :]
        out = tl.sum(state * query[None, :], axis=1)
        tl.store(output_ptr + value_offsets, out, mask=v_mask)

    tl.store(state_out_ptr + state_offsets, state, mask=v_mask[:, None])


@torch.inference_mode()
def recurrent_gdn_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, width = q.shape
    if q.device.type != "cuda":
        raise ValueError("inputs must be CUDA tensors")
    if width != D or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Phase 1 requires q/k/v [B,T,H,128]")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must be BF16")
    if alpha.shape != (batch, tokens, heads) or beta.shape != alpha.shape:
        raise ValueError("alpha/beta must be [B,T,H]")
    if alpha.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("alpha/beta must be FP32 activated values")
    if initial_state.shape != (batch, heads, D, D):
        raise ValueError("initial_state must be [B,H,128,128] V-first")
    if initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be FP32")

    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    block_v = 16
    grid = (batch * heads, triton.cdiv(D, block_v))
    _recurrent_gdn_kernel[grid](
        q,
        k,
        v,
        alpha,
        beta,
        initial_state,
        output,
        final_state,
        B=batch,
        T=tokens,
        H=heads,
        WIDTH=D,
        BLOCK_V=block_v,
        num_warps=4,
        num_stages=2,
    )
    return output, final_state
