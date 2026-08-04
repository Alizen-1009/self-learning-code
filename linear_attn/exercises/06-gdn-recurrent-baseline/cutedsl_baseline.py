from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream

D = 128
TILE_ROWS = 16


@cute.jit
def _dot8(lhs, row, rhs):
    return (
        (lhs[row, 0] * rhs[0] + lhs[row, 1] * rhs[1])
        + (lhs[row, 2] * rhs[2] + lhs[row, 3] * rhs[3])
    ) + (
        (lhs[row, 4] * rhs[4] + lhs[row, 5] * rhs[5])
        + (lhs[row, 6] * rhs[6] + lhs[row, 7] * rhs[7])
    )


@cute.jit
def _reduce_k(value):
    for offset in [8, 4, 2, 1]:
        value = value + cute.arch.shuffle_sync_bfly(
            value, offset=offset, mask=0xFFFFFFFF
        )
    return value


@cute.kernel
def _recurrent_gdn_cute_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    alpha: cute.Tensor,
    beta: cute.Tensor,
    state_in: cute.Tensor,
    output: cute.Tensor,
    state_out: cute.Tensor,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()

    num_v_tiles = D // TILE_ROWS
    v_tile = block % num_v_tiles
    bh = block // num_v_tiles
    head = bh % H
    batch = bh // H
    v_offset = v_tile * TILE_ROWS

    # D=128: 16 lane groups partition K into vec8; two lane rows partition V.
    k_lane = tid % 16
    v_lane = tid // 16
    values_per_thread = 4

    h_reg = cute.make_rmem_tensor((TILE_ROWS // 2, 8), cutlass.Float32)
    q_src = cute.make_rmem_tensor((values_per_thread,), cutlass.Float32)
    k_src = cute.make_rmem_tensor((values_per_thread,), cutlass.Float32)
    q_reg = cute.make_rmem_tensor((8,), cutlass.Float32)
    k_reg = cute.make_rmem_tensor((8,), cutlass.Float32)

    # Correctness-first scalar loads; later lessons can replace these with
    # vectorized copy atoms without changing the recurrence.
    for row in cutlass.range_constexpr(TILE_ROWS // 2):
        v_idx = v_offset + v_lane + 2 * row
        for elem in cutlass.range_constexpr(8):
            h_reg[row, elem] = state_in[
                batch, head, v_idx, k_lane * 8 + elem
            ].to(cutlass.Float32)

    for token in cutlass.range_constexpr(T):
        q_head = q[batch, token, head, None]
        k_head = k[batch, token, head, None]
        v_head = v[batch, token, head, None]

        for elem in cutlass.range_constexpr(values_per_thread):
            idx = tid * values_per_thread + elem
            q_src[elem] = q_head[idx].to(cutlass.Float32)
            k_src[elem] = k_head[idx].to(cutlass.Float32)

        # Broadcast the K-lane's contiguous vec8 to both V lane groups.
        for elem in cutlass.range_constexpr(8):
            source_lane = 2 * k_lane + elem // values_per_thread
            source_value = elem % values_per_thread
            q_reg[elem] = cute.arch.shuffle_sync(
                q_src[source_value], offset=source_lane, mask=0xFFFFFFFF
            )
            k_reg[elem] = cute.arch.shuffle_sync(
                k_src[source_value], offset=source_lane, mask=0xFFFFFFFF
            )

        a = alpha[batch, token, head].to(cutlass.Float32)
        b = beta[batch, token, head].to(cutlass.Float32)
        v_loaded = cutlass.Float32(0.0)
        if tid < TILE_ROWS:
            v_loaded = v_head[v_offset + tid].to(cutlass.Float32)

        for row in cutlass.range_constexpr(TILE_ROWS // 2):
            for elem in cutlass.range_constexpr(8):
                h_reg[row, elem] *= a

            prediction = _reduce_k(_dot8(h_reg, row, k_reg))
            v_idx = v_offset + v_lane + 2 * row
            value = cute.arch.shuffle_sync(
                v_loaded, offset=v_lane + 2 * row, mask=0xFFFFFFFF
            )
            residual = (value - prediction) * b

            for elem in cutlass.range_constexpr(8):
                h_reg[row, elem] += residual * k_reg[elem]

            out = _reduce_k(_dot8(h_reg, row, q_reg))
            if k_lane == row:
                output[batch, token, head, v_idx] = out.to(cutlass.BFloat16)

    for row in cutlass.range_constexpr(TILE_ROWS // 2):
        v_idx = v_offset + v_lane + 2 * row
        for elem in cutlass.range_constexpr(8):
            state_out[batch, head, v_idx, k_lane * 8 + elem] = h_reg[
                row, elem
            ]


@cute.jit
def _recurrent_gdn_cute_launch(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    alpha: cute.Tensor,
    beta: cute.Tensor,
    state_in: cute.Tensor,
    output: cute.Tensor,
    state_out: cute.Tensor,
    stream: cuda.CUstream,
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
):
    batch = q.shape[0]
    _recurrent_gdn_cute_kernel(
        q, k, v, alpha, beta, state_in, output, state_out, T, H
    ).launch(
        grid=[batch * H * (D // TILE_ROWS), 1, 1],
        block=[32, 1, 1],
        stream=stream,
    )


_COMPILED: dict[tuple[int, int, int], object] = {}


def _as_cute(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16, enable_tvm_ffi=True)


def _get_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state_in: torch.Tensor,
    output: torch.Tensor,
    state_out: torch.Tensor,
):
    batch, tokens, heads, _ = q.shape
    key = (batch, tokens, heads)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(
            _recurrent_gdn_cute_launch,
            _as_cute(q),
            _as_cute(k),
            _as_cute(v),
            _as_cute(alpha),
            _as_cute(beta),
            _as_cute(state_in),
            _as_cute(output),
            _as_cute(state_out),
            make_fake_stream(),
            tokens,
            heads,
            options="--enable-tvm-ffi --generate-line-info",
        )
        _COMPILED[key] = compiled
    return compiled


@torch.inference_mode()
def recurrent_gdn_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, tokens, heads, width = q.shape
    if width != D or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("Phase 1 requires q/k/v [B,T,H,128]")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q/k/v must be BF16")
    if alpha.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("alpha/beta must be activated FP32 values")
    if initial_state.shape != (batch, heads, D, D):
        raise ValueError("initial_state must be [B,H,128,128] V-first")
    if initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be FP32")

    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    compiled = _get_compiled(
        q, k, v, alpha, beta, initial_state, output, final_state
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    compiled(
        q, k, v, alpha, beta, initial_state, output, final_state, stream
    )
    return output, final_state
