from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream

D = 128
CHUNK = 64
TILE_ROWS = 16
WARPS_PER_CTA = D // TILE_ROWS  # 8 warps cover all 128 V rows.


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
    # XOR stays inside each 16-lane half-warp: lanes 0..15 and 16..31.
    for offset in [8, 4, 2, 1]:
        value = value + cute.arch.shuffle_sync_bfly(value, offset=offset, mask=0xFFFFFFFF)
    return value


@cute.kernel
def _persistent_recurrent_kernel(
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
    lane = tid % 32
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    # One CTA owns exactly one (batch, head); its 8 warps cover V=128.
    head = block % H
    batch = block // H
    v_offset = warp_idx * TILE_ROWS

    # Within each warp: 16 logical K slices x two V-lane rows.
    k_lane = lane % 16
    v_lane = lane // 16
    values_per_lane = 4

    state = cute.make_rmem_tensor((TILE_ROWS // 2, 8), cutlass.Float32)
    q_src = cute.make_rmem_tensor((values_per_lane,), cutlass.Float32)
    k_src = cute.make_rmem_tensor((values_per_lane,), cutlass.Float32)
    q_vec8 = cute.make_rmem_tensor((8,), cutlass.Float32)
    k_vec8 = cute.make_rmem_tensor((8,), cutlass.Float32)

    # Each thread owns 8 V rows x 8 K values = 64 FP32 state values.
    for row in cutlass.range_constexpr(TILE_ROWS // 2):
        v_idx = v_offset + v_lane + 2 * row
        for elem in cutlass.range_constexpr(8):
            state[row, elem] = state_in[batch, head, v_idx, k_lane * 8 + elem].to(cutlass.Float32)

    # This nested loop makes the FlashInfer-style scheduling visible:
    # one CTA keeps the whole head state on-chip while chunks advance serially.
    for chunk in cutlass.range_constexpr(T // CHUNK):
        for local_t in cutlass.range_constexpr(CHUNK):
            token = chunk * CHUNK + local_t
            q_head = q[batch, token, head, None]
            k_head = k[batch, token, head, None]
            v_head = v[batch, token, head, None]

            # Every warp redundantly loads q/k; its 32 lanes collectively load 128 values.
            for elem in cutlass.range_constexpr(values_per_lane):
                idx = lane * values_per_lane + elem
                q_src[elem] = q_head[idx].to(cutlass.Float32)
                k_src[elem] = k_head[idx].to(cutlass.Float32)

            # Repack adjacent pairs of 4-value lane fragments into vec8 K slices.
            for elem in cutlass.range_constexpr(8):
                source_lane = 2 * k_lane + elem // values_per_lane
                source_value = elem % values_per_lane
                q_vec8[elem] = cute.arch.shuffle_sync(q_src[source_value], offset=source_lane, mask=0xFFFFFFFF)
                k_vec8[elem] = cute.arch.shuffle_sync(k_src[source_value], offset=source_lane, mask=0xFFFFFFFF)

            a = alpha[batch, token, head].to(cutlass.Float32)
            b = beta[batch, token, head].to(cutlass.Float32)
            value_loaded = cutlass.Float32(0.0)
            if lane < TILE_ROWS:
                value_loaded = v_head[v_offset + lane].to(cutlass.Float32)

            for row in cutlass.range_constexpr(TILE_ROWS // 2):
                for elem in cutlass.range_constexpr(8):
                    state[row, elem] *= a

                prediction = _reduce_k(_dot8(state, row, k_vec8))
                v_idx = v_offset + v_lane + 2 * row
                value = cute.arch.shuffle_sync(value_loaded, offset=v_lane + 2 * row, mask=0xFFFFFFFF)
                residual = (value - prediction) * b

                for elem in cutlass.range_constexpr(8):
                    state[row, elem] += residual * k_vec8[elem]

                out = _reduce_k(_dot8(state, row, q_vec8))
                if k_lane == row:
                    output[batch, token, head, v_idx] = out.to(cutlass.BFloat16)

    for row in cutlass.range_constexpr(TILE_ROWS // 2):
        v_idx = v_offset + v_lane + 2 * row
        for elem in cutlass.range_constexpr(8):
            state_out[batch, head, v_idx, k_lane * 8 + elem] = state[row, elem]


@cute.jit
def _launch(
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
    _persistent_recurrent_kernel(q, k, v, alpha, beta, state_in, output, state_out, T, H).launch(
        grid=[batch * H, 1, 1],
        block=[WARPS_PER_CTA * 32, 1, 1],
        stream=stream,
    )


_CACHE: dict[tuple[int, int, int], object] = {}


def _tensor(x):
    return from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)


def _compiled(q, k, v, alpha, beta, state_in, output, state_out):
    B, T, H, _ = q.shape
    key = (B, T, H)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(
            _launch,
            _tensor(q), _tensor(k), _tensor(v), _tensor(alpha), _tensor(beta),
            _tensor(state_in), _tensor(output), _tensor(state_out), make_fake_stream(),
            T, H,
            options="--enable-tvm-ffi --generate-line-info",
        )
        _CACHE[key] = fn
    return fn


@torch.inference_mode()
def persistent_recurrent_gdn(q, k, v, alpha, beta, initial_state):
    B, T, H, width = q.shape
    if width != D or T % CHUNK:
        raise ValueError("requires D=128 and T multiple of 64")
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)
    fn = _compiled(q, k, v, alpha, beta, initial_state, output, final_state)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    fn(q, k, v, alpha, beta, initial_state, output, final_state, stream)
    return output, final_state
