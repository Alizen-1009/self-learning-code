"""FP32 RMSNorm with separate statistic and output kernels."""

import cutlass
import cutlass.cute as cute


THREADS = 256


@cute.kernel
def _rms_stats_kernel(x: cute.Tensor, stats: cute.Tensor,
                      N: cutlass.Constexpr[int], eps: cutlass.Constexpr[float]):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        squares = cutlass.Float32(0.0)
        for i in range(N):
            value = x[i]
            squares += value * value
        stats[0] = cute.rsqrt(squares / N + eps, fastmath=True)


@cute.kernel
def _rms_apply_kernel(x: cute.Tensor, weight: cute.Tensor, y: cute.Tensor,
                      stats: cute.Tensor, N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < N:
        y[i] = x[i] * stats[0] * weight[i]


