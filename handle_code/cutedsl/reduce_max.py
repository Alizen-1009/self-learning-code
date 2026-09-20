"""Scalar FP32 maximum reduction."""

import cutlass
import cutlass.cute as cute


@cute.kernel
def _reduce_max_kernel(x: cute.Tensor, out: cute.Tensor,
                       N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        value = cutlass.Float32(-3.4028234663852886e38)
        for i in range(N):
            current = x[i]
            value = value if value > current else current
        out[0] = value


