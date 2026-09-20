"""Row-major FP32 matrix-vector product, one thread per output row."""

import cutlass
import cutlass.cute as cute


THREADS = 128


@cute.kernel
def _sgemv_kernel(a: cute.Tensor, x: cute.Tensor, y: cute.Tensor,
                  M: cutlass.Constexpr[int], N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    row = bid * THREADS + tid
    if row < M:
        total = cutlass.Float32(0.0)
        for col in range(N):
            total += a[row, col] * x[col]
        y[row] = total

