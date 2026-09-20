"""Naive row-major FP32 matrix multiplication in CuTe DSL."""

import cutlass
import cutlass.cute as cute


BM = 16
BN = 16


@cute.kernel
def _matmul_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor,
                   M: cutlass.Constexpr[int], N: cutlass.Constexpr[int],
                   K: cutlass.Constexpr[int]):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    row = by * BM + ty
    col = bx * BN + tx
    if row < M and col < N:
        acc = cutlass.Float32(0.0)
        for k in range(K):
            acc += a[row, k] * b[k, col]
        c[row, col] = acc

