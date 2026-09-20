"""Inclusive int32 prefix sum, one independent prefix per output thread."""

import cutlass
import cutlass.cute as cute


THREADS = 128


@cute.kernel
def _prefix_kernel(src: cute.Tensor, dst: cute.Tensor, N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < N:
        total = cutlass.Int32(0)
        for j in range(i + 1):
            total += src[j]
        dst[i] = total


