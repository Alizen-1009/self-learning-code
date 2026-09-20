"""Row-major matrix transpose, with one CuTe DSL thread per output element."""

import cutlass
import cutlass.cute as cute


THREADS = 256


@cute.kernel
def _transpose_kernel(src: cute.Tensor, dst: cute.Tensor,
                      M: cutlass.Constexpr[int], N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    flat = bid * THREADS + tid
    if flat < M * N:
        out_row = flat // M
        out_col = flat % M
        dst[out_row, out_col] = src[out_col, out_row]

