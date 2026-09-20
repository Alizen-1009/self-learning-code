"""Numerically stable 1D softmax using an online (max, exp-sum) recurrence."""

import cutlass
import cutlass.cute as cute


THREADS = 256
NEG_MAX = -3.4028234663852886e38


@cute.kernel
def _stats_kernel(x: cute.Tensor, stats: cute.Tensor, N: cutlass.Constexpr[int]):
    # One thread computes the stable recurrence. A second kernel applies the
    # resulting statistics in parallel, avoiding a grid-wide synchronization.
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        m = cutlass.Float32(NEG_MAX)
        d = cutlass.Float32(0.0)
        for i in range(N):
            value = x[i]
            new_m = m if m > value else value
            d = d * cute.exp(m - new_m) + cute.exp(value - new_m)
            m = new_m
        stats[0] = m
        stats[1] = d


@cute.kernel
def _apply_kernel(x: cute.Tensor, y: cute.Tensor, stats: cute.Tensor,
                  N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < N:
        y[i] = cute.exp(x[i] - stats[0]) / stats[1]


