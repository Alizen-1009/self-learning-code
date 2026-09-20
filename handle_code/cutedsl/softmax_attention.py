"""The standard softmax and online attention-score examples from onlie_softmax.cu."""

import cutlass
import cutlass.cute as cute


THREADS = 128
NEG_MAX = -3.4028234663852886e38


@cute.kernel
def _softmax_kernel(x: cute.Tensor, y: cute.Tensor, N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < N:
        m = cutlass.Float32(NEG_MAX)
        for j in range(N):
            value = x[j]
            m = m if m > value else value
        total = cutlass.Float32(0.0)
        for j in range(N):
            total += cute.exp(x[j] - m)
        y[i] = cute.exp(x[i] - m) / total


@cute.kernel
def _attention_score_kernel(src: cute.Tensor, values: cute.Tensor,
                            out: cute.Tensor, N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        m = cutlass.Float32(NEG_MAX)
        denominator = cutlass.Float32(0.0)
        numerator = cutlass.Float32(0.0)
        for i in range(N):
            score = src[i]
            new_m = m if m > score else score
            old_scale = cute.exp(m - new_m)
            new_scale = cute.exp(score - new_m)
            denominator = denominator * old_scale + new_scale
            numerator = numerator * old_scale + values[i] * new_scale
            m = new_m
        out[0] = numerator / denominator


