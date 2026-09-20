"""Radix selection for the K-th largest signed int32 value."""

import cutlass
import cutlass.cute as cute


@cute.kernel
def _topk_kernel(x: cute.Tensor, out: cute.Tensor,
                 N: cutlass.Constexpr[int], K: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        # Flipping the sign bit maps signed int32 ordering to unsigned ordering.
        sign_bit = cutlass.Uint32(0x80000000)
        prefix = cutlass.Uint32(0)
        prefix_mask = cutlass.Uint32(0)
        rank = cutlass.Int32(K)

        for shift in (28, 24, 20, 16, 12, 8, 4, 0):
            chosen = False
            for digit in range(15, -1, -1):
                count = cutlass.Int32(0)
                for i in range(N):
                    ordered = x[i].to(cutlass.Uint32) ^ sign_bit
                    matches_prefix = (ordered & prefix_mask) == prefix
                    matches_digit = ((ordered >> shift) & cutlass.Uint32(15)) == digit
                    if matches_prefix:
                        if matches_digit:
                            count += 1

                if not chosen:
                    if rank > count:
                        rank -= count
                    else:
                        prefix = prefix | (cutlass.Uint32(digit) << shift)
                        prefix_mask = prefix_mask | (cutlass.Uint32(15) << shift)
                        chosen = True

        original = prefix ^ sign_bit
        out[0] = original.to(cutlass.Int32)

