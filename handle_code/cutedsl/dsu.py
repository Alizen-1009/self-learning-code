"""Disjoint-set union with a serial edge pass and parallel final compression."""

import cutlass
import cutlass.cute as cute


THREADS = 128


@cute.jit
def _find_root(parent: cute.Tensor, value: cutlass.Int32):
    root = value
    while parent[root] != root:
        root = parent[root]
    return root


@cute.kernel
def _init_kernel(parent: cute.Tensor, N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < N:
        parent[i] = i


@cute.kernel
def _union_kernel(parent: cute.Tensor, edges: cute.Tensor,
                  E: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    if tid == 0:
        for i in range(E):
            root_a = _find_root(parent, edges[i, 0])
            root_b = _find_root(parent, edges[i, 1])
            if root_a != root_b:
                high = root_a if root_a > root_b else root_b
                low = root_b if root_a > root_b else root_a
                parent[high] = low


@cute.kernel
def _compress_kernel(parent: cute.Tensor, N: cutlass.Constexpr[int]):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i = bid * THREADS + tid
    if i < N:
        parent[i] = _find_root(parent, cutlass.Int32(i))

