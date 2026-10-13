from __future__ import annotations

try:
    import torch
    import triton
    import triton.language as tl
except ModuleNotFoundError as exc:
    raise SystemExit("Install dependencies first: pip install torch triton") from exc


@triton.jit
def _softmax_kernel(
    x_ptr,
    y_ptr,
    n_cols,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute one softmax row per Triton program."""

    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x_row_ptr = x_ptr + row_id * stride_x_row + offsets
    y_row_ptr = y_ptr + row_id * stride_y_row + offsets

    # Masked values must not affect the reduction.  Using -inf makes the
    # padded elements disappear from both the max and the exp/sum reduction.
    x = tl.load(x_row_ptr, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)

    x_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - x_max)
    exp_sum = tl.sum(exp_x, axis=0)
    y = exp_x / exp_sum

    # Storing to the original dtype is safe because the reductions above use
    # FP32, while PyTorch's softmax also returns the input dtype by default.
    tl.store(y_row_ptr, y, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


LONG_SOFTMAX_THRESHOLD = 4096
LONG_SOFTMAX_BLOCK_SIZE = 1024


@triton.jit
def _softmax_chunk_stats_kernel(
    x_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute stable softmax statistics for one chunk of a 1D tensor."""

    chunk_id = tl.program_id(0)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)

    chunk_max = tl.max(x, axis=0)
    exp_x = tl.exp(x - chunk_max)
    chunk_sum = tl.sum(exp_x, axis=0)

    tl.store(partial_max_ptr + chunk_id, chunk_max)
    tl.store(partial_sum_ptr + chunk_id, chunk_sum)


@triton.jit
def _softmax_global_stats_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    global_max_ptr,
    global_sum_ptr,
    n_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    """Merge chunk statistics using the log-sum-exp rescaling identity."""

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_chunks

    partial_max = tl.load(partial_max_ptr + offsets, mask=mask, other=-float("inf"))
    partial_sum = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0)

    global_max = tl.max(partial_max, axis=0)
    global_sum = tl.sum(
        partial_sum * tl.exp(partial_max - global_max),
        axis=0,
    )

    tl.store(global_max_ptr, global_max)
    tl.store(global_sum_ptr, global_sum)


@triton.jit
def _softmax_long_output_kernel(
    x_ptr,
    y_ptr,
    global_max_ptr,
    global_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize one chunk using the global softmax statistics."""

    chunk_id = tl.program_id(0)
    offsets = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    global_max = tl.load(global_max_ptr)
    global_sum = tl.load(global_sum_ptr)
    y = tl.exp(x - global_max) / global_sum

    tl.store(y_ptr + offsets, y, mask=mask)


def _softmax_1d_long(x: torch.Tensor) -> torch.Tensor:
    """Three-kernel softmax path for a long 1D tensor."""

    x = x.contiguous()
    n_elements = x.numel()
    y = torch.empty_like(x)
    n_chunks = triton.cdiv(n_elements, LONG_SOFTMAX_BLOCK_SIZE)

    partial_max = torch.empty(n_chunks, device=x.device, dtype=torch.float32)
    partial_sum = torch.empty(n_chunks, device=x.device, dtype=torch.float32)
    global_max = torch.empty((), device=x.device, dtype=torch.float32)
    global_sum = torch.empty((), device=x.device, dtype=torch.float32)

    # Kernel 1: one program computes (local max, local sum) for one chunk.
    _softmax_chunk_stats_kernel[(n_chunks,)](
        x,
        partial_max,
        partial_sum,
        n_elements,
        BLOCK_SIZE=LONG_SOFTMAX_BLOCK_SIZE,
        num_warps=4,
    )

    # Kernel 2: one program merges all chunk statistics.  For N=100000,
    # n_chunks is only 98, so this reduction comfortably fits in one program.
    stats_block_size = _next_power_of_2(n_chunks)
    _softmax_global_stats_kernel[(1,)](
        partial_max,
        partial_sum,
        global_max,
        global_sum,
        n_chunks,
        BLOCK_SIZE=stats_block_size,
        num_warps=4,
    )

    # Kernel 3: each program writes one normalized chunk.
    _softmax_long_output_kernel[(n_chunks,)](
        x,
        y,
        global_max,
        global_sum,
        n_elements,
        BLOCK_SIZE=LONG_SOFTMAX_BLOCK_SIZE,
        num_warps=4,
    )
    return y


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Triton softmax over the last dimension of ``x``.

    The input is flattened to ``[n_rows, n_cols]``; each Triton program then
    loads one row, so rows are independent and no cross-program synchronization
    is needed.
    """

    if not x.is_cuda:
        raise ValueError("softmax() expects a CUDA tensor")
    if not x.is_floating_point():
        raise TypeError("softmax() expects a floating-point tensor")
    if x.ndim == 0:
        raise ValueError("softmax() expects a tensor with at least one dimension")
    if dim < 0:
        dim += x.ndim
    if dim != x.ndim - 1:
        raise NotImplementedError("this example only supports dim=-1")
    if x.shape[-1] == 0:
        raise ValueError("softmax() does not support an empty last dimension")
    if x.numel() == 0:
        return torch.empty_like(x)

    if x.ndim == 1 and x.shape[0] > LONG_SOFTMAX_THRESHOLD:
        return _softmax_1d_long(x)

    x_2d = x.contiguous().reshape(-1, x.shape[-1])
    y_2d = torch.empty_like(x_2d)
    n_rows, n_cols = x_2d.shape
    block_size = _next_power_of_2(n_cols)

    _softmax_kernel[(n_rows,)](
        x_2d,
        y_2d,
        n_cols,
        x_2d.stride(0),
        y_2d.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return y_2d.reshape(x.shape)


def main() -> None:
    torch.manual_seed(0)

    # 513 deliberately is not a power of two, so the masked tail is tested.
    x = torch.randn((4, 513), device="cuda", dtype=torch.float16)
    actual = softmax(x)
    expected = torch.softmax(x, dim=-1)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    print("triton softmax passed")
    print(actual[0, :8])

    x_long = torch.randn(100_000, device="cuda", dtype=torch.float16)
    actual_long = softmax(x_long)
    expected_long = torch.softmax(x_long, dim=-1)

    torch.testing.assert_close(actual_long, expected_long, rtol=1e-2, atol=1e-2)
    print("triton long 1D softmax passed")
    print(actual_long[:8])


if __name__ == "__main__":
    main()
