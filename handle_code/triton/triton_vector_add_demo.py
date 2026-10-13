try:
    import torch
    import triton
    import triton.language as tl
except ModuleNotFoundError as exc:
    raise SystemExit("Install dependencies first: pip install torch triton") from exc


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def main():
    n = 1024
    block = 256
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, block),)
    vector_add_kernel[grid](x, y, out, n, BLOCK=block)

    torch.testing.assert_close(out, x + y)
    print("triton vector add passed")
    print(out[:8])


if __name__ == "__main__":
    main()
