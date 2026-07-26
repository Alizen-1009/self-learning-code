try:
    import torch
    import tilelang
    import tilelang.language as T
    from tilelang import jit
except ModuleNotFoundError as exc:
    raise SystemExit("Install dependencies first: pip install torch tilelang") from exc


@jit
def vector_add(n: int, block: int = 256, dtype: str = "float32"):
    @T.prim_func
    def kernel(
        x: T.Tensor((n,), dtype),
        y: T.Tensor((n,), dtype),
        out: T.Tensor((n,), dtype),
    ):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                idx = bx * block + i
                out[idx] = x[idx] + y[idx]

    return kernel


def main():
    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)

    kernel = vector_add(n)
    kernel(x, y, out)

    torch.testing.assert_close(out, x + y)
    print("tilelang vector add passed")
    print(out[:8])


if __name__ == "__main__":
    main()
