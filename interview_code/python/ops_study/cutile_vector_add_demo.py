try:
    import cupy as cp
    import numpy as np
    import cuda.tile as ct
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Install dependencies first: pip install cuda-tile[tileiras] cupy-cuda13x numpy"
    ) from exc


@ct.kernel
def vector_add(x, y, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)

    x_tile = ct.load(x, index=(pid,), shape=(tile_size,))
    y_tile = ct.load(y, index=(pid,), shape=(tile_size,))
    result = x_tile + y_tile

    ct.store(out, index=(pid,), tile=result)


def main():
    n = 1024
    tile_size = 16
    grid = (ct.cdiv(n, tile_size), 1, 1)

    rng = cp.random.default_rng(123)
    x = rng.random(n, dtype=cp.float32)
    y = rng.random(n, dtype=cp.float32)
    out = cp.empty_like(x)

    ct.launch(cp.cuda.get_current_stream(), grid, vector_add, (x, y, out, tile_size))

    np.testing.assert_allclose(cp.asnumpy(out), cp.asnumpy(x + y), rtol=1e-5, atol=1e-5)
    print("cuTile vector add passed")
    print(cp.asnumpy(out[:8]))


if __name__ == "__main__":
    main()
