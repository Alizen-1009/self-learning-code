import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(X, W, Y, N: tl.constexpr, EPS: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # FP16 / BF16 输入转为 FP32 计算。
    x = tl.load(X + row * N + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    y = x * inv_rms * w
    tl.store(Y + row * N + cols, y, mask=mask)


def rmsnorm(x, weight, eps=1e-6):
    """前向：x 为连续 CUDA tensor (..., N)，weight 为 (N,)，均为 FP16 或 BF16。"""
    n = x.shape[-1]
    y = torch.empty_like(x)
    _rmsnorm_kernel[(x.numel() // n,)](
        x, weight, y, n, eps,
        BLOCK_SIZE=triton.next_power_of_2(n),
    )
    return y
