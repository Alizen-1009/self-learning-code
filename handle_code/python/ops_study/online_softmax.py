"""A small, readable online-softmax implementation for attention's P @ V."""

import torch


def online_softmax_pv(
    p: torch.Tensor,
    v: torch.Tensor,
    *,
    key_block_size: int = 64,
) -> torch.Tensor:
    """Return softmax(P, dim=-1) @ V without storing the full probability matrix.

    P is the pre-softmax score matrix QK^T, shape [num_queries, num_keys].
    Apply the attention scale to P before calling this function if needed.
    V has shape [num_keys, value_dim]. The output has shape
    [num_queries, value_dim].

    For each key block, maintain a row maximum m, softmax denominator l,
    and unnormalized output o. With alpha = exp(m_old - m_new) and
    w = exp(P_block - m_new), update l = alpha*l + sum(w) and
    o = alpha*o + w @ V_block. Normalize only after the final block.
    """
    if p.ndim != 2 or v.ndim != 2 or p.shape[1] != v.shape[0]:
        raise ValueError("expected P [M, N] and V [N, D]")
    if p.shape[1] == 0 or key_block_size <= 0:
        raise ValueError("num_keys and key_block_size must be positive")
    if p.device != v.device:
        raise ValueError("P and V must be on the same device")
    if not torch.is_floating_point(p) or not torch.is_floating_point(v):
        raise TypeError("P and V must be floating-point tensors")

    num_queries, num_keys = p.shape
    value_dim = v.shape[1]
    work_dtype = (
        torch.float64
        if p.dtype == torch.float64 or v.dtype == torch.float64
        else torch.float32
    )
    p_work = p.to(work_dtype)
    v_work = v.to(work_dtype)

    # Per query row: maximum m, denominator l, output numerator o.
    m = torch.full((num_queries, 1), -torch.inf, device=p.device, dtype=work_dtype)
    l = torch.zeros((num_queries, 1), device=p.device, dtype=work_dtype)
    o = torch.zeros((num_queries, value_dim), device=p.device, dtype=work_dtype)

    for start in range(0, num_keys, key_block_size):
        end = min(start + key_block_size, num_keys)
        scores = p_work[:, start:end]  # [M, B]
        values = v_work[start:end]  # [B, D]

        m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
        # Avoid -inf - (-inf) for an all-masked block.
        shift = torch.where(torch.isneginf(m_new), 0.0, m_new)
        alpha = torch.exp(m - shift)  # rescale the previous blocks
        weights = torch.exp(scores - shift)  # not yet normalized

        l = alpha * l + weights.sum(dim=-1, keepdim=True)
        o = alpha * o + weights @ values
        m = m_new

    # l=0 for an entirely masked row, yielding NaN like torch.softmax.
    return (o / l).to(v.dtype)


if __name__ == "__main__":
    # 2 queries, 5 keys, 3 value dimensions; B=2 requires three iterations.
    p = torch.tensor(
        [
            [1000.0, 1001.0, 999.0, -torch.inf, 997.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=torch.float32,
    )
    v = torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10

    actual = online_softmax_pv(p, v, key_block_size=2)
    expected = torch.softmax(p, dim=-1) @ v
    torch.testing.assert_close(actual, expected)
    print("online softmax P @ V passed")
    print(actual)
