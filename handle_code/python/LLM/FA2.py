from __future__ import annotations

import math

import torch


DEFAULT_QUERY_BLOCK_SIZE = 64
DEFAULT_KEY_BLOCK_SIZE = 64


def _attention_scale(q: torch.Tensor) -> float:
    return 1.0 / math.sqrt(q.shape[-1])


def _apply_causal_mask_block(
    scores: torch.Tensor,
    q_start: int,
    k_start: int,
) -> torch.Tensor:
    q_positions = torch.arange(q_start, q_start + scores.shape[-2], device=scores.device)[:, None]
    k_positions = torch.arange(k_start, k_start + scores.shape[-1], device=scores.device)[None, :]
    return torch.where(q_positions >= k_positions, scores, scores.new_full((), -1e6))


def _flash_attention_forward_single_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    query_block_size: int,
    key_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_queries, d_model = q.shape
    n_keys = k.shape[0]
    scale = _attention_scale(q)

    # Keep the accumulators in fp32 for stability even when the inputs are lower precision.
    q_work = q.to(torch.float32)
    k_work = k.to(torch.float32)
    v_work = v.to(torch.float32)

    out = torch.zeros((n_queries, d_model), device=q.device, dtype=torch.float32)
    max_scores = torch.full((n_queries, 1), float("-inf"), device=q.device, dtype=torch.float32)
    normalizers = torch.zeros((n_queries, 1), device=q.device, dtype=torch.float32)

    for q_start in range(0, n_queries, query_block_size):
        q_end = min(q_start + query_block_size, n_queries)
        q_block = q_work[q_start:q_end]

        out_block = torch.zeros((q_end - q_start, d_model), device=q.device, dtype=torch.float32)
        max_block = torch.full((q_end - q_start, 1), float("-inf"), device=q.device, dtype=torch.float32)
        norm_block = torch.zeros((q_end - q_start, 1), device=q.device, dtype=torch.float32)

        for k_start in range(0, n_keys, key_block_size):
            k_end = min(k_start + key_block_size, n_keys)
            k_block = k_work[k_start:k_end]
            v_block = v_work[k_start:k_end]

            scores_block = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale
            if is_causal:
                scores_block = _apply_causal_mask_block(scores_block, q_start=q_start, k_start=k_start)

            block_max = torch.max(scores_block, dim=-1, keepdim=True).values
            new_max = torch.maximum(max_block, block_max)
            exp_scale = torch.exp(max_block - new_max)
            probs_block = torch.exp(scores_block - new_max)

            norm_block = norm_block * exp_scale + torch.sum(probs_block, dim=-1, keepdim=True)
            out_block = out_block * exp_scale + torch.matmul(probs_block, v_block)
            max_block = new_max

        out[q_start:q_end] = out_block / norm_block
        max_scores[q_start:q_end] = max_block
        normalizers[q_start:q_end] = norm_block

    lse = max_scores.squeeze(-1) + torch.log(normalizers.squeeze(-1))
    return out.to(q.dtype), lse


def _flash_attention_forward_blockwise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    query_block_size: int,
    key_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    leading_shape = q.shape[:-2]
    n_queries, d_model = q.shape[-2:]

    q_flat = q.reshape(-1, n_queries, d_model)
    k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
    v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])

    outputs = []
    lses = []
    for batch_idx in range(q_flat.shape[0]):
        out, lse = _flash_attention_forward_single_batch(
            q_flat[batch_idx],
            k_flat[batch_idx],
            v_flat[batch_idx],
            is_causal=is_causal,
            query_block_size=query_block_size,
            key_block_size=key_block_size,
        )
        outputs.append(out)
        lses.append(lse)

    out = torch.stack(outputs, dim=0).reshape(*leading_shape, n_queries, d_model)
    lse = torch.stack(lses, dim=0).reshape(*leading_shape, n_queries)
    return out, lse


def _flash_attention_backward_single_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    is_causal: bool,
    query_block_size: int,
    key_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_queries, d_model = q.shape
    n_keys = k.shape[0]
    scale = _attention_scale(q)

    q_work = q.to(torch.float32)
    k_work = k.to(torch.float32)
    v_work = v.to(torch.float32)
    grad_out_work = grad_out.to(torch.float32)
    out_work = out.to(torch.float32)
    lse_work = lse.to(torch.float32)

    grad_q = torch.zeros_like(q_work)
    grad_k = torch.zeros_like(k_work)
    grad_v = torch.zeros_like(v_work)

    # Di = sum_d dO_i[d] * O_i[d], used by the softmax backward.
    row_dot = torch.sum(grad_out_work * out_work, dim=-1, keepdim=True)

    for q_start in range(0, n_queries, query_block_size):
        q_end = min(q_start + query_block_size, n_queries)
        q_block = q_work[q_start:q_end]
        grad_out_block = grad_out_work[q_start:q_end]
        lse_block = lse_work[q_start:q_end, None]
        row_dot_block = row_dot[q_start:q_end]

        grad_q_block = torch.zeros_like(q_block)

        for k_start in range(0, n_keys, key_block_size):
            k_end = min(k_start + key_block_size, n_keys)
            k_block = k_work[k_start:k_end]
            v_block = v_work[k_start:k_end]

            scores_block = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale
            if is_causal:
                scores_block = _apply_causal_mask_block(scores_block, q_start=q_start, k_start=k_start)

            probs_block = torch.exp(scores_block - lse_block)
            grad_v[k_start:k_end] += torch.matmul(probs_block.transpose(-1, -2), grad_out_block)

            grad_probs_block = torch.matmul(grad_out_block, v_block.transpose(-1, -2))
            grad_scores_block = probs_block * (grad_probs_block - row_dot_block)

            grad_q_block += torch.matmul(grad_scores_block, k_block) * scale
            grad_k[k_start:k_end] += torch.matmul(grad_scores_block.transpose(-1, -2), q_block) * scale

        grad_q[q_start:q_end] = grad_q_block

    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    query_block_size: int = DEFAULT_QUERY_BLOCK_SIZE,
    key_block_size: int = DEFAULT_KEY_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _flash_attention_forward_blockwise(
        q,
        k,
        v,
        is_causal=is_causal,
        query_block_size=query_block_size,
        key_block_size=key_block_size,
    )


def flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    is_causal: bool,
    query_block_size: int = DEFAULT_QUERY_BLOCK_SIZE,
    key_block_size: int = DEFAULT_KEY_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    leading_shape = q.shape[:-2]
    n_queries, d_model = q.shape[-2:]

    q_flat = q.reshape(-1, n_queries, d_model)
    k_flat = k.reshape(-1, k.shape[-2], k.shape[-1])
    v_flat = v.reshape(-1, v.shape[-2], v.shape[-1])
    grad_out_flat = grad_out.reshape(-1, n_queries, d_model)
    out_flat = out.reshape(-1, n_queries, d_model)
    lse_flat = lse.reshape(-1, n_queries)

    grad_qs = []
    grad_ks = []
    grad_vs = []
    for batch_idx in range(q_flat.shape[0]):
        grad_q, grad_k, grad_v = _flash_attention_backward_single_batch(
            q_flat[batch_idx],
            k_flat[batch_idx],
            v_flat[batch_idx],
            grad_out_flat[batch_idx],
            out_flat[batch_idx],
            lse_flat[batch_idx],
            is_causal=is_causal,
            query_block_size=query_block_size,
            key_block_size=key_block_size,
        )
        grad_qs.append(grad_q)
        grad_ks.append(grad_k)
        grad_vs.append(grad_v)

    grad_q = torch.stack(grad_qs, dim=0).reshape(*leading_shape, n_queries, d_model)
    grad_k = torch.stack(grad_ks, dim=0).reshape(*leading_shape, k.shape[-2], k.shape[-1])
    grad_v = torch.stack(grad_vs, dim=0).reshape(*leading_shape, v.shape[-2], v.shape[-1])
    return grad_q, grad_k, grad_v


class FlashAttentionPytorch(torch.autograd.Function):
    """
    Assignment-facing autograd.Function for the PyTorch FlashAttention path.

    Saved tensors are intentionally minimal and stable:
    - q, k, v for backward reconstruction
    - out for the softmax backward reduction term
    - lse so the tests can verify we saved log-sum-exp

    The forward pass now uses a blockwise online-softmax algorithm in the
    "outer-q" style, which avoids materializing the full attention matrix.
    Backward mirrors that structure blockwise as well, reconstructing only the
    local score tiles needed for each gradient update.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        out, lse = flash_attention_forward(q, k, v, is_causal=is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, out, lse)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, out, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = flash_attention_backward(
            q,
            k,
            v,
            grad_out,
            out,
            lse,
            is_causal=ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None


def get_flashattention_autograd_function_pytorch() -> type[FlashAttentionPytorch]:
    return FlashAttentionPytorch
