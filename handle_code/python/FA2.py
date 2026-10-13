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


class FlashAttentionPytorch(torch.autograd.Function):
    """Forward-only autograd.Function wrapper for blockwise FlashAttention."""

    @staticmethod
    def forward(
        _ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        out, _ = flash_attention_forward(q, k, v, is_causal=is_causal)
        return out


def get_flashattention_autograd_function_pytorch() -> type[FlashAttentionPytorch]:
    return FlashAttentionPytorch
