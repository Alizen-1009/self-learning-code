from __future__ import annotations

import torch

D = 128


def make_inputs(
    batch: int,
    tokens: int,
    heads: int,
    *,
    seed: int,
    random_initial_state: bool,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Create the Phase-1 contract: preprocessed q/k and activated alpha/beta."""
    generator = torch.Generator(device=device).manual_seed(seed)

    qf = torch.randn(batch, tokens, heads, D, device=device, generator=generator)
    kf = torch.randn(batch, tokens, heads, D, device=device, generator=generator)
    qf = torch.nn.functional.normalize(qf, dim=-1) * (D**-0.5)
    kf = torch.nn.functional.normalize(kf, dim=-1)

    q = qf.to(torch.bfloat16)
    k = kf.to(torch.bfloat16)
    v = torch.randn(
        batch, tokens, heads, D, device=device, dtype=torch.bfloat16, generator=generator
    )
    # Activated multiplicative forget gate and delta update rate.
    alpha = torch.empty(batch, tokens, heads, device=device, dtype=torch.float32)
    alpha.uniform_(0.80, 0.999, generator=generator)
    beta = torch.empty_like(alpha)
    beta.uniform_(0.10, 0.90, generator=generator)

    if random_initial_state:
        initial_state = torch.randn(
            batch, heads, D, D, device=device, dtype=torch.float32, generator=generator
        ) * 0.05
    else:
        initial_state = torch.zeros(
            batch, heads, D, D, device=device, dtype=torch.float32
        )

    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "alpha": alpha.contiguous(),
        "beta": beta.contiguous(),
        "initial_state": initial_state.contiguous(),
    }


@torch.inference_mode()
def recurrent_gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 recurrent oracle with V-first state [B,H,V,K].

    q is already L2-normalized and scaled by 1/sqrt(D); k is normalized;
    alpha/beta are already activated.
    """
    batch, tokens, heads, width = q.shape
    assert width == D and k.shape == q.shape and v.shape == q.shape
    assert alpha.shape == (batch, tokens, heads)
    assert beta.shape == alpha.shape
    assert initial_state.shape == (batch, heads, D, D)

    state = initial_state.float().clone()
    output = torch.empty_like(v)
    for t in range(tokens):
        qt = q[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        at = alpha[:, t, :, None, None]
        bt = beta[:, t, :, None]

        state = state * at
        prediction = torch.einsum("bhvk,bhk->bhv", state, kt)
        residual = (vt - prediction) * bt
        state = state + residual.unsqueeze(-1) * kt.unsqueeze(-2)
        output[:, t] = torch.einsum("bhvk,bhk->bhv", state, qt).to(v.dtype)

    return output, state


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = actual.float() - expected.float()
    max_abs = diff.abs().max().item()
    rel_l2 = (diff.norm() / expected.float().norm().clamp_min(1e-12)).item()
    return max_abs, rel_l2
