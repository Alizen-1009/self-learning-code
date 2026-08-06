from __future__ import annotations

import torch

D = 128
CHUNK = 64


def make_inputs(batch, tokens, heads, *, seed, random_initial_state, device="cuda"):
    if tokens % CHUNK:
        raise ValueError("T must be a multiple of 64")
    gen = torch.Generator(device=device).manual_seed(seed)
    qf = torch.randn(batch, tokens, heads, D, device=device, generator=gen)
    kf = torch.randn(batch, tokens, heads, D, device=device, generator=gen)
    q = (torch.nn.functional.normalize(qf, dim=-1) * D**-0.5).to(torch.bfloat16)
    k = torch.nn.functional.normalize(kf, dim=-1).to(torch.bfloat16)
    v = torch.randn(batch, tokens, heads, D, device=device, dtype=torch.bfloat16, generator=gen)
    alpha = torch.empty(batch, tokens, heads, device=device).uniform_(0.80, 0.999, generator=gen)
    beta = torch.empty_like(alpha).uniform_(0.10, 0.90, generator=gen)
    state = torch.zeros(batch, heads, D, D, device=device)
    if random_initial_state:
        state.normal_(generator=gen).mul_(0.05)
    return dict(q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), alpha=alpha.contiguous(), beta=beta.contiguous(), initial_state=state.float().contiguous())


@torch.inference_mode()
def recurrent_reference(q, k, v, alpha, beta, initial_state):
    state = initial_state.float().clone()
    output = torch.empty_like(v)
    for t in range(q.shape[1]):
        qt, kt, vt = q[:, t].float(), k[:, t].float(), v[:, t].float()
        state *= alpha[:, t, :, None, None]
        prediction = torch.einsum("bhvk,bhk->bhv", state, kt)
        residual = beta[:, t, :, None] * (vt - prediction)
        state += residual.unsqueeze(-1) * kt.unsqueeze(-2)
        output[:, t] = torch.einsum("bhvk,bhk->bhv", state, qt).to(v.dtype)
    return output, state


def metrics(actual, expected):
    diff = actual.float() - expected.float()
    return diff.abs().max().item(), (diff.norm() / expected.float().norm().clamp_min(1e-12)).item()
