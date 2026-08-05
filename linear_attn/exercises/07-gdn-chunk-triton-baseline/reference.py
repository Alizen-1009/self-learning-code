from __future__ import annotations

import torch

D = 128
CHUNK = 64


def make_inputs(batch, tokens, heads, *, seed, random_initial_state, device="cuda"):
    if tokens % CHUNK:
        raise ValueError("Lesson 7 requires T to be a multiple of 64")
    generator = torch.Generator(device=device).manual_seed(seed)
    qf = torch.randn(batch, tokens, heads, D, device=device, generator=generator)
    kf = torch.randn(batch, tokens, heads, D, device=device, generator=generator)
    q = (torch.nn.functional.normalize(qf, dim=-1) * (D**-0.5)).to(torch.bfloat16)
    k = torch.nn.functional.normalize(kf, dim=-1).to(torch.bfloat16)
    v = torch.randn(batch, tokens, heads, D, device=device, dtype=torch.bfloat16, generator=generator)
    alpha = torch.empty(batch, tokens, heads, device=device).uniform_(0.97, 0.999, generator=generator)
    beta = torch.empty_like(alpha).uniform_(0.10, 0.90, generator=generator)
    if random_initial_state:
        state = torch.randn(batch, heads, D, D, device=device, generator=generator) * 0.05
    else:
        state = torch.zeros(batch, heads, D, D, device=device)
    return dict(q=q.contiguous(), k=k.contiguous(), v=v.contiguous(), alpha=alpha.contiguous(), beta=beta.contiguous(), initial_state=state.float().contiguous())


@torch.inference_mode()
def recurrent_gdn_reference(q, k, v, alpha, beta, initial_state):
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


@torch.inference_mode()
def chunk_gdn_reference(q, k, v, alpha, beta, initial_state):
    """Exact C=64 chunk formulation, kept in FP32 as the algorithm oracle."""
    B, T, H, _ = q.shape
    state = initial_state.float().clone()
    output = torch.empty_like(v)
    eye = torch.eye(CHUNK, device=q.device, dtype=torch.float32)
    strict = torch.tril(torch.ones_like(eye), diagonal=-1)
    causal = torch.tril(torch.ones_like(eye))

    for start in range(0, T, CHUNK):
        stop = start + CHUNK
        for b in range(B):
            for h in range(H):
                qc = q[b, start:stop, h].float()
                kc = k[b, start:stop, h].float()
                vc = v[b, start:stop, h].float()
                ac = alpha[b, start:stop, h].float()
                bc = beta[b, start:stop, h].float()

                G = torch.log(ac).cumsum(0)
                rel = torch.exp(G[:, None] - G[None, :])
                lower = bc[:, None] * (kc @ kc.T) * rel * strict
                left = eye + lower
                ainv = torch.linalg.solve_triangular(left, eye, upper=False)

                U = ainv @ (bc[:, None] * vc)
                Kg = kc * torch.exp(G)[:, None]
                W = ainv @ (bc[:, None] * Kg)
                R = U - W @ state[b, h].T

                P = (qc @ kc.T) * rel * causal
                inter = (qc * torch.exp(G)[:, None]) @ state[b, h].T
                output[b, start:stop, h] = (inter + P @ R).to(v.dtype)

                Kend = kc * torch.exp(G[-1] - G)[:, None]
                state[b, h] = torch.exp(G[-1]) * state[b, h] + (Kend.T @ R).T
    return output, state


def error_metrics(actual, expected):
    diff = actual.float() - expected.float()
    return diff.abs().max().item(), (diff.norm() / expected.float().norm().clamp_min(1e-12)).item()
