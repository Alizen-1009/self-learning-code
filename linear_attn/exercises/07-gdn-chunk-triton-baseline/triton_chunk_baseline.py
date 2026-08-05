from __future__ import annotations

import torch
import triton
import triton.language as tl

# Triton 3.6 only permits captured globals instantiated as tl.constexpr.
D = tl.constexpr(128)
C = tl.constexpr(64)
D_INT = 128
C_INT = 64


@triton.jit
def _gate_kkt_kernel(k_ptr, alpha_ptr, beta_ptr, g_ptr, left_ptr, T: tl.constexpr, H: tl.constexpr, NT: tl.constexpr):
    pid = tl.program_id(0)
    chunk = pid % NT
    bh = pid // NT
    batch = bh // H
    head = bh % H
    rows = tl.arange(0, C)
    cols = tl.arange(0, D)
    token = chunk * C + rows

    k_off = ((batch * T + token[:, None]) * H + head) * D + cols[None, :]
    k = tl.load(k_ptr + k_off).to(tl.bfloat16)
    scalar_off = (batch * T + token) * H + head
    alpha = tl.load(alpha_ptr + scalar_off).to(tl.float32)
    beta = tl.load(beta_ptr + scalar_off).to(tl.float32)
    G = tl.cumsum(tl.log(alpha), axis=0)
    gram = tl.dot(k, tl.trans(k), out_dtype=tl.float32)
    r = rows[:, None]
    c = rows[None, :]
    strict = r > c
    rel = tl.exp(G[:, None] - G[None, :])
    left = tl.where(strict, beta[:, None] * rel * gram, 0.0)
    left += tl.where(r == c, 1.0, 0.0)

    matrix_base = pid * C * C
    tl.store(left_ptr + matrix_base + r * C + c, left)
    tl.store(g_ptr + scalar_off, G)


@triton.jit
def _solve_kernel(left_ptr, ainv_ptr):
    pid = tl.program_id(0)
    rows = tl.arange(0, C)
    r = rows[:, None]
    c = rows[None, :]
    left = tl.load(left_ptr + pid * C * C + r * C + c).to(tl.float32)
    inv = tl.where(r == c, 1.0, 0.0)

    # Row-wise forward substitution for a unit-lower triangular matrix.
    for i in range(1, C):
        row_i = tl.sum(tl.where(r == i, left, 0.0), axis=0)
        solved = -tl.sum(row_i[:, None] * inv, axis=0)
        solved = tl.where(rows < i, solved, tl.where(rows == i, 1.0, 0.0))
        inv = tl.where(r == i, solved[None, :], inv)

    tl.store(ainv_ptr + pid * C * C + r * C + c, inv.to(tl.bfloat16))


@triton.jit
def _wu_kernel(k_ptr, v_ptr, beta_ptr, g_ptr, ainv_ptr, w_ptr, u_ptr, T: tl.constexpr, H: tl.constexpr, NT: tl.constexpr):
    pid = tl.program_id(0)
    chunk = pid % NT
    bh = pid // NT
    batch = bh // H
    head = bh % H
    rows = tl.arange(0, C)
    width = tl.arange(0, D)
    token = chunk * C + rows
    data_off = ((batch * T + token[:, None]) * H + head) * D + width[None, :]
    scalar_off = (batch * T + token) * H + head

    k = tl.load(k_ptr + data_off).to(tl.bfloat16)
    v = tl.load(v_ptr + data_off).to(tl.bfloat16)
    beta = tl.load(beta_ptr + scalar_off).to(tl.float32)
    G = tl.load(g_ptr + scalar_off).to(tl.float32)
    rr = rows[:, None]
    cc = rows[None, :]
    ainv = tl.load(ainv_ptr + pid * C * C + rr * C + cc).to(tl.bfloat16)

    vb = (v.to(tl.float32) * beta[:, None]).to(tl.bfloat16)
    kbg = (k.to(tl.float32) * (beta * tl.exp(G))[:, None]).to(tl.bfloat16)
    u = tl.dot(ainv, vb, out_dtype=tl.float32)
    w = tl.dot(ainv, kbg, out_dtype=tl.float32)
    tl.store(u_ptr + data_off, u.to(tl.bfloat16))
    tl.store(w_ptr + data_off, w.to(tl.bfloat16))


@triton.jit
def _state_scan_kernel(k_ptr, g_ptr, w_ptr, u_ptr, state_in_ptr, h_ptr, r_ptr, state_out_ptr, T: tl.constexpr, H: tl.constexpr, NT: tl.constexpr, BV: tl.constexpr):
    bh = tl.program_id(0)
    v_tile = tl.program_id(1)
    batch = bh // H
    head = bh % H
    vv = v_tile * BV + tl.arange(0, BV)
    kk = tl.arange(0, D)
    vmask = vv < D
    state_off = ((batch * H + head) * D + vv[:, None]) * D + kk[None, :]
    state = tl.load(state_in_ptr + state_off, mask=vmask[:, None], other=0.0).to(tl.float32)

    rows = tl.arange(0, C)
    for chunk in range(NT):
        h_base = ((batch * NT + chunk) * H + head) * D * D
        tl.store(h_ptr + h_base + vv[:, None] * D + kk[None, :], state, mask=vmask[:, None])

        token = chunk * C + rows
        data_off = ((batch * T + token[:, None]) * H + head) * D
        w = tl.load(w_ptr + data_off + kk[None, :]).to(tl.bfloat16)
        u = tl.load(u_ptr + data_off + vv[None, :], mask=vmask[None, :], other=0.0).to(tl.float32)
        pred = tl.dot(w, tl.trans(state.to(tl.bfloat16)), out_dtype=tl.float32)
        residual = u - pred
        tl.store(r_ptr + data_off + vv[None, :], residual.to(tl.bfloat16), mask=vmask[None, :])

        G = tl.load(g_ptr + (batch * T + token) * H + head).to(tl.float32)
        G_last = tl.sum(tl.where(rows == C - 1, G, 0.0), axis=0)
        k = tl.load(k_ptr + data_off + kk[None, :]).to(tl.bfloat16)
        rend = (residual * tl.exp(G_last - G)[:, None]).to(tl.bfloat16)
        update = tl.dot(tl.trans(k), rend, out_dtype=tl.float32)
        state = state * tl.exp(G_last) + tl.trans(update)

    tl.store(state_out_ptr + state_off, state, mask=vmask[:, None])


@triton.jit
def _output_kernel(q_ptr, k_ptr, g_ptr, h_ptr, r_ptr, output_ptr, T: tl.constexpr, H: tl.constexpr, NT: tl.constexpr, BV: tl.constexpr):
    pid = tl.program_id(0)
    v_tile = tl.program_id(1)
    chunk = pid % NT
    bh = pid // NT
    batch = bh // H
    head = bh % H
    rows = tl.arange(0, C)
    kk = tl.arange(0, D)
    vv = v_tile * BV + tl.arange(0, BV)
    vmask = vv < D
    token = chunk * C + rows
    qk_off = ((batch * T + token[:, None]) * H + head) * D + kk[None, :]
    scalar_off = (batch * T + token) * H + head

    q = tl.load(q_ptr + qk_off).to(tl.bfloat16)
    k = tl.load(k_ptr + qk_off).to(tl.bfloat16)
    G = tl.load(g_ptr + scalar_off).to(tl.float32)
    h_base = ((batch * NT + chunk) * H + head) * D * D
    h = tl.load(h_ptr + h_base + vv[:, None] * D + kk[None, :], mask=vmask[:, None], other=0.0).to(tl.bfloat16)
    residual = tl.load(r_ptr + ((batch * T + token[:, None]) * H + head) * D + vv[None, :], mask=vmask[None, :], other=0.0).to(tl.bfloat16)

    qg = (q.to(tl.float32) * tl.exp(G)[:, None]).to(tl.bfloat16)
    inter = tl.dot(qg, tl.trans(h), out_dtype=tl.float32)
    scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    rr = rows[:, None]
    cc = rows[None, :]
    weighted = tl.where(rr >= cc, scores * tl.exp(G[:, None] - G[None, :]), 0.0)
    intra = tl.dot(weighted.to(tl.bfloat16), residual, out_dtype=tl.float32)
    out_off = ((batch * T + token[:, None]) * H + head) * D + vv[None, :]
    tl.store(output_ptr + out_off, inter + intra, mask=vmask[None, :])


@torch.inference_mode()
def chunk_gdn_triton(q, k, v, alpha, beta, initial_state):
    B, T, H, width = q.shape
    if width != D_INT or T % C_INT:
        raise ValueError("requires D=128 and T multiple of 64")
    NT = T // C_INT
    G = torch.empty_like(alpha)
    left = torch.empty(B, NT, H, C_INT, C_INT, device=q.device, dtype=torch.float32)
    ainv = torch.empty(B, NT, H, C_INT, C_INT, device=q.device, dtype=torch.bfloat16)
    w, u, residual = torch.empty_like(q), torch.empty_like(v), torch.empty_like(v)
    h = torch.empty(B, NT, H, D_INT, D_INT, device=q.device, dtype=torch.float32)
    output = torch.empty_like(v)
    final_state = torch.empty_like(initial_state)

    chunk_grid = (B * H * NT,)
    _gate_kkt_kernel[chunk_grid](k, alpha, beta, G, left, T=T, H=H, NT=NT, num_warps=4)
    _solve_kernel[chunk_grid](left, ainv, num_warps=8, num_stages=1)
    _wu_kernel[chunk_grid](k, v, beta, G, ainv, w, u, T=T, H=H, NT=NT, num_warps=8, num_stages=2)
    bv = 16
    _state_scan_kernel[(B * H, triton.cdiv(D_INT, bv))](k, G, w, u, initial_state, h, residual, final_state, T=T, H=H, NT=NT, BV=bv, num_warps=4, num_stages=2)
    _output_kernel[(B * H * NT, triton.cdiv(D_INT, bv))](q, k, G, h, residual, output, T=T, H=H, NT=NT, BV=bv, num_warps=4, num_stages=2)
    return output, final_state, {"G": G, "Ainv": ainv, "W": w, "U": u, "H": h, "R": residual}
