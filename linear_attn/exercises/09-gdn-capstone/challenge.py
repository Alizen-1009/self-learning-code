"""GDN capstone tasks. Fill the two TODO functions without reading grade.py first."""

from __future__ import annotations


def recurrent_gdn_step(state, q, k, v, alpha, beta):
    """Run one V-first GDN step using only Python lists.

    Args:
        state: [V][K]
        q, k: [K]
        v: [V]
        alpha, beta: scalars

    Returns:
        output: [V]
        new_state: [V][K]

    Do not mutate the input state.
    """
    # TODO: decay -> prediction -> residual -> state update -> output
    state = state * alpha
    prediction = state * k
    residual = v - prediction
    state = state + beta * residual
    out = state * q
    return state, out


def schedule_counts(batch, tokens, heads, *, chunk=64, value_dim=128, block_v=16):
    if tokens % chunk != 0:
        raise ValueError("tokens must be divisible by chunk")

    num_chunks = tokens // chunk
    value_tiles = (value_dim + block_v - 1) // block_v

    chunk_grid = batch * heads * num_chunks

    return {
        "num_chunks": num_chunks,
        "gate_kkt": chunk_grid,
        "solve": chunk_grid,
        "wu": chunk_grid,
        "state": batch * heads * value_tiles,
        "output": chunk_grid * value_tiles,
        "flashinfer_tiles": batch * heads,
    }
