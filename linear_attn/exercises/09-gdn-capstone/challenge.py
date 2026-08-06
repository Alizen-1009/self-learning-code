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
    raise NotImplementedError


def schedule_counts(batch, tokens, heads, *, chunk=64, value_dim=128, block_v=16):
    """Return program/CTA counts for the Lesson-7 FLA pipeline and FlashInfer.

    Require tokens % chunk == 0. Return a dict with keys:
      num_chunks, gate_kkt, solve, wu, state, output, flashinfer_tiles
    """
    # TODO: distinguish grids that contain num_chunks from the state grid.
    raise NotImplementedError
