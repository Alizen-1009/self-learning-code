from __future__ import annotations

import copy
import math

from challenge import recurrent_gdn_step, schedule_counts


def close_list(actual, expected, atol=1e-9):
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if isinstance(e, list):
            if not close_list(a, e, atol):
                return False
        elif not math.isclose(a, e, abs_tol=atol, rel_tol=atol):
            return False
    return True


def oracle_step(state, q, k, v, alpha, beta):
    s = [[alpha * x for x in row] for row in state]
    prediction = [sum(s[vv][kk] * k[kk] for kk in range(len(k))) for vv in range(len(v))]
    residual = [beta * (v[vv] - prediction[vv]) for vv in range(len(v))]
    for vv in range(len(v)):
        for kk in range(len(k)):
            s[vv][kk] += residual[vv] * k[kk]
    output = [sum(s[vv][kk] * q[kk] for kk in range(len(q))) for vv in range(len(v))]
    return output, s


def main():
    score = 0
    state = [[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]]
    q, k, v = [0.2, 0.5, -0.3], [0.6, -0.2, 0.1], [0.7, -0.4]
    original = copy.deepcopy(state)
    expected_o, expected_s = oracle_step(state, q, k, v, 0.9, 0.4)
    try:
        actual_o, actual_s = recurrent_gdn_step(state, q, k, v, 0.9, 0.4)
        if close_list(actual_o, expected_o) and close_list(actual_s, expected_s):
            print("[PASS] recurrent semantics (3 points)")
            score += 3
        else:
            print("[FAIL] recurrent semantics")
        if state == original:
            print("[PASS] input state remains unchanged (1 point)")
            score += 1
        else:
            print("[FAIL] input state was mutated")
    except Exception as error:
        print(f"[FAIL] recurrent task: {type(error).__name__}: {error}")

    try:
        actual = schedule_counts(2, 256, 8)
        expected = {
            "num_chunks": 4,
            "gate_kkt": 64,
            "solve": 64,
            "wu": 64,
            "state": 128,
            "output": 512,
            "flashinfer_tiles": 16,
        }
        if actual == expected:
            print("[PASS] scheduling counts (2 points)")
            score += 2
        else:
            print(f"[FAIL] scheduling counts: expected {expected}, got {actual}")
    except Exception as error:
        print(f"[FAIL] scheduling task: {type(error).__name__}: {error}")

    print(f"AUTO SCORE: {score}/6")
    print("Add the four written questions from README.md for a total score out of 10.")


if __name__ == "__main__":
    main()
