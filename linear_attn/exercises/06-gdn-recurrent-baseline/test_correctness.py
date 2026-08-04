from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import torch

from cutedsl_baseline import recurrent_gdn_cutedsl
from reference import error_metrics, make_inputs, recurrent_gdn_reference
from triton_baseline import recurrent_gdn_triton

CASES = [
    (1, 1, 1, False),
    (1, 4, 2, True),
    (2, 17, 4, False),
    (1, 64, 8, True),
]


class CaseTimeout(TimeoutError):
    pass


def _timeout_handler(_signum, _frame):
    raise CaseTimeout("case exceeded timeout")


def run_case(name, implementation, case, seed, timeout_sec):
    batch, tokens, heads, random_state = case
    inputs = make_inputs(
        batch,
        tokens,
        heads,
        seed=seed,
        random_initial_state=random_state,
    )
    expected_out, expected_state = recurrent_gdn_reference(**inputs)

    signal.alarm(timeout_sec)
    start = time.perf_counter()
    try:
        actual_out, actual_state = implementation(**inputs)
        torch.cuda.synchronize()
    finally:
        signal.alarm(0)
    elapsed = time.perf_counter() - start

    out_abs, out_rel = error_metrics(actual_out, expected_out)
    state_abs, state_rel = error_metrics(actual_state, expected_state)
    passed = out_rel < 0.02 and state_rel < 0.01
    return {
        "implementation": name,
        "shape": [batch, tokens, heads, 128],
        "random_initial_state": random_state,
        "output_max_abs": out_abs,
        "output_rel_l2": out_rel,
        "state_max_abs": state_abs,
        "state_rel_l2": state_rel,
        "elapsed_s_including_compile": elapsed,
        "status": "PASS" if passed else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=["triton", "cutedsl", "both"], default="both")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    signal.signal(signal.SIGALRM, _timeout_handler)
    selected = []
    if args.implementation in ("triton", "both"):
        selected.append(("triton", recurrent_gdn_triton))
    if args.implementation in ("cutedsl", "both"):
        selected.append(("cutedsl", recurrent_gdn_cutedsl))

    results = []
    for impl_name, impl in selected:
        for index, case in enumerate(CASES):
            try:
                result = run_case(impl_name, impl, case, seed=20260803 + index, timeout_sec=args.timeout)
            except Exception as error:  # keep the complete matrix visible
                result = {
                    "implementation": impl_name,
                    "shape": [case[0], case[1], case[2], 128],
                    "random_initial_state": case[3],
                    "status": "ERROR",
                    "error": f"{type(error).__name__}: {error}",
                }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "results": results,
    }
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    failures = [item for item in results if item["status"] != "PASS"]
    print(f"SUMMARY: {len(results) - len(failures)}/{len(results)} PASS", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
