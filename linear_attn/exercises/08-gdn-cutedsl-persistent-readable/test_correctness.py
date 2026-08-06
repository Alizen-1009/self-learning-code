from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from persistent_cutedsl import persistent_recurrent_gdn
from reference import make_inputs, metrics, recurrent_reference

CASES = [
    (1, 64, 1, False),
    (1, 64, 4, True),
    (1, 128, 2, False),
    (2, 128, 4, True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    results = []
    for idx, (B, T, H, random_state) in enumerate(CASES):
        inputs = make_inputs(B, T, H, seed=20260820 + idx, random_initial_state=random_state)
        expected_o, expected_s = recurrent_reference(**inputs)
        actual_o, actual_s = persistent_recurrent_gdn(**inputs)
        torch.cuda.synchronize()
        o_abs, o_rel = metrics(actual_o, expected_o)
        s_abs, s_rel = metrics(actual_s, expected_s)
        passed = o_rel < 0.01 and s_rel < 0.01
        result = {
            "shape": [B, T, H, 128],
            "random_initial_state": random_state,
            "output_max_abs": o_abs,
            "output_rel_l2": o_rel,
            "state_max_abs": s_abs,
            "state_rel_l2": s_rel,
            "status": "PASS" if passed else "FAIL",
        }
        print(json.dumps(result), flush=True)
        results.append(result)
    summary = {"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)), "schedule": "one CTA per batch-head, eight warps per CTA", "results": results}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    failures = [r for r in results if r["status"] != "PASS"]
    print(f"SUMMARY: {len(results)-len(failures)}/{len(results)} PASS")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
