from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from reference import chunk_gdn_reference, error_metrics, make_inputs, recurrent_gdn_reference
from triton_chunk_baseline import chunk_gdn_triton

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
        inputs = make_inputs(B, T, H, seed=20260810 + idx, random_initial_state=random_state)
        recurrent_out, recurrent_state = recurrent_gdn_reference(**inputs)
        chunk_ref_out, chunk_ref_state = chunk_gdn_reference(**inputs)
        triton_out, triton_state, _ = chunk_gdn_triton(**inputs)
        torch.cuda.synchronize()

        ref_o_abs, ref_o_rel = error_metrics(chunk_ref_out, recurrent_out)
        ref_s_abs, ref_s_rel = error_metrics(chunk_ref_state, recurrent_state)
        tri_o_abs, tri_o_rel = error_metrics(triton_out, recurrent_out)
        tri_s_abs, tri_s_rel = error_metrics(triton_state, recurrent_state)
        # Chunking is mathematically exact but changes BF16/FP32 reduction order.
        # Use the standard BF16 baseline threshold rather than bitwise equality.
        passed = ref_o_rel < 0.01 and ref_s_rel < 0.01 and tri_o_rel < 0.01 and tri_s_rel < 0.01
        result = {
            "shape": [B, T, H, 128],
            "random_initial_state": random_state,
            "pytorch_chunk_vs_recurrent": {"output_max_abs": ref_o_abs, "output_rel_l2": ref_o_rel, "state_max_abs": ref_s_abs, "state_rel_l2": ref_s_rel},
            "triton_chunk_vs_recurrent": {"output_max_abs": tri_o_abs, "output_rel_l2": tri_o_rel, "state_max_abs": tri_s_abs, "state_rel_l2": tri_s_rel},
            "status": "PASS" if passed else "FAIL",
        }
        results.append(result)
        print(json.dumps(result), flush=True)

    summary = {"device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)), "chunk_size": 64, "results": results}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    failed = [r for r in results if r["status"] != "PASS"]
    print(f"SUMMARY: {len(results)-len(failed)}/{len(results)} PASS")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
