# Lesson 6 Runtime: Dense Recurrent GDN Baselines

Phase-1 teaching contract for B300 / SM103:

- `q/k/v`: BF16 `[B,T,H,128]`
- `q`: already L2-normalized and scaled by `1/sqrt(128)`
- `k`: already L2-normalized
- `alpha/beta`: activated FP32 `[B,T,H]`
- state: FP32 V-first `[B,H,128,128]`
- output: BF16 `[B,T,H,128]`
- final state: FP32 `[B,H,128,128]`
- dense only; no varlen, state pool, GVA, or fused preprocessing

Files:

- `reference.py`: FP32 PyTorch recurrent oracle
- `triton_baseline.py`: one program per `(batch, head, V-tile)`
- `cutedsl_baseline.py`: one warp per 16 V rows, register-resident state tile
- `test_correctness.py`: zero/random state and boundary token-count cases
- `results.json`: generated on B300

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 test_correctness.py --implementation both --output results.json
```

This is a correctness baseline, not a performance claim. Compilation time is included only to make hangs visible and must not be interpreted as kernel latency.

B300 result: **8/8 PASS**. See `baseline_report.md` for the design and error matrix and `results.json` for raw output.
