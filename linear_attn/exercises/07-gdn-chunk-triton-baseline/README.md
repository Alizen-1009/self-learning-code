# Lesson 7 Runtime: C=64 Chunk GDN Triton Baseline

Uses the Lesson 6 dense preprocessed contract and implements an exact FLA-style multi-kernel chunk pipeline:

1. chunk-parallel gate cumsum + weighted KKT;
2. chunk-parallel unit-lower triangular inverse;
3. chunk-parallel W/U;
4. chunk-serial state scan producing H/R;
5. chunk-parallel output.

Scope:

- BF16 q/k/v `[B,T,H,128]`
- activated FP32 alpha/beta `[B,T,H]`
- FP32 V-first state `[B,H,128,128]`
- chunk size exactly 64
- T must be a multiple of 64
- dense MHA only; no tail, varlen, GVA, state pool, or fused preprocessing

Run on an idle B300 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python3 test_correctness.py --output results.json
```

The PyTorch chunk reference is first checked against the Lesson 6 recurrent oracle; the Triton pipeline is then checked against the same recurrent oracle.

B300 result: **4/4 PASS** with BF16 relative-L2 `< 0.01`. See `baseline_report.md` and raw `results.json`.
