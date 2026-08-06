# Readable FlashInfer-style Persistent CuTe DSL GDN

Teaching kernel for B300/SM103:

- grid: one CTA per `(batch, head)`;
- CTA: 8 warps, each warp owns 16 V rows;
- full `[128,128]` FP32 state is distributed across warp registers;
- CTA loops `chunk -> token` and keeps state on-chip;
- q/k are redundantly loaded by each warp;
- no KK/QK/Ainv/WU/UMMA/TMEM: this demonstrates scheduling and state residency, not production chunk acceleration.

Contract matches Lesson 6: preprocessed BF16 q/k/v, activated FP32 alpha/beta, FP32 V-first state, `D=128`, and `T` a multiple of 64.

Run on an idle B300 GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python3 test_correctness.py --output results.json
```

B300 result: **4/4 PASS**. See `baseline_report.md` and raw `results.json`.
