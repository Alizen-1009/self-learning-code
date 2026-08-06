# Readable Persistent CuTe DSL GDN Report

## Purpose

Demonstrate the FlashInfer-style scheduling decision in a compact runnable kernel:

```text
grid = B × H
one CTA owns one sequence/head
full [128,128] state remains on-chip
CTA loops chunk → token serially
```

This is recurrent math nested inside visible 64-token chunk loops. It intentionally does **not** implement the production 7-GEMM chunk algorithm, TMEM, TMA, Ainv, W/U, or warp specialization.

## CTA mapping

- CTA: 8 warps / 256 threads
- warp `w`: V rows `[16w, 16w+16)`
- each warp: logical `2 × 16` lane grid
- each thread: 8 V rows × 8 K values = 64 FP32 state values
- all 8 warps together own the complete FP32 `[128,128]` state
- q/k are redundantly loaded by every warp

## B300 environment

- CUDA Driver capability: 10.3 / SM103
- CUDA 13.2
- PyTorch 2.11.0a0
- Test GPU: index 0, checked idle before launch

## Correctness

| Shape `[B,T,H,D]` | State | output rel-L2 | state rel-L2 | Status |
|---:|---|---:|---:|---|
| `[1,64,1,128]` | zero | 4.075e-9 | 8.074e-8 | PASS |
| `[1,64,4,128]` | random | 1.402e-7 | 7.834e-8 | PASS |
| `[1,128,2,128]` | zero | 4.165e-5 | 7.521e-8 | PASS |
| `[2,128,4,128]` | random | 1.573e-5 | 8.076e-8 | PASS |

Result: **4/4 PASS**.

Maximum observed:

- output rel-L2: `4.165e-5`
- output max absolute: `6.104e-5`
- final-state rel-L2: `8.076e-8`
- final-state max absolute: `1.192e-7`

## What this proves

- A single CTA can own all V rows for one batch/head while preserving the Lesson 6 recurrence.
- Eight warps can keep disjoint state row tiles in registers without cross-warp state communication.
- Chunk progression can be made explicit inside the CTA without adding a chunk grid dimension.

## What this does not prove

- It is not chunk-parallel GDN: token recurrence remains serial.
- It does not use TMEM or UMMA and is not a FlashInfer performance substitute.
- It does not reduce repeated q/k loads across warps.
- No latency or utilization claim is made.

## Production comparison

FlashInfer SM103 replaces the simple recurrent body with KK/QK/Ainv/KS/QS/NV/dS UMMA stages and keeps state in TMEM. FlashInfer SM90 uses a 512-thread, four-warpgroup fused kernel with state in WGMMA register fragments. This teaching kernel isolates only the shared scheduling principle: one sequence/head CTA advances chunks serially while retaining state on-chip.
