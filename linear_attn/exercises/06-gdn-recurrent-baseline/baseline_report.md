# Dense Recurrent GDN Baseline Report

## Scope

Correctness-only Phase 1 for Lesson 6. Two independent GPU implementations share the same preprocessed-input contract and are compared against an FP32 PyTorch recurrent oracle.

Not included: varlen, GVA, state pools, raw gate activation, beta sigmoid, Q/K normalization, chunking, backward, or performance targets.

## B300 environment

- Pod alias: B300 ASI Pod, container `worker0`
- Runtime device name: `NVIDIA L20D` (masked/internal presentation)
- CUDA Driver / PyTorch compute capability: `10.3` (SM103)
- CUDA: `13.2`
- PyTorch: `2.11.0a0+a6c236b9fd.nv26.03.46836102`
- Triton: `3.6.0`
- Test GPU: physical index 0, selected only after confirming it was idle

## Contract

- `q/k/v`: BF16 `[B,T,H,128]`
- `q`: pre-normalized and scaled by `1/sqrt(128)`
- `k`: pre-normalized
- `alpha/beta`: activated FP32 `[B,T,H]`
- input/final state: FP32 V-first `[B,H,128,128]`
- output: BF16 `[B,T,H,128]`

Per token:

```text
state *= alpha
prediction = state @ k
residual = beta * (v - prediction)
state += residual outer k
output = state @ q
```

## Implementations

### Triton

`triton_baseline.py`

- Grid: `(B*H, ceil(V/16))`
- One program owns a `[16,128]` V-by-K state tile
- State tile accumulates in FP32
- Token dimension is a serial `tl.range(0,T)` recurrence
- Parallelism is batch × head × V tile

### CuTe DSL

`cutedsl_baseline.py`

- Grid: `B*H*(128/16)` one-warp CTAs
- One warp owns 16 V rows
- 16 lane groups partition K into contiguous vec8 segments; two lane rows partition V
- State tile remains in FP32 registers across the token loop
- Q/K are redistributed by warp shuffle and dot products use butterfly reduction
- This is a correctness-first register-tile kernel, not the production TMEM chunk kernel

## Correctness matrix

| Framework | Shape `[B,T,H,D]` | Initial state | output rel-L2 | state rel-L2 | Status |
|---|---:|---|---:|---:|---|
| Triton | `[1,1,1,128]` | zero | 0 | 0 | PASS |
| Triton | `[1,4,2,128]` | random | 0 | 5.822e-8 | PASS |
| Triton | `[2,17,4,128]` | zero | 2.820e-5 | 7.691e-8 | PASS |
| Triton | `[1,64,8,128]` | random | 6.792e-6 | 7.898e-8 | PASS |
| CuTe DSL | `[1,1,1,128]` | zero | 0 | 0 | PASS |
| CuTe DSL | `[1,4,2,128]` | random | 3.029e-9 | 5.908e-8 | PASS |
| CuTe DSL | `[2,17,4,128]` | zero | 2.803e-5 | 7.662e-8 | PASS |
| CuTe DSL | `[1,64,8,128]` | random | 7.571e-6 | 7.921e-8 | PASS |

Result: **8/8 PASS**.

Maximum observed:

- output relative L2: `2.820e-5`
- output max absolute error: `3.052e-5`
- final-state relative L2: `7.921e-8`
- final-state max absolute error: `8.941e-8`

The reported elapsed times in `results.json` include JIT compilation and are not kernel latency measurements.

## Knowledge sources

- `<gpu-wiki>/docs/nvidia/common/ref-docs/qwen3.5-gdn-principle-code-analysis.md`
- `<gpu-wiki>/docs/nvidia/blackwell/kernel-opt/kernels/gated-delta-net.md`
- `<gpu-wiki>/docs/nvidia/blackwell/kernel-opt/languages/triton-blackwell.md`
- `<gpu-wiki>/docs/nvidia/common/ref-docs/cutedsl/cutlass-cute-fundamentals.md`
- FlashInfer `kda_kernels/recurrent_kda.py` register-tile schedule, adapted to scalar GDN gate
- Existing sync-repo Triton recurrent GDN experiment for contract comparison

## Outcome

The recurrent semantics are now independently locked by PyTorch, Triton, and CuTe DSL on SM103. These implementations provide the correctness oracle for the next stage: chunked GDN.
