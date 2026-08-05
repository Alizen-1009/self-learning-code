# C=64 Chunk GDN Triton Baseline Report

## Scope

Correctness-first Lesson 7 implementation using the Lesson 6 dense preprocessed contract. The pipeline follows the FLA schedule and fixes chunk size at 64.

Stages:

1. chunk-parallel gate cumsum + weighted KKT;
2. chunk-parallel unit-lower triangular inverse;
3. chunk-parallel W/U;
4. chunk-serial state scan producing H/R;
5. chunk-parallel output.

Not included: tail chunks, varlen, GVA, state pool, preprocessing fusion, backward, CuTe chunk kernel, or performance claims.

## B300 environment

- CUDA Driver capability: 10.3 / SM103
- CUDA 13.2
- PyTorch 2.11.0a0
- Triton 3.6.0
- Test GPU: index 0, checked idle before final run

## Inputs

- q/k/v: BF16 `[B,T,H,128]`
- q/k pre-normalized; q pre-scaled
- activated alpha/beta: FP32 `[B,T,H]`
- state: FP32 V-first `[B,H,128,128]`
- `T % 64 == 0`

## Triton intermediates

| Tensor | Dtype | Shape | Producer | Consumer |
|---|---|---|---|---|
| G | FP32 | `[B,T,H]` | gate/KKT | W/U, state, output |
| left | FP32 | `[B,NT,H,64,64]` | gate/KKT | solve |
| Ainv | BF16 | `[B,NT,H,64,64]` | solve | W/U |
| W | BF16 | `[B,T,H,128]` | W/U | state scan |
| U | BF16 | `[B,T,H,128]` | W/U | state scan |
| H | FP32 | `[B,NT,H,128,128]` | state scan | output |
| R | BF16 | `[B,T,H,128]` | state scan | output |

## Parallel/serial structure

```text
gate/KKT  grid = B*H*NT               chunk parallel
solve     grid = B*H*NT               chunk parallel
W/U       grid = B*H*NT               chunk parallel
state     grid = B*H*Vtiles           for chunk: serial
output    grid = B*H*NT*Vtiles        chunk parallel
```

## Correctness

| Shape `[B,T,H,D]` | State | PyTorch chunk output rel-L2 | Triton output rel-L2 | Triton state rel-L2 | Status |
|---:|---|---:|---:|---:|---|
| `[1,64,1,128]` | zero | 1.275e-3 | 3.801e-3 | 3.224e-3 | PASS |
| `[1,64,4,128]` | random | 1.165e-3 | 4.073e-3 | 3.006e-3 | PASS |
| `[1,128,2,128]` | zero | 1.212e-3 | 4.079e-3 | 3.055e-3 | PASS |
| `[2,128,4,128]` | random | 1.218e-3 | 4.219e-3 | 3.062e-3 | PASS |

Result: **4/4 PASS** using BF16 relative-L2 threshold `< 0.01`.

Maximum observed:

- PyTorch chunk vs recurrent output rel-L2: `1.275e-3`
- Triton chunk vs recurrent output rel-L2: `4.219e-3`
- Triton chunk vs recurrent final-state rel-L2: `3.224e-3`
- Triton output max absolute error: `4.883e-4`
- Triton final-state max absolute error: `4.687e-3`

Chunking is algebraically exact but changes floating-point reduction order. Even the FP32 PyTorch chunk formulation differs from the token recurrence after BF16 q/k/v inputs, so bitwise equality is not a valid acceptance criterion.

## Issues found during B300 practice

1. Triton 3.6 rejected captured plain Python globals `C/D`; fixed by instantiating them as `tl.constexpr` globals and keeping separate host integers.
2. Triton rejected tensor negative indexing `G[-1]`; fixed by extracting the final lane with a masked reduction.
3. Initial `1e-5` threshold confused algebraic exactness with floating-point equality; corrected to the documented BF16 baseline threshold `1e-2`, with raw errors retained.

## Outcome

The C=64 multi-kernel Triton pipeline is numerically consistent with the Lesson 6 recurrent oracle on B300 SM103. It is intentionally materialization-heavy and provides the starting point for studying FLA memory traffic and later fusion—not a performance-optimized kernel.
