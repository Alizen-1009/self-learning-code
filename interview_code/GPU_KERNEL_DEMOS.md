# GPU Kernel Demo Map

This folder contains tiny starting points for common NVIDIA kernel stacks.

## What Each Tool Is

- cuBLAS: prebuilt NVIDIA BLAS kernels. You call GEMM directly.
- cuDNN: prebuilt NVIDIA deep learning kernels such as convolution, normalization, and attention-related building blocks.
- CUTLASS: C++ templates for building and customizing GEMM-like kernels.
- CuTe: the C++ layout/tensor/tile abstraction used inside modern CUTLASS.
- cuTile Python / CUDA Tile: NVIDIA's Python tile programming model built around CUDA Tile IR.
- Triton: Python JIT language/compiler for custom GPU kernels, common in PyTorch workflows.
- TileLang: Python DSL on top of TVM-style TIR with tile-level primitives such as `T.Kernel`, `T.copy`, and `T.gemm`.

## Files

- `cuda/mha_cublas.cu`: cuBLAS MHA-style dataflow demo.
- `cuda/prebuilt_libraries_demo.cu`: minimal cuBLAS GEMM plus cuDNN Conv2D demo.
- `cuda/cutlass_gemm_demo.cu`: CUTLASS row-major float GEMM.
- `cuda/cute_layout_demo.cu`: CuTe layout/indexing demo.
- `python/triton_vector_add_demo.py`: Triton vector add.
- `python/tilelang_vector_add_demo.py`: TileLang vector add.
- `python/cutile_vector_add_demo.py`: cuTile Python vector add.

## Example Commands

### Calling Prebuilt Libraries

`cuda/prebuilt_libraries_demo.cu` shows the "call an existing NVIDIA library" path:

- cuBLAS part: computes row-major `C = A * B` through `cublasSgemm`.
- cuDNN part: computes a tiny NCHW Conv2D through `cudnnConvolutionForward`.

```bash
nvcc -std=c++17 cuda/prebuilt_libraries_demo.cu -lcublas -lcudnn -o /tmp/prebuilt_libraries_demo
/tmp/prebuilt_libraries_demo
```

Expected output:

```text
cuBLAS GEMM C = A * B:
12 5 10
28 13 30
cuDNN Conv2D output:
-4 -4
-4 -4
```

CUTLASS and CuTe need a local CUTLASS checkout:

```bash
git clone https://github.com/NVIDIA/cutlass.git ~/cutlass
nvcc -std=c++17 cuda/cutlass_gemm_demo.cu -I$HOME/cutlass/include -I$HOME/cutlass/tools/util/include -o /tmp/cutlass_gemm_demo
/tmp/cutlass_gemm_demo

nvcc -std=c++17 cuda/cute_layout_demo.cu -I$HOME/cutlass/include -o /tmp/cute_layout_demo
/tmp/cute_layout_demo
```

Python demos:

```bash
pip install torch triton
python python/triton_vector_add_demo.py

pip install torch tilelang
python python/tilelang_vector_add_demo.py

pip install "cuda-tile[tileiras]" cupy-cuda13x numpy
python python/cutile_vector_add_demo.py
```
