# CuTe DSL kernels

This directory contains kernel definitions only. Host wrappers, launchers, and
PyTorch allocation code are omitted.

| File | Kernel(s) |
|---|---|
| `transpose.py` | `_transpose_kernel` |
| `prefix_sum.py` | `_prefix_kernel` |
| `online_softmax_1d.py` | `_stats_kernel`, `_apply_kernel` |
| `softmax_attention.py` | `_softmax_kernel`, `_attention_score_kernel` |
| `reduce_max.py` | `_reduce_max_kernel` |
| `sgemm.py` | `_matmul_kernel` |
| `sgemv.py` | `_sgemv_kernel` |
| `topk.py` | `_topk_kernel` |
| `dsu.py` | `_init_kernel`, `_union_kernel`, `_compress_kernel` |
| `rmsnorm.py` | `_rms_stats_kernel`, `_rms_apply_kernel` |

`dsu.py` keeps `_find_root` as a CuTe JIT helper used inside its kernels.
`online_softmax_1d.py` and `rmsnorm.py` require the stats kernel to run before
the output kernel on the same stream. These are readable learning kernels;
several use serial reductions or per-output loops and are not tuned for speed.
