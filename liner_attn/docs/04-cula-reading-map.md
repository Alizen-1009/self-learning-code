# 04｜cuLA 代码阅读指南

上游：[`inclusionAI/cuLA`](https://github.com/inclusionAI/cuLA)，当前快照记录在
`code/cula/UPSTREAM_REVISION.md`。

cuLA 是面向 Hopper/Blackwell 的 Linear Attention kernel 仓库，使用 **CuTe DSL + CUTLASS C++ + 少量 Triton 支撑 kernel**。它与 FLA 接口接近，但仍处于 early stage。

> 当前快照的成熟主线是 KDA 与 Lightning Attention。README roadmap 中“Modular GDN Forward / Backward”仍未完成，因此不要把共享的 `chunk_delta_h` 文件误认为完整的 cuLA GDN public backend。

## 1. 先看总体结构

依次阅读：

1. `code/cula/README.md`
2. `code/cula/REPO_LAYOUT.md`
3. `code/cula/USAGE.md`
4. `code/cula/cula/kda/README.md`
5. `code/cula/RECOMMENDED_CODING_STYLE.md`

先确认两条实现主线：

```text
Hopper SM90
├── CuTe DSL FlashKDA K1 + K2
└── CUTLASS C++ fully-fused prefill

Blackwell SM100/SM103
├── CuTe DSL modular state/output/backward kernels
└── CUTLASS C++ intra-chunk / WY recompute kernels
```

## 2. Public API 与 backend dispatch

阅读顺序：

```text
cula/kda/__init__.py
→ cula/backends.py
→ cula/kda/backends/__init__.py
→ cula/kda/backends/flashkda.py
→ cula/kda/backends/fully_fused.py
→ cula/kda/auto_route.py
```

需要回答：

- public API 怎样根据 GPU 架构和输入契约选择 backend？
- verifier 为什么要检查 dtype、head dimension、safe gate、state layout？
- “backend 不可用”和“输入不满足 specialization”如何区分？
- 为什么同一数学算子需要多个 kernel？

## 3. Hopper SM90：CuTe DSL K1/K2

主入口：

```text
code/cula/cula/kda/flashkda.py
code/cula/cula/ops/kda/sm90/fwd.py
code/cula/cula/ops/kda/sm90/k1.py
code/cula/cula/ops/kda/sm90/k2.py
code/cula/cula/ops/kda/sm90/_common.py
```

这条路径最适合与 `code/flashkda/csrc/smxx/` 横向比较。

### K1

观察：

- gate activation、q/k norm 与 beta sigmoid 在哪里完成；
- token/chunk 如何映射到 grid；
- `L`、`Mqk` 与 inverse 的 layout；
- shared-memory pipeline 与 MMA fragment；
- 为什么 chunk 内工作具有高并行度。

### K2

观察：

- sequence/head 级 chunk recurrence；
- state 的 dtype 与布局；
- K1 输出如何被 K2 消费；
- output/final state 写回；
- 为什么 K2 比 K1 更容易并行度不足。

Context Parallel 继续阅读：

```text
code/cula/cula/ops/kda/sm90/cp/plan.py
code/cula/cula/ops/kda/sm90/cp/pre_scan.py
code/cula/cula/ops/kda/sm90/cp/merge.py
code/cula/cula/ops/kda/sm90/cp/driver.py
```

## 4. Hopper SM90：CUTLASS C++ fully-fused

Python wrapper：

```text
code/cula/cula/kda/hopper_fused_fwd.py
code/cula/cula/kda/hopper_fused_fwd_opt.py
```

CUDA/CUTLASS：

```text
code/cula/csrc/api/kda_sm90.cu
code/cula/csrc/kda/sm90/prefill_kernel.hpp
code/cula/csrc/kda/sm90/prefill_kernel_kda_fwd_sm90.cuh
code/cula/csrc/kda/sm90/collective/mainloop_kda_fwd.hpp
code/cula/csrc/kda/sm90/kernel/kernel_kda_fwd.hpp
code/cula/csrc/kda/sm90/kernel/tile_scheduler.hpp
```

这条路径用于学习经典 CUTLASS 分层：

```text
host API
→ device adapter
→ kernel
→ collective/mainloop
→ TMA load/store
→ scheduler
```

对比 CuTe DSL K1/K2，记录 fully-fused 的收益与代价：launch 更少，但不同阶段的并行度和资源需求被绑定在同一个 kernel 中。

## 5. Blackwell SM100：模块化 KDA

Python orchestration：

```text
code/cula/cula/kda/chunk.py
code/cula/cula/kda/chunk_fwd.py
code/cula/cula/kda/chunk_intra.py
code/cula/cula/kda/wy_recompute.py
code/cula/cula/kda/chunk_bwd.py
```

CuTe DSL：

```text
code/cula/cula/ops/kda/sm100/delta_h.py
code/cula/cula/ops/kda/sm100/fwd_o.py
code/cula/cula/ops/kda/sm100/bwd_wy_dqkg.py
code/cula/cula/ops/kda/sm100/policy.py
```

CUTLASS C++：

```text
code/cula/csrc/kda/sm100/kda_fwd_intra_kernel_sm100.hpp
code/cula/csrc/kda/sm100/kda_fwd_intra_mainloop_sm100.hpp
code/cula/csrc/kda/sm100/kda_fwd_recomp_w_u_kernel_sm100.hpp
code/cula/csrc/kda/sm100/tile_scheduler.hpp
```

重点理解：

- 为什么 intra、state recurrence、output 和 backward 被拆成模块；
- SM100 UMMA 与 SM90 WGMMA 的代码组织差异；
- `policy.py` 如何按 workload 决定 intracard CP；
- 哪些 backward 仍复用 Triton/FLA，哪些已经迁移到 CuTe/C++。

## 6. KDA decode

入口：

```text
code/cula/cula/ops/kda/decode/cute.py
code/cula/cula/ops/kda/decode/mtp.py
code/cula/cula/ops/kda/decode/mtp_kvbuffer.py
code/cula/cula/ops/kda/decode/reference_fla.py
```

测试和 benchmark：

```text
code/cula/tests/test_kda_decode.py
code/cula/tests/test_kda_packed_decode.py
code/cula/tests/test_kda_decode_mtp.py
code/cula/benchmarks/bench_kda_decode.py
code/cula/benchmarks/bench_kda_packed_decode.py
code/cula/benchmarks/bench_kda_decode_mtp.py
```

将它与 FlashInfer `recurrent_kda.py` 对照，重点比较 state pool、packed input、MTP 和低延迟 tile 策略。

## 7. Lightning Attention（扩展阅读）

如果当前目标仅是 GDN/KDA，可以后读。它仍然很适合学习 CuTe DSL pipeline：

```text
code/cula/docs/lightning_attn_sm90.md
code/cula/cula/ops/lightning/prefill.py
code/cula/cula/ops/lightning/prefill_sm90.py
code/cula/cula/ops/lightning/sm90/schedule.py
code/cula/cula/ops/lightning/sm90/prefill_kernel.py
```

这里可以观察 persistent packed scheduling、warp-group 分工、WGMMA 数据流和 recurrent state placement。

## 8. 第一轮建议只读这些文件

如果一次不想读 189 个文件，先读以下 12 个：

```text
README.md
REPO_LAYOUT.md
cula/kda/README.md
cula/backends.py
cula/kda/flashkda.py
cula/ops/kda/sm90/fwd.py
cula/ops/kda/sm90/k1.py
cula/ops/kda/sm90/k2.py
cula/kda/chunk.py
cula/ops/kda/sm100/delta_h.py
cula/ops/kda/sm100/fwd_o.py
csrc/kda/sm90/collective/mainloop_kda_fwd.hpp
```

读完后应能画出 cuLA 的 backend dispatch 图，以及 SM90 K1/K2 和 SM100 modular 两条数据流。
