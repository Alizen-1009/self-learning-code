# 02｜代码阅读顺序

## 总原则

每读一个实现，都回答五个问题：

1. 输入契约是什么（shape、dtype、raw/activated gate）？
2. state 使用 `[K,V]` 还是 `[V,K]`？
3. token 维、head 维、K/V 维分别映射到 grid/CTA/warp/lane 的哪里？
4. 哪些中间量在 register、shared memory、global memory？
5. 它针对 training、prefill、decode 还是 speculative decode？

## 第一站：FLA——建立正确语义

### 1. 层级接线

先看模型如何生成 q/k/v/g/beta，再看算子：

- GDN：`code/fla/fla/layers/gated_deltanet.py`
- KDA：`code/fla/fla/layers/kda.py`

重点比较：

- GDN 的 `a_proj` 输出 `HV` 个标量；
- KDA 的 `f_proj` 输出 `HV×K` 个 gate；
- 长序列走 `chunk`，短序列 inference 走 `fused_recurrent`；
- q/k L2 norm、gate activation、beta sigmoid 可以融合进算子。

### 2. Naive reference

- `code/fla/fla/ops/gated_delta_rule/naive.py`
- `code/fla/fla/ops/kda/naive.py`

先只读 `naive_recurrent_*`，逐行对应递推公式；再读 `naive_chunk_*`，观察：

- chunk cumsum；
- 下三角 mask；
- `A/L` 矩阵；
- `w/u`（WY 表示）；
- chunk 间 state 更新。

### 3. Recurrent Triton

- GDN：`code/fla/fla/ops/gated_delta_rule/fused_recurrent.py`
- KDA：`code/fla/fla/ops/kda/fused_recurrent.py`

建议先追踪 KDA kernel 内这一条链：

```text
load q/k/v/g/beta
→ optional norm / gate / sigmoid
→ state *= exp(g)
→ residual = v - state·k
→ state += beta·outer(residual,k)
→ output = state·q
```

然后看 grid 如何按 `N × HV × V-tile × K-tile` 分割状态，以及 `state_v_first` 如何改变地址计算。

### 4. Chunk Triton（训练主线）

GDN 入口：

- `code/fla/fla/ops/gated_delta_rule/chunk.py`
- `code/fla/fla/ops/gated_delta_rule/chunk_fwd.py`
- `code/fla/fla/ops/gated_delta_rule/wy_fast.py`

KDA 入口：

- `code/fla/fla/ops/kda/chunk.py`
- `code/fla/fla/ops/kda/chunk_fwd.py`
- `code/fla/fla/ops/kda/chunk_intra.py`
- `code/fla/fla/ops/kda/wy_fast.py`
- `code/fla/fla/ops/kda/chunk_bwd.py`

KDA 还应读：`code/fla/.agents/skills/fla-kda/SKILL.md`。这里明确记录了 safe/non-safe gate 的指数偏移策略、两个 intra path 和 correctness 维度。

阅读时画出 forward 数据流：

```text
gate activation + local cumsum
→ intra-chunk Akk/Aqk
→ triangular solve / WY (w,u)
→ inter-chunk state scan
→ output
```

最后再读 backward；不要一开始同时追 forward/backward。

## 第二站：FlashKDA——理解专用 prefill 优化

推荐顺序：

1. `code/flashkda/docs/20260420-flashkda-v1-deep-dive.md`
2. `code/flashkda/tests/torch_ref.py`
3. `code/flashkda/flash_kda/__init__.py`
4. `code/flashkda/csrc/flash_kda.cpp`
5. `code/flashkda/csrc/smxx/fwd_launch.cu`
6. `code/flashkda/csrc/smxx/fwd_kernel1.cuh`
7. `code/flashkda/csrc/smxx/fwd_kernel2.cuh`
8. `code/flashkda/csrc/smxx/utils.cuh`
9. `code/flashkda/benchmarks/bench_fwd.py`

### K1 要看什么

K1 按 token/chunk 并行，负责：

- raw gate activation；
- q/k L2 normalization；
- decay 与 `L/Mqk` 构造；
- 16×16 inverse。

重点找 shared-memory lifetime union、`__launch_bounds__`、base-2 exponent、fp16 inverse。

### K2 要看什么

K2 按 sequence/head 运行，负责：

- chunk 间递推；
- output projection；
- state 累积与写回。

重点看 bf16 on-chip state、fp32 FMA、寄存器内转置 `MOVM_T`，以及为什么 K2 的并行度低于 K1。

### 当前快照的适用边界

FlashKDA backend 是 inference-only，要求 SM90+、CUDA 12.9+、bf16，当前固定 `K=V=128`，不支持 GVA。准确约束以：

- `code/flashkda/README.md`
- `code/fla/fla/ops/kda/backends/flash_kda.py`

为准。

## 第三站：cuLA——对照 Hopper 与 Blackwell

cuLA 同时包含 CuTe DSL 和 CUTLASS C++ 实现，重点对照 SM90 K1/K2、SM90 fully-fused 与 SM100 modular KDA。详细顺序见 [cuLA 代码阅读指南](04-cula-reading-map.md)。当前快照的独立 GDN public backend 尚在 roadmap 中，因此先把它作为 KDA/CuTe/CUTLASS 学习材料。

## 第四站：FlashInfer——理解 serving 约束

### GDN decode

入口：`code/flashinfer/flashinfer/gdn_decode.py`

内核：

- `gdn_kernels/gdn_decode_nontranspose.py`：K-major；
- `gdn_kernels/gdn_decode_pretranspose.py`：V-major；
- `gdn_kernels/gdn_decode_bf16_state.py`：bf16 state fast path；
- `gdn_kernels/gdn_decode_mtp.py`：multi-token/speculative path。

对照 benchmark reference：

- `code/flashinfer/benchmarks/gdn_triton_reference.py`
- `code/flashinfer/benchmarks/bench_gdn_decode.py`

重点不是只看单 token 公式，而是看 state pool、indices、原地更新、padding slot 和 speculative checkpoints。

### GDN prefill

入口：`code/flashinfer/flashinfer/gdn_prefill.py`

内核分两组：

- `gdn_kernels/blackwell/`：Blackwell chunk kernel；
- `gdn_kernels/delta_rule_dsl/`：SM90/SM120 CuTe DSL 与 context-parallel 路径。

重点看 architecture dispatch、varlen、state checkpoint、state pool indexed I/O，以及为何不同架构有不同 dtype/shape 限制。

### KDA decode

入口：`code/flashinfer/flashinfer/kda_decode.py`

实现：`code/flashinfer/flashinfer/kda_kernels/recurrent_kda.py`

文件开头已经给出 V-first 递推。继续比较两类 kernel：

- one-warp register-tile：足够多 sequence-head 时降低延迟；
- grouped-CTA：小 grid 或 multi-token 时复用 token preprocessing。

这是很好的 dispatch 学习案例：优化不是寻找一个“全局最快 kernel”，而是按并行度和工作负载选不同 kernel。

## 横向对照表

| 维度 | FLA | FlashKDA | cuLA | FlashInfer |
|---|---|---|---|---|
| 最佳入口 | naive + layer | deep dive + torch_ref | REPO_LAYOUT + KDA wrappers | public API + tests |
| 训练 backward | 完整 chunk 路径 | 不支持 | 部分 modular 路径 | 主要面向 inference |
| 长序列 prefill | Triton chunk | CUTLASS 两阶段 | SM90/SM100 CuTe + CUTLASS | 架构专用 CuTe/CUDA |
| decode | Triton recurrent | 非主线 | KDA/Lightning decode | GDN/KDA serving 主线 |
| varlen | 支持 | 支持 | 支持 | 支持并结合状态池 |
| 关键学习点 | 算法和通用性 | 极致融合与数值设计 | 架构对照、CP、模块化 | 调度、缓存、spec decode |

## 阅读产出模板

每读完一个 kernel，写一页笔记：

```text
文件 / 函数：
目标 workload：
输入 shape/dtype：
状态布局：
grid / block / warp 映射：
寄存器与 shared memory 占用对象：
global memory 读写：
融合了哪些前后处理：
数值近似：
预期瓶颈：
如何验证：
下一项可测优化：
```
