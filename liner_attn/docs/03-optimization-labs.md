# 03｜优化实验路线

## 0. 先固定实验契约

不要把 prefill 和 decode 混在同一张性能表。至少建立两组 workload：

### Decode

- `T=1`；再增加 `T=2/4/8` 模拟 speculative decode；
- 扫 batch/sequence-head 数量；
- `K=V=128` 起步；
- bf16 state 与 fp32 state 分开记录；
- 分开记录 `[K,V]` / `[V,K]` layout。

### Prefill

- `T=128, 512, 2048, 8192`；
- fixed length 与 ragged varlen；
- 从 `H=8/16/64` 扫并行度；
- chunk size 与 gate mode 单独记录；
- initial/final state on/off 分开记录。

每条结果写清 GPU、CUDA、PyTorch、commit、shape、dtype、warmup、repetition 和统计量，不能只写“快了 20%”。

## 1. Correctness baseline

优先使用上游 reference 和测试：

精选目录不是独立可安装的软件包。先在工作区之外 clone 并按对应 README 安装，再运行目标测试：

```bash
# 以下路径仅为示例
mkdir -p /tmp/linear-attn-upstreams
cd /tmp/linear-attn-upstreams

git clone https://github.com/fla-org/flash-linear-attention.git
git clone --recurse-submodules https://github.com/MoonshotAI/FlashKDA.git
git clone --recurse-submodules https://github.com/inclusionAI/cuLA.git
git clone https://github.com/flashinfer-ai/flashinfer.git

pytest -q flash-linear-attention/tests/ops/test_gdn.py
pytest -q flash-linear-attention/tests/ops/test_kda.py
bash FlashKDA/tests/test.sh
pytest -q cuLA/tests/test_kda_sm90_prefill_vs_fla.py       # SM90 示例
pytest -q cuLA/tests/test_kda_sm100_chunk_vs_naive.py     # SM100 示例
pytest -q flashinfer/tests/gdn flashinfer/tests/kda
```

真实命令会受 GPU 架构和依赖版本影响；失败时先确认上游 README 的安装要求，不要立即修改 kernel。

正确性矩阵至少覆盖：

- dense / varlen；
- full chunk / tail chunk；
- zero / random initial state；
- output final state on/off；
- raw / activated gate；
- beta logits / sigmoid beta；
- safe gate 边界 `lower_bound=-5` 与接近 0；
- 极端正负 raw gate；
- state layout 与 dtype；
- 若改训练路径，forward 与 backward 都要覆盖。

## 2. Lab A：复现 recurrent 公式

目标：能把 FLA naive 的每一行写成数学式，而不是先追性能。

步骤：

1. 读 `code/fla/fla/ops/gated_delta_rule/naive.py` 的 recurrent 函数；
2. 读 `code/fla/fla/ops/kda/naive.py` 的 recurrent 函数；
3. 标出两者唯一的核心差异：scalar gate vs per-K gate；
4. 手算 `K=2,V=2,T=2` 的状态；
5. 分别用 `[K,V]` 与 `[V,K]` 写出等价公式。

验收：能解释 residual、rank-1 update、beta 和 decay 各自的作用。

## 3. Lab B：分析 decode 的字节流

目标：理解为什么 decode 经常是 state-memory bound。

以 `K=V=128`、bf16 state 为例：单 head 状态大小为：

```text
128 × 128 × 2 bytes = 32 KiB
```

一次 token 至少涉及状态读写，另有 q/k/v/g/output。先写出理论最低字节量，再用 profiler 看实际 DRAM/L2 流量。注意：缓存命中、写回策略和 state 常驻 register/shared memory 会使“理论值”与硬件计数不同。

对比实验：

1. fp32 state vs bf16 state；
2. K-first vs V-first；
3. gate/norm/beta 分离 vs 融合；
4. one-warp vs grouped CTA；
5. 不同 sequence-head 并行度。

一次只改变一个类别。

## 4. Lab C：理解 FLA chunk/WY

目标：从串行 recurrence 过渡到 chunked GEMM。

追踪 KDA forward：

```text
chunk.py
→ gate.py
→ chunk_fwd.py
→ chunk_intra.py
→ wy_fast.py
→ common state/output kernels（在完整 FLA clone 中）
```

为一个 chunk 画出 `Akk`、`Aqk`、`w`、`u`、state 与 output 的 shape。然后回答：

- 哪些计算可按 token/chunk 并行？
- 哪部分仍必须按 chunk 顺序推进？
- 为什么 chunk 算法更适合 prefill，而 recurrent 更适合短 decode？
- backward 为什么倾向 recompute 中间量？

## 5. Lab D：复盘 FlashKDA 的 K1/K2 拆分

目标：理解“少一次 launch”不一定更快。

从 deep-dive 的结论出发，用 profiler 验证：

- K1 grid 约为 `N × H × num_chunks`，并行度高；
- K2 grid 约为 `N × H`，包含 chunk recurrence，并行度低；
- 单 kernel 融合会让高并行阶段被低并行阶段拖住；
- 拆成 K1/K2 后虽多一次 launch，但上游报告至少约 15% 端到端收益（只应视作该项目实测结论，需在你的 GPU 上重测）。

接着定位：

- K1 的 shared-memory reuse 与 occupancy；
- K2 的 register-file transpose；
- fp16 inverse、bf16 state、fp32 FMA 的精度职责；
- `exp2`、`tanh.approx` 的近似误差测试。

## 6. Lab E：研究 serving dispatch

目标：理解 kernel 之外的系统约束。

使用 FlashInfer KDA decode 做代码审查：

1. 什么条件选择 one-warp kernel？
2. 小 grid 为什么改用 grouped CTA？
3. state pool 与 `ssm_state_indices` 如何避免 gather/scatter？
4. speculative decode 如何选择已接受 token 对应的 checkpoint？
5. padding/负 index 的语义是什么？

再对 GDN prefill 检查 SM90、SM100、SM120 的 dispatch 和限制。把“算法不支持”和“当前 specialization 未实现”区分开。

## 7. Profile → 假设 → 单变量实验

每轮优化只写一个可证伪假设：

```text
证据：K1 shared-memory 限制为每 SM 只能驻留 1 CTA。
假设：复用两个生命周期不重叠的 buffer，可降到 2 CTA/SM 的阈值以下。
改动：只调整 shared-memory storage union，不改数学和 dtype。
验证：完整 correctness matrix。
性能：相同 workload/profile 配置，报告中位数与硬件计数。
结论：保留或回退。
```

可选指标：

- latency / tokens/s；
- achieved memory bandwidth；
- Tensor Core/FMA 利用率；
- occupancy、active warps；
- registers/thread、shared memory/CTA；
- DRAM/L2 bytes；
- launch 数与 launch 间空隙。

不要在没有 profiler 证据时同时调整 tile、dtype、layout 和 fusion。

## 8. 建议的八次学习安排

| 次数 | 内容 | 产出 |
|---|---|---|
| 1 | GDN/KDA recurrent 数学 | 手算与 shape 图 |
| 2 | FLA Triton recurrent | grid/layout/字节流分析 |
| 3 | FLA chunk/WY | chunk 数据流图 |
| 4 | FlashKDA K1 | inversion、数值、occupancy 笔记 |
| 5 | FlashKDA K2 | state、寄存器转置、低并行分析 |
| 6 | cuLA SM90/SM100 | CuTe DSL/CUTLASS 架构对照图 |
| 7 | FlashInfer serving | dispatch/state pool/spec decode 图 |
| 8 | 第一次 profile 实验 | 可复现 benchmark + 一条优化假设 |

完成这八步后，再选择一个明确目标：**GDN decode、GDN prefill、KDA decode 或 KDA prefill**。四者瓶颈不同，不建议一开始同时优化。
