# Linear Attention：GDN / KDA Kernel 学习区

本目录围绕 **Gated DeltaNet (GDN)** 与 **Kimi Delta Attention (KDA)**，对照学习
[FLA](https://github.com/fla-org/flash-linear-attention)、
[FlashKDA](https://github.com/MoonshotAI/FlashKDA)、
[FlashInfer](https://github.com/flashinfer-ai/flashinfer) 和
[cuLA](https://github.com/inclusionAI/cuLA) 的实现与优化。

> 目录名使用 `linear_attn`，正文统一使用 **linear attention**。

## 从这里开始

### 交互式课程

1. [学习使命](MISSION.md)：最终要独立实现并优化 GDN/KDA prefill kernel。
2. [第 1 课：从 Causal LA 到一个 Chunk](lessons/0001-causal-la-to-one-chunk.html)
3. [第 2 课：Scalar Decay 如何进入 Chunk](lessons/0002-scalar-decay-to-gated-chunk.html)
4. [第 3 课：Delta Rule 为什么需要三角求解](lessons/0003-delta-rule-to-wy.html)
5. [第 4 课：完整拼出 Gated Delta Chunk](lessons/0004-gated-delta-chunk.html)
6. [第 5 课：GDN Prefill 真实执行图](lessons/0005-gdn-prefill-execution-map.html)
7. [第 6 课：B300 Recurrent GDN Baseline](lessons/0006-b300-recurrent-gdn-baseline.html)
8. [第 6 课可运行代码](exercises/06-gdn-recurrent-baseline/README.md)
9. [第 7 课：C=64 GDN Chunk Triton Baseline](lessons/0007-c64-gdn-chunk-triton-baseline.html)
10. [第 7 课可运行代码](exercises/07-gdn-chunk-triton-baseline/README.md)
11. [第 8 课：Readable Persistent CuTe DSL GDN](lessons/0008-readable-persistent-cutedsl-gdn.html)
12. [第 8 课可运行代码](exercises/08-gdn-cutedsl-persistent-readable/README.md)
13. [第 9 课：GDN 毕业挑战](lessons/0009-gdn-capstone.html)
14. [第 9 课挑战代码](exercises/09-gdn-capstone/README.md)
15. [公式速查表](reference/linear-attention-equations.html)
16. [GDN Prefill 调度速查](reference/gdn-prefill-schedule.html)
17. [Recurrent GDN Baseline 速查](reference/gdn-recurrent-baseline.html)
18. [Chunk GDN Baseline 速查](reference/gdn-chunk-baseline.html)
19. [Persistent CuTe GDN 速查](reference/gdn-persistent-cutedsl.html)
20. [可信资源](RESOURCES.md)

### 原有阅读指南

1. [数学与核心概念](docs/01-foundations.md)：先弄清状态递推、GDN/KDA 的差别。
2. [代码阅读顺序](docs/02-code-reading-map.md)：从 reference 一路读到 Triton、CUTLASS、CuTe DSL。
3. [优化实验路线](docs/03-optimization-labs.md)：按 correctness → profile → 单变量优化推进。
4. [cuLA 代码阅读指南](docs/04-cula-reading-map.md)：对照 Hopper/Blackwell 的 CuTe DSL 与 CUTLASS C++ 实现。
5. [FLA vs FlashInfer GDN Prefill 全流程](docs/05-fla-vs-flashinfer-gdn-prefill.md)：逐步骤对照 multi-kernel 与 SM100/SM103 persistent kernel。
6. [上游版本与许可证](UPSTREAMS.md)：确认代码来源、commit 和更新方法。

## 目录结构

```text
linear_attn/
├── code/                        # 上游 GDN/KDA 精选源码
├── exercises/                   # 课程可运行代码与 B300 结果
│   ├── 06-gdn-recurrent-baseline/
│   ├── 07-gdn-chunk-triton-baseline/
│   ├── 08-gdn-cutedsl-persistent-readable/
│   └── 09-gdn-capstone/
├── lessons/                     # 交互式 HTML 课程
├── reference/                   # 可打印速查
├── docs/                        # 深入阅读材料
└── scripts/refresh_curated_sources.py
```

`code/` 下每个项目都有：

- 原项目的 `LICENSE`（FlashInfer 另含 `NOTICE`）；
- `UPSTREAM_REVISION.md`；
- `MANIFEST.sha256`；
- 保持上游相对路径不变的源码、测试和 benchmark。

精选快照不是独立可运行包。需要构建或运行完整上游测试时，请在本目录之外单独 clone；刷新脚本只使用临时 clone，完成后自动清理。

## 四个项目分别学什么

| 项目 | 最值得学习的部分 | 主要定位 |
|---|---|---|
| FLA | naive reference、chunk/WY 表示、训练反向、Triton recurrent | 算法语义与通用训练实现 |
| FlashKDA | chunk=16、K1/K2 拆分、混合精度、寄存器转置 | KDA prefill 的高性能 CUTLASS 实现 |
| FlashInfer | GDN prefill/decode、KDA decode、状态池、varlen、spec decode | 面向 serving 的架构特化实现 |
| cuLA | SM90/SM100 KDA、CuTe DSL、CUTLASS C++、Lightning Attention、CP | 面向 Hopper/Blackwell 的新一代 Linear Attention kernel；仍处于 early stage |

## 第一条主线

先不要直接扎进 CUDA。按下面顺序建立同一套心智模型：

```text
FLA naive recurrence
  → FLA fused recurrent Triton
  → FLA chunk/WY Triton
  → FlashKDA K1/K2 CUTLASS
  → cuLA 的 SM90/SM100 CuTe DSL 与 CUTLASS C++
  → FlashInfer decode/prefill serving kernels
```

贯穿始终只追踪四个量：`state`、`decay/gate`、`prediction error`、`output`。
