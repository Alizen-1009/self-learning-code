# AGENTS.md — 长期学习工作区

本仓库按学习主题分区。处理任务前，先确定目标目录并读取该目录内的说明文件。

## 顶层导航

- `handle_code/`：过去积累的手写代码，包含算法、C++、CUDA、Python 和量化练习。
- `gpu-kernel-learning/`：CUDA / Tensor Core / CUTLASS / MoE kernel 系统学习区。进入该主题时，先读 [`gpu-kernel-learning/AGENTS.md`](./gpu-kernel-learning/AGENTS.md)。

## 组织约定

- 新的大方向（例如 I/O、分布式系统）建立新的顶层目录，不混入现有 GPU 学习区。
- 每个主题独立维护目标、进度、资料和练习。
- 根目录只维护跨主题导航与仓库级配置。
