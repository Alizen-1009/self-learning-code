# Mission: 从 Linear Attention 到高性能 Chunk Kernel

## Why
为了能在真实 GPU 工作负载中独立理解、实现和优化 GDN/KDA prefill kernel，而不是只会调用现成算子。最终需要能从数学递推定位到 FLA、FlashKDA、cuLA 中的具体数据流和性能瓶颈。

## Success looks like
- 能从 causal Linear Attention 独立推导 recurrent 与 chunk 两种等价形式，并标注每个张量的 shape。
- 能写出 PyTorch reference，并用 recurrent baseline 验证 chunk forward 的正确性和数值误差。
- 能实现并调试一个 Triton chunk kernel，使用 profiler 判断计算、带宽、并行度或资源瓶颈。
- 能读懂并修改 FLA、FlashKDA、cuLA 的 GDN/KDA prefill 路径，用可复现实验决定保留或回退优化。

## Constraints
- 中文优先，数学必须落到代码、shape 和可运行练习。
- 每课只解决一个紧凑问题，通过检索练习而不是被动阅读建立长期记忆。
- 当前先按“熟悉 Python/PyTorch 和 CUDA 基础，尚未系统掌握 LA chunk 推导”设计；根据练习反馈调整。
- 本机没有 NVIDIA CUDA 环境；GPU kernel 实验需要在可用的 Hopper/Blackwell 机器上完成。

## Out of scope
- 在掌握 forward/prefill 主线前，不展开完整 backward、跨卡 Context Parallel 和模型训练配方。
- 不同时追逐所有 Linear Attention 变体；先用 vanilla LA 建立 chunk 模型，再进入 GDN，最后进入 KDA。
