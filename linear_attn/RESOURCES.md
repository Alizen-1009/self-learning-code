# Linear Attention → Chunk Kernel Resources

## Knowledge

- [Paper: _Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention_](https://arxiv.org/abs/2006.16236)
  Linear Attention 的基础来源。用于理解 feature map、causal state recurrence，以及为什么复杂度能从序列二次降为线性。
- [Paper: _Gated Linear Attention Transformers with Hardware-Efficient Training_](https://arxiv.org/abs/2312.06635)
  GLA 与硬件友好的 chunkwise parallel form。用于从 recurrent 表达过渡到可用矩阵乘并行化的 chunk 表达。
- [Paper: _Gated Delta Networks: Improving Mamba2 with Delta Rule_](https://arxiv.org/abs/2412.06464)
  GDN 的主要论文。用于学习 scalar forget gate、delta-rule residual 与状态更新。
- [Paper: _Kimi Linear: An Expressive, Efficient Attention Architecture_](https://arxiv.org/abs/2510.26692)
  KDA 的主要论文。用于理解 per-key-dimension gate、safe gate 和 chunk 算法所需的新数值约束。
- [Repository: FLA — flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
  算法语义与通用训练实现的主参考。优先读 naive reference，再读 Triton recurrent、chunk forward 和 backward。
- [FLA source: naive Linear Attention](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/linear_attn/naive.py)
  第一阶段代码锚点。用于把论文中的状态递推映射到最小 PyTorch 实现。
- [Local source: FLA naive GDN](code/fla/fla/ops/gated_delta_rule/naive.py)
  第二阶段代码锚点。用于对照 recurrent GDN 与 chunk GDN 的等价数据流。
- [Local source: FLA naive KDA](code/fla/fla/ops/kda/naive.py)
  第三阶段代码锚点。用于理解 per-K decay 如何改变 chunk 内 `Akk/Aqk` 构造。
- [Repository report: FlashKDA v1 deep dive](code/flashkda/docs/20260420-flashkda-v1-deep-dive.md)
  高性能 KDA prefill 的直接工程证据。用于学习 chunk=16、K1/K2 拆分、混合精度与寄存器转置。
- [NVIDIA CUTLASS: CuTe DSL](https://github.com/NVIDIA/cutlass/tree/main/python/CuTeDSL)
  后期实现参考。用于学习 layout algebra、TMA、MMA/WGMMA 与 CuTe DSL kernel 组织。
- [Repository: cuLA](https://github.com/inclusionAI/cuLA)
  Hopper/Blackwell Linear Attention 实现参考。用于横向比较 CuTe DSL、CUTLASS C++、模块化与 fused 路径。

## Wisdom (Communities)

- [FLA GitHub Issues](https://github.com/fla-org/flash-linear-attention/issues)
  适合检索 kernel correctness、shape、backend dispatch 和性能回归的真实讨论；提问前先提交最小复现和 profiler 证据。
- [cuLA GitHub Issues](https://github.com/inclusionAI/cuLA/issues)
  适合讨论 SM90/SM100 KDA、CuTe DSL 与 roadmap；项目处于 early stage，接口问题应优先核实当前 commit。
- [GPU MODE](https://discord.gg/gpumode)
  高性能 GPU kernel 实践社区。用于用 profiler 截图、最小 benchmark 和具体硬件计数向实践者验证优化判断。

## Gaps

- B300 已确认通过 ASI Pod `worker0` 远程运行，CUDA Driver/Triton 目标为 SM103；每次实验仍需记录当时的镜像、commit、空闲 GPU 和时钟状态，保证 benchmark 可复现。
- 尚未通过练习确认学习者对 CUDA memory hierarchy 和 Triton program model 的真实熟练度；第一个 baseline 应同时承担能力校准。
