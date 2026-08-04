# Teaching Notes

- 用户选择完整路线 D：数学推导 → PyTorch reference → Triton kernel → profile/优化 → 修改生产级 GDN/KDA prefill kernel。
- 中文优先；所有数学需要落到 shape、代码行和可验证练习。
- 当前能力未完整说明，暂按熟悉 Python/PyTorch 与 CUDA 基础、LA chunk 推导未系统掌握设计。
- 第一阶段顺序：vanilla causal LA → recurrent state → 单 chunk 分解 → scalar decay → delta/WY → gated delta 合并 → Triton baseline；之后进入 KDA。
- 用户在第 1 课后追问 causal mask M，说明下一课应显式区分 0/1 mask M 与带相对衰减的加权矩阵 D。
- 不因“看懂讲解”记录已掌握；必须由练习答案或代码产出提供证据后再写 learning record。
- 用户反馈前四课符号过多、主线不清。后续改为自顶向下工程教学：先给完整 GDN chunk pipeline 和源码调用图，再按需要补推导；避免把一个工程问题拆成多课公式。
- 用户当前重点转向 FlashInfer SM103 GDN 工程实现；需要明确区分算法层 chunk、grid 是否含 chunk 维，以及多 kernel/单 persistent kernel 两种调度。
- HTML 代码块需要明显的 Python 语法配色与高对比度，不能使用单一前景色。
- 第 5 课起以 grid、chunk loop、数据驻留和源码调用图为主；公式只在解释具体 kernel 数据依赖时出现。
- 目标环境固定为 B300 ASI Pod（容器 `worker0`），远程操作使用 `b300-pod` skill / `asicli console`。运行时信息：8× `NVIDIA L20D`（内部设备名），CUDA Driver API 与 PyTorch 均报告 compute capability 10.3，Triton target `arch=103`；CUDA 13.2、PyTorch 2.11.0a0、Triton 3.6.0。`nvidia-smi --query-gpu=compute_cap` 错报 8.9，不作为架构判断依据。
- B300 多人共用；每次 benchmark 前重新检查 GPU 空闲情况，只选择空闲卡，不触碰他人进程。
