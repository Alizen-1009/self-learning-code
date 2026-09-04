# Self Learning Code

面向 **GPU Kernel、Linear Attention 与推理系统** 的长期学习工作区。

这里不只是代码归档：每个专题都会同时维护学习目标、交互式课程、参考资料、练习代码和学习记录，让知识从“读过”逐步沉淀为“能推导、能实现、能验证”。

## 专题导航

| 目录 | 主题 | 当前重点 |
|---|---|---|
| [`tensor-core/`](./tensor-core/) | CUDA / Tensor Core / CUTLASS / CuTe | 从 `mma.sync`、`ldmatrix` 走向 Hopper `wgmma`、Blackwell `tcgen05` 与 MoE Kernel |
| [`linear_attn/`](./linear_attn/) | GDN / KDA Prefill | 从 recurrent、chunk / WY 推导走向 Triton 与 CuTe DSL Kernel，并在 B300 上验证 |
| [`replayssm/`](./replayssm/) | ReplaySSM / SSM Decode | 理解 state traffic、投机验证、cache / flush 不变量与端到端收益 |
| [`handle_code/`](./handle_code/) | 历史代码归档 | 算法题、C++、CUDA、Python 与量化练习 |

## 学习主线

```text
CUDA 性能模型与 Tensor Core
          ↓
Linear Attention 数学与 GDN/KDA Prefill Kernel
          ↓
ReplaySSM Decode、投机验证与 Cache 管理
          ↓
真实硬件上的 correctness、profiling 与优化决策
```

各专题彼此独立，但共享同一套方法：

1. 先建立公式、张量 shape 和执行路径的心智模型；
2. 再通过 reference、练习或真实源码验证理解；
3. 使用 correctness test 与 profiler 数据，而不是直觉判断优化；
4. 将关键纠误、实验结果和下一步记录到 `learning-records/`。

## 快速开始

仓库中的 Tensor Core 参考项目使用 Git submodule。首次克隆建议执行：

```bash
git clone --recurse-submodules git@github.com:Alizen-1009/self-learning-code.git
cd self-learning-code
```

如果已经克隆仓库，可补充初始化子模块：

```bash
git submodule update --init --recursive
```

然后根据目标进入对应专题（任选其一）：

```bash
cd tensor-core    # CUDA / Tensor Core / CUTLASS
# cd linear_attn  # GDN / KDA Prefill
# cd replayssm    # SSM Decode / ReplaySSM
```

课程与速查资料主要使用静态 HTML。可以直接用浏览器打开，也可以在仓库根目录启动本地服务器：

```bash
python3 -m http.server 8000
```

随后访问 `http://localhost:8000/`，进入对应专题的 `lessons/` 或 `reference/`。

## 从哪里开始

- **学习 Tensor Core 与现代 CUDA Kernel**：阅读 [`tensor-core/README.md`](./tensor-core/README.md)。
- **学习 GDN / KDA 及其 Prefill Kernel**：阅读 [`linear_attn/README.md`](./linear_attn/README.md)。
- **学习 ReplaySSM 与 Decode 优化**：先看 [`replayssm/MISSION.md`](./replayssm/MISSION.md)，再从 [`replayssm/lessons/0001-decode-byte-ledger.html`](./replayssm/lessons/0001-decode-byte-ledger.html) 开始。
- **查阅过去的手写代码**：进入 [`handle_code/`](./handle_code/)。

## 目录约定

专题通常按以下方式组织：

```text
<topic>/
├── README.md / MISSION.md   # 专题入口、目标与边界
├── NOTES.md                 # 学习偏好、决策与待办
├── lessons/                 # 编号课程
├── reference/               # 术语表与速查资料
├── exercises/ or code/      # 可运行练习与验证代码
├── learning-records/        # 掌握证据、纠误与进度
└── assets/                  # HTML 课程共享资源
```

新学习方向应建立独立的顶层目录，不与现有 GPU 专题混放；根目录只维护跨主题导航和仓库级配置。各目录的运行环境、依赖和实验方式以该专题内的说明为准。

## 外部源码

- `tensor-core/repos/` 中的参考仓库通过 Git submodule 固定到具体提交。
- `linear_attn/code/` 中的精选源码保留上游路径、版本信息、校验文件和原始许可证。
- 外部项目的版权与许可证以各自目录中的声明为准。
