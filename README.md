# Self Learning Workspace

这是一个长期学习仓库。不同主题在顶层独立组织，避免课程、参考资料和手写代码混在一起；以后学习 I/O、分布式系统等主题时，可以继续新增同级目录。

## 目录

| 路径 | 作用 |
|---|---|
| [`handle_code/`](./handle_code/) | 过去积累的手写代码，包括算法、C++、CUDA、Python 与量化练习 |
| [`gpu-kernel-learning/`](./gpu-kernel-learning/) | CUDA / Tensor Core / CUTLASS / MoE kernel 的系统学习工作区 |

## 后续主题约定

新学习方向优先建立独立的顶层目录，例如：

```text
self-learing/
├── handle_code/
├── gpu-kernel-learning/
├── io-systems-learning/           # 未来
└── distributed-systems-learning/  # 未来
```

每个主题自行维护 `README.md`、学习记录、练习和参考资料；仓库根目录只负责总导航和跨主题约定。

## 克隆

CUDA 学习区包含 Git submodule，首次克隆建议使用：

```bash
git clone --recurse-submodules git@github.com:Alizen-1009/cuda-self-learning-code.git
cd cuda-self-learning-code
```

已有仓库可执行：

```bash
git submodule update --init --recursive
```
