# SGLang CUDA Graph：代码阅读清单

> 阅读基准：SGLang `main` 的 `python/sglang/srt` 推理路径；路径相对 SGLang 仓库根目录。先读普通 decode 图，再看 prefill、speculative、breakable 等分支。用 `git rev-parse HEAD` 固定阅读版本。

## 固定例子

正在 decode 的请求数是 3，捕获的 batch-size 桶有 1、2、4、8。跟踪 3→4 的 padding：真实 `ForwardBatch`、捕获宽度 4、静态输入 buffer、attention metadata 和最终只返回 3 行结果。然后让第 4 个请求拥有不同的每请求 token 数，观察 `can_run_graph` 的限制。

## 最短阅读链

- [ ] **主入口与调用者**：读 `python/sglang/srt/model_executor/model_runner.py` 的 `init_cuda_graphs`、`init_decode_cuda_graph` 和实际 forward 中调用 `decode_cuda_graph_runner.can_run_graph/execute` 的分支。先确定何时捕获、何时 eager。
- [ ] **捕获哪些桶**：读 `python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py` 的 `get_batch_sizes_to_capture`、`BaseCudaGraphRunner._pad_to_bucket`；把 batch-size 对齐、最大请求数和 compile 桶与 capture 桶分开。
- [ ] **静态缓冲区与捕获准备**：读 `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` 的 `DecodeCudaGraphRunner.__init__`、`capture_prepare`。找 `capture_bs`、`captured_req_width`、`max_bs`、`DecodeInputBuffers` 和 attention backend 的 `init_cuda_graph_state`。
- [ ] **真正捕获**：同文件 `capture → capture_one_shape`。记录 warmup、dummy `ForwardBatch`、`init_forward_metadata_out_graph`、`run_once`、`backend.capture` 的顺序。遇到 `torch.compile` 时分清它与 CUDA Graph 的职责。
- [ ] **图后端**：读 `python/sglang/srt/model_executor/runner_backend/base_cuda_graph_backend.py`、`full_cuda_graph_backend.py`。找到 `torch.cuda.CUDAGraph` 的创建与 replay；再选读 `tc_piecewise_cuda_graph_backend.py`、`breakable_cuda_graph_backend.py`，看何时不是一个完整 forward 图。
- [ ] **每步如何喂新请求**：回到 `DecodeCudaGraphRunner.can_run_graph`、`load_batch`、`execute`。逐项追踪真实 bs、选择的 graph key、静态 buffer 原地更新、attention metadata 的图外/图内准备、`backend.replay`、输出裁剪。
- [ ] **KV 与线性状态的图内地址**：读所用 attention backend 的 `init_cuda_graph_state`、`init_forward_metadata_out_graph`；混合线性注意力模型再读 `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` 的 `_replay_metadata`。确认捕获的是 buffer 地址，replay 前会更新状态索引及内容。
- [ ] **扩展路径**：普通 decode 走通后看 `prefill_cuda_graph_runner.py` 和 `python/sglang/srt/speculative/*cuda_graph_runner.py`；这些路径有不同的图键和 token 宽度，不要套用普通 decode 的 1 token/req 假设。

## 阅读时回答

1. 为什么 batch size=3 可以 replay 4 的图？padding 出现在哪些 tensor 中？
2. `can_run_graph` 除了 bs 还检查什么？何时退回 eager？
3. 图外 metadata 准备与图内模型 forward 的界线在哪里？
4. 捕获图时 dummy 请求的 KV/状态索引与 replay 时真实请求的索引如何衔接？
5. FULL 与 piecewise/breakable 后端各自把什么作为可重放单元？

## 完成标志

能沿 `ModelRunner → DecodeCudaGraphRunner.can_run_graph → load_batch → backend.replay` 解释一次 decode，并明确区分捕获桶大小、逻辑 batch 大小、静态 buffer 容量和真实 KV 内容。
