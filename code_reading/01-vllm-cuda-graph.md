# vLLM CUDA Graph：代码阅读清单

> 阅读基准：vLLM `main`；开始阅读前用 `git rev-parse HEAD` 记录自己的提交号。下列路径相对 vLLM 仓库根目录。以符号搜索定位，避免更新后行号漂移。这里讨论 V1 GPU 推理主路径，不把 encoder、speculative、ubatching 等分支当作最短入门路径。

## 先抓住一个固定例子

设一次调度有 3 个 decode 请求，每个请求本轮执行 1 个 token；捕获桶为 4。重点跟踪：真实 `num_reqs=3`、逻辑 `num_tokens=3`、图输入容量 4、真实 KV 内容与 block table 在 replay 前如何更新。再把第 4 个请求改成一个 8-token prefill，观察 FULL、PIECEWISE、NONE 的选择。

## 最短阅读链

- [ ] **配置决定捕获集合**：读 `vllm/config/compilation.py` 中 `CUDAGraphMode`、`cudagraph_capture_sizes` 和 mode 配置。记录 FULL、PIECEWISE、混合模式各自允许的 batch descriptor。
- [ ] **构造调度键**：读 `vllm/v1/cudagraph_dispatcher.py` 的 `CudagraphDispatcher.initialize_cudagraph_keys`、`_create_padded_batch_descriptor`、`dispatch`、`get_capture_descs`。特别看 `num_tokens`、`num_reqs`、`uniform`、LoRA 如何进入键，为什么 FULL 先于 PIECEWISE 被尝试，以及何时退回 NONE。
- [ ] **GPU runner 的固定缓冲区**：读 `vllm/v1/worker/gpu_model_runner.py` 的 `GPUModelRunner.__init__`、`_update_states`、`_prepare_inputs`。把 CPU 侧请求状态、GPU 输入 tensor、KV block table、attention metadata 画成一条更新链；区分“地址固定”和“内容固定”。
- [ ] **捕获入口**：继续读同文件的 `capture_model`、`_capture_cudagraphs`、`_warmup_and_capture`、`_dummy_run`。按顺序记下捕获哪些 descriptor、dummy 输入如何成形、warmup 与真正 capture 分别做什么。
- [ ] **FULL 图实际存储与 replay**：读 `vllm/compilation/cuda_graph.py` 的 `CUDAGraphWrapper.__call__`、`CUDAGraphEntry`。定位 `torch.cuda.CUDAGraph()`、capture 上下文、`entry.cudagraph.replay()`、输出 buffer 和输入地址检查。
- [ ] **PIECEWISE 图边界**：读 `vllm/compilation/breakable_cudagraph.py` 以及 `vllm/compilation/cuda_graph.py`，找编译区域中哪些 op 被图覆盖、哪些 attention/backend 操作让图分段；不要把它理解成“一个完整 forward 图”。
- [ ] **请求实际执行时怎么选图**：回到 `GPUModelRunner.execute_model` 与调用 `cudagraph_dispatcher.dispatch` 的位置，追踪 `forward_context.cudagraph_runtime_mode` 到 wrapper 的调用。将“初始化时捕获”与“每步 replay 前更新输入”分成两列记。
- [ ] **后端限制**：顺着 `vllm/v1/attention/backends/` 的当前所选 backend 和 `build_for_cudagraph_capture` 看元数据形状、padding、支持的图模式；同一配置下 graph 选择可能因后端能力变化。

## 阅读时写下的答案

1. `BatchDescriptor` 的每个字段由哪个运行时值产生？3 个 decode 请求为什么可以使用容量 4 的图？
2. 一个新 batch 为什么可能命中 PIECEWISE，却不能命中 FULL？指出 `dispatch` 中的判断。
3. capture 时存下了什么：kernel 执行结构、tensor 地址、KV 数据、还是请求对象？为每一项找源码证据。
4. replay 前具体哪些 tensor 被原地填充？哪些输出要切回真实请求数？
5. `capture_model` 捕获失败、batch 超过最大桶、backend 不支持时分别如何回退？

## 完成标志

能从 `SchedulerOutput` 走到 `execute_model → dispatch → CUDAGraphWrapper.replay`，用 3→4 的例子说明 padding、地址和 KV 内容的关系；能指出 PIECEWISE 与 FULL 的实际代码边界。建议最后再看 `vllm/v1/worker/encoder_cudagraph.py`、speculative/LoRA 等特化路径。
