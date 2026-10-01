# vLLM × SGLang：核心机制对照阅读清单

> 对照同一模型、同一并发输入和相同 cache/page 配置来读；不要从数据结构名字直接推断性能。本文的“优先级”拆成请求队列策略、running/waiting 的调度顺序、prefill/decode 的执行顺序、内存不足时的牺牲顺序四件事。路径分别相对两个仓库根目录。

## 统一例子

设 A、B 共享 128 token 前缀，A 正在 decode；B 等待 32-token prefill；C 是新到的短请求。KV 只够再容纳部分 token。每读一项，都回答本轮选谁、复用多少、分配哪个物理槽、缺内存时谁让位、下轮怎样继续。

## 1. PagedAttention 与 RadixAttention

| 问题 | vLLM 阅读入口 | SGLang 阅读入口 |
| --- | --- | --- |
| 物理 KV 如何分页、block table 如何让 kernel 间接寻址 | `vllm/v1/core/block_pool.py`、`kv_cache_manager.py`、`vllm/v1/attention/backends/` | `python/sglang/srt/mem_cache/memory_pool.py`（按当前版本确认文件）、`managers/schedule_batch.py`、attention backend |
| 如何认定相同 token 前缀 | `block_pool.py` 的 block hash 链、`kv_cache_coordinator.py` 的多组命中 | `mem_cache/radix_cache.py` 的 `RadixKey`、`TreeNode`，当前混合模型再看 `unified_radix_cache.py` |
| 谁保留、锁住、驱逐已算好的 KV | `BlockPool`、`KVCacheManager` | `RadixCache`/`UnifiedRadixCache` 的 insert、lock ref、evict 与各 component |

- [ ] 画 A/B 的逻辑 token→物理 KV 映射，分别标注 vLLM block id 和 SGLang tree node/value（KV slot 索引）。
- [ ] 比较 prefix 粒度、page 对齐、锁与驱逐；在相同粒度、容量、策略下再讨论实际跳过的 prefill token。PagedAttention 是分页存储/寻址机制，RadixAttention 强调基于 radix tree 的前缀管理；两者不是互斥的“有分页/无分页”。

## 2. Continuous batching 与每轮组 batch

- [ ] **vLLM**：读 `vllm/v1/core/sched/scheduler.py::Scheduler.schedule`。按 `running` 循环→`waiting` 循环追踪 token budget、`max_num_active_reqs`、KV 分配、chunked prefill、preemption；再看 `GPUModelRunner.execute_model` 如何把本轮所选 token 变成模型输入。
- [ ] **SGLang**：读 `python/sglang/srt/managers/scheduler.py::get_next_batch_to_run`、`get_new_batch_prefill`、`update_running_batch`；读 `schedule_batch.py::ScheduleBatch.prepare_for_extend/prepare_for_decode/mix_with_running`。记录上一轮 prefill 怎样并入 running batch、本轮何时选新 prefill、何时执行 decode、mixed chunk 条件。
- [ ] 用 A/B/C 填两张逐轮表：`step | running | waiting | selected forward | tokens | cache hit | allocated slots | next state`。两者都能动态加入/移除请求；比较应落在代码中的决策和数据结构，而不是把 SGLang 说成“不使用 continuous batching”。

## 3. Decode 的 priority 到底是什么

- [ ] **vLLM 队列策略**：读 `vllm/config/scheduler.py` 的 `SchedulingPolicy`、`vllm/v1/core/sched/scheduler.py::schedule`。默认/显式 FCFS 与 PRIORITY 分开；当前主循环先调度 `running`，再考虑 `waiting`。在 PRIORITY 模式下看谁作为缺内存时的 preemption victim（`priority`、`arrival_time`）。这不是“所有 decode 永远比所有 prefill 高优先级”，因为 running 中也可能有 chunked prefill。
- [ ] **SGLang 队列策略**：读 `python/sglang/srt/managers/schedule_policy.py::SchedulePolicy.calc_priority`，比较 FCFS、LPM、DFS_WEIGHT、HRRN、LOF 和显式 priority。该函数主要给等待队列定序，不能直接当作 decode 的顺序。
- [ ] **SGLang 执行顺序**：读 `scheduler.py::get_next_batch_to_run`：通常先尝试 `get_new_batch_prefill`；有新 batch 时优先运行它，否则 `update_running_batch` 做 decode。`is_mixed_chunk` 时可把 running decode 与新 prefill 合并；延迟 prefill、内存/slot 约束、speculative/PD 模式会改变路径。内存不够时再看 `ScheduleBatch.retract_decode` 与 `_get_decode_retraction_order`。
- [ ] 固定同一批 A/B/C，区分“等待队列排序”“已有 decode 是否本轮执行”“OOM 时撤回谁”三个结论，不用一个 `priority` 标签概括。

## 4. 加做两个交叉对比

- [ ] **CUDA Graph**：对照 `01`、`02` 清单：谁选择捕获桶、谁更新静态输入、FULL/piecewise 的边界在哪里、哪些情况下 eager fallback。
- [ ] **线性注意力 prefix cache**：对照 `03`、`04` 清单：token 身份、Full/SWA KV、recurrent/conv checkpoint、对齐与驱逐。测量 skipped/recomputed prefill token、TTFT、占用量；不能只比 hash hit rate 或树节点数。

## 完成标志

能对 A/B/C 给出两套逐轮执行轨迹，并为每个不同结论指出具体函数。若想做性能判断，先固定模型、硬件、请求序列、并发、KV 容量、page/block 粒度、prefix cache 和 chunked prefill 配置，再采集 TTFT、ITL、吞吐、重算 token 与显存占用。
