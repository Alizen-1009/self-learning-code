# SGLang 线性注意力 Prefix Caching：代码阅读清单

> 阅读基准：`python/sglang/srt` 当前主路径。先看统一 radix cache 的 Mamba 组件，再看旧的 `radix_cache.py` 核心树概念。不要把 `RadixLinearAttention` 的算子封装误认为整个 prefix cache 的实现。

## 固定例子

两个请求共享 128 token，随后分叉。缓存树在 128 边界有 Full Attention KV 和线性层的递推状态检查点；若模型有短卷积，还需对应 conv 窗口。第三个请求复用该前缀时，追踪树匹配、状态槽分配、命中状态转入活跃槽、后续 decode 的写入与释放。分别记录 GPU KV slot、Mamba active slot、可选 checkpoint slot。

## 最短阅读链

- [ ] **模型状态是什么**：读 `python/sglang/srt/layers/attention/linear/gdn_backend.py`（KDA 模型则读 `kda_backend.py`）、`hybrid_linear_attn_backend.py` 和 `python/sglang/srt/layers/radix_linear_attention.py`。找 recurrent/conv state 的输入输出以及与 request slot 的关系。
- [ ] **前缀键与树的基本操作**：读 `python/sglang/srt/mem_cache/radix_cache.py` 的 `RadixKey.match/page_aligned`、`TreeNode`、`RadixCache.match_prefix/insert/_split_node`。`key` 是 token 段；`value` 是索引，不是直接存放一整份 GPU K/V。page_size > 1 时特别看对齐。
- [ ] **当前混合模型的 cache 入口**：读 `python/sglang/srt/mem_cache/unified_radix_cache.py` 的 `match_prefix`、`cache_finished_req`、`cache_unfinished_req`。确认 `tree_core.match_prefix` 之后各 component 的 `finalize_match_result_in_cache` 才决定最终可用命中。
- [ ] **Mamba 组件的检查点规则**：读 `python/sglang/srt/mem_cache/unified_cache/components/mamba.py`，结合 `unified_tree_core.py`。找树节点上的 Mamba state 与 Full/SWA component 的关系、分叉点如何选择 checkpoint、命中/驱逐/锁如何联动。
- [ ] **状态存储与复用**：读 `python/sglang/srt/mem_cache/allocator/mamba.py`、`mamba_checkpoint_pool.py`。后者的 int8 checkpoint pool 是可选配置；检查 `store_from_active`、`load_to_active` 如何一起处理 temporal 与 conv state，避免以为压缩缓存总是开启。
- [ ] **请求进入调度时实际命中**：读 `python/sglang/srt/managers/schedule_policy.py` 的 `match_prefix_for_req`、`SchedulePolicy._compute_prefix_matches`；然后读 `schedule_batch.py` 的 `Req.init_next_round_input` 与 `ScheduleBatch.prepare_for_extend`。看 `prefix_indices`、`mamba_branching_seqlen`、copy-on-write 源/目标槽和要重算的后缀。
- [ ] **继续 decode 与状态跟踪**：读 `schedule_batch.py` 的 `prepare_for_decode`、`hybrid_linear_attn_backend.py` 的 `_init_track_ssm_indices` 与 `_replay_metadata`。分清活跃 state 每轮更新、为了将来 prefix 命中而额外保存的 checkpoint，以及 CUDA Graph replay 所需静态地址。
- [ ] **验证读法**：用 `git ls-files 'test/*' | rg 'mamba|gdn|kda|radix|prefix'` 找用例，选“共享前缀后分叉”“命中后继续 decode”“cache eviction”“可选 int8 checkpoint”各一例，核对结果正确性与实际 skipped token 数。

## 必须回答

1. token 前缀树命中后，哪个 component 可能把可用长度缩短？
2. 树节点、GPU KV slot、Mamba active slot、可选 int8 checkpoint slot 各保存什么？
3. 分叉后为何要 copy-on-write？源状态何时可被另一请求复用，目标状态何时可写？
4. checkpoint 不在命中边界、SWA 窗口不完整或页面未对齐时，哪里发生重算？
5. `RadixLinearAttention.forward` 与 radix tree 的匹配/状态生命周期分别负责什么？

## 完成标志

能从 `Scheduler.get_new_batch_prefill → Req.init_next_round_input → UnifiedRadixCache.match_prefix → Mamba component → prepare_for_extend/linear backend` 追到真实 state，并说出一次命中的身份、驻留、状态完整性和页面边界。
