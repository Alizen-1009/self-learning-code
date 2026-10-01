# vLLM 线性注意力 Prefix Caching：代码阅读清单

> 阅读基准：V1 路径。vLLM 在多个状态空间/线性注意力模型中使用 Mamba 类缓存接口；不同模型的 GDN、KDA、Mamba2 状态形状和限制不相同。下面选一个 Full Attention + GDN 的混合模型作主线，勿把所有 `MambaSpec` 都当成同一个模型。

## 固定例子与正确性条件

请求 A、B 共享前 128 token，第 129 token 分叉。假定注意力 KV 的 128-token 前缀仍在 GPU。B 要跳过前 128 token，除了 token 身份与 KV 命中，还要在相同边界拿到 **线性层递推 state**；有短卷积的层还要有 **conv state**。任何一个状态缺失，就应缩短可复用长度或重算。记录“匹配长度”“状态检查点位置”“实际跳过的 token”三个不同数字。

## 最短阅读链

- [ ] **模型层要保存什么**：读 `vllm/model_executor/layers/mamba/gdn/base.py`，任选 `qwen_gdn_linear_attn.py` 或 `kimi_gdn_linear_attn.py`，列出 temporal/recurrent state、short-conv state 的形状与写回点。KDA 可对照 `vllm/model_executor/layers/mamba/kda_checkpoint.py`。
- [ ] **缓存配置与分组**：读 `vllm/v1/kv_cache_spec_registry.py`、`vllm/v1/core/kv_cache_coordinator.py` 的 `HybridKVCacheCoordinator`；找 Full Attention、SWA、Mamba 组如何各自报告 prefix 命中，以及最终可用长度如何收敛到共同边界。
- [ ] **状态块的所有权**：读 `vllm/v1/core/single_type_kv_cache_manager.py` 的 `MambaManager`，尤其 `find_longest_cache_hit`、`allocate_new_blocks`、`cache_blocks`。区分 `mamba_cache_mode` 的 `align` 与其他模式；看 checkpoint 对应的是哪个 block/结束 token 位置。
- [ ] **token 前缀身份**：读 `vllm/v1/core/block_pool.py` 的 `get_cached_block`、`cache_full_blocks`，及 request/block hash 生成处。画出父 hash + 当前 token block + 额外键如何形成链；命中 hash 不代表状态已完整或仍驻留。
- [ ] **新请求的命中与调度**：读 `vllm/v1/core/kv_cache_manager.py` 的 `get_computed_blocks`、`allocate_slots`；再读 `vllm/v1/core/sched/scheduler.py` 的 `_get_local_prefix_cache_hit`、`schedule`、`_mamba_block_aligned_split`。记录 prefix lookup、共享分叉边界、内部 prefill checkpoint 和实际 `num_computed_tokens` 的处理。
- [ ] **生成/导出 checkpoint**：读 `vllm/model_executor/layers/mamba/checkpoint.py` 的 `MambaPrefillCheckpointBuilder`/`Exporter`，再追到模型 backend 的实际 export；确认“为 checkpoint 分配 block”与“该步真的把 state 写入 block”是一致的。
- [ ] **命中后加载与继续写**：读 `vllm/v1/worker/gpu_model_runner.py` 的 `_update_states`、Mamba buffer 处理，以及 `vllm/v1/worker/mamba_utils.py`。追踪从缓存 block 到本轮计算 state 的索引或 copy-on-write，确认分叉请求不会互相覆盖状态。
- [ ] **验证边界**：阅读相关 `tests/v1/` 中含 `mamba`、`prefix`、`checkpoint` 的用例，优先找共同前缀后分叉、部分 block、SWA 混合、驱逐后重算。测试路径随版本变化，先用 `git ls-files 'tests/*' | rg 'mamba|prefix|checkpoint'` 定位。

## 必须回答

1. 128-token 命中时，各 KV 组与线性状态组分别返回多长？谁决定最终可跳过长度？
2. checkpoint 在 128 还是 127 结束位置？下一 token 的输入 state 从何处取？
3. `mamba_cache_mode=align` 下，为什么 hash 命中仍可能需要 block 对齐或部分尾巴重算？
4. 共享分叉点怎样保留状态，又怎样避免 A、B 在后续 decode 写同一个可变 state？
5. 若状态块被驱逐，哪些 token 必须重算？实际 TTFT 收益该用什么量度，而非只报 hash 命中率？

## 完成标志

能把 `Scheduler.schedule → KVCacheManager.get_computed_blocks → HybridKVCacheCoordinator → MambaManager → GPUModelRunner` 连起来，并用 A/B 分叉例子说清身份、驻留、完整状态和对齐这四层条件。
