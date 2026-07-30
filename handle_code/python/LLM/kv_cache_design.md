# Session 级 KV Cache：面试设计与 Python 模拟

## 1. 面试开场：先定义问题

可以先用下面这段话回答：

> 我会把系统分成上下文管理和 KV Cache 管理两层。上下文管理在 CPU 或外部存储中维护 session 的消息、token IDs、摘要和版本；KV Cache Manager 在 GPU 显存中维护模型各层已经计算好的 K/V 张量。每个 session 只保存逻辑 block table，不独占连续显存；真正的物理 blocks 由全局 Block Manager 统一管理。这样才能支持 PagedAttention、continuous batching、前缀复用、Copy-on-Write 和全局淘汰。

接着主动确认假设：

- 自部署 Decoder-only Transformer；
- 多用户、多 session；
- 对话主要是 append-only；
- 需要 continuous batching；
- 支持编辑、重新生成和分支；
- GPU 显存有限，必要时允许淘汰、重计算或 CPU swap。

## 2. 不要把四种状态混在一起

| 状态 | 保存内容 | 常见位置 | 生命周期 |
|---|---|---|---|
| Session 上下文 | messages、token IDs、摘要、版本 | CPU/Redis/数据库 | 可长期保存 |
| Session block table | logical block → physical block ID | 调度器 CPU 内存 | session/request 级 |
| KV Cache | 每层、每个 token 的 K/V 张量 | GPU，必要时 CPU | 临时、可重建 |
| 最终回答缓存 | query + config → answer | Redis/数据库 | 业务策略决定 |

KV Cache 不是聊天记录的唯一副本。KV 被淘汰后，系统必须能根据 token IDs 重新 prefill。

## 3. 总体架构

```text
Client
  │ session_id + new message
  ▼
Session / Context Manager
  ├── messages, token_ids, summary, context_version
  └── prompt construction / truncation / compaction
  │
  ▼
Tokenizer + Request Scheduler ───────────────┐
  │ prefill/decode request                  │ continuous batching
  ▼                                         │
Global KV Cache Manager                     │
  ├── session_id -> logical block table     │
  ├── prefix hash -> physical block IDs     │
  ├── ref_count / LRU / block state         │
  ├── GPU free-block pool                   │
  └── optional CPU swap pool                │
  │                                         │
  ▼                                         │
PagedAttention / Model Executor ◄───────────┘
  └── physical K/V tensors for every layer
```

关键原则：**session 拥有逻辑映射，全局池拥有物理显存。**

## 4. 需要哪些参数

### 4.1 模型决定的参数

- `num_layers`：Transformer 层数；
- `num_kv_heads`：KV heads 数量，GQA/MQA 下不同于 query heads；
- `head_dim`：每个 head 的维度；
- `dtype_bytes`：FP16/BF16 通常为 2，FP8/INT8 通常为 1；
- `tensor_parallel_size`：每张 GPU 保存多少 KV heads；
- `model_namespace`：模型权重、tokenizer、RoPE 配置和 adapter 的版本标识。

模型或影响 K/V 计算的配置变化后，旧 KV 不可以继续复用。

### 4.2 缓存和调度参数

- `block_size`：每个 block 容纳的 token 数，如 16 或 32；
- `num_gpu_blocks` / GPU memory utilization；
- CPU swap 空间大小；
- `max_context_tokens`；
- `max_batch_tokens`、`max_num_sequences`；
- prefill chunk 大小；
- 淘汰策略及 idle TTL；
- 是否允许 preemption、swap 或 recompute；
- tenant/session 配额与优先级。

block 小：内部碎片少、前缀命中粒度细，但 metadata 和调度开销大。block 大则相反。

### 4.3 单张 GPU 上一个 block 的显存估算

```text
bytes_per_block_per_gpu
= 2(K 和 V)
× num_layers
× block_size
× (num_kv_heads / tensor_parallel_size)
× head_dim
× dtype_bytes
```

总 blocks 不能简单使用“GPU 总显存 / block bytes”；还要减去模型权重、activation、CUDA graph、workspace 和安全水位。

示例：32 层、8 KV heads、head_dim=128、block size=16、BF16、TP=1：

```text
2 × 32 × 16 × 8 × 128 × 2 = 2 MiB / block / GPU
```

## 5. 需要存哪些数据

### 5.1 Session 元数据（CPU/持久层）

```text
SessionState
- session_id / tenant_id
- messages 或规范化后的 token_ids
- context_version
- logical_block_table: [physical_block_id, ...]
- num_computed_tokens
- sequence_length
- status: waiting/running/swapped/finished
- last_access_at / priority / TTL
- parent_session_id（可选，用于分支）
- summary 与 summary 覆盖范围（可选）
```

### 5.2 Physical Block 元数据（Block Manager）

```text
PhysicalBlock
- physical_block_id
- device: GPU:n / CPU
- block_state: FREE / ACTIVE / CACHED / SWAPPED
- valid_token_count
- ref_count
- prefix_hash / previous_hash
- model_namespace
- last_access_at
- dirty / transfer event（异步 swap 时需要）
```

### 5.3 真正的 KV 张量

概念形状可以写成：

```text
K_cache[layer][physical_block][local_kv_head][token_offset][head_dim]
V_cache[layer][physical_block][local_kv_head][token_offset][head_dim]
```

真实 kernel 可能为了访存合并和向量化使用不同 layout，但管理层不应依赖 Python 里的直觉布局。

### 5.4 Prefix Index

```text
(model_namespace, previous_prefix_hash, block_token_ids)
    -> physical_block_id
```

不能只用当前 block 的 token hash，否则相同 token 出现在不同历史位置时可能错误复用。生产环境还要处理 hash collision：hash 用于定位，token IDs 或强摘要用于校验。

## 6. 核心流程

### 6.1 创建新 session / Prefill

1. Context Manager 构建最终 token IDs；
2. 按 block size 从头查 prefix index；
3. 命中的完整 blocks 增加 `ref_count`；
4. 未命中的后缀申请物理 blocks 并执行 prefill；
5. 完整 block 加入 prefix index，最后一个 partial block 通常保持 session 私有；
6. scheduler 将 logical block table 交给 PagedAttention。

返回指标：`reused_tokens`、`computed_tokens`、分配 block 数和 prefill latency。

### 6.2 多轮 append

1. 从 session 上下文取得历史 token IDs 和 block table；
2. 新消息 tokenize 后追加；
3. 如果尾 block 未满且 `ref_count == 1`，可原地补齐；
4. 尾 block 已共享时先 Copy-on-Write；
5. 为剩余 token 申请新 blocks；
6. 只对新增 token 做 prefill，然后进入逐 token decode。

### 6.3 对话分支 / 重新生成

- 新分支复制逻辑 block table，而不是复制所有 K/V；
- 共享 blocks 的 `ref_count += 1`；
- 分支首次修改共享的 partial block 时执行 Copy-on-Write；
- 完整且不可变的 blocks 可以一直共享。

### 6.4 编辑历史消息

1. 重新 tokenize 编辑后的上下文；
2. 找到新旧 token IDs 的最长公共前缀；
3. 只复用公共前缀中的完整 blocks；
4. 修改点之后的 KV 全部失效并重新 prefill。

因为后续 token 的 K/V 依赖之前所有 token，不能只重算被编辑的那一句。

### 6.5 session 结束

- session 对所有 blocks 的 `ref_count -= 1`；
- partial 或不可复用 block 可以立即归还 free pool；
- 完整 prefix block 可进入 ref=0 的 cached 状态；
- 内存紧张时再由 LRU/LFU/cost-aware 策略淘汰。

## 7. 显存不足时怎么办

建议按以下顺序回答：

1. **Admission control**：请求进入运行队列前先估计所需 blocks，避免执行到一半才 OOM；
2. 回收已结束 request 的不可缓存 blocks；
3. 淘汰 `ref_count == 0` 的 inactive prefix blocks；
4. preempt 低优先级 sequence；
5. CPU swap，之后需要 PCIe/NVLink 搬回；
6. 直接丢弃 KV，保留 token IDs，未来重新 prefill；
7. 对上下文做 sliding window 或 summary compaction。

绝不能淘汰 `ref_count > 0` 的共享 block 而不更新所有引用者。

### 淘汰评分可以从简单到复杂

基础版本使用 LRU：

```text
victim = oldest(last_access_at), where ref_count == 0
```

生产版本可以使用 cost-aware score：

```text
value ≈ reuse_probability × recompute_tokens / block_bytes
```

长且常用的 system prompt 通常比短、一次性的用户后缀更值得保留。

## 8. 上下文管理策略

KV Cache 解决“重复计算”，不能解决无限增长的上下文。Context Manager 仍需管理：

- system prompt 和必要安全指令永久保留；
- 最近 N 轮原文保留；
- 较早内容摘要化；
- tool result 可提取结论后删除原始大 payload；
- RAG 文档只保留本轮真正需要的片段；
- 超过窗口时执行 sliding window 或 compaction；
- 每次改写上下文都增加 `context_version`，并使受影响后缀 KV 失效。

注意：摘要会改变 token 前缀，因此摘要位置之后的旧 KV 通常无法继续复用。

## 9. 一致性与安全边界

Prefix Cache 的 key 至少要区分：

- model weights/version；
- tokenizer/version；
- RoPE/position 配置；
- LoRA/adapter；
- 多模态输入及其预处理版本；
- 会影响 K/V 的 prompt 内容。

跨租户共享公共前缀可以提升命中率，但必须评估侧信道和数据隔离风险。保守方案是把 `tenant_id` 或 cache scope 放入 namespace；公共、审核过的 system prompt 可以单独进入 global scope。

## 10. 监控指标

- GPU KV 使用率、free/active/cached/swapped blocks；
- prefix hit tokens / prompt tokens；
- KV allocation failure 和 preemption 次数；
- eviction、swap-in/out bytes 和延迟；
- time to first token（TTFT）；
- inter-token latency（ITL）；
- prefill/decode tokens per second；
- recompute tokens；
- 每个 tenant/session 的 block 占用；
- context truncation/compaction 次数。

只看“cache hit request 数”不够，应优先观察 **命中的 token 数**，因为命中 4 tokens 和命中 4000 tokens 的价值完全不同。

## 11. 可选的最终回答缓存

最终回答缓存位于 KV Cache 之上：

```text
cache_key = hash(
    normalized_input,
    complete_context_or_context_version,
    model_version,
    sampling_parameters,
    tool_schema_version,
    retrieval_corpus_version,
)
```

只适合确定性较高、允许复用的任务。带实时数据、用户权限、随机采样或副作用工具调用的请求不能盲目返回旧答案。它优化的是整次请求；KV/prefix cache 优化的是模型 prefill 计算。

## 12. 面试中的两分钟完整回答

> 我不会给每个 session 分配一段连续显存，而是采用 paged KV cache。CPU 侧的 Context Manager 保存 messages、token IDs、context version 和摘要；GPU 侧由全局 Block Manager 预分配固定大小的物理 KV blocks。每个 session 只维护 logical-to-physical block table。
>
> 新请求到来后，我按完整 block 对 token prefix 做链式 hash 查找。命中就增加 ref count，只 prefill 未命中的后缀；完整 block 是不可变的，因此不同 session、公共 system prompt 和对话分支可以共享。最后一个 partial block 默认私有；如果分支共享了它，写入前执行 Copy-on-Write。
>
> 显存不足时先做 admission control，再淘汰 ref count 为 0 的 inactive prefix blocks。生产环境可以在 LRU 基础上考虑重计算成本和命中概率，也可以把低优先级 sequence swap 到 CPU，或者丢弃 KV 后根据 token IDs 重算。绝不把 KV 当成上下文的唯一存储。
>
> 历史消息被编辑时，我只复用最长公共前缀中的完整 blocks，修改点之后全部失效。模型、tokenizer、RoPE 或 adapter 变化时通过 namespace 整体隔离旧缓存。最后监控 prefix-hit tokens、GPU block 使用率、eviction、recompute、TTFT 和 ITL，验证缓存是否真正改善延迟和吞吐。

## 13. 面试官可能继续追问

### 为什么只复用完整 block？

完整 block 不可变，容易做 hash、引用计数和跨请求共享。partial block 会继续写入，直接共享会产生并发覆盖和一致性问题；可以私有化或使用 Copy-on-Write。

### 为什么不按 session 做 LRU？

一个 block 可能被多个 session 共享，session 级淘汰容易误删共享数据。物理 block 才是显存资源和引用计数的正确管理单位。

### KV Cache 能不能跨模型复用？

通常不能。K/V 是模型权重、层结构、位置编码和 adapter 共同计算出的中间状态。影响计算的任何配置改变都必须隔离 namespace。

### PagedAttention 解决了什么？

它通过 block table 让逻辑连续的 token 映射到不连续的物理显存，减少连续大块分配造成的外部碎片，并让 scheduler 能更灵活地分配、共享、回收和 swap KV blocks。

### Prefix cache 与 KV cache 有什么关系？

KV cache 是底层 K/V 张量；prefix cache 是跨请求查找和复用相同前缀 KV blocks 的管理机制。前者是数据，后者是索引、生命周期和复用策略。

## 14. 运行模拟代码

```bash
cd handle_code/python/LLM
python3 -m unittest -v test_kv_cache.py
```

核心文件：

- `kv_cache.py`：仅管理 metadata 的 Paged KV Cache 模拟；
- `test_kv_cache.py`：前缀复用、Copy-on-Write、编辑、LRU 和容量测试。

该模拟没有真实分配 GPU tensor，也没有执行 Attention；`PhysicalBlock` 只是实际 K/V block 的占位元数据。真实系统中，模型执行器负责写入预分配的 K/V tensor，Block Manager 只负责物理位置和生命周期。
