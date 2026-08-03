# FLA 与 FlashInfer SM100/SM103：GDN Prefill 全流程对比

> 范围：只比较 **Gated Delta Rule / GDN prefill forward**。不讨论 decode、backward、SM90/SM120 Context Parallel。
>
> 对应快照：FLA `0b346347379476548be0678ec597f9e14f148bf7`；FlashInfer `08ddfbcd2e89b2f4b68391825817909e30d445e2`。

## 1. 先给结论

两者执行同一套 GDN 语义：

```text
旧 state 衰减
→ K 从 state 预测 V
→ beta 缩放 prediction residual
→ residual 写回 state
→ Q 从更新后的 state 读取 output
```

主要区别不在数学，而在调度：

```text
FLA
  所有 chunks 并行准备 A/W/U
  → 单独做 chunk-state recurrence
  → 所有 chunks 并行计算 output
  → 中间张量经过 global memory

FlashInfer SM100/SM103
  一个 persistent CTA 负责一个 (sequence, head)
  → state 加载到 TMEM
  → CTA 内顺序处理全部 64-token chunks
  → 每个 chunk 的 7 组 GEMM、inverse、output、state update 全部融合
  → state 一直留在 TMEM
```

一句话概括：

> **FLA 用多 kernel 和中间张量换取通用性、训练能力及 chunk 维并行；FlashInfer 用单个架构特化 persistent kernel 换取 state 常驻片上、少 launch 和少 HBM 流量。**

---

## 2. 共同的 GDN token 语义

以 K-first state `S:[K,V]` 表示，一个 token 的逻辑步骤是：

```text
1. S_decay = alpha_t * S
2. prediction = k_t^T * S_decay
3. residual = beta_t * (v_t - prediction)
4. S = S_decay + k_t * residual^T
5. o_t = scale * q_t^T * S
```

其中：

- `alpha_t`：forget gate，通常在 `(0,1]`；
- `beta_t`：delta update rate；
- `prediction`：旧 state 对当前 value 的预测；
- `residual`：尚未被 state 表达的信息；
- `k_t residual^T`：rank-1 state correction。

两边都以 64-token chunk 为主要 prefill tile，但对 gate 的 API 表示不同：

| 项目 | FLA | FlashInfer prefill |
|---|---|---|
| gate public contract | 默认接收 log-space `g`；也可接收 raw gate 并在内部激活 | 接收乘法形式 `alpha`/forget gate，kernel 内取 log/cumprod |
| beta | 可接收 sigmoid 后 beta；也可在 wrapper 中 sigmoid | 接收已准备的 float32 update gate |
| chunk 内表示 | `G = cumsum(log(alpha))`，内部常转为 base-2 | `cumsumlog = cumsum(log(alpha))`，再得到 `cumprod` |

因此对比输出前必须先统一 gate/beta 语义，不能只看变量都叫 `g`、`beta`。

---

# Part A：FLA GDN Prefill Forward

## 3. FLA 总调用图

入口：

```text
fla.ops.gated_delta_rule.chunk_gated_delta_rule
```

核心 forward：

```text
ChunkGatedDeltaRuleFunction.forward
│
├─ 0. optional L2Norm(Q/K) + sigmoid(beta)
├─ 1. prepare chunk indices（varlen）
├─ 2. gate activation + chunk-local cumsum
├─ 3. intra-chunk KKT + triangular solve → A
├─ 4. A × betaV / A × betaK_g → U/W
├─ 5. chunk-to-chunk state recurrence → H/R/final_state
└─ 6. output kernel → O
```

主要入口文件：

- [`chunk.py`](../code/fla/fla/ops/gated_delta_rule/chunk.py)
- [`gate.py`](../code/fla/fla/ops/gated_delta_rule/gate.py)
- [`chunk_fwd.py`](../code/fla/fla/ops/gated_delta_rule/chunk_fwd.py)
- [`wy_fast.py`](../code/fla/fla/ops/gated_delta_rule/wy_fast.py)
- [上游 `common/chunk_delta_h.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py)
- [上游 `common/chunk_o.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_o.py)

## 4. FLA Step 0：输入预处理

位置：`ChunkGatedDeltaRuleFunction.forward()`。

可选操作：

```python
q, q_rstd = l2norm_fwd(q)
k, k_rstd = l2norm_fwd(k)
beta = fused_beta_sigmoid(beta_raw)
```

目的：

1. 对 Q/K 做 head-dimension L2 normalize；
2. 把 beta logits 转换为 update rate；
3. `allow_neg_eigval=True` 时可用 `2*sigmoid(beta)`；
4. 保存 backward 所需的 norm/beta 信息。

这一步不是 chunk 数学本身，而是模型输入 contract 的一部分。

## 5. FLA Step 1：准备 varlen chunk 索引

如果提供 `cu_seqlens`：

```python
chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
```

目的：

- 把 packed variable-length sequence 映射为 `(sequence_id, local_chunk_id)`；
- 防止 chunk 跨越 sequence 边界；
- tail chunk 使用 mask 处理不足 64 的 token。

固定长度时可直接用 `T/chunk_size` 推导 chunk 坐标。

## 6. FLA Step 2：gate 激活与 chunk-local cumsum

位置：`gdn_gate_chunk_cumsum()`。

若 `use_gate_in_kernel=True`，先计算 log-space gate：

```text
g = -exp(A_log) * softplus(raw_g + dt_bias)
```

然后每个 chunk 内：

```text
G[t] = g[0] + g[1] + ... + g[t]
```

代码会乘 `RCP_LN2`，后续用：

```text
exp2(G_i - G_j)
```

而不是直接 `exp`。

输入/输出：

```text
input : raw_g 或 log_g [B,T,HV]
output: cumulative G     [B,T,HV]，float32
```

并行度：

```text
num_chunks × B × HV
```

存储：`G` 被写入 global memory，供后续多个 kernel 使用。

## 7. FLA Step 3：KKT + gate/beta + triangular solve

位置：`chunk_gated_delta_rule_fwd_intra()`。

### 7.1 计算 chunk 内 key Gram

```text
KKT = K @ K^T                 [C,C]
```

其中：

```text
KKT[i,j] = k_i^T k_j
```

它表示 token `j` 的 state update 会被 token `i` 的 key 读取多少。

### 7.2 加入三种权重

```text
strict lower mask：只保留 j<i
relative decay   ：exp(G_i-G_j)
beta row scale   ：beta_i
```

得到 residual 依赖矩阵：

```text
A_lower[i,j] = beta_i * exp(G_i-G_j) * (k_i^T k_j),  j<i
```

### 7.3 求 unit-lower triangular inverse

```text
Ainv = (I + A_lower)^(-1)
```

`chunk_size=64` 时，FLA 使用融合 Triton kernel：

```text
chunk_gated_delta_rule_fwd_kkt_solve_kernel
```

它把 64×64 矩阵拆成：

```text
4 个 16×16 diagonal blocks
6 个 16×16 off-diagonal blocks
```

执行：

1. register 中计算 10 个 lower blocks；
2. 加 gate/beta/mask；
3. diagonal block forward substitution；
4. block merge；
5. 将完整 `Ainv` 写出。

输出：

```text
A: [B,T,HV,C]
```

逻辑上每个 chunk 是 `[C,C]`，只是沿 token 维打包保存。

并行度：

```text
num_chunks × B × HV
```

所有 chunks 的 A 都可独立计算。

## 8. FLA Step 4：生成 WY representation 的 W/U

位置：`recompute_w_u_fwd()`。

计算：

```text
U = Ainv @ (beta * V)
W = Ainv @ (beta * exp(G) * K)
```

含义：

```text
U：当前 chunk 自己希望写入的 value correction
W：incoming state 会对这些 correction 产生多少预测
R = U - W @ S_in：真正需要写入的 residual
```

输出：

```text
W: [B,T,HV,K]
U: [B,T,HV,V]
```

并行度：

```text
num_chunks × B × HV
```

存储：W/U 都写入 global memory。

为什么函数名叫 `recompute_w_u_fwd`：训练 backward 可以保留较少中间量，并在需要时根据 A/K/V/beta/G 重算 W/U。

## 9. FLA Step 5：chunk-to-chunk state recurrence

位置：`chunk_gated_delta_rule_fwd_h()`。

grid 主要按：

```text
sequence × value-head × V-tile
```

每个 program 持有一块 FP32 state accumulator，并在内部循环 sequence 的所有 chunks：

```python
for chunk in range(num_chunks):
    save chunk initial state H[chunk]
    prediction = W[chunk] @ state
    R = U[chunk] - prediction
    save R as v_new
    state *= chunk_total_decay
    state += K_end[chunk].T @ R
```

输入：

```text
K, W, U, G, initial_state
```

输出：

```text
H          : [B,num_chunks,HV,K,V] 或转置布局
v_new / R  : [B,T,HV,V]
final_state: [N,HV,K,V]（可选）
```

这里是 FLA forward 中真正的 chunk recurrence：

```text
state_0 → state_1 → state_2 → ...
```

但是可以同时并行：

- 不同 sequence；
- 不同 head；
- 不同 V tile。

存储特点：

- 每个 chunk 的 initial state H 写到 global memory；
- residual `v_new` 写到 global memory；
- 供最后 output kernel 和 backward 使用。

## 10. FLA Step 6：独立 output kernel

位置：`chunk_fwd_o()` / `chunk_fwd_kernel_o()`。

每个 program 对应：

```text
V-tile × chunk × batch-head
```

执行两部分：

### 10.1 Inter-chunk output

```text
O_inter = exp(G) * Q @ H_chunk
```

`H_chunk` 是 Step 5 保存的该 chunk initial state。

### 10.2 Intra-chunk output

```text
QK = Q @ K^T
QK *= exp(G_i-G_j)
QK *= causal_lower_mask(j<=i)
O_intra = QK @ R
```

最终：

```text
O = scale * (O_inter + O_intra)
```

输出：

```text
O: [B,T,HV,V]
```

所有 chunks 的 output 可以并行，因为 chunk initial state H 与 residual R 已经由 Step 5 准备完成。

## 11. FLA forward 的 global intermediates

典型需要物化：

| 中间量 | 大致 shape | 用途 |
|---|---|---|
| G | `[B,T,HV]` | relative decay |
| Ainv | `[B,T,HV,C]` | triangular representation / backward |
| W | `[B,T,HV,K]` | state prediction coefficient |
| U | `[B,T,HV,V]` | chunk-local corrected value |
| H | `[B,num_chunks,HV,K,V]` | 每个 chunk initial state |
| R / v_new | `[B,T,HV,V]` | output 与 state update |

优点：阶段清晰、chunk 并行度高、适合 autograd/backward。

代价：多个 kernel launch 和较多 HBM 中间流量。

---

# Part B：FlashInfer SM100/SM103 GDN Prefill Forward

## 12. FlashInfer 总调用图

Public API：

```text
flashinfer.gdn_prefill.chunk_gated_delta_rule
```

SM100/SM103 dispatch：

```text
chunk_gated_delta_rule_sm100
→ GatedDeltaNetChunkedKernel
```

文件：

- [`gdn_prefill.py`](../code/flashinfer/flashinfer/gdn_prefill.py)
- [`blackwell/gdn_prefill.py`](../code/flashinfer/flashinfer/gdn_kernels/blackwell/gdn_prefill.py)
- [`blackwell/gated_delta_net_chunked.py`](../code/flashinfer/flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py)
- [`blackwell/gated_delta_net_tile_scheduler.py`](../code/flashinfer/flashinfer/gdn_kernels/blackwell/gated_delta_net_tile_scheduler.py)

当前 SM100/SM103 路径的主要限制：

- CUDA 13+；
- `head_size == 128`；
- chunk size 固定 64；
- Q/K/V/output 为 fp16 或 bf16；
- state 可为 fp32、bf16、fp16 或支持的 fp8；
- 主要面向 forward inference；
- 支持 packed varlen、state pool indices、final state 和 checkpoints。

## 13. FlashInfer Step 0：host adapter 与 compile cache

`chunk_gated_delta_rule_sm100()` 做：

1. 检查 Q/K/V/state dtype 和 shape；
2. 判断 GQA/GVA/head mapping；
3. 根据静态配置建立 compile-cache key；
4. 将 PyTorch tensors 转为 CuTe tensors；
5. 标记 token 维动态、head/head_dim 静态；
6. 准备 TMA descriptor workspace；
7. 第一次调用时 `cute.compile()`，之后复用编译结果；
8. 启动 persistent kernel。

这一步不执行 GDN 数学，主要负责 specialization 与低开销 dispatch。

注意：当前 SM100/SM103 adapter 没有把 public API 的 `use_qk_l2norm_in_kernel` 传给这个 kernel。不要假设该 specialization 自动完成 Q/K L2Norm；调用方必须按当前 upstream contract/tests 准备输入。

## 14. FlashInfer Step 1：persistent tile scheduling

一个 logical tile 是：

```text
(sequence, output-head)
```

persistent grid：

```text
grid.x = min(num_sequences * num_output_heads, max_active_clusters)
```

每个 CTA：

1. 从 scheduler 领取一个 `(sequence, head)`；
2. 顺序处理该 tile 的全部 64-token chunks；
3. 完成后再领取下一个 tile。

chunk 不在 grid 中：

```text
并行维度：sequence × head
串行维度：同一 sequence/head 的 chunks
```

这样做是为了让 recurrent state 跨 chunk 常驻 TMEM。

## 15. FlashInfer Step 2：分配 SMEM/TMEM 与 warp specialization

一个 CTA 使用 12 个 warps（384 threads），逻辑角色为：

| 角色 | 主要工作 |
|---|---|
| compute group 0（4 warps） | gate pairwise、KK/QK epilogue、hierarchical inverse |
| compute group 1（4 warps） | residual、state decay/update、output epilogue |
| MMA issuer warp | 发射 KK/QK/KS/QS/NV 等矩阵指令 |
| TMA QKV warp | 通过 TMA 加载 Q/K/V，预取后续 chunk |
| 第二 MMA issuer | 发射 state/output 相关 GEMM |
| epilogue warp | gate/beta load 与 output store |

片上主要对象：

- Q/K/V stages：SMEM；
- gate cumsum/cumprod/beta：register + SMEM；
- A inverse / corrected values：复用 SMEM；
- KK/QK/KS/QS/NV accumulators：TMEM；
- recurrent state：TMEM；
- output staging：SMEM。

这些角色通过 mbarrier pipeline 重叠执行，所以后面的“Step 3～11”是**逻辑步骤**，硬件上并非完全串行。

## 16. FlashInfer Step 3：加载 initial state

若有 `initial_state`：

```text
GMEM state → registers → FP32 TMEM state
```

若无 initial state：TMEM state 初始化为 0。

对于 `128×128` FP32 state：

```text
128 * 128 * 4 B = 64 KiB / head
```

关键点：该 state 在 CTA 处理完整条 sequence 时一直保留在 TMEM，不会每个 chunk 都回写 global memory。

## 17. FlashInfer Step 4：TMA 加载当前/后续 chunk

每个 chunk 加载：

```text
Q: [64,128]
K: [64,128]
V: [64,128]
gate: [64]
beta: [64]
```

K/Q/V 使用多 stage buffer；kernel 会在计算当前 chunk 时预取后续 chunk，尽量重叠：

```text
GMEM → TMA → SMEM
```

varlen tail 由 bounded descriptor / predicate 零填充处理。

## 18. FlashInfer Step 5：gate cumsum 与 pairwise transfer

FlashInfer prefill 接收乘法 forget gate `alpha`。kernel 内计算：

```text
cumsumlog[t] = sum(log(alpha[0:t]))
cumprod[t] = exp(cumsumlog[t])
T_pairwise[i,j] = cumprod[i] / cumprod[j]
```

也就是：

```text
T_pairwise[i,j] = exp(G_i-G_j)
```

用途：

- 对 KK residual dependency 加相对 decay；
- 对 QK output score 加相对 decay；
- 把 incoming state 衰减到 chunk 内各位置；
- 把每个 residual 衰减到 chunk 末。

与 FLA 的区别：这些值只服务当前 chunk，主要留在 register/SMEM，不生成全序列 G tensor。

## 19. FlashInfer Step 6：GEMM 1 — KK

```text
W_kk = K @ K^T             [64,64]
```

作用：构造 chunk 内 residual dependency。

compute group 0 随后应用：

- strict lower mask；
- `T_pairwise` relative decay；
- beta row scaling。

对应 FLA：`chunk_scaled_dot_kkt`。

## 20. FlashInfer Step 7：GEMM 2 — QK

```text
W_qk = Q @ K^T             [64,64]
```

作用：准备 chunk 内 output score。

随后应用：

- causal lower mask（包含对角线）；
- relative decay；
- scale/beta 所需变换。

对应 FLA 最终 `chunk_fwd_o` 中的 QK，但 FlashInfer 在同一 persistent kernel 内提前计算并流水化。

## 21. FlashInfer Step 8：hierarchical triangular inverse

根据 weighted lower KK 构造：

```text
Ainv = (I + weighted_lower_KK)^(-1)
```

实现不是通用 dense inverse，而是分层 block inverse：

```text
8×8 → 16×16 → 32×32 → 64×64
```

compute group 0 负责 inverse，并将结果放入可被后续 MMA 消费的 SMEM layout。

对应 FLA：`KKT + solve_tril`。

区别：FlashInfer 不把 Ainv 写到 global memory，Ainv buffer 随后可复用。

## 22. FlashInfer Step 9：GEMM 3 — K × state

```text
KS = K @ S_prev             [64,128]
```

作用：计算 incoming recurrent state 对当前 64 个 token value 的预测。

随后结合 V、beta、gate/inverse，得到真正需要写入的 corrected value/residual。

对应 FLA：

```text
W @ state
R = U - W @ state
```

区别：FlashInfer 直接读取 TMEM state 做 GEMM，不显式物化 global W。

## 23. FlashInfer Step 10：GEMM 4 — Q × state

```text
QS = Q @ S_prev             [64,128]
```

作用：计算 incoming state 对当前 chunk output 的 inter-chunk contribution。

之后按 chunk 内累计 gate 对每一行缩放。

对应 FLA output kernel 中：

```text
exp(G) * Q @ H_chunk
```

## 24. FlashInfer Step 11：GEMM 5 — corrected/new value

```text
NV = Ainv @ value/residual-related input
```

逻辑作用：应用 triangular inverse，消除 chunk 内 residual 的逐 token 依赖，得到后续 output/state update 可消费的 corrected values。

对应 FLA 的组合：

```text
U = Ainv @ betaV
W = Ainv @ betaK_g
R = U - W @ state
```

FlashInfer 中这些步骤与 `KS`、gate/beta epilogue 融合，不生成完整的 global U/W/R tensors。

## 25. FlashInfer Step 12：GEMM 6 — intra-chunk output

```text
O_intra = weighted_QK @ corrected_value
```

作用：当前 chunk 内 token 对 output 的贡献。

对应 FLA：

```text
weighted causal QK @ v_new
```

## 26. FlashInfer Step 13：合并并写 output

```text
O = O_intra + gate_scaled(QS)
```

其中：

- `QS`：更早 chunks / initial state 的贡献；
- `O_intra`：当前 chunk 内的贡献。

output 经 SMEM staging，由 epilogue warp 写回 global memory。

对应 FLA：

```text
O = O_inter + O_intra
```

区别：FLA 等 state scan 完成后另起 output kernel；FlashInfer 在当前 chunk 内立即产生并存储 output。

## 27. FlashInfer Step 14：GEMM 7 — state update

```text
dS = K^T @ corrected_residual
S_next = chunk_total_decay * S_prev + dS
```

`S_next` 继续保存在 TMEM，然后 CTA 处理下一个 chunk：

```text
chunk 0 → TMEM S1
chunk 1 → TMEM S2
chunk 2 → TMEM S3
...
```

对应 FLA：`chunk_gated_delta_rule_fwd_h()`。

最关键差异：

- FLA 保存每个 chunk initial state H；
- FlashInfer 只在 CTA/TMEM 内传递 state，正常情况下只在 sequence 开始/结束访问 global state。

## 28. FlashInfer Step 15：checkpoint 与 final state

如果开启 checkpoint：按 `checkpoint_every_n_tokens` 将指定 chunk 边界 state 写到 global checkpoint buffer。

sequence 结束时：

```text
TMEM final state → registers/SMEM → output_state GMEM
```

若使用 `state_indices`：initial/final state 从 state pool 的指定 slot 读取/写回，避免调用方 gather/scatter。

---

# Part C：逐步骤对应

## 29. 数学步骤对照表

| GDN 任务 | FLA | FlashInfer SM100/SM103 |
|---|---|---|
| gate 表示 | log-space 或 raw→activation | multiplicative alpha |
| chunk cumsum | 独立 Triton kernel，G 写 HBM | CTA 内 register/SMEM |
| KK | Triton intra kernel | GEMM 1 |
| relative decay + beta | KKT kernel epilogue | compute group 0 epilogue |
| triangular inverse | 融合 KKT/solve，A 写 HBM | hierarchical inverse，片上 |
| U/W | 独立 Triton kernel，写 HBM | 与 KS/NV 流程融合，不全局物化 |
| state prediction | state kernel 的 `W @ state` | GEMM 3 `K @ TMEM state` |
| inter output | 独立 output kernel | GEMM 4 `Q @ TMEM state` |
| intra output | output kernel QK@R | GEMM 2 + GEMM 6 |
| state update | 独立 state recurrence kernel | GEMM 7，state 留 TMEM |
| chunk initial states | H 写 global | 不写，TMEM 内传递 |
| final state | state kernel 写 global | sequence 结束时写 global |

## 30. 调度时间线对照

### FLA

```text
所有 chunks：gate cumsum                         [Kernel 1]
        ↓ HBM G
所有 chunks：KKT + solve → A                    [Kernel 2]
        ↓ HBM A
所有 chunks：A×betaV / A×betaK_g → U/W          [Kernel 3]
        ↓ HBM U/W
每个 seq/head：循环 chunks，生成 H/R/final       [Kernel 4]
        ↓ HBM H/R
所有 chunks：QH + QKR → output                  [Kernel 5]
```

### FlashInfer SM100/SM103

```text
Persistent CTA(sequence, head)
    load state → TMEM
    for chunk:
        TMA load/prefetch Q/K/V/gate/beta
        gate cumsum / pairwise transfer
        KK + QK
        triangular inverse
        KS + QS
        corrected value
        intra + inter output → GMEM
        state update → TMEM
    store final state → GMEM
```

## 31. 并行度对照

| 阶段 | FLA | FlashInfer |
|---|---|---|
| gate/intra preparation | `B×H×num_chunks` | CTA 内 chunk loop |
| chunk state recurrence | `B×H×V_tiles`，内部循环 chunks | `B×H` persistent CTA，内部循环 chunks |
| output | `B×H×num_chunks×V_tiles` | CTA 内产生当前 chunk output |
| 主要 grid 并行来源 | batch、head、chunk、tile | sequence、head、persistent work queue |

FLA 更容易利用长 sequence 的 chunk 数增加并行度。FlashInfer 更依赖 `B×H` 足够大。

## 32. 内存流量对照

### FLA 主要代价

```text
G / A / W / U / H / R
```

需要在多个 kernel 间经过 global memory。

### FlashInfer 主要优势

```text
state、Ainv、accumulators、局部 corrected values
```

尽可能保留在 TMEM/SMEM/register，只写最终 output、final state 和可选 checkpoints。

### FlashInfer 的代价

- 一个 CTA 使用大量 SMEM/TMEM/register；
- occupancy 通常较低；
- kernel 非常复杂；
- chunk 维不能直接增加 CTA 数；
- `B×H` 较小时可能并行度不足。

## 33. 为什么 FLA 不直接照搬 FlashInfer？

FLA 需要：

- training backward；
- 更多 K/V dimensions；
- 多种 GPU/backend；
- autotune；
- varlen/GVA/不同 layout；
- 保存或重算 backward intermediates。

将所有步骤融合进单个 Blackwell kernel，会显著增加寄存器/SMEM/TMEM压力和 backward 复杂度。

## 34. 为什么 FlashInfer 不直接照搬 FLA？

FlashInfer serving 更关心：

- forward inference latency/throughput；
- state pool 原地读写；
- 少 kernel launch；
- 少 HBM intermediate；
- 固定 `D=128` 的架构 specialization；
- checkpoint 和 serving cache 集成。

因此让 state 常驻 TMEM 并融合完整 chunk pipeline 更符合其目标。

---

# Part D：阅读与验证

## 35. 推荐阅读顺序

### FLA

```text
1. chunk_gated_delta_rule_fwd                总 pipeline
2. gdn_gate_chunk_cumsum                     gate
3. chunk_gated_delta_rule_fwd_intra          KKT/solve/WU
4. recompute_w_u_fwd                         W/U
5. chunk_gated_delta_rule_fwd_h              state recurrence
6. chunk_fwd_o                               output
```

阅读时只追踪：

```text
G → Ainv → W/U → H/R → O/final_state
```

### FlashInfer SM100/SM103

```text
1. gdn_prefill.py                            dispatch/contract
2. blackwell/gdn_prefill.py                  compile cache/adapter
3. gated_delta_net_tile_scheduler.py         CTA work mapping
4. gated_delta_net_chunked.py 文件顶部       7 GEMM 总览
5. kernel()                                  pipeline/warp roles
6. compute_group_0_pair                      gate/KK/QK/inverse
7. compute_group_1_chunk                     residual/state/output
8. epilogue_warp                             output store
```

阅读时只追踪：

```text
GMEM Q/K/V → SMEM
state → TMEM
KK/QK/Ainv/KS/QS/NV/dS
output → GMEM
state → next chunk / final GMEM
```

## 36. Correctness 对比前必须固定

- FLA 的 gate 是 raw、log-space，还是已激活？
- FlashInfer 的 alpha 是否和 `exp(FLA g)` 一致？
- beta 是 logits 还是 sigmoid 后值？
- Q/K 是否已 L2 normalize？
- state 是 K-first 还是 V-first？
- initial state dtype 是否一致？
- tail chunk 和 varlen 边界是否一致？
- output scale 是否一致？

否则 output 不一致不代表 kernel 错误。

## 37. 最终心智模型

```text
FLA：先“准备每个 chunk 的数学表示”，再做 state scan 和 output。

FlashInfer：让一个 CTA 拿着 state，把每个 chunk 从输入到 output/state 一次做完。
```

两者不是谁“更正确”，而是在不同约束下选择了不同的数据驻留与并行策略。
