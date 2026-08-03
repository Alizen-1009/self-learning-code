# 01｜数学与核心概念

## 1. 统一记号

对单个 batch、单个 head，令：

- `q_t, k_t ∈ R^K`；
- `v_t, o_t ∈ R^V`；
- 状态 `S_t ∈ R^(K×V)`（K-first 表示）；
- 更新率 `β_t` 通常由 sigmoid 得到；
- `g_t` 是 log-space forget gate，因此实际衰减是 `exp(g_t)`。

不同项目可能把状态保存成 `[K,V]` 或转置后的 `[V,K]`。两种布局数学等价，读代码时必须先确认布局，不能只看变量名 `h`/`state`。

## 2. GDN：每个 head 一个标量衰减

GDN 的 gate 对一个 head 是标量。K-first 写法为：

```text
S̄_t = exp(g_t) · S_(t-1)
r_t = v_t - k_tᵀ S̄_t
S_t = S̄_t + β_t · k_t r_tᵀ
o_t = (scale · q_t)ᵀ S_t
```

含义：

1. 旧记忆先整体衰减；
2. 用旧状态预测当前 value；
3. 用预测误差 `r_t` 做一次 rank-1 delta-rule 更新；
4. query 从更新后的状态读取结果。

FLA 常见的原始 gate 激活为：

```text
g_t = -exp(A_log) · softplus(a_t + dt_bias)
β_t = sigmoid(b_t)
```

所以 `g_t ≤ 0`，`exp(g_t) ∈ (0,1]`。

最清楚的 reference：

- `code/fla/fla/ops/gated_delta_rule/naive.py`
- `code/flashinfer/tests/gdn/reference_delta_rule.py`

## 3. KDA：把标量衰减升级为 K 维向量

KDA 的核心变化不是 delta update，而是 **每个 key 维度有独立衰减**：

```text
g_t ∈ R^K
S̄_t = diag(exp(g_t)) · S_(t-1)       # K-first
```

其余递推不变：

```text
r_t = v_t - k_tᵀ S̄_t
S_t = S̄_t + β_t · k_t r_tᵀ
o_t = (scale · q_t)ᵀ S_t
```

若状态使用 V-first `[V,K]`，同一公式写成：

```text
S̄_t = S_(t-1) · diag(exp(g_t))
r_t = v_t - S̄_t k_t
S_t = S̄_t + β_t · r_t k_tᵀ
o_t = S_t (scale · q_t)
```

这正是 FlashKDA 与 FlashInfer KDA decode 中最常见的布局。

KDA 原始 gate 通常为：

```text
g_t = -exp(A_log) · softplus(f_t + dt_bias)   # 每个 K 维度不同
```

safe-gate 路径改成有界形式：

```text
g_t = lower_bound · sigmoid(exp(A_log) · (f_t + dt_bias))
```

当 `lower_bound=-5` 时，`g_t ∈ [-5,0)`。有界范围让 chunk 内指数计算更容易控制，也为 FlashKDA 的 16-token Tensor Core 路径提供数值前提。

最清楚的 reference：

- `code/fla/fla/ops/kda/naive.py`
- `code/flashkda/tests/torch_ref.py`（刻意模拟 kernel 的近似指令与精度）

## 4. GDN 与 KDA 的本质差异

| 项目 | GDN | KDA |
|---|---|---|
| forget gate 形状 | `[B,T,HV]` | `[B,T,HV,K]` |
| 状态衰减 | 整个矩阵乘标量 | 状态每个 K 列/行独立衰减 |
| 表达能力 | 较低，衰减便宜 | 较高，gate 带宽与计算更多 |
| recurrent decode | GEMV + rank-1 update | 仍是 GEMV + rank-1 update，但多 K 维 gate |
| chunk prefill | 可用块矩阵/WY 形式 | 可用块矩阵/WY 形式，但指数范围更难处理 |

因此优化时不能把 KDA 当成“多读一个 gate 的 GDN”；KDA 的逐维衰减会影响数据布局、chunk 内矩阵构造和数值稳定性。

## 5. 为什么区分 decode 与 prefill

### Decode / recurrent（通常 `T=1` 或很小）

每个 token 都要读写 `K×V` 状态，主体是 GEMV 和 rank-1 update。以 `K=V=128` 为例，每 head 的 bf16 状态为 32 KiB，fp32 为 64 KiB（只计算状态本身）。常见瓶颈是：

- 状态读写带宽；
- 小 batch 时并行度不足；
- gate、norm、beta 等额外 kernel launch；
- state layout 导致的不合并访问或转置。

### Prefill / chunk（`T` 很长）

逐 token recurrence 并行性差。chunk 算法把序列分块：

1. chunk 内构造下三角依赖矩阵；
2. 用三角求解或有限 Neumann 展开得到等价的 WY 表示；
3. chunk 内大量工作转为 GEMM，使用 Tensor Core；
4. 只在 chunk 之间保留状态递推。

FLA 默认常见 chunk 为 32/64；当前 FlashKDA v1 固定用 16。后者不是越小越好，而是在数值范围、16×16 逆、Tensor Core 映射和并行度之间做出的选择。

## 6. 数值语义必须先固定

比较 kernel 前至少固定这些契约：

- `g` 是 raw gate 还是已经在 log-space；
- `beta` 是 logits 还是 sigmoid 后的值；
- q/k 是否在 kernel 内做 L2 normalize；
- state 是 fp32 还是 bf16；
- state 是 `[K,V]` 还是 `[V,K]`；
- dense 还是 varlen；
- 是否使用 initial/final state；
- safe gate 与 `lower_bound` 是否启用。

只要其中一项不一致，输出差异就不能直接归因于“kernel 不正确”。

## 7. 第一组自测问题

读完 reference 后，先不看答案，自己说明：

1. 为什么 residual 必须用衰减后的状态计算？
2. GDN 标量 gate 在 KDA 公式中对应哪种特殊情况？
3. `[K,V]` 与 `[V,K]` 各自让哪一维连续？
4. decode 为什么通常不像 prefill 那样容易喂满 Tensor Core？
5. 为什么 chunk 越大，KDA 的指数数值范围越难控制？
