# Mission: ReplaySSM —— 从算法层看懂 SSM decode 的重构

## Why

我已经能独立推导并实现 GDN 的 chunk prefill kernel（见 `../linear_attn`）。但 prefill 只是一半：真实推理服务的成本大头在 decode，而 decode 侧的 recurrent state 搬运、投机解码回滚、串行验证是三个我完全没碰过的瓶颈。ReplaySSM 用「cache 输入而非 cache state」一次性回答了这三个问题，我要先把这套算法层的账算清楚，才有资格谈 kernel 实现或生产集成。

## Success looks like

- 能在白纸上写出 summary route 与 history route 的等价推导，并说明 `(v k^T) q = v (k^T q)` 这一步结合律为什么是整个方法的支点。
- 能独立算出 decode 的**账本**：per head per step 的 state traffic 从 `8dn` 降到 `4dn`，state 构造 FLOPs 比值 `dn/(d+n)`，以及投机解码引入的 `T(T+h)` 二次项。
- 能解释 GDN 为什么必须 cache `u` 而不是 `v`，为什么它走 state-and-output route 而 Mamba-2 走 output-only route，并把 `u` 和我在 WY 表示里学的 delta-rule 修正项对应起来。
- 能推导投机验证的 `u_s = R_s - Σ_{s'<s} A_{s,s'} u_{s'}`，并说明为什么一次 `T×T` 三角求解能替掉串行循环。
- 能画出 cache / flush 状态机（含 `h + 2T > L` 判据、ring buffer 指针回滚），并说明 cache 长度 `L` 的取值上界由什么决定。

## Constraints

- 中文优先。每个结论必须落到公式、张量 shape 或可算的数字，不接受"大概更快"这类说法。
- 每课只解决一个紧凑问题，用检索练习而不是被动阅读建立长期记忆。
- 起点假设：GDN chunk 数学（含 WY / UT transform）、Triton chunk kernel、CuTe persistent kernel 已掌握；decode 侧调度、投机解码、KV/state cache 管理未系统接触。
- 一手材料以 Tri Dao 的博客为准，不使用参数化记忆补全公式。目前**没有对应 arXiv 论文**，博客即 primary source。

## Out of scope

- 本阶段不写 kernel。Triton / CuTe 实现、B300 benchmark 留到算法层账本算清之后。
- 不展开 SGLang / vLLM 的完整 scheduler 与 radix cache 设计；只在解释 flush 边界时点到为止。
- 不追 KDA 的 decode 变体（官方实现也还没做）。
