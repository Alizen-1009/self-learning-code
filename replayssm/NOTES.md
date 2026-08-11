# Teaching Notes — replayssm

## 用户背景（承接 `../linear_attn`）

- 已完成 9 课：causal LA → scalar decay → delta rule/WY → gated delta chunk → GDN prefill 执行图 → B300 recurrent baseline → C64 Triton chunk pipeline → CuTe DSL persistent kernel → capstone。
- 已掌握证据：Triton 与 CuTe 版本在 B300 上 correctness 全 PASS（output rel-L2 至 `4.2e-3`，state rel-L2 至 `3.2e-3`）。
- **未接触**：decode 侧调度、投机解码、state/KV cache 管理、ring buffer。这是本工作区的全部新增面。

## 教学偏好（沿用并已验证）

- 中文优先；数学必须落到 shape、代码行、可算的数字。
- 反感符号堆砌与"一个工程问题拆成多课公式"。偏好自顶向下：先给完整 pipeline 与账本，再按需补推导。
- HTML 代码块需要明显的 Python 语法配色与高对比度，不能单一前景色（`assets/code-highlight.js` 已满足）。
- 不因"看懂讲解"记录已掌握；必须有练习答案或代码产出作为证据才写 learning record。
- **符号预算（2026-08-11 重写第 1 课后确立，见 `learning-records/0001`）**：每课 header 声明符号数，>12 就拆课；新符号首次出现必须当场承担推理，只用一次的一律移交 reference；正文禁用内联下标，下标只进公式框。能写成数字的写成数字。
  - 补充（见 `learning-records/0003`）：**引入新符号前先 `grep` 全工作区确认未被占用**（第 3 课的 `L` 撞了 buffer 容量，已改 `C`）。预算可以破，但必须在讲义里写明破了、为什么、代价是什么。
- **命名对照**（三方易撞，固定如下）：T×T 耦合矩阵 = `C`（博客 `A_{s,s'}`）· 衰减门 = `a`（博客 `A`/`e^g`）· buffer 容量 = `L`（不动）· 第 5 课 Amdahl 符号 = `f` / `x` / `E`（已 grep 确认未撞名；`s` 被第 3 课投机下标占用，故局部加速用 `x` 不用 `s`）。

## 本工作区决策

- 用户选择 **「先建立算法层理解」**，明确本阶段不写 kernel。第 1 课起以推导 + 账本 + 检索练习为主。
- 用户开场即要求两件事：ReplaySSM 与 GDN 的关系、学习路线。两者已在会话中回答；GDN 的 `u` 替换正式教学放在第 2 课。
- `assets/` 直接复用 `../linear_attn/assets/` 的 `course.css`、`quiz.js`、`code-highlight.js`，保证两个工作区视觉与交互一致。**`course.css` 与 linear_attn 逐字节相同，永不在此修改** —— 本工作区的新样式一律进独立组件表。
- 本工作区新增组件：`assets/ledger.css`（账本视觉语言：字节预算条 `.budget`、路线对比 `.routes`、账本表 `.ledger`、三答案卡 `.answers`、旁注 `.aside`、支点公式 `.equation.is-hero`、形状判定 `.drill`）、`assets/ledger-calc.js`（`data-ledger-calc`，d/n/h 三滑块搬运账）、`assets/shape-drill.js`（`data-shape-drill`，每项 `data-ok` + `data-why`，点击即刻反馈）、`assets/spec-calc.js`（`data-spec-calc`，T/h 投机开销账）、`assets/flush-calc.js`（`data-flush-calc`，d/n/L/T 的 flush 周期与 L 两侧约束）、`assets/amdahl-calc.js`（`data-amdahl-calc`，f/x 双滑块的端到端加速账，含时间分解条与上限）。后续课程优先复用这六个。
- `ledger.css` 另含状态机图 `.fsm`（`.fsm-step` / `.is-guard` / `.is-flush` / `.fsm-branch`）与不变量卡 `.invariants`，第 5 课若要画 Amdahl 分解可复用 `.ledger` 与 `.budget`。
- ⚠️ **`flush-calc.js` 的分工不要合并**：搬运账固定按非投机稳态算，`T` 只驱动下界。混进去会得到 8×–20× 的假收益（见 `learning-records/0004`）。
- 两个 reference 卡都要 link `ledger.css`（公式卡已用 `.ledger`/`.invariants`/`.is-hero`）。
- 三角矩阵可视化直接复用 `course.css` 的 `.matrix-table` + `.diagonal`/`.history`/`.future` —— 那是 linear_attn 为 prefill 衰减矩阵 `D` 设计的，在这里刚好把「同一个 UT transform」的连续性看出来。
- **`code/verify_replayssm_identities.py`**：讲义所有代数断言的可执行证据（fp64，容差 1e-13，含三条「必须失败」的反例测试：照搬 `v` 展开、违反 flush 判据、Amdahl 区间错配）。**规矩：先让脚本 PASS，再写进讲义。**第 1 课那句错话就是因为没走这一步。

## 路线图（algorithm-first）

| # | 课程 | 单一收获 | 状态 |
|---|------|----------|------|
| 1 | 一个矩阵，从来不需要存在（`0001-decode-byte-ledger.html`） | 能说出 decode 单步收支（128 KiB 换 512 B），并写出结合律支点 | 已重写交付 2026-08-11 |
| 2 | 修一个被破坏的形状（`0002-gdn-pseudo-value.html`） | 能推出 `u = β(v − α·Sk)`，对应到 WY 修正项，并说明代价**不是带宽**而是串行纠缠 | 已交付 2026-08-11 |
| 3 | 一次三角求解，砍断串行链（`0003-triangular-solve-speculative.html`） | 能分离出 `R` 与 `C`、写出 `(I+C)U = R`，并说清收益是「压缩」而非「消灭」串行。已补上 `w(i→j)` 箭头记号与 `2Th+T²` 账本 | 已交付 2026-08-11 |
| 4 | 三个不变量（`0004-cache-flush-invariants.html`） | 能凭记忆画出状态机、说出三条不变量「破了会怎样」、从两侧夹出 `L`。已补 `L* = √(2dn/(d+n))` | 已交付 2026-08-11 |
| 5 | 账本 vs 实测：为什么 MoE 端到端只有 2.3%（`0005-amdahl-ledger-vs-measured.html`） | 能用 `f = (1−1/E)/(1−1/x)` 反推占比、解释 kernel↔端到端落差、区分 Amdahl 管辖的时间轴与不受压制的显存/解锁轴 | 已交付 2026-08-11 |

**算法层五课已收官。**之后若用户想动手，再开 kernel 实现分支（先 profile 目标模型的 `f` → Triton decode kernel → B300 benchmark），届时需更新 MISSION 的 Out of scope。

## 教学轴：数学 vs 簿记（第 1 课后追问引出）

用户读完第 1 课后问「主要是省写入的 HBM 吗，那怎么保证计算正确性」。第一半推对了（省的正好是写那一半）。这暴露了一条第 1 课没有覆盖的轴：

- **数学侧是恒等变换**，不是近似。最容易误解的点是「窗口不是截断」—— 窗口之前的全部历史精确活在 `S_0` 里，和 sliding-window attention 完全不同。这一条要在第 2 课开头再强调一次。
- **风险全在簿记**：`h + 2T > L` 是正确性不变量而非性能旋钮（判错 → ring buffer 覆盖仍需的输入 → 静默算错）；`write_pos` 每 forward 只前进一次；flush 边界必须对齐 radix cache snapshot 边界。**第 4 课围绕这三个不变量设计。**
- **浮点侧**：buffer 内 token 的连续舍入次数比 baseline 少（衰减由前缀和一次算出），可能更准；但 `pre` 前缀和有 cancellation / 下溢风险，flush 顺带把范围夹在 ≤ `L` 个 token 内。此结论是推理，未实测，第 5 课或 kernel 分支需要用 recurrent baseline 验证。

用户对「恒等 vs 近似」的区分反应很快，可以把 ZPD 再往上调一档。

## 待处理：linear_attn 09-gdn-capstone 答案有问题

用户提交的 `recurrent_gdn_step` 无法通过。题目要求「only Python lists」，`state: [V][K]`、`q, k: [K]`、`v: [V]`，返回 `output, new_state`。当前答案有三处问题：

1. 用了 numpy 风格的逐元素运算符处理嵌套 list —— `state * alpha` 对 list 和 float 会直接 TypeError。
2. 语义错：`prediction` 应是 `state @ k` → `[V]`；state 更新应是 `state + beta * outer(residual, k)`，当前缺了外积，维度对不上。
3. 返回顺序与 docstring 相反（写成了 `return state, out`）。

`schedule_counts` 未核对。**下次会话优先处理这个**：它是 GDN milestone 的证据来源，没过就不能记 learning record。

## 待办 / 悬空问题

- 重写第 1 课时 `tridao.me` 与 `dao-lab.ai` 均被网络策略拦截，**无法重新核对博客原文**。第 1、2 课只复用了先前会话已抽取并写入 reference 的事实，未新增任何未核实断言；本课程自己补的账（buffer 读取字节、GDN 的算术翻倍）均在讲义中显式标注，且经脚本验证。下次能联网时回查三点：① `A` 的参数化（对角/标量 —— **已不再阻塞**，见 learning-record 0002）；② 博客是否讨论过 buffer 读取开销；③ 博客怎么称呼 `u`，以及它归类 state-and-output 的理由是否与第 2 课的推导一致。

- 官方 repo 未能在本机 clone（Bash 受工作区边界限制，`/tmp` 不可写）。需要时改到 `replayssm/code/` 下拉取。
- ~~`A` 的具体参数化（对角/标量）~~ **已不再阻塞**：第 2 课判定练习证明对角衰减同样可展开（偏差 `4e-16`），结论不变。仍待回查原文确认。
- **回查清单（等能访问 tridao.me 时一次性做）**：① 博客怎么称呼 `u`；② 投机复杂度 `T(T+h)` 算一条还是两条读出路径（决定它与本课程 `2Th+T²` 谁是原意）；③「为什么是 `2T` 而不是 `T`」是否有原文说明（第 4 课的解释是重建）；④ 博客是否讨论过 buffer 读取开销。
- **未实测的推理**（需 recurrent baseline 在长序列上验证，留给第 5 课或 kernel 分支）：flush 把衰减前缀和跨度夹在 ≤ `L` 内，可能顺带改善数值稳定性。
