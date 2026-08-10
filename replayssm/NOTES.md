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

## 本工作区决策

- 用户选择 **「先建立算法层理解」**，明确本阶段不写 kernel。第 1 课起以推导 + 账本 + 检索练习为主。
- 用户开场即要求两件事：ReplaySSM 与 GDN 的关系、学习路线。两者已在会话中回答；GDN 的 `u` 替换正式教学放在第 2 课。
- `assets/` 直接复用 `../linear_attn/assets/` 的 `course.css`、`quiz.js`、`code-highlight.js`，保证两个工作区视觉与交互一致。若本工作区需要新组件（如 ring buffer 动画），新增到本目录的 `assets/`。

## 路线图（algorithm-first）

| # | 课程 | 单一收获 | 状态 |
|---|------|----------|------|
| 1 | Decode 的账本：summary route vs history route | 能写出 output-only 公式，并算出 `8dn→4dn` 与 `dn/(d+n)` | 已交付 |
| 2 | GDN 为什么必须 cache `u` | 把 `u` 对应到 WY 修正项，说明 state-and-output route 的必要性 | 待开 |
| 3 | 投机验证的 `T×T` 三角求解 | 从 `u_s = R_s − Σ A_{s,s'} u_{s'}` 推到一次三角求解 | 待开 |
| 4 | Cache / flush 状态机 | 画出含 `h+2T>L` 判据与指针回滚的完整状态机，定出 `L` 的上界 | 待开 |
| 5 | 账本 vs 实测：为什么 MoE 端到端只有 2.3% | 用 Amdahl 解释 kernel 加速与端到端加速的落差 | 待开 |

之后若用户想动手，再开 kernel 实现分支（Triton decode kernel → B300 benchmark），届时需更新 MISSION 的 Out of scope。

## 待办 / 悬空问题

- 官方 repo 未能在本机 clone（Bash 受工作区边界限制，`/tmp` 不可写）。需要时改到 `replayssm/code/` 下拉取。
- `pre_j = Σ_{i≤j} Δ_i` 与 `s_{j,t}` 的定义已从博客取得原文，但 `A` 的具体参数化（对角/标量）未在博客明确，第 2 课前需回查 Appendix。
