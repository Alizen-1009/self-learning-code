# ReplaySSM Resources

## Knowledge

- [Blog: "ReplaySSM: Cache SSM Inputs, Not State" — Tri Dao](https://tridao.me/blog/2026/replayssm/)
  **本主题的 primary source。** 目前没有对应 arXiv 论文，所有公式与数字都以这篇为准。章节顺序：
  1. State Space Models (SSMs) — 2. Three challenges in SSM decoding — 3. The idea: don't store the state, cache the recent inputs — 4. One change, three answers — 5. Algorithm — 6. Evaluation — 7. Conclusion and the future — 8. Appendix。
  Use for: history route 推导、output-only 公式、GDN 的 `u` 替换、cache/flush 判据、traffic 与 FLOP 账本、benchmark 数字。
  （[dao-lab.ai 镜像](https://dao-lab.ai/blog/2026/replayssm/) 内容相同。）

- [Code: Johnny-Liou/ReplaySSM](https://github.com/Johnny-Liou/ReplaySSM.git)
  官方参考实现。Use for: 核对公式到代码的映射、看 ring buffer 的实际数据布局。（算法层阶段只做对照，不深读。）

- [RFC: Porting ReplaySSM to SGLang — sgl-project/sglang#28511](https://github.com/sgl-project/sglang/issues/28511)
  生产集成视角。关键事实：新 kernel `fused_recurrent_gdn_replayssm_decode`；改动落在 `mem_cache/memory_pool.py`、`attention/linear/gdn_backend.py`、`attention/mamba/mamba2_metadata.py`；开关 `--enable-gdn-replayssm`，buffer 长度 `--gdn-replayssm-cache-len`；`write_pos` 游标每次 forward 只前进一次以兼容 CUDA graph；Phase 1 只做 GDN，KDA 的 gating 未实现。microbench kernel 1.2–1.5×，但 Qwen3.5 这类 MoE 端到端只有 ~2.3% TPOT 提升（dense GDN 模型收益应更大）。
  Use for: 理解 flush 边界为什么要和 radix cache 的 snapshot 边界对齐；理解「kernel 加速 ≠ 端到端加速」。

- [Paper: "Snakes and Ladders: Accelerating SSM Inference with Speculative Decoding" — Wu et al., PMLR v262](https://proceedings.mlr.press/v262/wu24a.html)
  ReplaySSM 之前的 baseline 思路（每个 draft token 存 state 快照）。Use for: 对照理解 ReplaySSM 省掉的到底是什么。

## Gaps

- 缺少 ReplaySSM 的 **backward / training** 视角材料（博客只讲 inference）。若后续要碰训练侧再搜。
- 缺少 KDA decode 侧的公开讨论（官方与 SGLang 都标记未实现）。
- 尚未核实官方 repo 的实际文件结构（本机 clone 受工作区边界限制，待在允许的目录内拉取）。

## Wisdom (Communities)

- [sgl-project/sglang Issues & Discussions](https://github.com/sgl-project/sglang/issues)
  ReplaySSM 集成的实际讨论就发生在这里（#28511）。Use for: 提出实现层疑问、看 reviewer 关心什么。
- [flash-linear-attention (FLA) repo discussions](https://github.com/fla-org/flash-linear-attention/discussions)
  GDN / linear attention kernel 作者与使用者聚集地。Use for: 递推形式、数值精度、kernel 取舍的同行校验。

> 用户尚未表态是否愿意参与社区。首次需要「智慧」层反馈（例如实现取舍的判断）时再确认。
