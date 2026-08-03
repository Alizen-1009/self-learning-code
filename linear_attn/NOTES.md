# Teaching Notes

- 用户选择完整路线 D：数学推导 → PyTorch reference → Triton kernel → profile/优化 → 修改生产级 GDN/KDA prefill kernel。
- 中文优先；所有数学需要落到 shape、代码行和可验证练习。
- 当前能力未完整说明，暂按熟悉 Python/PyTorch 与 CUDA 基础、LA chunk 推导未系统掌握设计。
- 第一阶段顺序：vanilla causal LA → recurrent state → 单 chunk 分解 → scalar decay → delta/WY；之后进入 KDA。
- 用户在第 1 课后追问 causal mask M，说明下一课应显式区分 0/1 mask M 与带相对衰减的加权矩阵 D。
- 不因“看懂讲解”记录已掌握；必须由练习答案或代码产出提供证据后再写 learning record。
