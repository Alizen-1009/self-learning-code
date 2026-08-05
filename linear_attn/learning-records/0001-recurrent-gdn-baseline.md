# Recurrent GDN baseline 与 GPU 映射已建立

用户已完成第 6 课，并通过连续追问厘清了 V-first state、Q/K/V 与 state 的关系、一个 CTA 负责的 V-row tile、warp lane 映射、state_in/state_out，以及 Triton `num_warps` / `num_stages` 的含义。后续课程可以假设其已掌握 recurrent GDN 的核心语义与 correctness baseline 结构，但尚未证明能独立修改 kernel 并定位错误。
