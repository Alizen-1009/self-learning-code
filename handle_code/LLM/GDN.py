import torch


def gdn_step(q, k, v, state, alpha, beta):
    """单 token、单 head 的 Gated DeltaNet 核心运算。

    q, k: [Dk]；v: [Dv]；state: [Dk, Dv]。
    alpha: 衰减系数，范围 [0, 1]；beta: 更新强度，范围 [0, 1]。
    输入 q/k 已完成需要的归一化；这里省略投影、卷积和输出门控。
    alpha 是实际衰减系数；若输入是 log gate g，先计算 alpha = exp(g)。
    """
    state = alpha * state  # [Dk, Dv]：遗忘部分旧记忆
    error = v - k @ state  # [Dv]：目标值 - 旧记忆的预测
    delta = beta * error  # [Dv]：控制写入强度
    state = state + torch.outer(k, delta)  # [Dk, Dv]：外积更新记忆
    out = q @ state  # [Dv]：从新记忆中读取
    return out, state


if __name__ == "__main__":
    q = torch.tensor([1.0, 0.0])
    k = torch.tensor([1.0, 0.0])
    v = torch.tensor([2.0, 4.0])
    state = torch.eye(2)

    out, state = gdn_step(q, k, v, state, alpha=0.5, beta=0.5)
    print("out:", out)  # tensor([1.2500, 2.0000])
    print("state:", state)  # tensor([[1.2500, 2.0000], [0.0000, 0.5000]])
    # 连续处理序列时，把返回的 state 传给下一个 token。
