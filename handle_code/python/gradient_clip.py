import math
import torch


@torch.no_grad()
def clip_grad_norm_(parameters, max_norm=1.0):
    """按全局 L2 范数原地裁剪梯度，返回裁剪前的范数。

    在 loss.backward() 之后、optimizer.step() 之前调用。
    支持单进程下多个设备上的稠密实数梯度；不做跨进程归约。
    """
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("max_norm 必须是有限的非负数")
    grads = [p.grad for p in parameters if p.grad is not None]

    # 把所有梯度看成一个长向量：norm = sqrt(sum(g_i ** 2))。
    # 转 FP32 计算，每个设备上的平方和取回 CPU 后相加。
    total_norm = math.sqrt(sum(g.float().square().sum().item() for g in grads))
    if not math.isfinite(total_norm):
        raise ValueError("梯度范数出现 NaN 或 Inf")

    scale = min(1.0, max_norm / (total_norm + 1e-6))
    for grad in grads:
        grad.mul_(scale)
    return total_norm


# 使用示例：
# loss.backward()
# norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
# optimizer.step()
