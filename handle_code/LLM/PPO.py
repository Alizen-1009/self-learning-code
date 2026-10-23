"""最小 PPO Demo：让策略学会在两个动作中选择奖励更高的动作。

环境只有一个状态：
    action=0 -> reward=0
    action=1 -> reward=1

运行：uv run --with torch --with numpy python PPO.py
"""

import torch
from torch.distributions import Categorical


def ppo_loss(new_log_prob, old_log_prob, advantage, clip_eps=0.2):
    """PPO 的 clipped policy loss。"""
    ratio = torch.exp(new_log_prob - old_log_prob)
    objective1 = ratio * advantage
    objective2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    return -torch.min(objective1, objective2).mean()


def train():
    torch.manual_seed(0)

    # 两个可学习的数，softmax 后就是选择两个动作的概率。
    logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([logits], lr=0.1)

    for update in range(30):
        # 1. 使用旧策略采样一批动作。
        old_policy = Categorical(logits=logits.detach())
        actions = old_policy.sample((64,))
        old_log_prob = old_policy.log_prob(actions)

        # 2. action=1 得 1 分，action=0 得 0 分。
        rewards = actions.float()
        advantage = rewards - rewards.mean()  # 最简单的 baseline

        # 3. 同一批数据重复更新几次，PPO clip 限制策略变化幅度。
        for _ in range(4):
            new_policy = Categorical(logits=logits)
            new_log_prob = new_policy.log_prob(actions)
            loss = ppo_loss(new_log_prob, old_log_prob, advantage)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if update % 5 == 0 or update == 29:
            probability = torch.softmax(logits, dim=0)[1].item()
            print(f"update={update:2d}, P(action=1)={probability:.3f}")


if __name__ == "__main__":
    train()
