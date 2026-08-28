"""最小 GRPO Demo：用组内相对奖励学习选择高奖励动作。

环境只有一个状态：
    action=0 -> reward=0
    action=1 -> reward=1

GRPO 不需要 Critic；它在一组采样结果内标准化奖励来得到 advantage。
运行：uv run --with torch --with numpy python GRPO.py
"""

import torch
from torch.distributions import Categorical


def group_advantages(rewards, eps=1e-8):
    """GRPO：用组内均值和标准差计算相对 advantage。"""
    return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + eps)


def grpo_loss(new_log_prob, old_log_prob, advantage, clip_eps=0.2):
    """GRPO 使用与 PPO 相同形式的 clipped policy loss。"""
    ratio = torch.exp(new_log_prob - old_log_prob)
    objective1 = ratio * advantage
    objective2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
    return -torch.min(objective1, objective2).mean()


def train():
    torch.manual_seed(0)

    # 两个 logits 经 softmax 后，表示选择两个动作的概率。
    logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([logits], lr=0.1)

    for update in range(30):
        # 1. 针对同一个问题，一次采样一组答案（这里答案就是动作）。
        old_policy = Categorical(logits=logits.detach())
        actions = old_policy.sample((64,))
        old_log_prob = old_policy.log_prob(actions)

        # 2. 给答案打分，并用组内相对表现计算 advantage，无需 Critic。
        rewards = actions.float()
        advantages = group_advantages(rewards)

        # 3. 在同一组数据上做多次 clipped policy update。
        for _ in range(4):
            new_policy = Categorical(logits=logits)
            new_log_prob = new_policy.log_prob(actions)
            loss = grpo_loss(new_log_prob, old_log_prob, advantages)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if update % 5 == 0 or update == 29:
            probability = torch.softmax(logits, dim=0)[1].item()
            print(f"update={update:2d}, P(action=1)={probability:.3f}")


if __name__ == "__main__":
    train()
