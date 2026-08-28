import unittest

import torch

from GRPO import grpo_loss, group_advantages


class GRPOTest(unittest.TestCase):
    def test_advantage_is_relative_to_the_group(self):
        rewards = torch.tensor([0.0, 1.0])

        advantages = group_advantages(rewards)

        torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0]))

    def test_clips_an_overly_large_policy_update(self):
        new_log_prob = torch.log(torch.tensor([2.0]))
        old_log_prob = torch.log(torch.tensor([1.0]))
        advantage = torch.tensor([1.0])

        loss = grpo_loss(new_log_prob, old_log_prob, advantage, clip_eps=0.2)

        self.assertAlmostEqual(loss.item(), -1.2, places=6)


if __name__ == "__main__":
    unittest.main()
