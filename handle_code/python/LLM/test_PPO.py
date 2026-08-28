import unittest

import torch

from PPO import ppo_loss


class PPOLossTest(unittest.TestCase):
    def test_clips_an_overly_large_policy_update(self):
        new_log_prob = torch.log(torch.tensor([2.0]))
        old_log_prob = torch.log(torch.tensor([1.0]))
        advantage = torch.tensor([1.0])

        loss = ppo_loss(new_log_prob, old_log_prob, advantage, clip_eps=0.2)

        self.assertAlmostEqual(loss.item(), -1.2, places=6)


if __name__ == "__main__":
    unittest.main()
