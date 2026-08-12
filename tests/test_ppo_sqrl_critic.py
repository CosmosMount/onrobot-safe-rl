from __future__ import annotations

import unittest

import torch

from rl.qsafe.ppo_sqrl_critic import (
    PpoSqrlCriticConfig,
    PpoSqrlSafetyCritic,
    sqrl_bellman_target,
)


class PpoSqrlCriticTest(unittest.TestCase):
    def test_action_model_changes_with_action_and_state_model_does_not(self):
        history = torch.randn(3, 5, 46)
        first = torch.zeros(3, 12)
        second = torch.ones(3, 12)
        action = PpoSqrlSafetyCritic(PpoSqrlCriticConfig(mode="action"))
        state = PpoSqrlSafetyCritic(PpoSqrlCriticConfig(mode="state_only"))
        self.assertFalse(torch.equal(action(history, first), action(history, second)))
        torch.testing.assert_close(state(history, first), state(history, second))

    def test_bellman_cost_and_timeout_boundaries(self):
        target = sqrl_bellman_target(
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([True, False, False]),
            torch.tensor([False, True, False]),
            torch.tensor([0.8, 0.8, 0.8]),
            gamma_safe=0.7,
        )
        torch.testing.assert_close(target, torch.tensor([1.0, 0.0, 0.56]))


if __name__ == "__main__":
    unittest.main()
