from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from reproductions.sqrl_go2.algo.safety_policy import SafetyPolicy


class Actor(nn.Module):
    def sample(self, observation, deterministic=False):
        del deterministic
        values = torch.tensor([[0.8], [0.2], [-0.3]], dtype=torch.float32)
        return values[:observation.shape[0]], torch.zeros(observation.shape[0])


class Risk(nn.Module):
    def forward(self, observation, action):
        del observation
        return action[:, 0].abs()


def preview(actions):
    return SimpleNamespace(
        requested=actions.copy(), critic_actions=actions.copy(),
        q_targets=actions.copy())


def test_rejection_sampling_returns_first_accepted_candidate():
    policy = SafetyPolicy(Actor(), Risk(), epsilon=0.25,
                          max_candidates=3, device=torch.device("cpu"))
    result = policy.select(np.zeros(2, np.float32), preview)
    np.testing.assert_allclose(result.critic_action, [0.2])
    assert result.accepted
    assert result.candidate_count == 2


def test_all_rejected_falls_back_to_minimum_risk_only():
    policy = SafetyPolicy(Actor(), Risk(), epsilon=0.1,
                          max_candidates=3, device=torch.device("cpu"))
    result = policy.select(np.zeros(2, np.float32), preview)
    np.testing.assert_allclose(result.critic_action, [0.2])
    assert result.no_safe_candidate
    assert result.candidate_count == 3
