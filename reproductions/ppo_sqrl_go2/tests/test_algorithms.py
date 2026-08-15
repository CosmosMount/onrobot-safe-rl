from __future__ import annotations

import numpy as np
import pytest
import torch

from reproductions.ppo_sqrl_go2.buffers import (
    SafetyTransition, TaskRollout, VectorRecentSafetyBuffer)
from reproductions.ppo_sqrl_go2.dual import ProjectedDual, frozen_qsafe_penalty
from reproductions.ppo_sqrl_go2.masking import select_masked_actions
from reproductions.ppo_sqrl_go2.mjlab import target_order_action
from reproductions.ppo_sqrl_go2.qsafe import (
    SafetyCriticConfig, SafetyQNetwork, safety_bellman_target)


def test_task_rollout_rejects_masked_source():
    rollout = TaskRollout(1, 2, 3, 5, 4, 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="masked safety"):
        rollout.add(
            actor_observation=torch.zeros(2, 3),
            critic_observation=torch.zeros(2, 5),
            qsafe_observation=torch.zeros(2, 4), action=torch.zeros(2, 1),
            log_probability=torch.zeros(2), value=torch.zeros(2),
            reward=torch.zeros(2), done=torch.zeros(2, dtype=torch.bool),
            mean=torch.zeros(2, 1), std=torch.ones(2, 1), source="safety")


def test_vector_safety_buffer_retains_complete_trajectories():
    buffer = VectorRecentSafetyBuffer(
        2, 3, 2, 4, 3, 1, device=torch.device("cpu"), seed=3)
    kwargs = {
        "observation": torch.zeros(2, 4),
        "policy_observation": torch.zeros(2, 3),
        "action": torch.zeros(2, 1),
        "next_observation": torch.ones(2, 4),
        "next_policy_observation": torch.ones(2, 3),
    }
    assert buffer.add_batch(**kwargs, cost=torch.tensor([0, 0]),
                            terminated=torch.tensor([0, 0]),
                            truncated=torch.tensor([0, 0])) == 0
    assert buffer.add_batch(**kwargs, cost=torch.tensor([1, 0]),
                            terminated=torch.tensor([1, 0]),
                            truncated=torch.tensor([0, 1])) == 2
    assert buffer.trajectory_count == 2
    assert buffer.total_falls == 1
    assert buffer.sample(4)["next_policy_observation"].shape == (4, 3)


def test_bellman_cost_and_boundaries():
    result = safety_bellman_target(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.8, 0.8, 0.8]), 0.7)
    assert torch.allclose(result, torch.tensor([1.0, 0.56, 0.56]))


class _Risk(torch.nn.Module):
    def forward(self, observation, action):
        return action[:, 0]


def test_mask_uses_first_safe_and_minimum_fallback():
    actions = torch.tensor([
        [[0.7], [0.1], [0.0]],
        [[0.8], [0.9], [0.6]],
    ])
    selected = select_masked_actions(
        torch.zeros(2, 4),
        sample_policy_actions=lambda observation, count: actions,
        project_for_critic=lambda value: value,
        critic=_Risk(), epsilon=0.2, candidates=3)
    assert torch.equal(selected.policy_action[:, 0], torch.tensor([0.1, 0.6]))
    assert torch.equal(selected.attempts, torch.tensor([2, 3]))
    assert torch.equal(selected.no_safe, torch.tensor([False, True]))


def test_critic_action_uses_real_normalized_projection():
    action = torch.tensor([[2.0, -2.0] + [0.0] * 10])
    projected = target_order_action(action)
    assert float(projected.max()) == 1.0
    assert float(projected.min()) == -1.0


def test_frozen_qsafe_passes_gradient_only_to_action():
    critic = SafetyQNetwork(SafetyCriticConfig(
        observation_dim=4, action_dim=1, hidden_dims=(4,)))
    critic.requires_grad_(False)
    action = torch.zeros(3, 1, requires_grad=True)
    penalty, _ = frozen_qsafe_penalty(
        critic, torch.zeros(3, 4), action,
        epsilon=0.1, dual_value=1.0)
    penalty.backward()
    assert action.grad is not None
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_dual_is_projected():
    dual = ProjectedDual(learning_rate=0.5, initial_value=0.1)
    assert dual.update(-1.0) == 0.0
    assert dual.update(2.0) == 1.0
