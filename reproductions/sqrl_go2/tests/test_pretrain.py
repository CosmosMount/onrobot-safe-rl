import numpy as np
import torch

from reproductions.sqrl_go2.algo.buffers import (
    ReplayBuffer, SafetyReplayBuffer, Transition)
from reproductions.sqrl_go2.algo.pretrain import SQRLPretrainer
from reproductions.sqrl_go2.algo.sac import SACConfig, VanillaSAC
from reproductions.sqrl_go2.algo.safety_critic import (
    SafetyCriticConfig, SafetyCriticLearner)
from reproductions.sqrl_go2.algo.safety_policy import SafetyPolicy


def transition(*, terminated=False, truncated=False):
    return Transition(
        np.zeros(2, np.float32), np.zeros(1, np.float32), 0.0,
        np.ones(2, np.float32), float(terminated), terminated, truncated)


def test_task_and_safety_data_never_cross_replay_boundaries():
    torch.manual_seed(1)
    sac = VanillaSAC(SACConfig(
        observation_dim=2, action_dim=1, hidden_dims=(8, 8)))
    safety = SafetyCriticLearner(SafetyCriticConfig(
        observation_dim=2, action_dim=1, hidden_dims=(8, 8)))
    task = ReplayBuffer(10, (2,), 1, 1)
    safe = SafetyReplayBuffer(2, 2)
    policy = SafetyPolicy(sac.actor, safety.critic, 0.1, 2, sac.device)
    trainer = SQRLPretrainer(
        sac, safety, task, safe, policy, batch_size=1,
        minimum_task_transitions=99, minimum_safety_transitions=1,
        task_steps_per_cycle=1, safety_trajectories_per_cycle=1,
        safety_updates_per_cycle=1)
    trainer.observe(transition(truncated=True))
    assert trainer.phase == "safety"
    assert len(task) == 1 and len(safe) == 0
    trainer.observe(transition(truncated=True))
    assert trainer.phase == "task"
    assert len(task) == 1 and len(safe) == 1
