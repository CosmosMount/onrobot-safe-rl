from types import SimpleNamespace

import numpy as np
import torch

from reproductions.sqrl_go2.algo.buffers import (
    ReplayBuffer, SafetyReplayBuffer, Transition)
from reproductions.sqrl_go2.algo.checkpoint import (
    load_pretrain_checkpoint, save_pretrain_checkpoint)
from reproductions.sqrl_go2.algo.finetune import SafetyLagrange
from reproductions.sqrl_go2.algo.pretrain import SQRLPretrainer
from reproductions.sqrl_go2.algo.sac import SACConfig, VanillaSAC
from reproductions.sqrl_go2.algo.safety_critic import (
    SafetyCriticConfig, SafetyCriticLearner)
from reproductions.sqrl_go2.algo.safety_policy import SafetyPolicy
from reproductions.sqrl_go2.algo.target import TargetTrainer


def models(seed):
    torch.manual_seed(seed)
    sac = VanillaSAC(SACConfig(
        observation_dim=3, action_dim=1, hidden_dims=(8, 8)))
    safety = SafetyCriticLearner(SafetyCriticConfig(
        observation_dim=3, action_dim=1, hidden_dims=(8, 8)))
    return sac, safety


def preview(actions):
    actions = np.asarray(actions, np.float32)
    return SimpleNamespace(
        requested=actions, critic_actions=actions,
        q_targets=actions)


def rollout_transition(step, action):
    observation = np.asarray([step % 5, 0.0, 1.0], np.float32)
    next_observation = np.asarray([(step + 1) % 5, 0.0, 1.0], np.float32)
    done = step % 5 == 4
    # Sparse natural failure on every second completed episode.
    failure = done and (step // 5) % 2 == 1
    return Transition(
        observation, np.asarray(action, np.float32).reshape(1),
        float(action[0]), next_observation, float(failure),
        terminated=failure, truncated=done and not failure)


def test_pretrain_checkpoint_clones_into_all_target_loops(tmp_path):
    sac, safety = models(1)
    task = ReplayBuffer(100, (3,), 1, 1)
    safe = SafetyReplayBuffer(3, 2)
    policy = SafetyPolicy(sac.actor, safety.critic, 0.1, 3, sac.device)
    pretrainer = SQRLPretrainer(
        sac, safety, task, safe, policy, batch_size=2,
        minimum_task_transitions=2, minimum_safety_transitions=2,
        task_steps_per_cycle=5, safety_trajectories_per_cycle=1,
        safety_updates_per_cycle=1)
    for step in range(20):
        observation = np.asarray([step % 5, 0.0, 1.0], np.float32)
        decision = pretrainer.decide(observation, preview)
        pretrainer.observe(rollout_transition(step, decision.critic_action))
    assert len(task) > 0 and len(safe) > 0

    checkpoint = tmp_path / "pretrain.pt"
    save_pretrain_checkpoint(checkpoint, sac, safety, {"step": 20})
    transferred_actors = []
    for index, branch in enumerate(("sac_transfer", "sqrl_mask", "sqrl_full")):
        target_sac, target_safety = models(10 + index)
        load_pretrain_checkpoint(
            checkpoint, target_sac,
            None if branch == "sac_transfer" else target_safety, branch)
        transferred_actors.append(
            [value.clone() for value in target_sac.actor.state_dict().values()])
        target_policy = (
            None if branch == "sac_transfer" else SafetyPolicy(
                target_sac.actor, target_safety.critic, 0.1, 3,
                target_sac.device))
        lagrange = (
            SafetyLagrange(0.0, 3e-4, target_sac.device)
            if branch == "sqrl_full" else None)
        target = TargetTrainer(
            target_sac, ReplayBuffer(20, (3,), 1, index), branch=branch,
            batch_size=2, minimum_transitions=2,
            safety=None if branch == "sac_transfer" else target_safety,
            policy=target_policy, lagrange=lagrange, epsilon=0.1)
        for step in range(3):
            observation = np.asarray([step, 0.0, 1.0], np.float32)
            decision = target.decide(observation, preview)
            metrics = target.observe(rollout_transition(step, decision.critic_action))
        assert "sac/actor_loss" in metrics
        if branch == "sqrl_full":
            assert "sqrl/nu" in metrics
    for branch_actor in transferred_actors[1:]:
        for reference, candidate in zip(transferred_actors[0], branch_actor):
            torch.testing.assert_close(reference, candidate)
