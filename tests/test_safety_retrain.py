from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.transition import Transition, zero_costs
from jaxrl.agents.sac.droq.learner import DroQLearner
from jaxrl.agents.safety_critic import SafetyCritic
from jaxrl.data.replay_buffer import ReplayBuffer
from jaxrl.data.safety_replay import SafetyReplayManager
from jaxrl.env.specs import BoxSpec
from learner.checkpoint import save_training_snapshot
from learner.safety_dataset import (merge_episode_artifacts,
                                    save_safety_episode_artifact,
                                    split_episode_artifacts)
from learner.safety_retrain import run_safety_retrain
from train.config import TrainConfig


def _transition(value: float, *, unsafe: bool = False,
                done: bool = False) -> Transition:
    return Transition(
        observation=np.full(4, value, dtype=np.float32),
        requested_action=np.full(2, value * 0.1, dtype=np.float32),
        projected_action=np.full(2, value * 0.1, dtype=np.float32),
        executed_q_target=np.full(2, value * 0.1, dtype=np.float32),
        reward=float(value),
        costs=zero_costs(),
        next_observation=np.full(4, value + 1, dtype=np.float32),
        terminated=done and unsafe,
        truncated=done and not unsafe,
        unsafe_label=unsafe,
        near_failure_label=unsafe,
    )


def _episode_replay(values, *, unsafe_last: bool) -> SafetyReplayManager:
    replay = SafetyReplayManager(
        recent_capacity=50, failure_capacity=50, boundary_capacity=50,
        recovery_capacity=20, all_capacity=100, failure_history=3,
        n_step=3, seed=3)
    for index, value in enumerate(values):
        last = index == len(values) - 1
        replay.insert(
            _transition(float(value), unsafe=last and unsafe_last, done=last),
            behavior_noise_std=0.4)
    return replay


class SafetyRetrainTest(unittest.TestCase):

    def test_split_and_merge_by_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for episode, (seed, failure) in enumerate(
                    [(10, True), (10, False), (11, True), (12, False)]):
                replay = _episode_replay([0, 1, 2], unsafe_last=failure)
                paths.append(save_safety_episode_artifact(
                    tmp,
                    episode_index=episode,
                    safety_replay_state=replay.state_dict(),
                    metadata={
                        'episode_index': episode,
                        'rollout_seed': seed,
                        'outcome': 'failure' if failure else 'success',
                        'steps': 3,
                        'unsafe_steps': int(failure),
                        'episode_return': 3.0,
                        'action_noise_std': 0.4,
                        'safety_mask': True,
                        'safety_mask_epsilon': 0.15,
                        'source_checkpoint': 'ckpt.pkl',
                    }))
            train, val, held = split_episode_artifacts(
                paths, held_out_seeds={11})
            self.assertEqual(held, {11})
            self.assertEqual(len(val), 1)
            self.assertEqual(len(train), 3)
            merged = SafetyReplayManager(
                recent_capacity=200, failure_capacity=200,
                boundary_capacity=200, recovery_capacity=50,
                all_capacity=400, failure_history=3, n_step=3, seed=0)
            stats = merge_episode_artifacts(train, merged)
            self.assertEqual(stats['episodes_loaded'], 3)
            self.assertGreater(stats['replay_sizes']['all'], 0)

    def test_offline_retrain_writes_new_checkpoint(self):
        obs_spec = BoxSpec(shape=(4,), dtype=np.float32)
        action_spec = BoxSpec(
            shape=(2,), dtype=np.float32,
            low=np.full(2, -1.0, dtype=np.float32),
            high=np.full(2, 1.0, dtype=np.float32))
        agent = DroQLearner.create(
            0, obs_spec, action_spec, hidden_dims=(8, 8), num_qs=2,
            critic_dropout_rate=None)
        safety = SafetyCritic.create(
            1, 4, 2, hidden_dims=(8, 8), learning_rate=1e-3)
        replay = ReplayBuffer(obs_spec, action_spec, 32)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = save_training_snapshot(
                root / 'ckpt', agent=agent, replay_buffer=replay,
                safety_critic=safety, step=100)
            dataset = root / 'dataset'
            for episode, (seed, failure) in enumerate(
                    [(1, True), (2, False), (3, True), (4, False)]):
                ep = _episode_replay([0.0, 1.0, 2.0], unsafe_last=failure)
                save_safety_episode_artifact(
                    dataset,
                    episode_index=episode,
                    safety_replay_state=ep.state_dict(),
                    metadata={
                        'episode_index': episode,
                        'rollout_seed': seed,
                        'outcome': 'failure' if failure else 'success',
                        'steps': 3,
                        'unsafe_steps': int(failure),
                        'episode_return': 3.0,
                        'action_noise_std': 0.4,
                        'safety_mask': True,
                        'safety_mask_epsilon': 0.15,
                        'source_checkpoint': str(ckpt),
                    })
            out = root / 'out'
            cfg = TrainConfig(
                safety_recent_capacity=100,
                safety_failure_capacity=100,
                safety_boundary_capacity=100,
                safety_recovery_capacity=50,
                buffer_size=32,
                safety_failure_history=3,
                safety_critic_n_step=3,
                safety_critic_batch_size=8,
                safety_critic_hidden_dims=(8, 8),
                safety_critic_learning_rate=1e-3,
            )
            droq_cfg = {
                'hidden_dims': (8, 8),
                'num_qs': 2,
                'critic_dropout_rate': None,
            }
            rc = run_safety_retrain(
                cfg, droq_cfg,
                checkpoint=str(ckpt),
                dataset_dir=dataset,
                retrain_steps=5,
                held_out_seeds={4},
                save_dir=str(out),
                log_interval=5)
            self.assertEqual(rc, 0)
            snapshots = list(out.glob('training_snapshot_*.pkl'))
            self.assertEqual(len(snapshots), 1)
            self.assertTrue(
                (out / 'safety_retrain_report_000000000105.json').exists())


if __name__ == '__main__':
    unittest.main()
