from __future__ import annotations

import unittest
import tempfile

import jax
import numpy as np

from jaxrl.agents.sac.droq.learner import DroQLearner
from jaxrl.agents.safety_critic import (SafetyCritic,
                                        binary_prediction_metrics)
from jaxrl.env.specs import BoxSpec
from jaxrl.data.replay_buffer import ReplayBuffer
from learner.checkpoint import (restore_training_snapshot,
                                save_training_snapshot)


class SafetyCriticTest(unittest.TestCase):

    def test_update_does_not_modify_reward_agent(self):
        obs_spec = BoxSpec(shape=(5,), dtype=np.float32)
        action_spec = BoxSpec(shape=(2,), dtype=np.float32,
                              low=np.full(2, -1.0),
                              high=np.full(2, 1.0))
        agent = DroQLearner.create(
            0, obs_spec, action_spec, hidden_dims=(16, 16),
            num_qs=2, critic_dropout_rate=None)
        safety = SafetyCritic.create(
            1, 5, 2, hidden_dims=(16, 16), learning_rate=1e-3)
        batch = {
            'observations': np.ones((8, 5), dtype=np.float32),
            'actions': np.zeros((8, 2), dtype=np.float32),
            'next_observations': np.ones((8, 5), dtype=np.float32) * 2,
            'n_step_next_observations': (
                np.ones((8, 5), dtype=np.float32) * 3),
            'unsafe_labels': np.asarray(
                [0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32),
            'n_step_unsafe_labels': np.asarray(
                [0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32),
            'n_step_masks': np.asarray(
                [1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32),
            'n_step_steps': np.full(8, 8, dtype=np.int32),
            'future_failure_labels': np.asarray(
                [0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32),
            'behavior_noise_std': np.full(8, 0.35, dtype=np.float32),
            'source_ids': np.asarray(
                [0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int8),
        }
        actor_before = jax.tree_util.tree_map(
            np.asarray, agent.actor.params)
        updated, info = SafetyCritic.update(safety, agent.actor, batch)
        self.assertTrue(np.isfinite(float(info['safety_critic_loss'])))
        self.assertTrue(np.isfinite(float(info['safety_future_bce'])))
        self.assertTrue(np.isfinite(float(info['safety_td_loss'])))
        self.assertFalse(all(
            np.array_equal(a, b) for a, b in zip(
                jax.tree_util.tree_leaves(safety.critic.params),
                jax.tree_util.tree_leaves(updated.critic.params))))
        for before, after in zip(
                jax.tree_util.tree_leaves(actor_before),
                jax.tree_util.tree_leaves(agent.actor.params)):
            np.testing.assert_array_equal(before, np.asarray(after))

    def test_prediction_metrics(self):
        metrics = binary_prediction_metrics(
            [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(metrics['Q_safe_AUROC'], 1.0)
        self.assertEqual(metrics['Q_safe_average_precision'], 1.0)
        self.assertTrue(np.isfinite(metrics['Q_safe_calibration_ece']))

    def test_checkpoint_round_trip(self):
        obs_spec = BoxSpec(shape=(5,), dtype=np.float32)
        action_spec = BoxSpec(shape=(2,), dtype=np.float32)
        agent = DroQLearner.create(
            0, obs_spec, action_spec, hidden_dims=(8, 8), num_qs=2)
        safety = SafetyCritic.create(1, 5, 2, hidden_dims=(8, 8))
        replay = ReplayBuffer(obs_spec, action_spec, 10)
        expected = safety.predict(
            np.ones((2, 5), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_training_snapshot(
                tmp, agent=agent, replay_buffer=replay,
                safety_critic=safety, step=4)
            restored = restore_training_snapshot(
                path, agent=agent,
                replay_buffer=ReplayBuffer(obs_spec, action_spec, 10),
                safety_critic=SafetyCritic.create(
                    99, 5, 2, hidden_dims=(8, 8)))
            actual = restored['safety_critic'].predict(
                np.ones((2, 5), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    unittest.main()
