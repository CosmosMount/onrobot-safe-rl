from __future__ import annotations

import unittest

import jax
import numpy as np

from jaxrl.agents.action_masking import select_masked_action
from jaxrl.agents.sac.droq.learner import DroQLearner
from jaxrl.agents.safety_critic import SafetyCritic
from jaxrl.env.specs import BoxSpec


class ActionMaskingTest(unittest.TestCase):

    def test_selects_bounded_candidate_and_reports_metrics(self):
        obs_spec = BoxSpec((5,), np.float32)
        action_spec = BoxSpec(
            (2,), np.float32, np.full(2, -1), np.full(2, 1))
        agent = DroQLearner.create(
            0, obs_spec, action_spec, hidden_dims=(16, 16), num_qs=2,
            critic_dropout_rate=None)
        safety = SafetyCritic.create(1, 5, 2, hidden_dims=(16, 16))
        action, info, rng = select_masked_action(
            agent, safety, np.zeros(5, dtype=np.float32),
            jax.random.PRNGKey(2), num_candidates=16,
            epsilon_safe=0.3, action_noise_std=0.5)
        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))
        self.assertGreaterEqual(info['mask_rejected_fraction'], 0.0)
        self.assertLessEqual(info['mask_rejected_fraction'], 1.0)
        self.assertGreaterEqual(info['selected_Q_safe'], 0.0)
        self.assertLessEqual(info['selected_Q_safe'], 1.0)
        self.assertEqual(np.asarray(rng).shape, (2,))

    def test_rejects_invalid_candidate_count(self):
        with self.assertRaises(ValueError):
            select_masked_action(
                None, None, np.zeros(1), jax.random.PRNGKey(0),
                num_candidates=0)


if __name__ == '__main__':
    unittest.main()
