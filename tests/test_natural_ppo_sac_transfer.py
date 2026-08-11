from __future__ import annotations

import unittest

import numpy as np

from safety_data.natural_ppo_sac_transfer import (
    reconstruct_ordered_histories_and_h96_labels,
    validate_ordered_replay_continuity,
)


class NaturalPpoSacTransferTest(unittest.TestCase):
    def test_history_never_crosses_episode_and_timeout_is_censored(self):
        observation = np.repeat(
            np.arange(12, dtype=np.float32)[:, None], 46, axis=1)
        terminated = np.zeros(12, dtype=bool)
        truncated = np.zeros(12, dtype=bool)
        terminated[5] = True
        truncated[11] = True
        histories, label, eligible = reconstruct_ordered_histories_and_h96_labels(
            observation, terminated, truncated, horizon=4)
        self.assertEqual(histories[0, :, 0].tolist(), [0.0] * 5)
        self.assertEqual(histories[6, :, 0].tolist(), [6.0] * 5)
        self.assertTrue(label[2])
        self.assertTrue(eligible[2])
        self.assertFalse(label[8])
        self.assertFalse(eligible[8])

    def test_full_horizon_survival_is_negative(self):
        observation = np.zeros((8, 46), dtype=np.float32)
        histories, label, eligible = reconstruct_ordered_histories_and_h96_labels(
            observation, np.zeros(8, bool), np.zeros(8, bool), horizon=4)
        self.assertEqual(histories.shape, (8, 5, 46))
        self.assertFalse(label.any())
        self.assertEqual(eligible.tolist(), [True] * 5 + [False] * 3)

    def test_continuity_ignores_reset_boundary_but_rejects_gap(self):
        observation = np.zeros((4, 46), dtype=np.float32)
        next_observation = observation.copy()
        terminated = np.asarray([False, True, False, False])
        truncated = np.zeros(4, bool)
        next_observation[1] = 4.0
        validate_ordered_replay_continuity(
            observation, next_observation, terminated, truncated)
        next_observation[2] = 1.0
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_ordered_replay_continuity(
                observation, next_observation, terminated, truncated)


if __name__ == "__main__":
    unittest.main()
