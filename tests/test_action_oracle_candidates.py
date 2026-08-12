from __future__ import annotations

import unittest

import numpy as np

from runtime.inference.actions import ActionApplier
from safety_data.action_oracle_candidates import (
    ACTION_ORACLE_KINDS,
    build_action_oracle_candidates,
)
from safety_data.candidates import EvidenceCandidateConfig, build_evidence_candidates


class ActionOracleCandidatesTest(unittest.TestCase):
    def setUp(self):
        init = np.asarray([0.05, 0.70, -1.40] * 4, dtype=np.float32)
        self.applier = ActionApplier(
            init_qpos=init,
            action_offset=np.asarray([0.2, 0.4, 0.4] * 4, dtype=np.float32),
            joint_min=np.full(12, -3.0, dtype=np.float32),
            joint_max=np.full(12, 3.0, dtype=np.float32),
            max_joint_delta=None,
            action_filter=None,
        )
        self.local = build_evidence_candidates(
            nominal=np.full(12, 0.1, dtype=np.float32),
            deterministic_mean=np.zeros(12, dtype=np.float32),
            previous_requested=np.full(12, -0.1, dtype=np.float32),
            actor_samples=np.stack([
                np.full(12, value, dtype=np.float32)
                for value in (0.2, -0.2, 0.3, -0.3)]),
            action_applier=self.applier,
            current_qpos=np.asarray([0.05, 0.70, -1.40] * 4, dtype=np.float32),
            candidate_seed=17,
            config=EvidenceCandidateConfig(),
        )

    def test_appends_state_dependent_executable_actions(self):
        history = np.zeros((5, 46), dtype=np.float32)
        history[:, :12] = np.asarray([0.05, 0.70, -1.40] * 4)
        history[:, 30] = 1.0
        history[-1, 24:26] = [1.0, -0.7]
        result = build_action_oracle_candidates(
            self.local, observation_history=history,
            action_applier=self.applier)
        self.assertEqual(result.requested.shape, (24, 12))
        self.assertEqual(tuple(result.kind), ACTION_ORACLE_KINDS)
        np.testing.assert_array_equal(result.requested[:16], self.local.requested)
        self.assertGreaterEqual(result.valid_count, 12)
        self.assertFalse(np.allclose(result.requested[16], result.requested[17]))
        self.assertTrue(np.all(result.requested >= -1.0))
        self.assertTrue(np.all(result.requested <= 1.0))

    def test_rejects_non_deployable_history(self):
        with self.assertRaisesRegex(ValueError, r"\[5,46\]"):
            build_action_oracle_candidates(
                self.local, observation_history=np.zeros((4, 46)),
                action_applier=self.applier)


if __name__ == "__main__":
    unittest.main()
