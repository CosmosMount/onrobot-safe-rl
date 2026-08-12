from __future__ import annotations

import unittest

import numpy as np

from safety_data.action_oracle_analysis import (
    _posthoc_state_oracle,
    _same_crn_outcome_oracle,
    _selected_candidate,
)


class ActionOracleAnalysisTest(unittest.TestCase):
    def test_candidate_is_selected_on_discovery_not_audit(self):
        # Candidate one wins discovery but loses audit. The analyzer must not
        # peek at audit outcomes and switch to candidate two.
        fall = np.asarray([[[1, 1, 1, 1],
                            [0, 0, 1, 1],
                            [1, 1, 0, 0]]], dtype=bool)
        mask = np.ones((1, 3), dtype=bool)
        action = np.zeros((1, 3, 2), dtype=np.float32)
        action[0, 1] = 0.1
        action[0, 2] = 0.2
        selected = _selected_candidate(fall, mask, action)
        self.assertEqual(selected.tolist(), [1])

    def test_tie_prefers_minimum_nominal_deviation(self):
        fall = np.zeros((1, 3, 4), dtype=bool)
        mask = np.ones((1, 3), dtype=bool)
        action = np.asarray([[[0.0, 0.0], [0.2, 0.2], [0.1, 0.1]]])
        selected = _selected_candidate(fall, mask, action)
        self.assertEqual(selected.tolist(), [0])

    def test_posthoc_oracle_uses_all_replicas_and_reports_state_risk(self):
        fall = np.asarray([[[1, 1, 1, 1],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]]], dtype=bool)
        mask = np.ones((1, 3), dtype=bool)
        action = np.asarray([[[0.0], [0.1], [0.2]]], dtype=np.float32)
        selected, nominal, oracle = _posthoc_state_oracle(fall, mask, action)
        self.assertEqual(selected.tolist(), [2])
        np.testing.assert_allclose(nominal, [1.0])
        np.testing.assert_allclose(oracle, [0.25])

    def test_posthoc_oracle_ignores_masked_candidate(self):
        fall = np.asarray([[[1, 1], [0, 0], [1, 0]]], dtype=bool)
        mask = np.asarray([[True, False, True]])
        action = np.asarray([[[0.0], [0.1], [0.2]]], dtype=np.float32)
        selected, _, oracle = _posthoc_state_oracle(fall, mask, action)
        self.assertEqual(selected.tolist(), [2])
        np.testing.assert_allclose(oracle, [0.5])

    def test_same_crn_oracle_is_explicitly_per_realization(self):
        fall = np.asarray([[[1, 1], [0, 1], [1, 0]]], dtype=bool)
        mask = np.ones((1, 3), dtype=bool)
        nominal, oracle = _same_crn_outcome_oracle(fall, mask)
        np.testing.assert_allclose(nominal, [1.0])
        np.testing.assert_allclose(oracle, [0.0])


if __name__ == "__main__":
    unittest.main()
