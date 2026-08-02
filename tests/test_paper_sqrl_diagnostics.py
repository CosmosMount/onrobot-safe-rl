from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_paper_sqrl_snapshot import _summarize
from scripts.recalibrate_paper_sqrl_safety import _auc


class SafetyRecalibrationMetricsTest(unittest.TestCase):
    def test_auc_handles_perfect_order_and_ties(self):
        labels = np.asarray([False, False, True, True])
        self.assertEqual(_auc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertEqual(_auc(labels, np.ones(4)), 0.5)


class SnapshotCausalMetricsTest(unittest.TestCase):
    def test_horizon_summary_is_paired_and_counts_only_prior_actions(self):
        records = [
            {
                "nominal": {"failure_step": 6},
                "closed_loop": {
                    "failure_step": None,
                    "replacement_steps": [1, 9],
                    "no_safe_steps": [1, 10],
                },
            },
            {
                "nominal": {"failure_step": None},
                "closed_loop": {
                    "failure_step": 7,
                    "replacement_steps": [2, 12],
                    "no_safe_steps": [],
                },
            },
        ]
        result = _summarize(records, "closed_loop", 8)
        self.assertEqual(result["nominal_failures"], 1)
        self.assertEqual(result["selected_failures"], 1)
        self.assertEqual(result["improved"], 1)
        self.assertEqual(result["worsened"], 1)
        self.assertEqual(result["total_replacements"], 2)
        self.assertEqual(result["total_no_safe"], 1)


if __name__ == "__main__":
    unittest.main()
