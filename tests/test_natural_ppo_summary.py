from __future__ import annotations

import unittest

import numpy as np

from scripts.summarize_natural_ppo_collection import _age_counts, _tilt


class NaturalPpoSummaryTest(unittest.TestCase):
    def test_age_counts_use_registered_left_closed_buckets(self):
        self.assertEqual(_age_counts(np.asarray([
            0, 999_999, 1_000_000, 1_999_999, 2_000_000,
            5_000_000, 10_000_000, 20_000_000, 30_000_000,
        ])), {"0": 2, "1": 2, "2": 1, "3": 1, "4": 1, "5": 2})

    def test_tilt_uses_wxyz_quaternion(self):
        angle = np.pi / 3
        quaternion = np.asarray([
            [1.0, 0.0, 0.0, 0.0],
            [np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0],
        ])
        np.testing.assert_allclose(_tilt(quaternion), [0.0, angle], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
