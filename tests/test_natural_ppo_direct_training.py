from __future__ import annotations

import unittest

import numpy as np

from safety_data.natural_ppo_direct_training import (
    binary_auc,
    expected_calibration_error,
    paired_accuracy,
)


class DirectPpoQSafeMetricsTest(unittest.TestCase):
    def test_auc_handles_ties_by_average_rank(self):
        label = np.asarray([False, True, False, True])
        self.assertEqual(binary_auc(label, [0.0, 1.0, 0.5, 0.5]), 0.875)
        self.assertEqual(binary_auc(label, [0.5, 0.5, 0.5, 0.5]), 0.5)

    def test_pair_accuracy_is_pair_macro(self):
        pairs = np.asarray([b"a", b"a", b"b", b"b"])
        label = np.asarray([False, True, False, True])
        self.assertEqual(paired_accuracy(pairs, label, [0.1, 0.9, 0.8, 0.2]), 0.5)

    def test_ece_is_zero_for_exact_bin_frequencies(self):
        self.assertAlmostEqual(expected_calibration_error(
            np.asarray([False, True]), np.asarray([0.0, 1.0])), 0.0)


if __name__ == "__main__":
    unittest.main()
