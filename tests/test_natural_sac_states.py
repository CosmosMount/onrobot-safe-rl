from __future__ import annotations

import unittest

import numpy as np

from safety_data.natural_sac_states import episode_h96_labels


class NaturalSacStateLabelTest(unittest.TestCase):
    def test_fall_episode_labels_last_h96_and_keeps_earlier_negatives(self):
        eligible, label, steps = episode_h96_labels(100, failed=True)
        self.assertTrue(np.all(eligible))
        self.assertEqual(int(label.sum()), 96)
        self.assertFalse(bool(label[3]))
        self.assertTrue(bool(label[4]))
        self.assertEqual(int(steps[4]), 96)
        self.assertEqual(int(steps[-1]), 1)

    def test_timeout_censors_last_h95_states(self):
        eligible, label, steps = episode_h96_labels(100, failed=False)
        self.assertEqual(int(eligible.sum()), 5)
        self.assertFalse(bool(label.any()))
        self.assertTrue(np.all(steps == 96))

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            episode_h96_labels(0, failed=True)


if __name__ == "__main__":
    unittest.main()
