from __future__ import annotations

import unittest

import numpy as np
import torch

from safety_data.mjlab_natural_falls import (
    CAPTURE_RING_STEPS,
    ordered_ring_indices,
    target_fall_predicate,
)


class MjlabNaturalFallHelpersTest(unittest.TestCase):
    def test_ring_indices_are_chronological_after_wrap(self):
        np.testing.assert_array_equal(ordered_ring_indices(3, 5), [0, 1, 2])
        np.testing.assert_array_equal(ordered_ring_indices(7, 5), [2, 3, 4, 0, 1])
        self.assertGreater(CAPTURE_RING_STEPS, 96)

    def test_target_predicate_uses_height_roll_and_pitch(self):
        qpos = torch.zeros((4, 19), dtype=torch.float64)
        qpos[:, 2] = 0.4
        qpos[:, 3] = 1.0
        qpos[1, 2] = 0.17
        angle = 1.1
        qpos[2, 3] = np.cos(angle / 2)
        qpos[2, 4] = np.sin(angle / 2)
        qpos[3, 3] = np.cos(angle / 2)
        qpos[3, 5] = np.sin(angle / 2)
        self.assertEqual(target_fall_predicate(qpos).tolist(),
                         [False, True, True, True])


if __name__ == "__main__":
    unittest.main()
