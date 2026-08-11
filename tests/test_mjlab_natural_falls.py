from __future__ import annotations

import unittest

import numpy as np
import torch

from safety_data.mjlab_natural_falls import (
    CAPTURE_RING_STEPS,
    MJLAB_TO_TARGET_JOINT,
    ordered_ring_indices,
    target_order_action_and_qtarget,
    target_fall_predicate,
)


class MjlabNaturalFallHelpersTest(unittest.TestCase):
    def test_current_action_and_absolute_target_use_target_joint_order(self):
        action = torch.arange(12, dtype=torch.float32).reshape(1, 12)
        bias = torch.full_like(action, 0.1)
        requested, target = target_order_action_and_qtarget(
            action, scale=0.25, offset=1.0, encoder_bias=bias)
        expected = action[:, MJLAB_TO_TARGET_JOINT]
        self.assertTrue(torch.equal(requested, expected))
        self.assertTrue(torch.allclose(target, expected * 0.25 + 0.9))

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
