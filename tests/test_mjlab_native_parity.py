from __future__ import annotations

import unittest

import numpy as np

from scripts.validate_mjlab_native_parity import _fall


class MjlabNativeParityHelpersTest(unittest.TestCase):
    def test_fall_predicate_matches_height_roll_and_pitch_contract(self):
        qpos = np.zeros((4, 19), dtype=np.float64)
        qpos[:, 2] = 0.4
        qpos[:, 3] = 1.0
        qpos[1, 2] = 0.17
        angle = 1.1
        qpos[2, 3] = np.cos(angle / 2)
        qpos[2, 4] = np.sin(angle / 2)
        qpos[3, 3] = np.cos(angle / 2)
        qpos[3, 5] = np.sin(angle / 2)
        self.assertEqual(_fall(qpos).tolist(), [False, True, True, True])


if __name__ == "__main__":
    unittest.main()
