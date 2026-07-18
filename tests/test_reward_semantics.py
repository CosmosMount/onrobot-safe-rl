from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from train.obs import get_run_reward_from_state, get_terminal_penalty
from train.types import RobotState


def _cfg():
    return SimpleNamespace(
        move_speed=0.5,
        reward_min_forward_vel=0.04,
        reward_upright_min_cos=0.8660254,
        fall_terminal_penalty=-10.0,
    )


def _state(quat):
    return RobotState(
        imu_quat=np.asarray(quat, dtype=np.float32),
        imu_gyro=np.zeros(3, dtype=np.float32),
        body_velocity=np.asarray([0.5, 0.0, 0.0], dtype=np.float32),
    )


class RewardSemanticsTest(unittest.TestCase):

    def test_upright_forward_motion_keeps_forward_reward(self):
        reward, info = get_run_reward_from_state(
            _state([1.0, 0.0, 0.0, 0.0]), _cfg())

        self.assertAlmostEqual(info["upright_gate"], 1.0)
        self.assertAlmostEqual(info["forward_term"], 1.0)
        self.assertAlmostEqual(reward, 10.0)

    def test_belly_up_motion_cannot_receive_forward_reward(self):
        reward, info = get_run_reward_from_state(
            _state([0.0, 1.0, 0.0, 0.0]), _cfg())

        self.assertAlmostEqual(info["body_up_cos"], -1.0)
        self.assertAlmostEqual(info["upright_gate"], 0.0)
        self.assertAlmostEqual(info["forward_term"], 0.0)
        self.assertLessEqual(reward, 0.0)

    def test_terminal_penalty_does_not_apply_to_time_limit(self):
        self.assertEqual(
            get_terminal_penalty(terminated=False, cfg=_cfg()), 0.0)
        self.assertEqual(
            get_terminal_penalty(terminated=True, cfg=_cfg()), -10.0)


if __name__ == "__main__":
    unittest.main()
