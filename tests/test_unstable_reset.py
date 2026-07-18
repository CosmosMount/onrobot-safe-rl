from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from train.env import Go2Env, UnstableResetError
from train.obs import is_pose_stable


def _state(*, joint_q=None):
    return SimpleNamespace(
        joint_q=np.asarray(
            np.zeros(12, dtype=np.float32) if joint_q is None else joint_q
        ),
        imu_quat=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


class UnstableResetTest(unittest.TestCase):

    def _env(self, *, abort_on_unstable_reset=True):
        env = Go2Env.__new__(Go2Env)
        env.cfg = SimpleNamespace(
            init_qpos=np.zeros(12, dtype=np.float32),
            imu_upside_down_up_cos=-0.7,
        )
        env.reset_hold_steps = 3
        env.reset_joint_tolerance = 0.30
        env.recovery_stable_steps = 2
        env.abort_on_unstable_reset = abort_on_unstable_reset
        env.control_dt = 0.0
        env._state_reader = Mock()
        env._policy_client = Mock()
        env._send_standup_request = Mock()
        return env

    @patch("train.env.is_belly_up", return_value=True)
    @patch("train.env.quat_to_euler_xyz", return_value=(3.14, 0.0, 0.0))
    @patch("train.env.is_pose_stable", return_value=False)
    def test_failed_standup_aborts_before_policy_mode(
            self, _stable, _euler, _belly):
        env = self._env()
        env._state_reader.get_state.return_value = _state()

        with self.assertRaisesRegex(
                UnstableResetError, "Policy rollout was aborted"):
            env._wait_standup(with_recovery=True)

        self.assertEqual(env._send_standup_request.call_count, 3)
        env._policy_client.send_target.assert_not_called()

    @patch("train.env.is_pose_stable",
           side_effect=(False, True, True))
    def test_stable_standup_enters_policy_mode(self, _stable):
        env = self._env()
        state = _state()
        env._state_reader.get_state.return_value = state

        result = env._wait_standup(with_recovery=False)

        self.assertIs(result, state)
        env._policy_client.send_target.assert_called_once()

    @patch("train.env.is_belly_up", return_value=True)
    @patch("train.env.is_pose_stable",
           side_effect=(False, False, False, False, True, True))
    def test_standup_that_ends_belly_up_escalates_to_recovery(
            self, _stable, _belly):
        env = self._env()
        state = _state()
        env._state_reader.get_state.return_value = state

        env._wait_standup(with_recovery=False)

        requests = [
            call.kwargs["with_recovery"]
            for call in env._send_standup_request.call_args_list
        ]
        self.assertEqual(requests, [False, False, False, True, True, True])
        env._policy_client.send_target.assert_called_once()

    @patch("train.env.is_pose_stable", return_value=False)
    def test_legacy_opt_out_is_explicit(self, _stable):
        env = self._env(abort_on_unstable_reset=False)
        state = _state()
        env._state_reader.get_state.return_value = state

        env._wait_standup(with_recovery=False)

        env._policy_client.send_target.assert_called_once()

    @patch("train.obs.is_fallen_risk", return_value=False)
    def test_reset_can_use_independent_joint_tolerance(self, _fallen):
        cfg = SimpleNamespace(
            init_qpos=np.zeros(12, dtype=np.float32),
            joint_tolerance=0.20,
        )
        state = _state(joint_q=np.asarray(
            [0.252] + [0.0] * 11, dtype=np.float32))

        self.assertFalse(is_pose_stable(state, cfg))
        self.assertTrue(
            is_pose_stable(state, cfg, joint_tolerance=0.30))

    def test_logical_timeout_reset_does_not_restore_fall_grace(self):
        env = self._env()
        env.reset_grace_steps = 20
        env.cfg.num_joints = 12
        env._standup_active = False
        env._action_filter = None
        env._ensure_connected = Mock()
        env._init_action_filter = Mock()
        env._state_reader.get_state.return_value = _state()

        with patch("train.env.build_observation",
                   return_value=np.zeros(58, dtype=np.float32)):
            env.reset(standup=False, grace_period=False)

        self.assertEqual(env._steps_since_reset, 21)


if __name__ == "__main__":
    unittest.main()
