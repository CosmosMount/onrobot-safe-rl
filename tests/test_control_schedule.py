import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from train.env import Go2Env
from train.types import RobotState


class _Clock:

    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, duration):
        self.now += duration

    def advance(self, duration):
        self.now += duration


class ControlScheduleTest(unittest.TestCase):

    def _env(self):
        env = Go2Env.__new__(Go2Env)
        env.cfg = SimpleNamespace(
            num_joints=12,
            init_qpos=np.zeros(12, dtype=np.float32),
            sport_state_max_age_ms=250.0,
        )
        env.control_dt = 0.05
        env.max_episode_steps = 400
        env.max_joint_delta = None
        env.reset_grace_steps = 20
        env.standup_timeout_steps = 200
        env.recovery_stable_steps = 10
        env._steps_since_reset = 0
        env._step_count = 0
        env._standup_active = False
        env._standup_with_recovery = False
        env._standup_step_count = 0
        env._standup_stable_count = 0
        env._belly_up_count = 0
        env._not_belly_up_count = 0
        env._last_policy_send_time = None
        env._prev_requested_action = np.zeros(12, dtype=np.float32)
        env._prev_executed_action = np.zeros(12, dtype=np.float32)
        env._action_filter = None
        env._ensure_connected = Mock()
        env._policy_client = Mock()
        env._state_reader = Mock()
        env._state_reader.get_state.return_value = RobotState()
        env._state_reader.sport_state_age.return_value = 0.0
        env._send_q_target = Mock(
            return_value=np.zeros(12, dtype=np.float32))
        env._send_standup_request = Mock()
        env._tick_standup = Mock()
        env._resume_policy = Mock()
        return env

    def _step(self, env, clock, callback=None):
        with (
            patch('train.env.time.perf_counter',
                  side_effect=clock.perf_counter),
            patch('train.env.time.sleep', side_effect=clock.sleep),
            patch('train.env.action_to_qpos',
                  return_value=np.zeros(12, dtype=np.float32)),
            patch('train.env.qpos_to_action',
                  return_value=np.zeros(12, dtype=np.float32)),
            patch('train.env.build_observation',
                  return_value=np.zeros(58, dtype=np.float32)),
            patch('train.env.get_run_reward_from_state',
                  return_value=(0.0, {'forward_velocity': 0.0})),
            patch('train.env.get_terminal_penalty', return_value=0.0),
            patch('train.env.is_belly_up', return_value=False),
            patch('train.env.is_fallen', return_value=False),
        ):
            return env.step(
                np.zeros(12, dtype=np.float32), during_hold=callback)

    def test_update_runs_inside_remaining_control_hold(self):
        env = self._env()
        clock = _Clock()
        callback = Mock(side_effect=lambda: clock.advance(0.015))

        _, _, _, _, first_info = self._step(env, clock, callback)
        _, _, _, _, second_info = self._step(env, clock)

        callback.assert_called_once()
        self.assertAlmostEqual(clock.now, 0.100, places=6)
        self.assertAlmostEqual(
            first_info['control_hold_overrun_ms'], 0.0, places=6)
        self.assertAlmostEqual(
            second_info['action_interval_ms'], 50.0, places=6)
        self.assertAlmostEqual(
            second_info['action_frequency_hz'], 20.0, places=6)

    def test_update_overrun_is_measured_without_negative_sleep(self):
        env = self._env()
        clock = _Clock()

        _, _, _, _, info = self._step(
            env, clock, lambda: clock.advance(0.060))

        self.assertAlmostEqual(clock.now, 0.060, places=6)
        self.assertAlmostEqual(
            info['control_hold_overrun_ms'], 10.0, places=6)

    def test_recovery_frame_never_invokes_update_callback(self):
        env = self._env()
        env._standup_active = True
        clock = _Clock()
        callback = Mock()

        _, _, _, _, info = self._step(env, clock, callback)

        callback.assert_not_called()
        self.assertFalse(info['policy_step'])
        self.assertAlmostEqual(clock.now, 0.050, places=6)


if __name__ == '__main__':
    unittest.main()
