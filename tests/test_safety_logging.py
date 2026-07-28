from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from collector.transition_builder import build_transition
from common.transition import COST_KEYS, TerminationReason
from train.safety import safety_signals
from train.types import RobotState
from train.config import load_app_config
from train.env import Go2Env


def _cfg():
    return SimpleNamespace(
        joint_min=np.full(12, -1.0, dtype=np.float32),
        joint_max=np.full(12, 1.0, dtype=np.float32),
        success_orientation_rad=0.5,
        fallen_risk_rad=0.35,
        imu_upright_up_cos=0.5,
        imu_upside_down_up_cos=-0.7,
        safety_joint_limit_margin_rad=0.1,
        safety_joint_velocity_rad_s=25.0,
        safety_torque_nm=35.0,
        safety_power_w=350.0,
        safety_impact_accel_m_s2=8.0,
        safety_angular_velocity_rad_s=5.0,
        safety_base_height_m=0.18,
    )


class SafetyLoggingTest(unittest.TestCase):

    def test_nominal_state_has_complete_finite_costs(self):
        state = RobotState(
            joint_q=np.zeros(12, dtype=np.float32),
            joint_dq=np.ones(12, dtype=np.float32),
            joint_tau=np.full(12, 2.0, dtype=np.float32),
            imu_accel=np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
            world_position=np.asarray([0.0, 0.0, 0.30], dtype=np.float32),
        )
        info = safety_signals(
            state, _cfg(), terminated=False, recovering=False,
            intervention_mask=False)

        self.assertEqual(set(info['costs']), set(COST_KEYS))
        self.assertTrue(all(np.isfinite(v) for v in info['costs'].values()))
        self.assertFalse(info['unsafe_label'])
        self.assertFalse(info['near_failure_label'])
        self.assertGreater(info['costs']['torque_cost'], 0.0)
        self.assertGreater(info['costs']['power_cost'], 0.0)
        self.assertEqual(info['costs']['slip_cost'], 0.0)

    def test_boundary_and_failure_labels(self):
        # 0.4 rad roll is over the risk threshold but below termination.
        state = RobotState(
            imu_quat=np.asarray([np.cos(0.2), np.sin(0.2), 0.0, 0.0],
                                dtype=np.float32),
            imu_accel=np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
            world_position=np.asarray([0.0, 0.0, 0.30], dtype=np.float32),
        )
        boundary = safety_signals(
            state, _cfg(), terminated=False, recovering=False,
            intervention_mask=False)
        self.assertTrue(boundary['near_failure_label'])
        self.assertFalse(boundary['unsafe_label'])

        # 0.6 rad roll exceeds the task termination threshold.
        state.imu_quat = np.asarray(
            [np.cos(0.3), np.sin(0.3), 0.0, 0.0], dtype=np.float32)
        failure = safety_signals(
            state, _cfg(), terminated=True, recovering=True,
            intervention_mask=True)
        self.assertTrue(failure['unsafe_label'])
        self.assertTrue(failure['near_failure_label'])
        self.assertEqual(failure['costs']['impact_cost'], 1.0)
        self.assertEqual(failure['costs']['intervention_cost'], 1.0)

    def test_transition_preserves_legacy_replay_and_adds_safety_view(self):
        costs = {key: (0.25 if key == 'tilt_cost' else 0.0)
                 for key in COST_KEYS}
        transition = build_transition(
            np.zeros(3), np.zeros(2), 1.0, np.ones(3), True,
            {
                'terminated': True,
                'costs': costs,
                'unsafe_label': True,
                'near_failure_label': True,
            })

        self.assertEqual(set(transition.replay_dict()), {
            'observations', 'actions', 'rewards', 'masks', 'dones',
            'next_observations',
        })
        self.assertEqual(transition.termination_reason,
                         TerminationReason.EXCESSIVE_TILT)
        safety = transition.safety_replay_dict()
        self.assertEqual(safety['costs']['tilt_cost'], 0.25)
        self.assertEqual(safety['unsafe_labels'], 1.0)
        self.assertEqual(safety['near_failure_labels'], 1.0)
        self.assertIn('episode_ids', safety)
        self.assertIn('policy_versions', safety)
        self.assertIn('command_speeds', safety)

    def test_explicit_failure_reason_cannot_be_overridden_by_false_label(self):
        transition = build_transition(
            np.zeros(3), np.zeros(2), 0.0, np.ones(3), True,
            {
                'terminated': True,
                'termination_reason': int(TerminationReason.MOTOR_FAULT),
                'unsafe_label': False,
                'near_failure_label': False,
            })
        self.assertTrue(transition.unsafe_label)
        self.assertTrue(transition.near_failure_label)

    def test_scripted_motion_frames_are_drained_as_interventions(self):
        cfg, _, _ = load_app_config(path='config/go2.yaml')
        env = Go2Env.__new__(Go2Env)
        env.cfg = cfg
        env._recovery_transitions = []
        before = RobotState(world_position=np.asarray(
            [0.0, 0.0, 0.12], dtype=np.float32))
        after = RobotState(world_position=np.asarray(
            [0.0, 0.0, 0.20], dtype=np.float32))
        env._record_recovery_transition(before, after)

        transitions = env.drain_recovery_transitions()
        self.assertEqual(len(transitions), 1)
        self.assertTrue(transitions[0].intervention_mask)
        self.assertEqual(transitions[0].costs['intervention_cost'], 1.0)
        self.assertEqual(env.drain_recovery_transitions(), [])


if __name__ == '__main__':
    unittest.main()
