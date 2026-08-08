from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from runtime.inference.actions import action_to_qpos
from safety_data.native import evaluate_same_state_group
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


MODEL = Path(
    "/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml")


class MujocoSnapshotEnvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mujoco  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("mujoco is not installed") from exc
        if not MODEL.exists():
            raise unittest.SkipTest(f"Go2 MJCF is unavailable: {MODEL}")
        cls.robot, cls.train, _ = load_app_config(
            "config/go2_50hz_safe_adaptive_gated_v3.yaml", agent="safe_droq")

    def env(self, *, filtered: bool = False) -> MujocoSnapshotEnv:
        result = MujocoSnapshotEnv(
            MODEL,
            self.robot,
            policy_frequency=self.train.control_frequency,
            max_joint_delta=self.train.max_joint_delta,
            use_action_filter=filtered,
        )
        result.reset_standing(settle_seconds=0.02, rng=np.random.default_rng(7))
        return result

    def test_reset_uses_absolute_init_qpos_in_corrected_observation(self):
        env = self.env()
        history = env.record_observation()
        self.assertEqual(history.shape, (5, 46))
        np.testing.assert_array_equal(
            history[..., -12:],
            np.broadcast_to(self.robot.init_qpos, (5, 12)))
        with self.assertRaisesRegex(ValueError, "normalized policy action"):
            env.observation(np.zeros(12, dtype=np.float32))

    def test_unfiltered_step_reports_requested_executed_and_q_target(self):
        env = self.env()
        action = np.linspace(-0.7, 0.7, 12, dtype=np.float32)
        result = env.step(action)
        expected = action_to_qpos(
            action,
            init_qpos=self.robot.init_qpos,
            action_offset=self.robot.action_offset,
            joint_min=self.robot.joint_min,
            joint_max=self.robot.joint_max,
        )
        np.testing.assert_array_equal(result.application.action_requested, action)
        np.testing.assert_allclose(
            result.application.action_executed, action, rtol=0.0, atol=2e-7)
        np.testing.assert_array_equal(result.application.action_q_target, expected)
        np.testing.assert_array_equal(env.observation()[-12:], expected)

    def test_compound_snapshot_restores_filter_and_history_bit_exactly(self):
        env = self.env(filtered=True)
        env.record_observation()
        snapshot = env.capture()
        snapshot_hash = snapshot.compound_sha256()
        before = env.observation()
        action = np.linspace(-0.5, 0.5, 12, dtype=np.float32)
        first = env.step(action)
        first_history = env.record_observation()
        first_state = env.capture().integration_state.copy()

        env.restore(snapshot)
        self.assertEqual(env.capture().compound_sha256(), snapshot_hash)
        np.testing.assert_array_equal(env.observation(), before)
        second = env.step(action)
        second_history = env.record_observation()
        second_state = env.capture().integration_state.copy()
        np.testing.assert_array_equal(first_state, second_state)
        np.testing.assert_array_equal(first_history, second_history)
        np.testing.assert_array_equal(
            first.application.action_q_target,
            second.application.action_q_target)

    def test_native_group_reuses_crn_for_duplicate_candidates(self):
        env = self.env(filtered=True)
        env.record_observation()
        snapshot = env.capture()
        action = np.linspace(-0.3, 0.3, 12, dtype=np.float32)

        class StatefulContinuation:
            def __init__(self):
                self.counter = 0

            def capture_branch_state(self):
                return self.counter

            def restore_branch_state(self, state):
                self.counter = state

            def __call__(self, history, step, rng):
                del history, step
                self.counter += 1
                return (rng.normal(0.0, 0.01, size=12)
                        + self.counter * 1e-4).astype(np.float32)

        continuation = StatefulContinuation()
        result = evaluate_same_state_group(
            env,
            snapshot,
            np.stack([action, action]),
            np.asarray([1001, 1002], dtype=np.int64),
            horizon_steps=3,
            continuation_policy=continuation,
        )
        self.assertEqual(continuation.counter, 0)
        np.testing.assert_array_equal(result.fall[0], result.fall[1])
        np.testing.assert_array_equal(
            result.first_failure_step[0], result.first_failure_step[1])
        np.testing.assert_array_equal(result.max_tilt_rad[0], result.max_tilt_rad[1])
        np.testing.assert_array_equal(result.min_height_m[0], result.min_height_m[1])
        np.testing.assert_array_equal(
            result.candidate_q_target[0], result.candidate_q_target[1])

        with self.assertRaisesRegex(ValueError, "integer array"):
            evaluate_same_state_group(
                env,
                snapshot,
                np.stack([action, action]),
                np.asarray([1001.0, 1002.0]),
                horizon_steps=1,
                continuation_policy=StatefulContinuation(),
            )

    def test_fingerprint_locks_runtime_pd_gains(self):
        fingerprint = self.env().simulator_fingerprint()
        self.assertEqual(fingerprint["kp"], [60.0] * 12)
        self.assertEqual(fingerprint["kd"], [5.0] * 12)
        self.assertEqual(len(fingerprint["mjcf_xml_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
