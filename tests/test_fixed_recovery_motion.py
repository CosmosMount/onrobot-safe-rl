from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from safety_data.fixed_recovery_motion import (
    FixedRecoveryConfig,
    FixedRecoveryExecutor,
    FixedRecoveryMotion,
    RecoveryStage,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_CONFIG = ROOT / "runtime/control/go2/go2.yaml"
CONTROLLER_SOURCE = ROOT / "runtime/control/go2/motions/src/recovery.cpp"


class FixedRecoveryMotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FixedRecoveryConfig.from_controller_yaml(CONTROLLER_CONFIG)

    def test_manifest_binds_original_nonpolicy_sequence(self) -> None:
        manifest = self.config.manifest(
            control_hz=500.0,
            controller_yaml=CONTROLLER_CONFIG,
            controller_source=CONTROLLER_SOURCE,
        )
        self.assertEqual(manifest["schema_version"], "qsafe.original_go2_fixed_recovery.v1")
        self.assertEqual(manifest["implementation"], "Fold->Above->SwingDown->Push")
        self.assertFalse(manifest["learned_policy_used"])
        self.assertEqual(manifest["control_hz"], 500.0)
        self.assertEqual(len(manifest["contract_sha256"]), 64)
        self.assertEqual(manifest["kp"], 100.0)
        self.assertEqual(manifest["kd"], 8.0)
        self.assertEqual(
            set(manifest["authoritative_files"]), {
                "controller_yaml", "controller_yaml_sha256",
                "recovery_cpp", "recovery_cpp_sha256",
            })
        self.assertTrue(all(
            len(value) == 64
            for key, value in manifest["authoritative_files"].items()
            if key.endswith("sha256")))

    def test_exact_stage_boundaries_match_cpp_tick_contract(self) -> None:
        motion = FixedRecoveryMotion(self.config, control_hz=500.0)
        measured = np.asarray(
            [0.05, 0.70, -1.40] * 4, dtype=np.float32)
        motion.reset(measured)
        records = []
        while True:
            result = motion.update(measured)
            records.append(result)
            measured = result.q_target
            if result.done:
                break
            self.assertLess(len(records), 2500)

        # C++ changes stage only after emitting each boundary tick.  The next
        # update starts the following interpolation at local tick zero.
        self.assertEqual(records[0].stage_executed, RecoveryStage.FOLD)
        np.testing.assert_array_equal(records[0].q_target, np.asarray(
            [0.05, 0.70, -1.40] * 4, dtype=np.float32))
        self.assertEqual(records[475].stage_executed, RecoveryStage.FOLD)
        self.assertEqual(records[476].stage_executed, RecoveryStage.ABOVE)
        self.assertEqual(records[876].stage_executed, RecoveryStage.ABOVE)
        self.assertEqual(records[877].stage_executed, RecoveryStage.SWING_DOWN)
        self.assertEqual(records[1427].stage_executed, RecoveryStage.SWING_DOWN)
        self.assertEqual(records[1428].stage_executed, RecoveryStage.PUSH)
        self.assertEqual(records[-1].stage_executed, RecoveryStage.PUSH)
        self.assertEqual(records[-1].tick, 1703)
        self.assertEqual(len(records), 1704)
        np.testing.assert_array_equal(records[-1].q_target, self.config.push_jpos)

    def test_swing_down_waits_for_measured_joints(self) -> None:
        motion = FixedRecoveryMotion(self.config, control_hz=500.0)
        measured = np.asarray([0.05, 0.70, -1.40] * 4, dtype=np.float32)
        motion.reset(measured)
        for _ in range(1600):
            result = motion.update(measured)
            if result.stage_executed is not RecoveryStage.SWING_DOWN:
                measured = result.q_target
        self.assertEqual(motion.capture_state().stage, RecoveryStage.SWING_DOWN)
        result = motion.update(self.config.swing_down_jpos)
        self.assertEqual(result.stage_executed, RecoveryStage.SWING_DOWN)
        self.assertEqual(motion.capture_state().stage, RecoveryStage.PUSH)

    def test_push_changes_only_configured_calf_targets(self) -> None:
        motion = FixedRecoveryMotion(self.config, control_hz=500.0)
        motion.reset(np.asarray([0.05, 0.70, -1.40] * 4, dtype=np.float32))
        measured = motion.capture_state().initial_jpos
        first_push = None
        while first_push is None:
            result = motion.update(measured)
            measured = result.q_target
            if result.stage_executed is RecoveryStage.PUSH:
                first_push = result.q_target
        np.testing.assert_array_equal(first_push, self.config.swing_down_jpos)
        final = first_push
        for _ in range(151):
            result = motion.update(measured)
            measured = result.q_target
            final = result.q_target
        np.testing.assert_array_equal(final[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10]],
                                      self.config.swing_down_jpos[
                                          [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]])
        np.testing.assert_allclose(final[[8, 11]], self.config.push_jpos[[8, 11]])

    def test_capture_restore_replays_identical_targets(self) -> None:
        motion = FixedRecoveryMotion(self.config, control_hz=500.0)
        measured = np.asarray([0.05, 0.70, -1.40] * 4, dtype=np.float32)
        motion.reset(measured)
        for _ in range(600):
            measured = motion.update(measured).q_target
        state = motion.capture_state()
        expected = [motion.update(measured).q_target for _ in range(10)]
        motion.restore_state(state)
        actual = [motion.update(measured).q_target for _ in range(10)]
        np.testing.assert_array_equal(actual, expected)

    def test_executor_applies_ten_500hz_targets_per_policy_interval(self) -> None:
        class FakeEnv:
            def __init__(self, joint_q):
                self.model = type("Model", (), {
                    "opt": type("Opt", (), {"timestep": 0.002})(),
                })()
                self.policy_frequency = 50.0
                self.joint_q = np.asarray(joint_q, dtype=np.float32).copy()
                self.targets = []

            def robot_state(self):
                return type("State", (), {"joint_q": self.joint_q.copy()})()

            def step_recovery_target(self, q_target, *, kp, kd):
                self.targets.append((np.asarray(q_target).copy(), kp, kd))
                self.joint_q = np.asarray(q_target, dtype=np.float32).copy()
                return len(self.targets)

        env = FakeEnv([0.05, 0.70, -1.40] * 4)
        executor = FixedRecoveryExecutor(FixedRecoveryMotion(self.config))
        executor.start(env)
        result = executor.policy_interval(env)
        self.assertEqual(len(result), 10)
        self.assertEqual(len(env.targets), 10)
        self.assertTrue(all(kp == 100.0 and kd == 8.0 for _, kp, kd in env.targets))
        self.assertEqual([item.motion.tick for item in result], list(range(10)))
        np.testing.assert_array_equal(result[-1].motion.q_target, env.targets[-1][0])


if __name__ == "__main__":
    unittest.main()
