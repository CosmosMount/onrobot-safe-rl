from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from runtime.inference.actions import action_to_qpos
from safety_data.native import ReplicaSeedBundle, evaluate_same_state_group
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

    def test_recursive_mjcf_dependency_requires_audit_marker_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.xml"
            audit = root / "source-7801.audit.npz"
            model.write_text(
                '<mujoco><include file="source-7801.audit.npz"/></mujoco>',
                encoding="utf-8",
            )
            audit.write_bytes(b"must-not-read")
            original_read_bytes = Path.read_bytes

            def guarded_read(path: Path):
                if path == audit:
                    raise AssertionError("audit dependency must not be read")
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", guarded_read):
                with self.assertRaisesRegex(
                        PermissionError, "audit-consumed marker"):
                    MujocoSnapshotEnv._xml_dependency_hash(model)

    def test_reset_uses_absolute_init_qpos_in_corrected_observation(self):
        env = self.env()
        history = env.record_observation()
        self.assertEqual(history.shape, (5, 46))
        np.testing.assert_array_equal(
            history[..., -12:],
            np.broadcast_to(self.robot.init_qpos, (5, 12)))
        with self.assertRaisesRegex(ValueError, "normalized policy action"):
            env.observation(np.zeros(12, dtype=np.float32))

    def test_measurement_height_is_base_body_not_offset_imu_site(self):
        env = self.env()
        measurement = env.measurement()
        expected = float(env.data.xpos[env.base_body_id, 2])
        imu_site_height = float(env.robot_state().world_position[2])
        self.assertEqual(measurement.height_m, expected)
        self.assertGreater(abs(imu_site_height - expected), 0.01)

    def test_base_body_height_failure_uses_strict_point18_boundary(self):
        env = self.env()
        env.data.qpos[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        env.data.qvel[:] = 0.0
        for height, expected_failure in ((0.179, True), (0.180, False)):
            with self.subTest(height=height):
                env.data.qpos[2] = height
                env.mujoco.mj_forward(env.model, env.data)
                measurement = env.measurement()
                self.assertAlmostEqual(measurement.height_m, height, places=12)
                self.assertEqual(measurement.failure, expected_failure)
                self.assertLess(
                    measurement.tilt_rad,
                    float(env.cfg.fallen_orientation_rad),
                )

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
        self.assertEqual(result.seed_contract, "legacy_equal_seeds_v1")
        np.testing.assert_array_equal(result.crn_id, [1001, 1002])
        np.testing.assert_array_equal(result.rollout_seed, result.crn_id)
        np.testing.assert_array_equal(result.perturbation_seed, result.crn_id)
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

    def test_recovery_option_l1_is_backward_compatible_and_restores_state(self):
        env = self.env(filtered=True)
        env.record_observation()
        snapshot = env.capture()
        snapshot_hash = snapshot.compound_sha256()
        nominal = np.zeros(12, dtype=np.float32)
        recovery = nominal.copy()
        recovery[0] = 0.25
        candidates = np.stack([nominal, recovery])

        class StatefulContinuation:
            def __init__(self):
                self.counter = 17

            def capture_branch_state(self):
                return self.counter

            def restore_branch_state(self, state):
                self.counter = state

            def __call__(self, history, step, rng):
                del history, step
                self.counter += 1
                return (rng.normal(0.0, 0.01, size=12)
                        + self.counter * 1e-5).astype(np.float32)

        continuation = StatefulContinuation()
        legacy = evaluate_same_state_group(
            env,
            snapshot,
            candidates,
            np.asarray([1101, 1102], dtype=np.int64),
            horizon_steps=4,
            continuation_policy=continuation,
        )
        self.assertEqual(continuation.counter, 17)
        self.assertEqual(env.capture().compound_sha256(), snapshot_hash)

        explicit_l1 = evaluate_same_state_group(
            env,
            snapshot,
            candidates,
            np.asarray([1101, 1102], dtype=np.int64),
            horizon_steps=4,
            continuation_policy=continuation,
            option_steps=np.ones(2, dtype=np.int64),
        )
        self.assertEqual(continuation.counter, 17)
        self.assertEqual(env.capture().compound_sha256(), snapshot_hash)
        for name in (
                "candidate_requested", "candidate_executed",
                "candidate_q_target", "fall", "first_failure_step",
                "max_tilt_rad", "min_height_m", "crn_id", "rollout_seed",
                "perturbation_seed"):
            with self.subTest(field=name):
                np.testing.assert_array_equal(
                    getattr(legacy, name), getattr(explicit_l1, name))
        self.assertEqual(legacy.seed_contract, explicit_l1.seed_contract)

    def test_recovery_option_l3_has_exact_decay_and_paired_crn(self):
        env = self.env(filtered=True)
        env.record_observation()
        snapshot = env.capture()
        snapshot_hash = snapshot.compound_sha256()
        nominal = np.zeros(12, dtype=np.float32)
        recovery = nominal.copy()
        recovery[0] = 1.0
        candidates = np.stack([nominal, recovery])
        action_trace = []
        continuation_trace = []
        original_step = env.step

        def recording_step(action):
            action_trace.append(np.asarray(action, dtype=np.float32).copy())
            return original_step(action)

        def continuation(history, step, rng):
            del history, step
            action = rng.normal(0.0, 0.01, size=12).astype(np.float32)
            action[0] += 0.5
            continuation_trace.append(action.copy())
            return action

        env.step = recording_step  # type: ignore[method-assign]
        result = evaluate_same_state_group(
            env,
            snapshot,
            candidates,
            ReplicaSeedBundle(
                crn_id=np.asarray([21, 22], dtype=np.int64),
                rollout_seed=np.asarray([121, 122], dtype=np.int64),
                perturbation_seed=np.asarray([221, 222], dtype=np.int64),
            ),
            horizon_steps=4,
            continuation_policy=continuation,
            option_steps=np.asarray([1, 3], dtype=np.int64),
        )
        self.assertFalse(np.any(result.fall))
        self.assertEqual(env.capture().compound_sha256(), snapshot_hash)

        actions = np.asarray(action_trace).reshape(2, 2, 4, 12)
        continuations = np.asarray(continuation_trace).reshape(2, 2, 3, 12)
        np.testing.assert_array_equal(continuations[0], continuations[1])
        for replica_index in range(2):
            np.testing.assert_array_equal(actions[0, replica_index, 0], nominal)
            np.testing.assert_array_equal(actions[1, replica_index, 0], recovery)
            np.testing.assert_array_equal(
                actions[0, replica_index, 1:],
                continuations[0, replica_index])
            np.testing.assert_allclose(
                actions[1, replica_index, 1],
                np.clip(
                    continuations[1, replica_index, 0]
                    + (2.0 / 3.0) * (recovery - nominal),
                    -1.0,
                    1.0),
                rtol=0.0,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                actions[1, replica_index, 2],
                np.clip(
                    continuations[1, replica_index, 1]
                    + (1.0 / 3.0) * (recovery - nominal),
                    -1.0,
                    1.0),
                rtol=0.0,
                atol=1e-7,
            )
            np.testing.assert_array_equal(
                actions[1, replica_index, 3],
                continuations[1, replica_index, 2])

    def test_explicit_replica_seed_streams_are_isolated_and_paired(self):
        env = self.env(filtered=True)
        env.record_observation()
        snapshot = env.capture()
        action = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
        candidates = np.stack([action, action])

        def run(bundle):
            continuation_draws = []
            disturbance_draws = []

            def continuation(history, step, rng):
                del history, step
                draw = int(rng.integers(0, 2**31))
                continuation_draws.append(draw)
                value = ((draw % 1001) / 1000.0 - 0.5) * 0.02
                return np.full(12, value, dtype=np.float32)

            def disturbance(target_env, step, rng):
                del target_env, step
                disturbance_draws.append(int(rng.integers(0, 2**31)))

            evaluation = evaluate_same_state_group(
                env,
                snapshot,
                candidates,
                bundle,
                horizon_steps=3,
                continuation_policy=continuation,
                disturbance_program=disturbance,
            )
            self.assertEqual(len(continuation_draws), 2 * 2 * 2)
            self.assertEqual(len(disturbance_draws), 2 * 2 * 3)
            continuation_trace = np.asarray(continuation_draws).reshape(2, 2, 2)
            disturbance_trace = np.asarray(disturbance_draws).reshape(2, 2, 3)
            np.testing.assert_array_equal(
                continuation_trace[0], continuation_trace[1])
            np.testing.assert_array_equal(
                disturbance_trace[0], disturbance_trace[1])
            np.testing.assert_array_equal(evaluation.fall[0], evaluation.fall[1])
            np.testing.assert_array_equal(
                evaluation.first_failure_step[0],
                evaluation.first_failure_step[1])
            return evaluation, continuation_trace, disturbance_trace

        base = ReplicaSeedBundle(
            crn_id=np.asarray([11, 12], dtype=np.int64),
            rollout_seed=np.asarray([101, 102], dtype=np.int64),
            perturbation_seed=np.asarray([201, 202], dtype=np.int64),
        )
        base_result, base_continuation, base_disturbance = run(base)

        rollout_changed = ReplicaSeedBundle(
            crn_id=base.crn_id,
            rollout_seed=np.asarray([301, 302], dtype=np.int64),
            perturbation_seed=base.perturbation_seed,
        )
        _, changed_continuation, unchanged_disturbance = run(rollout_changed)
        self.assertFalse(np.array_equal(base_continuation, changed_continuation))
        np.testing.assert_array_equal(base_disturbance, unchanged_disturbance)

        perturbation_changed = ReplicaSeedBundle(
            crn_id=base.crn_id,
            rollout_seed=base.rollout_seed,
            perturbation_seed=np.asarray([401, 402], dtype=np.int64),
        )
        _, unchanged_continuation, changed_disturbance = run(
            perturbation_changed)
        np.testing.assert_array_equal(base_continuation, unchanged_continuation)
        self.assertFalse(np.array_equal(base_disturbance, changed_disturbance))

        identity_changed = ReplicaSeedBundle(
            crn_id=np.asarray([71, 72], dtype=np.int64),
            rollout_seed=base.rollout_seed,
            perturbation_seed=base.perturbation_seed,
        )
        identity_result, same_continuation, same_disturbance = run(identity_changed)
        np.testing.assert_array_equal(base_continuation, same_continuation)
        np.testing.assert_array_equal(base_disturbance, same_disturbance)
        np.testing.assert_array_equal(identity_result.crn_id, [71, 72])
        np.testing.assert_array_equal(base_result.crn_id, [11, 12])
        self.assertEqual(base_result.seed_contract, "explicit_three_stream_v1")

    def test_replica_seed_bundle_rejects_invalid_group_arrays(self):
        valid = np.asarray([1, 2], dtype=np.int64)
        cases = (
            ("one-dimensional integer", dict(
                crn_id=np.asarray([1.0, 2.0]),
                rollout_seed=valid,
                perturbation_seed=valid)),
            ("one-dimensional integer", dict(
                crn_id=np.asarray([[1, 2]], dtype=np.int64),
                rollout_seed=valid,
                perturbation_seed=valid)),
            ("nonnegative", dict(
                crn_id=np.asarray([-1, 2], dtype=np.int64),
                rollout_seed=valid,
                perturbation_seed=valid)),
            ("unique across replicas", dict(
                crn_id=np.asarray([1, 1], dtype=np.int64),
                rollout_seed=valid,
                perturbation_seed=valid)),
            ("identical one-dimensional shapes", dict(
                crn_id=valid,
                rollout_seed=np.asarray([3], dtype=np.int64),
                perturbation_seed=valid)),
        )
        for expected, values in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(
                    ValueError, expected):
                ReplicaSeedBundle(**values)

    def test_native_group_rejects_invalid_horizon_and_candidate_values(self):
        env = self.env(filtered=True)
        env.record_observation()
        snapshot = env.capture()
        action = np.zeros(12, dtype=np.float32)
        candidates = np.stack([action, action])

        def continuation(history, step, rng):
            del history, step, rng
            return action

        for horizon in (True, 1.5, 32767):
            with self.subTest(horizon=horizon), self.assertRaisesRegex(
                    ValueError, "horizon_steps"):
                evaluate_same_state_group(
                    env,
                    snapshot,
                    candidates,
                    np.asarray([11], dtype=np.int64),
                    horizon_steps=horizon,  # type: ignore[arg-type]
                    continuation_policy=continuation,
                )
        invalid_nonfinite = candidates.copy()
        invalid_nonfinite[1, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "finite actions"):
            evaluate_same_state_group(
                env,
                snapshot,
                invalid_nonfinite,
                np.asarray([11], dtype=np.int64),
                horizon_steps=2,
                continuation_policy=continuation,
            )
        invalid_bounds = candidates.copy()
        invalid_bounds[1, 3] = 1.1
        with self.assertRaisesRegex(ValueError, r"normalized \[-1, 1\]"):
            evaluate_same_state_group(
                env,
                snapshot,
                invalid_bounds,
                np.asarray([11], dtype=np.int64),
                horizon_steps=2,
                continuation_policy=continuation,
            )
        invalid_options = (
            (np.asarray([1.0, 2.0]), "one-dimensional integer"),
            (np.asarray([[1, 2]], dtype=np.int64), "one-dimensional integer"),
            (np.asarray([1], dtype=np.int64), "one-dimensional integer"),
            (np.asarray([2, 2], dtype=np.int64), "nominal candidate"),
            (np.asarray([1, 0], dtype=np.int64), r"\[1, 4\]"),
            (np.asarray([1, 5], dtype=np.int64), r"\[1, 4\]"),
        )
        for option_steps, expected in invalid_options:
            with self.subTest(option_steps=option_steps), self.assertRaisesRegex(
                    ValueError, expected):
                evaluate_same_state_group(
                    env,
                    snapshot,
                    candidates,
                    np.asarray([11], dtype=np.int64),
                    horizon_steps=2,
                    continuation_policy=continuation,
                    option_steps=option_steps,
                )

    def test_fingerprint_locks_runtime_pd_gains(self):
        fingerprint = self.env().simulator_fingerprint()
        self.assertEqual(fingerprint["kp"], [60.0] * 12)
        self.assertEqual(fingerprint["kd"], [5.0] * 12)
        self.assertEqual(len(fingerprint["mjcf_xml_sha256"]), 64)
        self.assertEqual(fingerprint["failure_measurement"], {
            "height_reference": "base_link_body_origin_world_z",
            "cadence": "post_policy_step_after_all_low_level_substeps",
            "low_level_substeps_per_policy_step": fingerprint["substeps"],
        })


if __name__ == "__main__":
    unittest.main()
