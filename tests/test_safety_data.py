from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safety_data.legacy import audit_legacy_p17
from safety_data.metrics import _equal_mass_ece, evaluate_predictions
from safety_data.paths import ProtectedEvidencePathError, assert_development_path
from safety_data.schema import (
    DatasetValidationError,
    GroupedBranchDataset,
    PRIVILEGED_SCHEMA_VERSION,
    PrivilegedBranchView,
    SCHEMA_VERSION,
    audit_split_disjointness,
)


def synthetic_dataset(*, split: str = "development_train", offset: int = 0):
    groups, candidates, replicas, horizon = 4, 3, 4, 32
    target = np.asarray([
        [1.00, 0.00, 0.50],
        [0.75, 0.25, 0.50],
        [0.50, 0.50, 0.50],
        [0.25, 0.75, 0.00],
    ], dtype=np.float32)
    fall = np.zeros((groups, candidates, replicas), dtype=bool)
    for group in range(groups):
        for candidate in range(candidates):
            fall[group, candidate, :int(target[group, candidate] * replicas)] = True
    q_send = np.broadcast_to(
        np.asarray([0.05, 0.7, -1.4] * 4, dtype=np.float32),
        (groups, 5, 12)).copy()
    observations = np.zeros((groups, 5, 46), dtype=np.float32)
    observations[..., -12:] = q_send
    nominal = np.linspace(-0.2, 0.2, groups * 12, dtype=np.float32).reshape(groups, 12)
    requested = np.repeat(nominal[:, None, :], candidates, axis=1)
    requested[:, 1] = np.clip(requested[:, 1] + 0.1, -1.0, 1.0)
    requested[:, 2] = np.clip(requested[:, 2] - 0.1, -1.0, 1.0)
    init_qpos = np.asarray([0.05, 0.7, -1.4] * 4, dtype=np.float32)
    action_offset = np.asarray([0.2, 0.4, 0.4] * 4, dtype=np.float32)
    q_target = init_qpos[None, None, :] + requested * action_offset[None, None, :]
    first_failure = np.where(fall, 2, horizon + 1).astype(np.int16)
    base = offset * 10_000
    arrays = {
        "group_id": np.asarray([f"group-{base + i}" for i in range(groups)]),
        "state_hash": np.asarray([
            hashlib.sha256(f"state-{base + i}".encode()).hexdigest()
            for i in range(groups)]),
        "trajectory_id": np.asarray([f"trajectory-{base + i}" for i in range(groups)]),
        "episode_id": np.arange(base, base + groups, dtype=np.int64),
        "episode_step": np.arange(groups, dtype=np.int32),
        "policy_training_seed": np.arange(base + 10, base + 10 + groups),
        "source_seed": np.arange(base + 20, base + 20 + groups),
        "policy_source": np.asarray(["frozen_sac"] * groups),
        "command_vx": np.full(groups, 0.30, dtype=np.float32),
        "acceptance_probability": np.ones(groups, dtype=np.float32),
        "obs_history": observations,
        "q_send_history": q_send,
        "nominal_action_requested": nominal,
        "candidate_requested": requested,
        "candidate_executed": requested.copy(),
        "candidate_q_target": q_target,
        "candidate_kind": np.asarray([
            ["nominal", "local_plus", "local_minus"] for _ in range(groups)]),
        "candidate_mask": np.ones((groups, candidates), dtype=bool),
        "fall": fall,
        "first_failure_step": first_failure,
        "max_tilt_rad": np.where(fall, 1.2, 0.1).astype(np.float32),
        "min_height_m": np.where(fall, 0.1, 0.3).astype(np.float32),
        "crn_id": np.arange(
            base + 100, base + 100 + groups * replicas).reshape(groups, replicas),
        "rollout_seed": np.arange(
            base + 200, base + 200 + groups * replicas).reshape(groups, replicas),
        "perturbation_seed": np.arange(
            base + 300, base + 300 + groups * replicas).reshape(groups, replicas),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "feature_view": "deployable",
        "horizon_steps": horizon,
        "generator_commit": "synthetic-test",
        "simulator_fingerprint": {"backend": "synthetic", "version": 1},
        "source_policy": {"name": "frozen_sac", "sha256": "source"},
        "continuation_policy": {"name": "frozen_sac", "sha256": "continuation"},
        "candidate_protocol": {"nominal_index": 0, "count": candidates},
        "fall_definition": {
            "max_abs_roll_pitch_rad": 1.047198,
            "min_base_height_m": 0.18,
        },
        "observation_contract": {
            "frames": 5,
            "dimension": 46,
            "tail_semantic": "previous_absolute_action_q_target",
        },
        "action_application_contract": {
            "q_target_semantic": "absolute_joint_position_sent",
            "init_qpos": [0.05, 0.7, -1.4] * 4,
            "action_offset": [0.2, 0.4, 0.4] * 4,
            "joint_min": [-1.05, -1.57, -2.72] * 4,
            "joint_max": [1.05, 3.49, -0.84] * 4,
        },
        "state_hash_contract": "sha256_compound_snapshot_v1",
    }
    return GroupedBranchDataset(manifest, arrays), target


class GroupedBranchDatasetTest(unittest.TestCase):
    def test_round_trip_is_hashed_and_pickle_free(self):
        dataset, _ = synthetic_dataset()
        report = dataset.validate()
        self.assertEqual(report["groups"], 4)
        with tempfile.TemporaryDirectory() as directory:
            path = dataset.save(Path(directory) / "development.npz")
            restored = GroupedBranchDataset.load(path)
        self.assertEqual(restored.manifest["content_sha256"], report["content_sha256"])
        np.testing.assert_array_equal(restored["fall"], dataset["fall"])

    def test_rejects_four_frames_and_normalized_tail(self):
        dataset, _ = synthetic_dataset()
        dataset.arrays["obs_history"] = dataset["obs_history"][:, :4]
        with self.assertRaisesRegex(DatasetValidationError, "obs_history shape"):
            dataset.validate()

        dataset, _ = synthetic_dataset()
        dataset.arrays["obs_history"][..., -12:] = 0.0
        dataset.arrays["q_send_history"][...] = 0.0
        with self.assertRaisesRegex(DatasetValidationError, "physical joint bounds"):
            dataset.validate()

    def test_rejects_privileged_features_in_deployable_file(self):
        dataset, _ = synthetic_dataset()
        dataset.arrays["privileged_qpos"] = np.zeros((4, 19), np.float32)
        with self.assertRaisesRegex(DatasetValidationError, "physically separate"):
            dataset.validate()

    def test_privileged_view_is_separate_and_identity_aligned(self):
        dataset, _ = synthetic_dataset()
        view = PrivilegedBranchView(
            manifest={
                "schema_version": PRIVILEGED_SCHEMA_VERSION,
                "feature_view": "privileged_diagnostic_only",
            },
            group_id=dataset["group_id"].copy(),
            state_hash=dataset["state_hash"].copy(),
            features=np.arange(16, dtype=np.float32).reshape(4, 4),
            feature_names=np.asarray(["qpos", "qvel", "contact", "com"]),
        )
        report = view.validate(dataset)
        self.assertEqual(report["features"], 4)
        with tempfile.TemporaryDirectory() as directory:
            deployable_path = dataset.save(Path(directory) / "deployable.npz")
            privileged_path = view.save(Path(directory) / "privileged.npz")
            deployable = GroupedBranchDataset.load(deployable_path)
            restored = PrivilegedBranchView.load(
                privileged_path, deployable=deployable)
        np.testing.assert_array_equal(restored.features, view.features)
        view.manifest.pop("content_sha256")
        view.state_hash[0] = "wrong-state"
        with self.assertRaisesRegex(DatasetValidationError, "state_hash"):
            view.validate(dataset)

    def test_split_audit_checks_trajectory_state_seed_and_crn(self):
        train, _ = synthetic_dataset()
        validation, _ = synthetic_dataset(
            split="development_validation", offset=1)
        report = audit_split_disjointness([train, validation])
        self.assertEqual(report["pairs_checked"], 1)
        validation.arrays["trajectory_id"][0] = train["trajectory_id"][0]
        with self.assertRaisesRegex(DatasetValidationError, "trajectory_id leaks"):
            audit_split_disjointness([train, validation])

    def test_content_hash_detects_in_memory_tamper_after_load(self):
        dataset, _ = synthetic_dataset()
        with tempfile.TemporaryDirectory() as directory:
            path = dataset.save(Path(directory) / "development.npz")
            restored = GroupedBranchDataset.load(path)
        restored.arrays["max_tilt_rad"][0, 0, 0] += 0.01
        with self.assertRaisesRegex(DatasetValidationError, "content hash mismatch"):
            restored.validate()

    def test_on_disk_dataset_without_hash_is_rejected(self):
        dataset, _ = synthetic_dataset()
        with tempfile.TemporaryDirectory() as directory:
            source = dataset.save(Path(directory) / "source.npz")
            with np.load(source, allow_pickle=False) as payload:
                arrays = {
                    name: payload[name].copy()
                    for name in payload.files if name != "manifest_json"}
                manifest = json.loads(str(payload["manifest_json"].item()))
            manifest.pop("content_sha256")
            target = Path(directory) / "missing-hash.npz"
            np.savez_compressed(
                target,
                manifest_json=np.asarray(json.dumps(manifest)),
                **arrays,
            )
            with self.assertRaisesRegex(DatasetValidationError, "requires.*hash"):
                GroupedBranchDataset.load(target)


class MetricsTest(unittest.TestCase):
    def test_perfect_empirical_predictor_has_known_group_macro_metrics(self):
        dataset, target = synthetic_dataset()
        result = evaluate_predictions(dataset, target)
        self.assertAlmostEqual(result["pair_accuracy_group_macro"], 1.0)
        self.assertAlmostEqual(result["strong_pair_accuracy_group_macro"], 1.0)
        self.assertAlmostEqual(result["brier_group_macro"], 1.0 / 6.0)
        self.assertAlmostEqual(result["empirical_risk_mse_group_macro"], 0.0)
        self.assertAlmostEqual(result["ece_equal_mass"], 0.0)
        self.assertAlmostEqual(result["nominal_fall_risk"], 0.625)
        self.assertAlmostEqual(result["selected_fall_risk"], 0.1875)
        self.assertAlmostEqual(result["top1_absolute_reduction"], 0.4375)
        self.assertAlmostEqual(result["oracle_gap_capture"], 1.0)
        self.assertLess(result["ece_max_bin_mass_error"], 1e-12)

    def test_ece_is_invariant_to_row_order_inside_prediction_ties(self):
        prediction = np.full(4, 0.5)
        weight = np.ones(4)
        first, first_mass = _equal_mass_ece(
            prediction, np.asarray([0.0, 0.0, 1.0, 1.0]), weight, 2)
        second, second_mass = _equal_mass_ece(
            prediction, np.asarray([0.0, 1.0, 0.0, 1.0]), weight, 2)
        self.assertAlmostEqual(first, 0.0)
        self.assertAlmostEqual(first, second)
        np.testing.assert_allclose(first_mass, second_mass, rtol=0.0, atol=1e-15)

    def test_extreme_acceptance_probabilities_keep_ipw_finite(self):
        dataset, target = synthetic_dataset()
        dataset.arrays["acceptance_probability"] = np.asarray(
            [1e-30, 1.0, 1.0, 1.0], dtype=np.float64)
        result = evaluate_predictions(dataset, target)
        self.assertTrue(np.isfinite(result["brier_group_macro"]))
        self.assertTrue(np.isfinite(result["ece_equal_mass"]))
        self.assertLess(result["ece_max_bin_mass_error"], 1e-12)

    def test_candidate_rows_are_not_used_as_pair_macro_units(self):
        dataset, target = synthetic_dataset()
        prediction = target.copy()
        prediction[0] = prediction[0, ::-1]
        result = evaluate_predictions(dataset, prediction)
        self.assertGreater(result["pair_groups"], 1)
        self.assertLess(result["pair_accuracy_group_macro"], 1.0)
        self.assertGreaterEqual(result["pair_accuracy_group_macro"], 0.0)


class LeakageAndLegacyTest(unittest.TestCase):
    def test_protected_paths_are_case_insensitive_and_fail_before_io(self):
        for path in (
            "/tmp/sealed/result.npz",
            "/tmp/Formal_Test.npz",
            "/tmp/FORMAL_DATA/report.json",
            "/tmp/formal200/metrics.json",
        ):
            with self.assertRaises(ProtectedEvidencePathError):
                assert_development_path(path)

    def test_legacy_adapter_is_opt_in_and_never_evidence_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez_compressed(
                path,
                observation_histories=np.zeros((2, 4, 46), np.float32),
                actions=np.zeros((2, 12), np.float32),
            )
            with self.assertRaisesRegex(ValueError, "opt-in"):
                audit_legacy_p17(path)
            report = audit_legacy_p17(path, acknowledge_legacy=True)
        self.assertFalse(report["evidence_eligible"])
        self.assertIn("history_is_not_5x46", report["issues"])


if __name__ == "__main__":
    unittest.main()
