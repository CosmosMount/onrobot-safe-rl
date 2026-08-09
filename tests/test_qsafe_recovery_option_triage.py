from __future__ import annotations

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from safety_data.label_reliability import LabelReliabilityError
from safety_data.recovery_option_triage import evaluate_recovery_option_triage
from safety_data.recovery_options import (
    RECOVERY_OPTION_COUNT,
    RECOVERY_OPTION_KINDS,
    RECOVERY_OPTION_STEPS,
    RecoveryOptionCandidateConfig,
)
from scripts.collect_recovery_option_triage import (
    _bind_cohort_lock,
    _load_locked_protocol,
    _output_bundle,
)


PROTOCOL = Path("config/qsafe_recovery_option_triage_v2.yaml")


def _partition() -> dict[str, object]:
    return {
        "schema_version": "qsafe.independent_replica_partition.v2",
        "assignment_timing": "before_candidate_outcomes",
        "axis": "replica",
        "ordering": "discovery_then_audit",
        "discovery_indices": [0, 1],
        "audit_indices": [2, 3],
        "discovery_replicas": 2,
        "audit_replicas": 2,
        "exhaustive": True,
    }


def _protocol() -> dict:
    value = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    value = copy.deepcopy(value)
    collection = value["collection"]
    collection["source_seeds"] = [1, 2, 3]
    collection["groups_per_source_seed"] = 2
    collection["total_groups"] = 6
    collection["total_replicas"] = 4
    collection["replica_partition"] = _partition()
    gates = value["triage_gates"]
    gates["data"] = {
        "min_independent_groups": 6,
        "min_trajectory_clusters": 6,
        "required_source_seeds": [1, 2, 3],
        "candidates_per_group": 29,
        "discovery_replicas": 2,
        "audit_replicas": 2,
    }
    gates["one_step_A"] = {
        "min_audit_absolute_reduction": 0.20,
        "min_reduction_lcb": 0.0,
        "min_discovery_to_audit_pair_order_agreement": 0.0,
        "require_each_source_seed_same_positive_direction": True,
    }
    gates["multistep_B"] = {
        "min_audit_absolute_reduction": 0.50,
        "min_reduction_lcb": 0.0,
        "min_improvement_over_locked_L1": 0.20,
        "min_improvement_over_L1_lcb": 0.0,
        "require_each_source_seed_same_positive_direction": True,
    }
    value["statistics"]["bootstrap"].update({
        "replicates": 100,
        "seed": 11,
    })
    return value


class StubDataset:
    def __init__(self, fall: np.ndarray):
        fall = np.asarray(fall, dtype=np.int8)
        groups = fall.shape[0]
        option_steps = np.asarray(RECOVERY_OPTION_STEPS, dtype=np.int8)
        kinds = np.asarray(RECOVERY_OPTION_KINDS, dtype=str)
        self.arrays = {
            "fall": fall,
            "candidate_mask": np.ones(
                (groups, RECOVERY_OPTION_COUNT), dtype=bool),
            "acceptance_probability": np.ones(groups, dtype=np.float64),
            "trajectory_id": np.asarray([
                f"trajectory-{index}" for index in range(groups)]),
            "source_seed": np.asarray([1, 1, 2, 2, 3, 3]),
            "candidate_option_steps": np.repeat(
                option_steps[None, :], groups, axis=0),
            "candidate_kind": np.repeat(kinds[None, :], groups, axis=0),
        }
        self.manifest = {
            "candidate_protocol": (
                RecoveryOptionCandidateConfig().manifest_protocol()),
            "collection_protocol": {"replica_partition": _partition()},
            "content_sha256": "a" * 64,
        }

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def validate(self):
        return {"valid": True}


def _base_fall() -> np.ndarray:
    # Nominal and every option initially fall on every replica.
    return np.ones((6, RECOVERY_OPTION_COUNT, 4), dtype=np.int8)


class RecoveryOptionTriageTest(unittest.TestCase):
    def test_discovery_selects_l3_and_independent_audit_passes(self):
        fall = _base_fall()
        # K29 template-major slots: first template has L1/L2/L3/L4 at 1..4.
        fall[:, 1, :] = np.asarray([0, 0, 0, 1])  # L1 audit effect 0.5
        fall[:, 2, :] = np.asarray([0, 1, 1, 1])  # L2 discovery score 0.5
        fall[:, 3, :] = np.asarray([0, 0, 0, 0])  # L3 score/effect 1.0
        fall[:, 4, :] = np.asarray([0, 1, 1, 1])  # L4 discovery score 0.5

        report = evaluate_recovery_option_triage(
            StubDataset(fall), _protocol())

        self.assertEqual(report["selection"]["selected_multistep_duration"], 3)
        self.assertAlmostEqual(
            report["one_step_A"]["audit_absolute_reduction"], 0.5)
        self.assertAlmostEqual(
            report["multistep_B"]["audit_absolute_reduction"], 1.0)
        self.assertAlmostEqual(
            report["multistep_B"]["improvement_over_locked_L1"], 0.5)
        self.assertTrue(report["one_step_A"]["pass"])
        self.assertTrue(report["multistep_B"]["pass"])
        self.assertEqual(
            report["decision"],
            "authorize_fresh_multistep_protocol_preregistration_only")
        self.assertFalse(report["model_training_authorized"])
        self.assertFalse(report["phase2_authorized"])

    def test_audit_failure_does_not_fall_through_to_safer_runner_up(self):
        fall = _base_fall()
        fall[:, 1, :] = np.asarray([0, 0, 0, 1])  # L1 passes.
        fall[:, 2, :] = np.asarray([0, 0, 1, 1])  # L2 wins discovery, audit 0.
        fall[:, 3, :] = np.asarray([0, 1, 0, 0])  # L3 runner-up, audit effect 1.
        fall[:, 4, :] = np.asarray([1, 1, 0, 0])

        report = evaluate_recovery_option_triage(
            StubDataset(fall), _protocol())

        self.assertEqual(report["selection"]["selected_multistep_duration"], 2)
        self.assertEqual(
            report["multistep_B"]["audit_absolute_reduction"], 0.0)
        self.assertFalse(report["multistep_B"]["pass"])
        self.assertEqual(
            report["decision"],
            "authorize_fresh_high_replica_one_step_preregistration_only")

    def test_independent_audit_zero_stops_model_scaling(self):
        fall = _base_fall()
        # Several discovery-only minima create a large biased discovery Oracle,
        # while every audit outcome equals nominal.
        fall[:, 1, :2] = 0
        fall[:, 2, :2] = 0
        fall[:, 3, :2] = 0
        fall[:, 4, :2] = 0

        report = evaluate_recovery_option_triage(
            StubDataset(fall), _protocol())

        self.assertEqual(report["one_step_A"]["audit_absolute_reduction"], 0.0)
        self.assertEqual(report["multistep_B"]["audit_absolute_reduction"], 0.0)
        self.assertTrue(report["no_headroom_stop"])
        self.assertEqual(
            report["decision"],
            "stop_model_scaling_and_redesign_recovery_action_library")

    def test_rejects_candidate_order_or_partition_drift(self):
        dataset = StubDataset(_base_fall())
        dataset.arrays["candidate_option_steps"][:, [1, 2]] = (
            dataset.arrays["candidate_option_steps"][:, [2, 1]])
        with self.assertRaisesRegex(
                LabelReliabilityError, "candidate_option_steps"):
            evaluate_recovery_option_triage(dataset, _protocol())

        dataset = StubDataset(_base_fall())
        dataset.manifest["collection_protocol"]["replica_partition"][
            "assignment_timing"] = "after_candidate_outcomes"
        with self.assertRaisesRegex(LabelReliabilityError, "assignment_timing"):
            evaluate_recovery_option_triage(dataset, _protocol())


class RecoveryOptionTriageGovernanceTest(unittest.TestCase):
    def test_canonical_protocol_matches_implemented_candidates(self):
        protocol = _load_locked_protocol()
        self.assertEqual(
            protocol["collection"]["candidates"],
            RecoveryOptionCandidateConfig().manifest_protocol(),
        )

    def test_output_root_and_cohort_lock_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            args = SimpleNamespace(
                output=str(root / "seed7601.npz"),
                privileged_output=None,
                report=None,
            )
            outputs = _output_bundle(args, artifact_root=root)
            self.assertTrue(all(path.parent == root for path in outputs))

            outside = SimpleNamespace(
                output=str(Path(directory) / "outside.npz"),
                privileged_output=None,
                report=None,
            )
            with self.assertRaisesRegex(ValueError, "locked artifact root"):
                _output_bundle(outside, artifact_root=root)

            lock = root / "cohort-lock.json"
            expected = _bind_cohort_lock(
                lock,
                generator_commit="abc123",
                protocol_sha256="1" * 64,
                source_seeds=[7601, 7602, 7603],
            )
            self.assertEqual(
                _bind_cohort_lock(
                    lock,
                    generator_commit="abc123",
                    protocol_sha256="1" * 64,
                    source_seeds=[7601, 7602, 7603],
                ),
                expected,
            )
            with self.assertRaisesRegex(RuntimeError, "different protocol/commit"):
                _bind_cohort_lock(
                    lock,
                    generator_commit="different",
                    protocol_sha256="1" * 64,
                    source_seeds=[7601, 7602, 7603],
                )

if __name__ == "__main__":
    unittest.main()
