from __future__ import annotations

import json
import unittest

import numpy as np

from safety_data.label_reliability import (
    LabelReliabilityError,
    PARTITION_SCHEMA_VERSION,
    evaluate_independent_replica_label_gate,
)


def _partition(replicas: int = 4) -> dict[str, object]:
    half = replicas // 2
    return {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "assignment_timing": "before_candidate_outcomes",
        "axis": "replica",
        "ordering": "discovery_then_audit",
        "discovery_indices": list(range(half)),
        "audit_indices": list(range(half, replicas)),
        "discovery_replicas": half,
        "audit_replicas": replicas - half,
        "exhaustive": True,
    }


def _thresholds(**changes) -> dict[str, object]:
    result: dict[str, object] = {
        "min_discovery_to_audit_absolute_reduction": 0.20,
        "min_reduction_ci_low": 0.0,
        "min_pair_order_agreement": 0.75,
        "min_pair_order_agreement_ci_low": 0.50,
        "bootstrap_replicates": 200,
        "bootstrap_seed": 17,
        "confidence_level": 0.90,
    }
    result.update(changes)
    return result


class StubDataset:
    def __init__(
        self,
        fall: np.ndarray,
        *,
        acceptance_probability: np.ndarray | None = None,
        trajectory_id: np.ndarray | None = None,
        candidate_mask: np.ndarray | None = None,
        partition: dict[str, object] | None = None,
    ):
        fall = np.asarray(fall)
        groups, candidates, replicas = fall.shape
        self.arrays = {
            "fall": fall,
            "candidate_mask": (
                np.ones((groups, candidates), dtype=bool)
                if candidate_mask is None else np.asarray(candidate_mask)),
            "acceptance_probability": (
                np.ones(groups, dtype=np.float64)
                if acceptance_probability is None
                else np.asarray(acceptance_probability)),
            "trajectory_id": (
                np.asarray([f"trajectory-{index}" for index in range(groups)])
                if trajectory_id is None else np.asarray(trajectory_id)),
        }
        self.manifest = {
            "collection_protocol": {
                "replica_partition": (
                    _partition(replicas) if partition is None else partition),
            },
        }
        self.validation_error: Exception | None = None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def validate(self):
        if self.validation_error is not None:
            raise self.validation_error
        return {"valid": True}


class IndependentReplicaLabelGateTest(unittest.TestCase):
    def test_primary_uses_discovery_for_selection_and_audit_for_evaluation(self):
        # Candidate 1 is independently safer than nominal in both halves.
        # Candidate 2 looks safer only in discovery and is worse in audit.
        per_group = np.asarray([
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1]],
        ], dtype=np.int8)
        dataset = StubDataset(np.repeat(per_group, 6, axis=0))

        report = evaluate_independent_replica_label_gate(
            dataset, _thresholds(min_pair_order_agreement=0.65))

        self.assertAlmostEqual(
            report["primary"]["discovery_to_audit_absolute_reduction"], 0.5)
        self.assertAlmostEqual(
            report["primary"][
                "uniform_tie_expected_selected_audit_fall_risk"], 0.5)
        self.assertEqual(report["primary"]["groups_with_discovery_min_tie"], 6)
        self.assertAlmostEqual(
            report["pair_order_agreement"]["estimate"], 2.0 / 3.0)
        self.assertEqual(report["pair_order_agreement"]["tie_comparisons"], 12)
        self.assertTrue(report["pass"])
        self.assertTrue(all(report["checks"].values()))
        json.dumps(report, allow_nan=False)

    def test_uniform_ties_prevent_candidate_row_order_from_changing_primary(self):
        original = np.asarray([
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1]],
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 1, 1]],
        ], dtype=np.int8)
        swapped = original[:, [0, 2, 1], :]
        loose = _thresholds(
            min_discovery_to_audit_absolute_reduction=-1.0,
            min_reduction_ci_low=-1.0,
            min_pair_order_agreement=0.0,
            min_pair_order_agreement_ci_low=0.0,
        )
        first = evaluate_independent_replica_label_gate(
            StubDataset(original), loose)
        second = evaluate_independent_replica_label_gate(
            StubDataset(swapped), loose)
        self.assertEqual(first["primary"], second["primary"])
        self.assertEqual(
            first["pair_order_agreement"], second["pair_order_agreement"])

    def test_ipw_is_group_macro_and_bootstrap_resamples_trajectory_clusters(self):
        # Group 0 has +1 reduction, group 1 has -1.  With p=[1, .25], IPW
        # weights are [1, 4], hence (1 - 4) / 5 = -0.6.  Repeated groups in
        # the same trajectory remain together in cluster resampling.
        fall = np.asarray([
            [[1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 0, 0], [0, 0, 1, 1]],
        ], dtype=np.int8)
        report = evaluate_independent_replica_label_gate(
            StubDataset(
                fall,
                acceptance_probability=np.asarray([1.0, 0.25]),
                trajectory_id=np.asarray(["positive", "negative"])),
            _thresholds(
                min_discovery_to_audit_absolute_reduction=-1.0,
                min_reduction_ci_low=-1.0,
                min_pair_order_agreement=0.0,
                min_pair_order_agreement_ci_low=0.0,
            ),
        )
        self.assertAlmostEqual(
            report["primary"]["discovery_to_audit_absolute_reduction"], -0.6)
        self.assertAlmostEqual(report["ipw_effective_groups"], 25.0 / 17.0)
        interval = report["primary"]["confidence_interval"]
        self.assertLessEqual(interval["low"], -1.0)
        self.assertGreaterEqual(interval["high"], 1.0)

    def test_pair_ties_are_half_credit_and_full_oracle_is_diagnostic_only(self):
        fall = np.asarray([
            [[1, 0, 1, 0], [0, 1, 1, 0]],
            [[1, 0, 1, 0], [0, 1, 1, 0]],
        ], dtype=np.int8)
        report = evaluate_independent_replica_label_gate(
            StubDataset(fall),
            _thresholds(
                min_discovery_to_audit_absolute_reduction=-1.0,
                min_reduction_ci_low=-1.0,
                min_pair_order_agreement=0.5,
                min_pair_order_agreement_ci_low=0.5,
            ),
        )
        self.assertEqual(report["pair_order_agreement"]["estimate"], 0.5)
        self.assertEqual(report["pair_order_agreement"]["tie_score"], 0.5)
        diagnostic = report["diagnostics_not_gate_eligible"][
            "biased_same_replica_full_oracle"]
        self.assertFalse(diagnostic["gate_eligible"])
        self.assertIn("optimistically biased", diagnostic["bias_warning"])
        self.assertNotIn("biased_same_replica_full_oracle", report["checks"])


class IndependentReplicaFailClosedTest(unittest.TestCase):
    def setUp(self):
        self.fall = np.asarray([
            [[1, 1, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0]],
        ], dtype=np.int8)

    def assert_partition_invalid(self, change: dict[str, object], pattern: str):
        partition = _partition()
        partition.update(change)
        with self.assertRaisesRegex(LabelReliabilityError, pattern):
            evaluate_independent_replica_label_gate(
                StubDataset(self.fall, partition=partition), _thresholds())

    def test_rejects_missing_unknown_or_invalid_thresholds(self):
        missing = _thresholds()
        missing.pop("confidence_level")
        with self.assertRaisesRegex(
                LabelReliabilityError, "missing=.*confidence_level"):
            evaluate_independent_replica_label_gate(StubDataset(self.fall), missing)

        with self.assertRaisesRegex(LabelReliabilityError, "unknown=.*extra"):
            evaluate_independent_replica_label_gate(
                StubDataset(self.fall), _thresholds(extra=1))
        with self.assertRaisesRegex(LabelReliabilityError, "bootstrap_replicates"):
            evaluate_independent_replica_label_gate(
                StubDataset(self.fall), _thresholds(bootstrap_replicates=True))
        with self.assertRaisesRegex(LabelReliabilityError, "confidence_level"):
            evaluate_independent_replica_label_gate(
                StubDataset(self.fall), _thresholds(confidence_level=float("nan")))

    def test_rejects_partition_not_declared_before_outcomes_or_not_exhaustive(self):
        self.assert_partition_invalid(
            {"assignment_timing": "after_candidate_outcomes"},
            "assignment_timing")
        self.assert_partition_invalid({"exhaustive": False}, "exhaustive")
        self.assert_partition_invalid(
            {"discovery_indices": [0, 2], "audit_indices": [1, 3]},
            "discovery_then_audit")
        self.assert_partition_invalid(
            {"audit_indices": [1, 2], "audit_replicas": 2}, "overlap")
        self.assert_partition_invalid(
            {"schema_version": "qsafe.partition.v1"}, "schema_version")

    def test_rejects_unknown_partition_key_and_dataset_validation_failure(self):
        partition = _partition()
        partition["posthoc"] = True
        with self.assertRaisesRegex(LabelReliabilityError, "unknown=.*posthoc"):
            evaluate_independent_replica_label_gate(
                StubDataset(self.fall, partition=partition), _thresholds())

        dataset = StubDataset(self.fall)
        dataset.validation_error = ValueError("upstream dataset invalid")
        with self.assertRaisesRegex(ValueError, "upstream dataset invalid"):
            evaluate_independent_replica_label_gate(dataset, _thresholds())

    def test_rejects_nonbinary_fall_bad_probability_and_single_cluster(self):
        nonbinary = self.fall.copy()
        nonbinary[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "fall labels must be binary"):
            evaluate_independent_replica_label_gate(
                StubDataset(nonbinary), _thresholds())
        with self.assertRaisesRegex(ValueError, "acceptance_probability"):
            evaluate_independent_replica_label_gate(
                StubDataset(
                    self.fall,
                    acceptance_probability=np.asarray([1.0, 0.0])),
                _thresholds())
        with self.assertRaisesRegex(ValueError, "at least two trajectories"):
            evaluate_independent_replica_label_gate(
                StubDataset(
                    self.fall,
                    trajectory_id=np.asarray(["same", "same"])),
                _thresholds())


if __name__ == "__main__":
    unittest.main()
