from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from safety_data.paths import workflow_evidence_read_scope
from safety_data.schema import (
    CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS,
    CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS,
    CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION,
    DatasetValidationError,
)
from tests.test_safety_data import synthetic_dataset


def closed_loop_dataset():
    dataset, _ = synthetic_dataset()
    groups = dataset.group_count
    candidates = len(CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS)
    dataset.manifest["horizon_steps"] = 96
    dataset.manifest["candidate_protocol"] = {
        "protocol_version": CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION,
        "count": candidates,
        "ordered_names": list(CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS),
        "behavior_steps_array": "candidate_behavior_steps",
        "behavior_override_steps": list(
            CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS),
    }
    for name in (
        "candidate_requested", "candidate_executed", "candidate_q_target",
    ):
        dataset.arrays[name] = np.repeat(
            dataset.arrays[name][:, :1], candidates, axis=1)
    dataset.arrays["nominal_action_requested"] = dataset.arrays[
        "candidate_requested"][:, 0].copy()
    dataset.arrays["candidate_kind"] = np.repeat(
        np.asarray(CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS, dtype=str)[None, :],
        groups,
        axis=0,
    )
    dataset.arrays["candidate_mask"] = np.ones(
        (groups, candidates), dtype=bool)
    for name in (
        "fall", "first_failure_step", "max_tilt_rad", "min_height_m",
    ):
        dataset.arrays[name] = np.repeat(
            dataset.arrays[name][:, :1], candidates, axis=1)
    dataset.arrays["first_failure_step"][~dataset.arrays["fall"]] = 97
    dataset.arrays["candidate_behavior_steps"] = np.repeat(
        np.asarray(CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS, dtype=np.int16)[
            None, :],
        groups,
        axis=0,
    )
    return dataset


class ClosedLoopRecoverySchemaTest(unittest.TestCase):
    def test_locked_k9_behavior_contract_validates(self):
        dataset = closed_loop_dataset()
        report = dataset.validate()
        self.assertEqual(report["max_candidates"], 9)
        self.assertEqual(report["groups"], 4)

    def test_behavior_steps_require_v3_and_exact_order(self):
        dataset = closed_loop_dataset()
        dataset.manifest["candidate_protocol"] = {
            "nominal_index": 0,
            "count": 9,
        }
        with self.assertRaisesRegex(
                DatasetValidationError, "requires the closed-loop recovery"):
            dataset.validate()

        dataset = closed_loop_dataset()
        dataset.arrays["candidate_behavior_steps"][:, 1] = 11
        with self.assertRaisesRegex(DatasetValidationError, "locked K9 order"):
            dataset.validate()

    def test_legacy_and_behavior_duration_arrays_are_mutually_exclusive(self):
        dataset = closed_loop_dataset()
        dataset.arrays["candidate_option_steps"] = np.ones(
            (dataset.group_count, dataset.candidate_count), dtype=np.int8)
        with self.assertRaisesRegex(DatasetValidationError, "mutually exclusive"):
            dataset.validate()

    def test_behavior_array_is_horizon_bounded(self):
        dataset = closed_loop_dataset()
        dataset.manifest["horizon_steps"] = 25
        dataset.arrays["first_failure_step"][~dataset.arrays["fall"]] = 26
        with self.assertRaisesRegex(DatasetValidationError, "recovery behaviors"):
            dataset.validate()

    def test_manifest_order_and_array_kinds_both_bind_identity(self):
        dataset = closed_loop_dataset()
        swapped = copy.deepcopy(dataset.manifest["candidate_protocol"])
        swapped["ordered_names"][1], swapped["ordered_names"][2] = (
            swapped["ordered_names"][2], swapped["ordered_names"][1])
        dataset.manifest["candidate_protocol"] = swapped
        with self.assertRaisesRegex(DatasetValidationError, "locked closed-loop"):
            dataset.validate()

    def test_locked_audit_load_requires_bound_marker_and_rejects_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = closed_loop_dataset().save(root / "staging.npz")
            audit = root / "source-7801.audit.npz"
            staging.rename(audit)
            with workflow_evidence_read_scope(
                    workflow="objective1_closed_loop_recovery_triage_v3",
                    role="audit",
                    path=audit):
                with self.assertRaisesRegex(
                        DatasetValidationError, "audit-consumed.*missing"):
                    type(closed_loop_dataset()).load(audit)

            alias = root / "innocent-looking.npz"
            alias.symlink_to(audit)
            with self.assertRaisesRegex(DatasetValidationError, "symlink"):
                type(closed_loop_dataset()).load(alias)

            forged = {
                "schema_version": (
                    "qsafe.closed_loop_recovery_triage.audit_consumed.v1"),
                "protocol_name": "objective1_closed_loop_recovery_triage_v3",
                "protocol_contract_sha256": "a" * 64,
                "protocol_file_sha256": "b" * 64,
                "selection_lock_sha256": "c" * 64,
                "audit_identifier": "d" * 64,
                "created_at_utc": "2026-08-09T00:00:00+00:00",
                "status": "irreversibly_consumed_before_outcome_read",
            }
            (root / "audit-consumed.json").write_text(
                json.dumps(forged), encoding="utf-8")
            with workflow_evidence_read_scope(
                    workflow="objective1_closed_loop_recovery_triage_v3",
                    role="audit",
                    path=audit):
                with self.assertRaisesRegex(
                        DatasetValidationError, "selection lock"):
                    type(closed_loop_dataset()).load(audit)

    def test_audit_basename_as_ancestor_is_rejected_before_lstat(self):
        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory) / "source-7801.audit.npz" / "child.npz")
            with mock.patch.object(
                    Path, "lstat",
                    side_effect=AssertionError("audit path must not be probed")):
                with self.assertRaisesRegex(
                        DatasetValidationError, "path ancestor"):
                    type(closed_loop_dataset()).load(path)


if __name__ == "__main__":
    unittest.main()
