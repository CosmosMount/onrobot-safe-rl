from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from safety_data.paths import (
    ProtectedEvidencePathError,
    STAGE_B_FROZEN_INPUT_NAMES,
    STAGE_B_MODEL_TEST_REPORT_SCHEMA,
    STAGE_B_PROTOCOL_NAME,
    _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS,
    _canonical_control_json,
    require_workflow_authorized_or_safe_input,
)
from safety_data.schema import GroupedBranchDataset
from safety_data.stage_b_paths import (
    compile_stage_b_model_test_commitment,
    consume_stage_b_model_test,
    create_stage_b_model_test_producer_attempt,
    stage_b_evidence_read_scope,
    stage_b_model_test_producer_read_scope,
)
from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_CONTRACT_SHA256 as PARENT_PROTOCOL_CONTRACT_SHA256,
    PROTOCOL_FILE_SHA256 as PARENT_PROTOCOL_FILE_SHA256,
)
from safety_data.state_dependent_recovery_v5_stage_b import (
    EXECUTION_PROTOCOL_CONTRACT_SHA256,
    EXECUTION_PROTOCOL_FILE_SHA256,
    REDUCED7_AMENDMENT_CONTRACT_SHA256,
    REDUCED7_AMENDMENT_FILE_SHA256,
    STAGE_A_DISPOSITION_COMMIT,
    STAGE_A_REPORT_SHA256,
)
from tests.test_safety_data import synthetic_dataset


class StageBEvidencePathTest(unittest.TestCase):
    def _model_test_fixture(
        self,
        root: Path,
        *,
        valid_deployable: bool = False,
    ) -> tuple[Path, Path, dict[str, str], str]:
        stage_b = root / "stage-b"
        model_test = stage_b / "model-test"
        model_test.mkdir(parents=True)
        deployable = model_test / "labels-r64-deployable.npz"
        for relative in sorted(_STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS):
            if relative == "stage-b/model-test/attempt-started.json":
                continue
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"fixture:{relative}\n".encode("ascii"))
        if valid_deployable:
            ordinary = synthetic_dataset()[0].save(root / "ordinary.npz")
            deployable.unlink()
            ordinary.rename(deployable)
        with mock.patch(
                "safety_data.stage_b_paths.require_clean_stage_b_generator",
                return_value="e" * 40):
            attempt = create_stage_b_model_test_producer_attempt(
                attempt_path=model_test / "attempt-started.json",
                generator_commit="e" * 40,
                created_at_utc="2026-08-10T00:00:00+00:00",
            )
        artifacts = [
            {
                "kind": _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS[relative],
                "path": relative,
                "sha256": hashlib.sha256(
                    (root / relative).read_bytes()).hexdigest(),
            }
            for relative in sorted(_STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS)
        ]
        report = {
            "schema_version": STAGE_B_MODEL_TEST_REPORT_SCHEMA,
            "parent_protocol_name": STAGE_B_PROTOCOL_NAME,
            "parent_protocol_contract_sha256": (
                PARENT_PROTOCOL_CONTRACT_SHA256),
            "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
            "execution_protocol_name": (
                "objective1_state_dependent_recovery_qsafe_v5_stage_b_execution"),
            "execution_protocol_contract_sha256": (
                EXECUTION_PROTOCOL_CONTRACT_SHA256),
            "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
            "roster_amendment_contract_sha256": (
                REDUCED7_AMENDMENT_CONTRACT_SHA256
            ),
            "roster_amendment_file_sha256": (
                REDUCED7_AMENDMENT_FILE_SHA256
            ),
            "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
            "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
            "generator_commit": "e" * 40,
            "role": "model_test",
            "source_seeds": [8701, 8702, 8711, 8712, 8721, 8722],
            "groups": 384,
            "admission_replicas": 32,
            "label_replicas": 64,
            "evidence_artifacts": artifacts,
            "producer_attempt_sha256": attempt["attempt_sha256"],
            "status": "complete_evidence_hashes_only",
            "created_at_utc": "2026-08-10T00:00:00+00:00",
        }
        report_path = model_test / "report.json"
        report_path.write_bytes(_canonical_control_json(report))
        frozen = {name: "9" * 64 for name in STAGE_B_FROZEN_INPUT_NAMES}
        return report_path, deployable, frozen, str(attempt["attempt_sha256"])

    def test_all_five_roles_require_exact_kind_and_path(self):
        roles = (
            "fit",
            "probability_calibration",
            "uncertainty_calibration",
            "selector_calibration",
            "model_test",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "stage-b"
            for role in roles:
                role_directory = role.replace("_", "-")
                replicas = 64 if role == "model_test" else 32
                path = root / role_directory / f"labels-r{replicas}-deployable.npz"
                capability = f"stage_b_{role}_label"
                with self.subTest(role=role):
                    with self.assertRaisesRegex(
                            ProtectedEvidencePathError, "scoped workflow/role"):
                        require_workflow_authorized_or_safe_input(
                            path, allowed_roles=(capability,))
                    if role == "model_test":
                        with self.assertRaisesRegex(
                                ProtectedEvidencePathError,
                                "dedicated producer capability"):
                            with stage_b_evidence_read_scope(
                                    scientific_role=role,
                                    evidence_kind="label",
                                    path=path):
                                pass
                    else:
                        with stage_b_evidence_read_scope(
                                scientific_role=role,
                                evidence_kind="label",
                                path=path):
                            self.assertEqual(
                                require_workflow_authorized_or_safe_input(
                                    path, allowed_roles=(capability,)),
                                path.absolute(),
                            )
                    wrong_directory = (
                        "probability-calibration" if role == "fit" else "fit")
                    wrong = root / wrong_directory / path.name
                    with self.assertRaisesRegex(
                            ProtectedEvidencePathError, "exact scientific-role"):
                        with stage_b_evidence_read_scope(
                                scientific_role=role,
                                evidence_kind="label",
                                path=wrong):
                            pass

    def test_stage_b_label_schema_load_uses_role_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary = synthetic_dataset()[0].save(root / "ordinary.npz")
            target = root / "stage-b/fit/labels-r32-deployable.npz"
            target.parent.mkdir(parents=True)
            ordinary.rename(target)
            with self.assertRaisesRegex(
                    Exception, "scoped workflow/role"):
                GroupedBranchDataset.load(target)
            with stage_b_evidence_read_scope(
                    scientific_role="fit",
                    evidence_kind="label",
                    path=target):
                loaded = GroupedBranchDataset.load(target)
            self.assertEqual(loaded.group_count, 4)

    def test_model_test_blind_producer_has_separate_precommit_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_test = root / "stage-b/model-test"
            model_test.mkdir(parents=True)
            leaf = model_test / "source-8701.labels-r64.npz"
            synthetic_dataset()[0].save(root / "ordinary.npz").rename(leaf)
            with mock.patch(
                    "safety_data.stage_b_paths.require_clean_stage_b_generator",
                    return_value="e" * 40):
                attempt = create_stage_b_model_test_producer_attempt(
                    attempt_path=model_test / "attempt-started.json",
                    generator_commit="e" * 40,
                    created_at_utc="2026-08-10T00:00:00+00:00",
                )
                with self.assertRaisesRegex(
                        ProtectedEvidencePathError, "dedicated producer"):
                    with stage_b_evidence_read_scope(
                            scientific_role="model_test",
                            evidence_kind="label",
                            path=leaf):
                        pass
                with stage_b_model_test_producer_read_scope(
                        attempt_path=model_test / "attempt-started.json",
                        expected_attempt_sha256=attempt["attempt_sha256"],
                        evidence_paths=[leaf]) as checked:
                    self.assertEqual(checked, [leaf.absolute()])
                    self.assertEqual(
                        GroupedBranchDataset.load(checked[0]).group_count, 4)

                (model_test / "report.json").write_bytes(b"published\n")
                with self.assertRaisesRegex(
                        ProtectedEvidencePathError, "revoked by report"):
                    with stage_b_model_test_producer_read_scope(
                            attempt_path=model_test / "attempt-started.json",
                            expected_attempt_sha256=attempt["attempt_sha256"],
                            evidence_paths=[leaf]):
                        pass

    def test_committed_unconsumed_denies_before_every_outcome_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, deployable, frozen, attempt_sha = (
                self._model_test_fixture(root))
            compile_stage_b_model_test_commitment(
                report_path=report,
                commitment_path=root / "stage-b/model-test-committed.json",
                expected_producer_attempt_sha256=attempt_sha,
                created_at_utc="2026-08-10T00:01:00+00:00",
            )
            with mock.patch.object(
                    Path, "exists",
                    side_effect=AssertionError("outcome exists probed")), (
                        mock.patch.object(
                            Path, "lstat",
                            side_effect=AssertionError("outcome lstat probed"))), (
                        mock.patch.object(
                            Path, "stat",
                            side_effect=AssertionError("outcome stat probed"))), (
                        mock.patch.object(
                            Path, "open",
                            side_effect=AssertionError("outcome open probed"))), (
                        mock.patch.object(
                            Path, "resolve",
                            side_effect=AssertionError("outcome resolved"))), (
                        mock.patch(
                            "safety_data.paths._regular_sha256_no_symlink",
                            side_effect=AssertionError("outcome hashed"))), (
                        mock.patch(
                            "safety_data.schema.np.load",
                            side_effect=AssertionError("outcome np.load called"))):
                with self.assertRaisesRegex(
                        Exception, "embargoed between commitment"):
                    with stage_b_evidence_read_scope(
                            scientific_role="model_test",
                            evidence_kind="label",
                            path=deployable):
                        GroupedBranchDataset.load(deployable)

    def test_compiler_reads_only_strict_outcome_free_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, _, frozen, attempt_sha = self._model_test_fixture(root)
            original = __import__(
                "safety_data.stage_b_paths", fromlist=[
                    "_regular_bytes_no_symlink"])._regular_bytes_no_symlink
            opened: list[Path] = []

            def record(path, name):
                opened.append(Path(path))
                return original(path, name)

            with mock.patch(
                    "safety_data.stage_b_paths._regular_bytes_no_symlink",
                    side_effect=record):
                result = compile_stage_b_model_test_commitment(
                    report_path=report,
                    commitment_path=root / "stage-b/model-test-committed.json",
                    expected_producer_attempt_sha256=attempt_sha,
                    created_at_utc="2026-08-10T00:01:00+00:00",
                )
            self.assertEqual(opened, [report.absolute()])
            self.assertEqual(len(result["commitment_sha256"]), 64)

    def test_extra_outcome_summary_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, _, frozen, attempt_sha = self._model_test_fixture(root)
            value = json.loads(report.read_text(encoding="utf-8"))
            value["fall_rate"] = 0.5
            report.write_bytes(_canonical_control_json(value))
            with self.assertRaisesRegex(
                    ProtectedEvidencePathError, "extra or missing fields"):
                compile_stage_b_model_test_commitment(
                    report_path=report,
                    commitment_path=root / "stage-b/model-test-committed.json",
                    expected_producer_attempt_sha256=attempt_sha,
                )

    def test_consumed_marker_precedes_first_read_and_survives_crash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, deployable, frozen, attempt_sha = self._model_test_fixture(
                root, valid_deployable=True)
            result = compile_stage_b_model_test_commitment(
                report_path=report,
                commitment_path=root / "stage-b/model-test-committed.json",
                expected_producer_attempt_sha256=attempt_sha,
                created_at_utc="2026-08-10T00:01:00+00:00",
            )
            consumed = root / "stage-b/model-test-consumed.json"
            import safety_data.paths as guarded_paths
            original_hash = guarded_paths._regular_sha256_no_symlink

            def assert_marker_then_hash(path, name):
                self.assertTrue(os.path.lexists(consumed))
                return original_hash(path, name)

            with mock.patch(
                    "safety_data.stage_b_paths.require_clean_stage_b_generator",
                    return_value="7" * 40), self.assertRaisesRegex(
                        RuntimeError, "simulated evaluator crash"):
                with mock.patch(
                        "safety_data.paths._regular_sha256_no_symlink",
                        side_effect=assert_marker_then_hash), (
                        consume_stage_b_model_test(
                            commitment_path=(
                                root / "stage-b/model-test-committed.json"),
                            consumed_path=consumed,
                            expected_commitment_sha256=result[
                                "commitment_sha256"],
                            prerequisite_artifact_sha256=frozen,
                            evaluator_clean_commit="7" * 40,
                            evidence_paths=[deployable],
                            created_at_utc="2026-08-10T00:02:00+00:00",
                        )):
                    loaded = GroupedBranchDataset.load(deployable)
                    self.assertEqual(loaded.group_count, 4)
                    raise RuntimeError("simulated evaluator crash")
            self.assertTrue(consumed.is_file())
            with mock.patch(
                    "safety_data.stage_b_paths.require_clean_stage_b_generator",
                    return_value="7" * 40), self.assertRaisesRegex(
                        ProtectedEvidencePathError, "already been consumed"):
                with consume_stage_b_model_test(
                    commitment_path=root / "stage-b/model-test-committed.json",
                    consumed_path=consumed,
                    expected_commitment_sha256=result["commitment_sha256"],
                    prerequisite_artifact_sha256=frozen,
                    evaluator_clean_commit="7" * 40,
                    evidence_paths=[deployable],
                ):
                    pass


if __name__ == "__main__":
    unittest.main()
