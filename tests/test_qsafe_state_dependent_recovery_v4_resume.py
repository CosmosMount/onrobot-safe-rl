from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

from safety_data.closed_loop_recovery_collector import canonical_protocol_sha256
import safety_data.state_dependent_recovery_v4 as v4
import scripts.merge_state_dependent_recovery_v4 as merge_v4


_CLEAN_COMMIT = "d" * 40
_PROTOCOL_FILE_SHA256 = "a" * 64


def _spec_and_protocol(root: Path):
    protocol = yaml.safe_load(v4.PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = copy.deepcopy(protocol)
    protocol["collection"]["artifact_root"] = str(root)
    seed_age = {
        int(seed): int(policy["training_step"])
        for policy in protocol["early_task_policies"]
        for seed in policy["source_seeds"]
    }
    spec = {
        "protocol": protocol,
        "collection": protocol["collection"],
        "protocol_contract_sha256": canonical_protocol_sha256(protocol),
        "data_gate": protocol["triage_gates"]["data"],
        "seed_age": seed_age,
        "age_strata": v4.AGE_STRATA,
        # A one-group-per-source miniature preserves every structural relation
        # while keeping the recovery-reader unit fixture small.
        "groups": len(v4.SOURCE_SEEDS),
        "groups_per_seed": 1,
        "required_seeds": tuple(v4.SOURCE_SEEDS),
        "admission_replicas": 32,
        "discovery_replicas": 64,
        "audit_replicas": 64,
        "horizon": 96,
    }
    return spec, protocol


def _audit_denied_lock(spec):
    protocol = spec["protocol"]
    collection = spec["collection"]
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    next_seed = 1 << 63

    def seed_vector(count):
        nonlocal next_seed
        result = list(range(next_seed, next_seed + count))
        next_seed += count
        return result

    risk_row = [0.0, *[float(index) / 16.0 for index in range(1, 9)]]
    group_selection = []
    replica_partition = []
    for index, source_seed in enumerate(spec["required_seeds"]):
        group_id = f"group-{index}"
        group_selection.append({
            "group_index": index,
            "group_id": group_id,
            "state_hash": digest(f"state-{index}"),
            "trajectory_id": f"trajectory-{index}",
            "source_seed": source_seed,
            "policy_age": spec["seed_age"][source_seed],
            "admission_falls": 6,
            "discovery_candidate_risk": list(risk_row),
            "discovery_minimizer_indices": [0],
            "discovery_minimizer_names": [v4.CANDIDATE_NAMES[0]],
            "uniform_weights": [1.0],
        })
        replica_partition.append({
            "group_index": index,
            "group_id": group_id,
            "admission_crn_ids": seed_vector(spec["admission_replicas"]),
            "admission_rollout_seeds": seed_vector(
                spec["admission_replicas"]),
            "admission_perturbation_seeds": seed_vector(
                spec["admission_replicas"]),
            "discovery_crn_ids": seed_vector(spec["discovery_replicas"]),
            "discovery_rollout_seeds": seed_vector(
                spec["discovery_replicas"]),
            "discovery_perturbation_seeds": seed_vector(
                spec["discovery_replicas"]),
            "discovery_candidate_seed": seed_vector(1)[0],
            "audit_crn_ids": seed_vector(spec["audit_replicas"]),
            "audit_rollout_seeds": seed_vector(spec["audit_replicas"]),
            "audit_perturbation_seeds": seed_vector(
                spec["audit_replicas"]),
            "audit_candidate_seed": seed_vector(1)[0],
        })

    role_commitments = {role: [] for role in v4._READINESS_ROLES}
    for ordinal, source_seed in enumerate(spec["required_seeds"]):
        for role in v4._READINESS_ROLES:
            role_commitments[role].append({
                "ordinal": ordinal,
                "source_seed": source_seed,
                "policy_training_step": spec["seed_age"][source_seed],
                "path": f"opaque-{role}-{source_seed}",
                "file_sha256": digest(f"{role}-file-{source_seed}"),
                "content_sha256": digest(f"{role}-content-{source_seed}"),
            })

    source_records = []
    for ordinal, source_seed in enumerate(spec["required_seeds"]):
        outputs = {
            role: {
                "path": role_commitments[role][ordinal]["path"],
                "file_sha256": role_commitments[role][ordinal]["file_sha256"],
                "content_sha256": role_commitments[role][ordinal][
                    "content_sha256"],
            }
            for role in v4._READINESS_ROLES
        }
        validations = {
            "admission": {
                "proposals": 1,
                "accepted": 1,
                "content_sha256": outputs["admission"]["content_sha256"],
            },
            "admission_privileged": {
                "proposals": 1,
                "content_sha256": outputs["admission_privileged"][
                    "content_sha256"],
            },
            "discovery": {
                "groups": 1,
                "max_candidates": 9,
                "replicas": spec["discovery_replicas"],
                "horizon_steps": spec["horizon"],
                "content_sha256": outputs["discovery"]["content_sha256"],
            },
            "discovery_privileged": {
                "groups": 1,
                "content_sha256": outputs["discovery_privileged"][
                    "content_sha256"],
            },
            "audit": {
                "groups": 1,
                "max_candidates": 9,
                "replicas": spec["audit_replicas"],
                "horizon_steps": spec["horizon"],
                "content_sha256": outputs["audit"]["content_sha256"],
            },
            "audit_privileged": {
                "groups": 1,
                "content_sha256": outputs["audit_privileged"][
                    "content_sha256"],
            },
        }
        source_records.append({
            "ordinal": ordinal,
            "source_seed": source_seed,
            "policy_training_step": spec["seed_age"][source_seed],
            "protocol_file_sha256": _PROTOCOL_FILE_SHA256,
            "protocol_contract_sha256": spec["protocol_contract_sha256"],
            "generator_commit": _CLEAN_COMMIT,
            "collection_report_path": f"opaque-report-{source_seed}",
            "collection_report_file_sha256": digest(
                f"report-{source_seed}"),
            "cohort_lock": {
                "path": "opaque-cohort-lock",
                "file_sha256": digest("cohort-file"),
                "contract_sha256": digest("cohort-contract"),
            },
            "attempt_marker": {
                "path": f"opaque-attempt-{source_seed}",
                "file_sha256": digest(f"attempt-file-{source_seed}"),
                "contract_sha256": digest(f"attempt-contract-{source_seed}"),
            },
            "outputs": outputs,
            "validations": validations,
        })
    readiness_manifest = {
        "schema_version": v4._v3.COLLECTION_READINESS_SCHEMA_VERSION,
        "protocol_name": v4.PROTOCOL_NAME,
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": _PROTOCOL_FILE_SHA256,
        "generator_commit": _CLEAN_COMMIT,
        "artifact_root": str(v4._artifact_root(protocol)),
        "required_source_seeds": list(spec["required_seeds"]),
        "source_records": source_records,
        "role_commitments": role_commitments,
    }
    admission_file_sha256 = digest("merged-admission-file")
    admission_content_sha256 = digest("merged-admission-content")
    discovery_file_sha256 = digest("merged-discovery-file")
    discovery_content_sha256 = digest("merged-discovery-content")
    merge_manifest = {
        "schema_version": (
            "qsafe.closed_loop_recovery_triage.merge_readiness.v1"),
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "collection_readiness_sha256": v4._v3.canonical_sha256(
            readiness_manifest),
        "admission_merge_report": {
            "path": "opaque-admission-merge-report",
            "file_sha256": digest("admission-merge-report"),
            "output_file_sha256": admission_file_sha256,
            "output_content_sha256": admission_content_sha256,
        },
        "discovery_merge_report": {
            "path": "opaque-discovery-merge-report",
            "file_sha256": digest("discovery-merge-report"),
            "output_file_sha256": discovery_file_sha256,
            "output_content_sha256": discovery_content_sha256,
        },
    }
    source_seed_array = np.asarray(spec["required_seeds"], dtype=np.int64)
    nominal_risk = np.zeros(spec["groups"], dtype=np.float64)
    informativeness = v4._v3._discovery_informativeness(
        nominal_risk, source_seed_array, spec)
    global_table = [{
        "candidate_index": index,
        "candidate_name": v4.CANDIDATE_NAMES[index],
        "equal_seed_discovery_risk": risk_row[index],
    } for index in range(1, 9)]
    return {
        "schema_version": v4._v3.SELECTION_LOCK_SCHEMA_VERSION,
        "protocol_name": v4.PROTOCOL_NAME,
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": _PROTOCOL_FILE_SHA256,
        "generator_commit": _CLEAN_COMMIT,
        "candidate_library_sha256": v4._v3.canonical_sha256(
            collection["candidates"]),
        "policy_bundle_sha256": v4._v3.canonical_sha256({
            "policy_config": protocol["policy_config"],
            "early_task_policies": protocol["early_task_policies"],
            "mature_recovery_policy": protocol["mature_recovery_policy"],
        }),
        "created_at_utc": "2026-08-09T00:00:00+00:00",
        "input_artifacts": {
            "admission": {
                "filename": collection["admission_deployable_filename"],
                "file_sha256": admission_file_sha256,
                "content_sha256": admission_content_sha256,
                "proposal_count": spec["groups"],
            },
            "discovery": {
                "filename": collection["discovery_filename"],
                "file_sha256": discovery_file_sha256,
                "content_sha256": discovery_content_sha256,
            },
        },
        "candidate_order": list(v4.CANDIDATE_NAMES),
        "selected_global_candidate": {
            "candidate_index": 1,
            "candidate_name": v4.CANDIDATE_NAMES[1],
            "selection_scope": "eight_nonnominal_candidates",
            "exact_tie_break": "locked_candidate_order",
            "discovery_candidate_table": global_table,
        },
        "bootstrap": protocol["statistics"]["bootstrap"],
        "triage_gates": protocol["triage_gates"],
        "selection_semantics": copy.deepcopy(v4.SELECTION_SEMANTICS),
        "audit_identifier": "c" * 64,
        "audit_authorized": False,
        "audit_runner_up_policy": "forbidden",
        "data_gate": {
            "structural_contract_pass": True,
            "independent_groups": spec["groups"],
            "unique_state_fingerprints": spec["groups"],
            "unique_trajectory_fingerprints": spec["groups"],
            "groups_per_source_seed": spec["groups_per_seed"],
            "required_source_seeds": list(v4.SOURCE_SEEDS),
            "candidates": 9,
            "admission_replicas": 32,
            "discovery_replicas": 64,
            "audit_replicas_preassigned": 64,
            "horizon_policy_steps": 96,
            "discovery_informativeness": {
                **informativeness,
            },
            "pass": False,
        },
        "group_selection": group_selection,
        "group_selection_sha256": v4._v3.canonical_sha256(group_selection),
        "replica_partition": replica_partition,
        "replica_partition_sha256": v4._v3.canonical_sha256(
            replica_partition),
        "collection_readiness_sha256": v4._v3.canonical_sha256(
            readiness_manifest),
        "collection_readiness_manifest": readiness_manifest,
        "merge_readiness_sha256": v4._v3.canonical_sha256(merge_manifest),
        "merge_readiness_manifest": merge_manifest,
        "expected_audit_shards": copy.deepcopy(role_commitments["audit"]),
    }


class StateDependentRecoveryV4ResumeTest(unittest.TestCase):
    def _write_lock(self, root: Path, lock):
        path = root / "selection-lock.json"
        payload = v4._v3._canonical_json_bytes(lock)
        path.write_bytes(payload)
        return path, hashlib.sha256(payload).hexdigest()

    def test_resume_is_idempotent_and_never_reaches_an_audit_surface(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, protocol = _spec_and_protocol(root)
            lock_path, lock_sha256 = self._write_lock(
                root, _audit_denied_lock(spec))
            artifact_path = lambda value, **kwargs: Path(value)
            with mock.patch.object(
                    v4, "_validate_protocol", return_value=spec), \
                    mock.patch.object(
                        v4, "_require_clean_head_protocol_binding",
                        return_value=(_CLEAN_COMMIT, _PROTOCOL_FILE_SHA256)), \
                    mock.patch.object(
                        v4._v3, "_artifact_path", side_effect=artifact_path), \
                    mock.patch.object(
                        v4, "_locked_audit_paths_before_consumption",
                        side_effect=AssertionError("audit path derived")), \
                    mock.patch.object(
                        v4._v3, "_expected_audit_shard_paths",
                        side_effect=AssertionError("audit path constructed")), \
                    mock.patch.object(
                        v4._v3, "_canonical_embedded_path",
                        side_effect=AssertionError("embedded path parsed")), \
                    mock.patch.object(
                        v4._v3, "_load_audit_shards_after_consumption",
                        side_effect=AssertionError("audit outcome opened")):
                first = v4.resume_state_dependent_discovery_failure_report(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock_sha256,
                )
                second = v4.resume_state_dependent_discovery_failure_report(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock_sha256,
                )

            self.assertEqual(first, second)
            self.assertEqual(first["decision"], "no_model_training")
            self.assertFalse(first["audit_authorized"])
            self.assertFalse(first["objective1_pass"])
            report_path = root / protocol["collection"][
                "triage_report_filename"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["audit_opened_for_analysis"])
            self.assertFalse(report["audit_consumed"])
            self.assertEqual(
                hashlib.sha256(report_path.read_bytes()).hexdigest(),
                first["stage_A_failure_report_sha256"],
            )

    def test_reader_rejects_self_rehashed_schema_and_semantic_tampering(self):
        def rehash_readiness(lock):
            readiness_sha256 = v4._v3.canonical_sha256(
                lock["collection_readiness_manifest"])
            lock["collection_readiness_sha256"] = readiness_sha256
            lock["merge_readiness_manifest"][
                "collection_readiness_sha256"] = readiness_sha256
            lock["merge_readiness_sha256"] = v4._v3.canonical_sha256(
                lock["merge_readiness_manifest"])

        def missing_readiness_field(lock):
            del lock["collection_readiness_manifest"][
                "source_records"][0]["outputs"]
            rehash_readiness(lock)

        def wrong_global_selection(lock):
            lock["selected_global_candidate"]["candidate_index"] = 2
            lock["selected_global_candidate"]["candidate_name"] = (
                v4.CANDIDATE_NAMES[2])

        def wrong_per_state_selection(lock):
            group = lock["group_selection"][0]
            group["discovery_minimizer_indices"] = [1]
            group["discovery_minimizer_names"] = [v4.CANDIDATE_NAMES[1]]
            group["uniform_weights"] = [1.0]
            lock["group_selection_sha256"] = v4._v3.canonical_sha256(
                lock["group_selection"])

        def forged_informativeness(lock):
            lock["data_gate"]["discovery_informativeness"][
                "overall_equal_seed_nominal_risk"] = 0.5

        def mismatched_input_identity(lock):
            lock["input_artifacts"]["discovery"]["content_sha256"] = "e" * 64

        def duplicate_replica_seed(lock):
            record = lock["replica_partition"][0]
            record["audit_crn_ids"][1] = record["audit_crn_ids"][0]
            lock["replica_partition_sha256"] = v4._v3.canonical_sha256(
                lock["replica_partition"])

        def missing_v4_seed_tag(lock):
            lock["replica_partition"][0]["audit_candidate_seed"] = 7
            lock["replica_partition_sha256"] = v4._v3.canonical_sha256(
                lock["replica_partition"])

        def missing_top_level_field(lock):
            del lock["input_artifacts"]

        cases = (
            ("missing_top", missing_top_level_field, "top-level mapping"),
            ("readiness_schema", missing_readiness_field,
             "source_records.*extra or missing"),
            ("input_identity", mismatched_input_identity,
             "merge/input artifact"),
            ("global_selection", wrong_global_selection, "global selection"),
            ("per_state_selection", wrong_per_state_selection,
             "per-state minimizers"),
            ("informativeness", forged_informativeness,
             "informativeness record"),
            ("replica_seed", duplicate_replica_seed, "audit_crn_ids"),
            ("seed_tag", missing_v4_seed_tag, "at least"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _ = _spec_and_protocol(root)
            for name, mutate, message in cases:
                with self.subTest(name=name):
                    lock = _audit_denied_lock(spec)
                    mutate(lock)
                    lock_path, lock_sha256 = self._write_lock(root, lock)
                    with self.assertRaisesRegex(
                            v4.StateDependentRecoveryV4Error, message):
                        v4._read_audit_denied_selection_lock(
                            lock_path,
                            expected_sha256=lock_sha256,
                            spec=spec,
                            clean_commit=_CLEAN_COMMIT,
                            protocol_file_sha256=_PROTOCOL_FILE_SHA256,
                        )
                    lock_path.unlink()

    def test_resume_rejects_hash_gate_drift_and_existing_report_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, protocol = _spec_and_protocol(root)
            lock_path, lock_sha256 = self._write_lock(
                root, _audit_denied_lock(spec))
            artifact_path = lambda value, **kwargs: Path(value)
            patches = (
                mock.patch.object(v4, "_validate_protocol", return_value=spec),
                mock.patch.object(
                    v4, "_require_clean_head_protocol_binding",
                    return_value=(_CLEAN_COMMIT, _PROTOCOL_FILE_SHA256)),
                mock.patch.object(
                    v4._v3, "_artifact_path", side_effect=artifact_path),
            )
            with patches[0], patches[1], patches[2]:
                with self.assertRaisesRegex(
                        v4.StateDependentRecoveryV4Error, "file hash differs"):
                    v4.resume_state_dependent_discovery_failure_report(
                        protocol=protocol,
                        selection_lock_path=lock_path,
                        expected_selection_lock_sha256="0" * 64,
                    )
                result = v4.resume_state_dependent_discovery_failure_report(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock_sha256,
                )
                Path(result["stage_A_failure_report"]).write_text(
                    "{}\n", encoding="utf-8")
                with self.assertRaisesRegex(
                        v4.StateDependentRecoveryV4Error,
                        "existing .* differs"):
                    v4.resume_state_dependent_discovery_failure_report(
                        protocol=protocol,
                        selection_lock_path=lock_path,
                        expected_selection_lock_sha256=lock_sha256,
                    )

    def test_resume_rejects_audit_authorized_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, protocol = _spec_and_protocol(root)
            lock = _audit_denied_lock(spec)
            lock["audit_authorized"] = True
            lock_path, lock_sha256 = self._write_lock(root, lock)
            artifact_path = lambda value, **kwargs: Path(value)
            with mock.patch.object(
                    v4, "_validate_protocol", return_value=spec), \
                    mock.patch.object(
                        v4, "_require_clean_head_protocol_binding",
                        return_value=(_CLEAN_COMMIT, _PROTOCOL_FILE_SHA256)), \
                    mock.patch.object(
                        v4._v3, "_artifact_path", side_effect=artifact_path):
                with self.assertRaisesRegex(
                        v4.StateDependentRecoveryV4Error,
                        "audit_authorized mismatch"):
                    v4.resume_state_dependent_discovery_failure_report(
                        protocol=protocol,
                        selection_lock_path=lock_path,
                        expected_selection_lock_sha256=lock_sha256,
                    )

    def test_resume_cli_bypasses_collection_readiness_and_audit_paths(self):
        protocol = {"protocol_name": v4.PROTOCOL_NAME}
        rendered = {
            "selection_lock_sha256": "b" * 64,
            "decision": "no_model_training",
            "audit_authorized": False,
            "objective1_pass": False,
            "phase2_authorized": False,
        }
        with mock.patch(
                "sys.argv",
                ["merge_state_dependent_recovery_v4.py",
                 "resume-denied-report", "--selection-lock-sha256",
                 "b" * 64]), \
                mock.patch.object(
                    merge_v4,
                    "load_state_dependent_recovery_v4_protocol",
                    return_value=protocol), \
                mock.patch.object(
                    merge_v4, "_paths",
                    return_value={"selection_lock": Path("selection-lock.json")}), \
                mock.patch.object(
                    merge_v4,
                    "resume_state_dependent_discovery_failure_report",
                    return_value=rendered) as resume, \
                mock.patch.object(
                    merge_v4,
                    "validate_state_dependent_collection_readiness",
                    side_effect=AssertionError("collection readiness opened")), \
                mock.patch.object(
                    merge_v4, "_create_lock",
                    side_effect=AssertionError("lock/audit route reached")), \
                mock.patch("builtins.print") as output:
            self.assertEqual(merge_v4.main(), 0)
        resume.assert_called_once_with(
            protocol=protocol,
            selection_lock_path=Path("selection-lock.json"),
            expected_selection_lock_sha256="b" * 64,
        )
        payload = json.loads(output.call_args.args[0])
        self.assertEqual(payload, rendered)


if __name__ == "__main__":
    unittest.main()
