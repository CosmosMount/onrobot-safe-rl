#!/usr/bin/env python3
"""Irreversibly consume and evaluate the canonical V5 Stage-B Model-Test.

The only production argument is the previously published commitment SHA-256.
Bootstrap counts, RNG, quantiles, gates, paths, and output names are compiled
into the frozen protocol and numerical module; there is deliberately no CLI
surface that can alter them after seeing Model-Test outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from rl.qsafe.artifact import LoadedQSafeArtifact, load_qsafe_artifact
from rl.qsafe.data import TorchGroupedView
from rl.qsafe.recovery_calibration import (
    STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES,
    STAGE_B_SELECTOR_BOOTSTRAP_SEED,
    SignedConformalCalibration,
)
from rl.qsafe.recovery_model_test import (
    STAGE_B_MODEL_TEST_BOOTSTRAP_REPLICATES,
    STAGE_B_MODEL_TEST_BOOTSTRAP_SEED,
    evaluate_stage_b_model_test,
)
from rl.qsafe.recovery_placebo import MatchedRandomPlaceboBundle
from rl.qsafe.recovery_program import RECOVERY_PROGRAM_VIEW
from rl.qsafe.recovery_selector import RecoverySelectorBundle
from safety_data.paths import (
    STAGE_B_EXECUTION_PROTOCOL_NAME,
    STAGE_B_FROZEN_INPUT_NAMES,
    STAGE_B_PROTOCOL_NAME,
    _validate_stage_b_model_test_commitment,
)
from safety_data.schema import GroupedBranchDataset
from safety_data.stage_b_paths import consume_stage_b_model_test
from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_CONTRACT_SHA256 as PARENT_PROTOCOL_CONTRACT_SHA256,
    PROTOCOL_FILE_SHA256 as PARENT_PROTOCOL_FILE_SHA256,
    load_state_dependent_recovery_v5_protocol,
)
from safety_data.state_dependent_recovery_v5_stage_b import (
    CHECKPOINT_STEPS,
    EXECUTION_PROTOCOL_CONTRACT_SHA256,
    EXECUTION_PROTOCOL_FILE_SHA256,
    EXECUTION_PROTOCOL_NAME,
    GROUPS_PER_SOURCE,
    HORIZON_POLICY_STEPS,
    LABEL_REPLICAS,
    RECOVERY_LIBRARY_FINGERPRINT_SHA256,
    ROLE_ACTOR_SEEDS,
    ROLE_ORDER,
    ROLE_SOURCE_SEEDS,
    SPLIT_COLLISION_DIMENSIONS,
    SPLIT_IDENTITY_SOURCE_FIELDS,
    STAGE_A_DISPOSITION_COMMIT,
    STAGE_A_REPORT_SHA256,
    TRAJECTORY_FINGERPRINT_ARRAY,
    TRAJECTORY_FINGERPRINT_CONTRACT,
    assignment_for,
    load_stage_b_execution_protocol,
    load_stage_b_reduced7_amendment,
    require_clean_stage_b_generator,
    stage_b_artifact_root,
    validate_stage_a_authorization,
)
from train.state_dependent_recovery_v5_stage_b_actor_bank import (
    actor_identity_for,
    load_reduced7_actor_bank_manifest,
)


_REPORT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_report.v1")
_MODEL_TEST_LABEL_RELATIVE = (
    "stage-b/model-test/labels-r64-deployable.npz")
_FROZEN_PATHS = {
    "actor_bank_manifest": "actor-bank-manifest.json",
    "split_disjointness_report": "stage-b-split-disjointness-report.json",
    "normalization_report": "normalization-fit-only-report.json",
    "qsafe_artifact": "qsafe-artifact/manifest.json",
    "probability_calibration_report": "probability-calibration-report.json",
    "uncertainty_calibration_report": "uncertainty-calibration-report.json",
    "selector_search_report": "selector-search-report.json",
    "recovery_selector_bundle": "recovery-selector-bundle.json",
    "matched_random_placebo_bundle": "matched-random-placebo-bundle.json",
}
_REPORT_SCHEMAS = {
    "normalization_report": (
        "qsafe.state_dependent_recovery_v5.stage_b.normalization_fit_only.v1"),
    "probability_calibration_report": (
        "qsafe.state_dependent_recovery_v5.stage_b.probability_calibration.v1"),
    "uncertainty_calibration_report": (
        "qsafe.state_dependent_recovery_v5.stage_b.uncertainty_calibration.v1"),
    "selector_search_report": (
        "qsafe.state_dependent_recovery_v5.stage_b.selector_search.v1"),
}
class StageBModelTestError(RuntimeError):
    """The one-shot Stage-B evaluator failed closed."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-test-commitment-sha256",
        required=True,
        help="Exact SHA-256 printed by the blind Model-Test commitment compiler",
    )
    return parser


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n").encode("utf-8")


def _canonical_object_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)[:-1]).hexdigest()


def _ordered_text_sha256(value: Any) -> str:
    array = np.asarray(value).astype(str).reshape(-1)
    digest = hashlib.sha256(b"qsafe.ordered_text_vector.v1\0")
    for item in array:
        encoded = item.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _require_real_ancestor_chain(root: Path, path: Path, name: str) -> None:
    if not path.is_absolute() or not root.is_absolute() or path != root and (
            root not in path.parents):
        raise StageBModelTestError(f"{name} escapes the canonical Stage-B root")
    current = root
    relative = path.relative_to(root)
    components = relative.parts[:-1]
    for component in components:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise StageBModelTestError(f"{name} parent is missing") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise StageBModelTestError(
                f"{name} parent must be a real directory")


def _regular_bytes(path: Path, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageBModelTestError(
                    f"{name} must be a single-link regular file")
            return stream.read()
    except StageBModelTestError:
        raise
    except OSError as exc:
        raise StageBModelTestError(
            f"{name} is missing, unreadable, or a symlink") from exc


def _regular_sha256(path: Path, name: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageBModelTestError(
                    f"{name} must be a single-link regular file")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            return digest.hexdigest()
    except StageBModelTestError:
        raise
    except OSError as exc:
        raise StageBModelTestError(
            f"{name} is missing, unreadable, or a symlink") from exc


def _read_json(path: Path, name: str) -> tuple[dict[str, Any], str]:
    raw = _regular_bytes(path, name)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StageBModelTestError(f"{name} is not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise StageBModelTestError(f"{name} must contain one JSON object")
    return dict(decoded), hashlib.sha256(raw).hexdigest()


def _validate_self_hashed_report(
    value: Mapping[str, Any],
    *,
    schema_version: str,
    name: str,
) -> str:
    if value.get("schema_version") != schema_version:
        raise StageBModelTestError(f"{name} schema version drifted")
    observed = value.get("report_sha256")
    basis = dict(value)
    basis.pop("report_sha256", None)
    expected = _canonical_object_sha256(basis)
    if not _is_sha256(observed) or observed != expected:
        raise StageBModelTestError(f"{name} canonical self-hash mismatch")
    if "pass" in value and value.get("pass") is not True:
        raise StageBModelTestError(f"{name} records a failed prerequisite gate")
    return str(observed)


def _commitment_label_sha256(
    commitment_path: Path,
    expected_commitment_sha256: str,
) -> tuple[str, str]:
    commitment, raw_sha256 = _read_json(
        commitment_path, "Stage-B Model-Test commitment")
    if raw_sha256 != expected_commitment_sha256:
        raise StageBModelTestError("Model-Test commitment SHA-256 mismatch")
    if _regular_bytes(
            commitment_path, "Stage-B Model-Test commitment") != (
                _canonical_json_bytes(commitment)):
        raise StageBModelTestError(
            "Stage-B Model-Test commitment is not canonical JSON")
    try:
        checked = _validate_stage_b_model_test_commitment(commitment)
    except Exception as exc:
        raise StageBModelTestError(
            "Stage-B Model-Test commitment contract is invalid") from exc
    expected_identity = {
        "parent_protocol_name": STAGE_B_PROTOCOL_NAME,
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "execution_protocol_name": STAGE_B_EXECUTION_PROTOCOL_NAME,
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256),
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
    }
    for name, expected in expected_identity.items():
        if checked.get(name) != expected:
            raise StageBModelTestError(
                f"Stage-B Model-Test commitment {name} drifted")
    records = checked["evidence_artifacts"]
    assert isinstance(records, list)  # Strict validator above.
    matching = [
        record for record in records
        if isinstance(record, Mapping)
        and record.get("path") == _MODEL_TEST_LABEL_RELATIVE
        and record.get("kind") == "label"
    ]
    if len(matching) != 1 or not _is_sha256(matching[0].get("sha256")):
        raise StageBModelTestError(
            "Model-Test commitment does not bind the canonical label array")
    generator_commit = checked.get("generator_commit")
    if not isinstance(generator_commit, str) or len(generator_commit) != 40:
        raise StageBModelTestError(
            "Model-Test commitment generator commit is invalid")
    return str(matching[0]["sha256"]), generator_commit


def _validate_split_disjointness_report(
    value: Mapping[str, Any],
    *,
    generator_commit: str,
    actor_bank_manifest_file_sha256: str,
    actor_bank_contract_sha256: str,
    committed_model_test_label_sha256: str,
) -> str:
    """Validate the exact outcome-blind, five-role split proof topology."""
    frozen_identity = {
        "parent_protocol_name": STAGE_B_PROTOCOL_NAME,
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "execution_protocol_name": STAGE_B_EXECUTION_PROTOCOL_NAME,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256),
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
        "generator_commit": generator_commit,
    }
    expected_fields = {
        "schema_version",
        *frozen_identity,
        "actor_bank_manifest_file_sha256",
        "actor_bank_contract_sha256",
        "role_order",
        "role_aggregate_labels",
        "role_aggregate_admissions",
        "identity_proof",
        "partition_rng_proof",
        "model_test_source",
        "identity_proof_outcome_columns_read",
        "blind_mechanical_merge_outcome_statistics_computed",
        "pass",
        "report_sha256",
    }
    if set(value) != expected_fields:
        raise StageBModelTestError(
            "split disjointness report has extra or missing fields")
    report_sha256 = _validate_self_hashed_report(
        value,
        schema_version=(
            "qsafe.state_dependent_recovery_v5."
            "stage_b_split_disjointness_bound.v3"),
        name="split disjointness report",
    )
    for name, expected in frozen_identity.items():
        if value.get(name) != expected:
            raise StageBModelTestError(
                f"split disjointness report {name} drifted")
    if value.get("actor_bank_manifest_file_sha256") != (
            actor_bank_manifest_file_sha256) or value.get(
                "actor_bank_contract_sha256") != actor_bank_contract_sha256:
        raise StageBModelTestError(
            "split disjointness report actor-bank binding drifted")
    if value.get("role_order") != list(ROLE_ORDER) or value.get(
            "model_test_source") != (
                "in_memory_merged_dataset_and_staged_label_bytes_"
                "before_role_report") or value.get(
                    "identity_proof_outcome_columns_read") is not False or (
                        value.get(
                            "blind_mechanical_merge_outcome_statistics_computed")
                        is not False) or value.get("pass") is not True:
        raise StageBModelTestError(
            "split disjointness report top-level gate drifted")

    aggregates = value.get("role_aggregate_labels")
    if not isinstance(aggregates, list) or len(aggregates) != len(ROLE_ORDER):
        raise StageBModelTestError(
            "split disjointness report aggregate roster drifted")
    for record, role in zip(aggregates, ROLE_ORDER, strict=True):
        replicas = LABEL_REPLICAS[role]
        expected_path = (
            f"stage-b/{role.replace('_', '-')}/"
            f"labels-r{replicas}-deployable.npz")
        expected_groups = len(ROLE_SOURCE_SEEDS[role]) * GROUPS_PER_SOURCE[role]
        if not isinstance(record, Mapping) or set(record) != {
                "role", "path", "file_sha256", "content_sha256", "groups",
                "role_report_file_sha256"} or record.get("role") != role or (
                    record.get("path") != expected_path) or record.get(
                        "groups") != expected_groups or not _is_sha256(
                            record.get("file_sha256")) or not _is_sha256(
                                record.get("content_sha256")):
            raise StageBModelTestError(
                f"split disjointness report {role} aggregate drifted")
        role_report_sha256 = record.get("role_report_file_sha256")
        if role == "model_test":
            if role_report_sha256 is not None or record.get(
                    "file_sha256") != committed_model_test_label_sha256:
                raise StageBModelTestError(
                    "split disjointness report Model-Test aggregate drifted")
        elif not _is_sha256(role_report_sha256):
            raise StageBModelTestError(
                f"split disjointness report {role} role report drifted")

    admissions = value.get("role_aggregate_admissions")
    if not isinstance(admissions, list) or len(admissions) != len(ROLE_ORDER):
        raise StageBModelTestError(
            "split disjointness report admission roster drifted")
    for record, role in zip(admissions, ROLE_ORDER, strict=True):
        expected_path = (
            f"stage-b/{role.replace('_', '-')}/admission-r32.npz")
        if not isinstance(record, Mapping) or set(record) != {
                "role", "admission_path", "admission_file_sha256",
                "admission_content_sha256", "admission_proposals"} or (
                    record.get("role") != role or record.get(
                        "admission_path") != expected_path or not _is_sha256(
                            record.get("admission_file_sha256")) or not _is_sha256(
                                record.get("admission_content_sha256")) or (
                                    isinstance(record.get("admission_proposals"), bool)
                                    or not isinstance(
                                        record.get("admission_proposals"), int)
                                    or record.get("admission_proposals") <= 0)):
            raise StageBModelTestError(
                f"split disjointness report {role} admission aggregate drifted")

    proof = value.get("identity_proof")
    expected_proof_fields = {
        "schema_version", "dimensions", "identity_array_fields", "roles",
        "pairs_checked", "pairs", "outcome_columns_read", "pass",
        "report_sha256",
    }
    if not isinstance(proof, Mapping) or set(proof) != expected_proof_fields:
        raise StageBModelTestError(
            "nested split identity proof has extra or missing fields")
    _validate_self_hashed_report(
        proof,
        schema_version=(
            "qsafe.state_dependent_recovery_v5."
            "stage_b_split_disjointness.v2"),
        name="nested split identity proof",
    )
    if proof.get("dimensions") != list(SPLIT_COLLISION_DIMENSIONS) or proof.get(
            "identity_array_fields") != dict(
                SPLIT_IDENTITY_SOURCE_FIELDS) or proof.get(
                "pairs_checked") != 10 or proof.get(
                "outcome_columns_read") is not False or proof.get(
                    "pass") is not True:
        raise StageBModelTestError("nested split identity proof gate drifted")
    roles = proof.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(ROLE_ORDER):
        raise StageBModelTestError("nested split role roster drifted")
    for role in ROLE_ORDER:
        record = roles.get(role)
        if not isinstance(record, Mapping) or set(record) != {
                "groups", "source_seeds", "actor_training_seeds",
                "identity_commitment_sha256", "outcome_columns_read"} or (
                    record.get("groups") != len(ROLE_SOURCE_SEEDS[role]) *
                    GROUPS_PER_SOURCE[role]) or record.get(
                        "source_seeds") != sorted(ROLE_SOURCE_SEEDS[role]) or (
                            record.get("actor_training_seeds") != sorted(
                                ROLE_ACTOR_SEEDS[role])) or not _is_sha256(
                                    record.get(
                                        "identity_commitment_sha256")) or (
                                            record.get(
                                                "outcome_columns_read") is not
                                            False):
            raise StageBModelTestError(
                f"nested split identity for {role} drifted")
    pairs = proof.get("pairs")
    expected_pairs = [
        (left, right)
        for index, left in enumerate(ROLE_ORDER)
        for right in ROLE_ORDER[index + 1:]
    ]
    if not isinstance(pairs, list) or len(pairs) != len(expected_pairs):
        raise StageBModelTestError("nested split pair roster drifted")
    for record, (left, right) in zip(pairs, expected_pairs, strict=True):
        collisions = record.get("collision_counts") if isinstance(
            record, Mapping) else None
        if not isinstance(record, Mapping) or set(record) != {
                "left", "right", "collision_counts", "pass"} or record.get(
                    "left") != left or record.get("right") != right or (
                        record.get("pass") is not True) or not isinstance(
                            collisions, Mapping) or set(collisions) != set(
                                SPLIT_COLLISION_DIMENSIONS) or any(
                                    isinstance(count, bool) or not isinstance(
                                        count, int) or count != 0
                                    for count in collisions.values()):
            raise StageBModelTestError(
                f"nested split pair {left}/{right} drifted")

    partition = value.get("partition_rng_proof")
    if not isinstance(partition, Mapping) or set(partition) != {
            "schema_version", "domains", "namespaces", "pairs_checked",
            "pairs", "outcome_columns_read", "pass", "report_sha256"}:
        raise StageBModelTestError("partition RNG proof schema drifted")
    _validate_self_hashed_report(
        partition,
        schema_version=(
            "qsafe.state_dependent_recovery_v5."
            "stage_b_partition_rng_disjointness.v1"),
        name="partition RNG proof",
    )
    expected_domains = [
        f"{role}/{partition_name}"
        for role in ROLE_ORDER
        for partition_name in ("admission", "label")
    ]
    expected_namespaces = {
        "admission": [
            "admission_crn_id", "admission_rollout_seed",
            "admission_perturbation_seed", "admission_candidate_seed",
        ],
        "label": ["crn_id", "rollout_seed", "perturbation_seed", "candidate_seed"],
    }
    ppairs = partition.get("pairs")
    if partition.get("domains") != expected_domains or partition.get(
            "namespaces") != expected_namespaces or partition.get(
                "pairs_checked") != 45 or partition.get(
                    "outcome_columns_read") is not False or partition.get(
                        "pass") is not True or not isinstance(ppairs, list) or len(
                            ppairs) != 45:
        raise StageBModelTestError("partition RNG proof gate drifted")
    expected_domain_pairs = [
        (left, right)
        for index, left in enumerate(expected_domains)
        for right in expected_domains[index + 1:]
    ]
    for record, (left, right) in zip(ppairs, expected_domain_pairs, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
                "left", "right", "collision_count", "pass"} or record.get(
                    "left") != left or record.get("right") != right or record.get(
                        "collision_count") != 0 or record.get("pass") is not True:
            raise StageBModelTestError(
                f"partition RNG pair {left}/{right} drifted")
    return report_sha256


def _require_generator_commit_alignment(
    *,
    evaluator_commit: str,
    actor_bank_generator_commit: str,
    commitment_generator_commit: str,
) -> None:
    commits = (
        evaluator_commit,
        actor_bank_generator_commit,
        commitment_generator_commit,
    )
    if any(not isinstance(value, str) or len(value) != 40 for value in commits) or (
            len(set(commits)) != 1):
        raise StageBModelTestError(
            "evaluator, actor-bank, and commitment generator commits differ")


@dataclass(frozen=True)
class _FrozenPrerequisites:
    hashes: dict[str, str]
    paths: dict[str, Path]
    actor_bank: dict[str, Any]
    reports: dict[str, dict[str, Any]]
    report_identity_sha256: dict[str, str]
    selector_bundle: RecoverySelectorBundle
    placebo_bundle: MatchedRandomPlaceboBundle
    artifact: LoadedQSafeArtifact
    artifact_manifest_identity_sha256: str


def _load_frozen_prerequisites(
    *,
    stage_b_root: Path,
    commitment_sha256: str,
    committed_model_test_label_sha256: str,
) -> _FrozenPrerequisites:
    if set(_FROZEN_PATHS) != set(STAGE_B_FROZEN_INPUT_NAMES):
        raise AssertionError("evaluator prerequisite roster drifted")
    paths = {
        name: stage_b_root / relative
        for name, relative in _FROZEN_PATHS.items()
    }
    for name, path in paths.items():
        _require_real_ancestor_chain(stage_b_root, path, name)
    raw_json: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        value, digest = _read_json(path, name)
        raw_json[name] = value
        hashes[name] = digest

    actor_bank = load_reduced7_actor_bank_manifest(
        paths["actor_bank_manifest"],
        expected_bindings={
            "manifest_file_sha256": hashes["actor_bank_manifest"],
            "protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
            "protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
            "execution_supplement_file_sha256": (
                EXECUTION_PROTOCOL_FILE_SHA256),
            "execution_supplement_contract_sha256": (
                EXECUTION_PROTOCOL_CONTRACT_SHA256),
            "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        },
    )
    generator_commit = actor_bank.get("stage_b_generator_commit")
    if not isinstance(generator_commit, str) or len(generator_commit) != 40:
        raise StageBModelTestError("actor-bank generator commit is invalid")

    split = raw_json["split_disjointness_report"]
    if _regular_bytes(
            paths["split_disjointness_report"],
            "split disjointness report") != _canonical_json_bytes(split):
        raise StageBModelTestError(
            "split disjointness report is not canonical JSON")
    actor_bank_contract_sha256 = actor_bank.get(
        "actor_bank_contract_sha256")
    if not _is_sha256(actor_bank_contract_sha256):
        raise StageBModelTestError("actor-bank contract SHA-256 is invalid")
    split_sha = _validate_split_disjointness_report(
        split,
        generator_commit=generator_commit,
        actor_bank_manifest_file_sha256=hashes["actor_bank_manifest"],
        actor_bank_contract_sha256=str(actor_bank_contract_sha256),
        committed_model_test_label_sha256=(
            committed_model_test_label_sha256),
    )

    identity_sha: dict[str, str] = {
        "split_disjointness_report": split_sha,
    }
    reports: dict[str, dict[str, Any]] = {
        "split_disjointness_report": split,
    }
    for name, schema in _REPORT_SCHEMAS.items():
        report = raw_json[name]
        if _regular_bytes(paths[name], name.replace("_", " ")) != (
                _canonical_json_bytes(report)):
            raise StageBModelTestError(
                f"{name.replace('_', ' ')} is not canonical JSON")
        identity_sha[name] = _validate_self_hashed_report(
            report, schema_version=schema, name=name.replace("_", " "))
        reports[name] = report

    normalization_report = reports["normalization_report"]
    if normalization_report.get("source_role") != "fit" or (
            normalization_report.get("privileged_features_absent") is not True):
        raise StageBModelTestError(
            "normalization report is not the frozen deployable fit-only result")
    probability_report = reports["probability_calibration_report"]
    if probability_report.get("source_role") != "probability_calibration" or (
            probability_report.get("normalization_report_sha256") !=
            identity_sha["normalization_report"]) or (
                probability_report.get("normalization_report_file_sha256") !=
                hashes["normalization_report"]):
        raise StageBModelTestError(
            "probability calibration does not bind fit-only normalization")
    uncertainty_report = reports["uncertainty_calibration_report"]
    if uncertainty_report.get("source_role") != "uncertainty_calibration" or (
            uncertainty_report.get("probability_calibration_report_sha256") !=
            identity_sha["probability_calibration_report"]) or (
                uncertainty_report.get(
                    "probability_calibration_report_file_sha256") !=
                hashes["probability_calibration_report"]):
        raise StageBModelTestError(
            "uncertainty calibration does not bind probability calibration")
    selector_report = reports["selector_search_report"]
    if selector_report.get("source_role") != "selector_calibration" or (
            selector_report.get("probability_calibration_report_sha256") !=
            identity_sha["probability_calibration_report"]) or (
                selector_report.get("uncertainty_calibration_report_sha256") !=
                identity_sha["uncertainty_calibration_report"]) or (
                    selector_report.get(
                        "uncertainty_calibration_report_file_sha256") !=
                    hashes["uncertainty_calibration_report"]) or (
                        selector_report.get("development_decision") !=
                        "freeze_selected_selector") or (
                            selector_report.get("model_test_outcomes_read") is not
                            False) or selector_report.get(
                                "model_test_consumed") is not False:
        raise StageBModelTestError(
            "selector report did not freeze a result-blind feasible selector")

    report_frozen_identity = normalization_report.get("frozen_identity")
    if not isinstance(report_frozen_identity, Mapping) or any(
            report.get("frozen_identity") != report_frozen_identity
            for report in (
                probability_report, uncertainty_report, selector_report)):
        raise StageBModelTestError(
            "development reports do not share one frozen identity")
    aggregate_by_role = {
        str(record["role"]): record
        for record in split["role_aggregate_labels"]
    }
    for name, role in (
        ("normalization_report", "fit"),
        ("probability_calibration_report", "probability_calibration"),
        ("uncertainty_calibration_report", "uncertainty_calibration"),
        ("selector_search_report", "selector_calibration"),
    ):
        report = reports[name]
        aggregate = aggregate_by_role[role]
        if report.get("source_array_sha256") != aggregate.get(
                "file_sha256") or report.get(
                    "source_content_sha256") != aggregate.get(
                        "content_sha256"):
            raise StageBModelTestError(
                f"{name.replace('_', ' ')} source binding drifted")

    if _regular_bytes(
            paths["recovery_selector_bundle"], "recovery selector bundle") != (
                _canonical_json_bytes(raw_json["recovery_selector_bundle"])):
        raise StageBModelTestError("recovery selector bundle is not canonical JSON")
    selector = RecoverySelectorBundle.from_dict(
        raw_json["recovery_selector_bundle"])
    if selector.probability_calibration_report_sha256 != identity_sha[
            "probability_calibration_report"] or (
                selector.uncertainty_calibration_report_sha256 != identity_sha[
                    "uncertainty_calibration_report"]) or (
                        selector.selector_search_report_sha256 != identity_sha[
                            "selector_search_report"]):
        raise StageBModelTestError(
            "selector bundle report-hash provenance is inconsistent")
    signed_payload = uncertainty_report.get("signed_conformal")
    try:
        families = signed_payload["families"]
        offsets = signed_payload["offsets"]
        conformal = SignedConformalCalibration(
            nominal_lower=offsets["nominal_lower"],
            risk_upper=np.asarray(offsets["risk_upper"], dtype=np.float64),
            benefit_lower=np.asarray(
                offsets["benefit_lower"], dtype=np.float64),
            group_count=signed_payload["group_count"],
            option_rank=families["risk_upper"]["one_based_rank"],
            nominal_rank=families[
                "nominal_trigger_lower"]["one_based_rank"],
            execution_lock_sha256=signed_payload["execution_lock_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StageBModelTestError(
            "signed conformal calibration payload is invalid") from exc
    if conformal.report_payload() != signed_payload or conformal.group_count != (
            len(ROLE_SOURCE_SEEDS["uncertainty_calibration"])
            * GROUPS_PER_SOURCE["uncertainty_calibration"]) or (
                conformal.option_rank != 383) or conformal.nominal_rank != 366 or (
                    conformal.execution_lock_sha256 !=
                    EXECUTION_PROTOCOL_CONTRACT_SHA256) or not np.array_equal(
                        conformal.risk_upper,
                        selector.offsets.risk_upper) or not np.array_equal(
                            conformal.benefit_lower,
                            selector.offsets.benefit_lower) or (
                                conformal.nominal_lower !=
                                selector.offsets.nominal_lower):
        raise StageBModelTestError(
            "selector offsets differ from frozen signed conformal calibration")
    selector_search = selector_report.get("selector_search")
    if not isinstance(selector_search, Mapping):
        raise StageBModelTestError("selector search payload is missing")
    selector_bootstrap = selector_search.get("bootstrap")
    selector_feasibility = selector_search.get("feasibility")
    if selector_search.get("execution_lock_sha256") != (
            EXECUTION_PROTOCOL_CONTRACT_SHA256) or selector_search.get(
                "grid_points_exact") != 100 or not isinstance(
                    selector_bootstrap, Mapping) or selector_bootstrap.get(
                        "replicates") != (
                            STAGE_B_SELECTOR_BOOTSTRAP_REPLICATES) or (
                                selector_bootstrap.get("seed") !=
                                STAGE_B_SELECTOR_BOOTSTRAP_SEED) or (
                                    selector_bootstrap.get(
                                        "rng_bit_generator") !=
                                    "numpy_PCG64") or selector_bootstrap.get(
                                        "quantile_method") != "linear" or not (
                                            isinstance(
                                                selector_feasibility,
                                                Mapping)) or dict(
                                                    selector_feasibility) != {
                                                        "minimum_absolute_reduction": 0.03,
                                                        "simultaneous_lcb_strictly_positive": True,
                                                        "maximum_intervention_rate": 0.35,
                                                    }:
        raise StageBModelTestError(
            "selector search bootstrap or frozen gates drifted")
    selected_index = selector_search.get("selected_grid_index")
    if isinstance(selected_index, bool) or not isinstance(selected_index, int) or (
            selected_index < 0 or selected_index >= 100):
        raise StageBModelTestError(
            "selector search did not freeze one feasible grid point")
    rows = selector_search.get("rows")
    if not isinstance(rows, list) or len(rows) != 100 or not isinstance(
            rows[selected_index], Mapping) or rows[selected_index].get(
                "grid_index") != selected_index or rows[selected_index].get(
                    "feasible") is not True or rows[selected_index].get(
                        "selector_config") != selector.to_dict()[
                            "selector_config"]:
        raise StageBModelTestError(
            "selector bundle differs from the frozen feasible search row")

    if _regular_bytes(
            paths["matched_random_placebo_bundle"],
            "matched-random placebo bundle") != _canonical_json_bytes(
                raw_json["matched_random_placebo_bundle"]):
        raise StageBModelTestError(
            "matched-random placebo bundle is not canonical JSON")
    placebo = MatchedRandomPlaceboBundle.from_dict(
        raw_json["matched_random_placebo_bundle"])
    if placebo.selector_bundle_sha256 != selector.bundle_sha256 or not (
            placebo.fit_metrics.eligible) or placebo.execution_lock_sha256 != (
                EXECUTION_PROTOCOL_CONTRACT_SHA256) or (
                    placebo.fit_rng_assignment_count != len(
                        ROLE_SOURCE_SEEDS["selector_calibration"])
                    * GROUPS_PER_SOURCE["selector_calibration"]):
        raise StageBModelTestError(
            "matched-random placebo is not eligible/bound to the selector")

    artifact_manifest = raw_json["qsafe_artifact"]
    artifact_manifest_identity_sha256 = _canonical_object_sha256(
        artifact_manifest)
    artifact = load_qsafe_artifact(
        paths["qsafe_artifact"].parent,
        device="cpu",
        expected_manifest_sha256=artifact_manifest_identity_sha256,
    )
    artifact.require_live_integrity()
    provenance = artifact.manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise StageBModelTestError("Q_safe artifact provenance is missing")
    expected_frozen_identity_fields = {
        "protocol_name",
        "parent_protocol_file_sha256",
        "parent_protocol_contract_sha256",
        "execution_protocol_file_sha256",
        "execution_protocol_contract_sha256",
        "stage_a_report_sha256",
        "stage_a_disposition_commit",
        "recovery_library_fingerprint_sha256",
        "generator_commit",
        "model_test_commitment_file_sha256",
        "model_test_outcomes_read",
        "model_test_consumed",
        "actor_bank_manifest",
        "split_disjointness_report",
        "development_role_inputs",
    }
    if set(report_frozen_identity) != expected_frozen_identity_fields or any(
            provenance.get(name) != value
            for name, value in report_frozen_identity.items()):
        raise StageBModelTestError(
            "development-report frozen identity differs from Q_safe artifact")
    required_identity = {
        "protocol_name": EXECUTION_PROTOCOL_NAME,
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256),
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
        "recovery_library_fingerprint_sha256": (
            RECOVERY_LIBRARY_FINGERPRINT_SHA256),
        "generator_commit": generator_commit,
        "model_test_commitment_file_sha256": commitment_sha256,
        "model_test_outcomes_read": False,
        "model_test_consumed": False,
        "command_vx": 0.30,
        "action_view": RECOVERY_PROGRAM_VIEW,
        "recovery_selector_bundle_sha256": selector.bundle_sha256,
        "matched_random_placebo_bundle_sha256": placebo.bundle_sha256,
    }
    for name, expected in required_identity.items():
        if provenance.get(name) != expected:
            raise StageBModelTestError(
                f"Q_safe artifact provenance {name} drifted")
    if provenance.get("recovery_selector_bundle") != selector.to_dict():
        raise StageBModelTestError(
            "Q_safe artifact does not embed the exact selector bundle")

    development_inputs = report_frozen_identity.get("development_role_inputs")
    if not isinstance(development_inputs, Mapping) or set(
            development_inputs) != set(ROLE_ORDER[:-1]):
        raise StageBModelTestError(
            "frozen development-role input roster drifted")
    for role in ROLE_ORDER[:-1]:
        binding = development_inputs.get(role)
        aggregate = aggregate_by_role[role]
        directory = role.replace("_", "-")
        replicas = LABEL_REPLICAS[role]
        expected_binding_fields = {
            "role", "relative_path", "file_sha256", "content_sha256",
            "groups", "candidates", "replicas", "source_seeds",
            "actor_training_seeds", "role_report", "completion_marker",
        }
        binding_valid = isinstance(binding, Mapping) and all((
            set(binding) == expected_binding_fields,
            binding.get("role") == role,
            binding.get("relative_path") == (
                f"{directory}/labels-r{replicas}-deployable.npz"),
            binding.get("file_sha256") == aggregate.get("file_sha256"),
            binding.get("content_sha256") == aggregate.get("content_sha256"),
            binding.get("groups") == aggregate.get("groups"),
            binding.get("candidates") == 9,
            binding.get("replicas") == replicas,
            binding.get("source_seeds") == sorted(ROLE_SOURCE_SEEDS[role]),
            binding.get("actor_training_seeds") == sorted(
                ROLE_ACTOR_SEEDS[role]),
        ))
        if not binding_valid:
            raise StageBModelTestError(
                f"frozen development input {role} drifted")
        role_report_binding = binding.get("role_report")
        completion_binding = binding.get("completion_marker")
        if not isinstance(role_report_binding, Mapping) or set(
                role_report_binding) != {"relative_path", "file_sha256"} or (
                    role_report_binding.get("relative_path") !=
                    f"{directory}/report.json") or role_report_binding.get(
                        "file_sha256") != aggregate.get(
                            "role_report_file_sha256") or not isinstance(
                                completion_binding, Mapping) or set(
                                    completion_binding) != {
                                        "relative_path", "file_sha256"} or (
                                            completion_binding.get(
                                                "relative_path") !=
                                            f"{directory}/completed.json") or (
                                                not _is_sha256(
                                                    completion_binding.get(
                                                        "file_sha256"))):
            raise StageBModelTestError(
                f"frozen development control binding {role} drifted")

    probability_model = probability_report.get("model")
    temperature_calibration = probability_report.get("temperature_calibration")
    artifact_members = artifact.manifest.get("members")
    if not isinstance(probability_model, Mapping) or set(probability_model) != {
            "network_config", "training_config", "loss_config", "members"} or (
                probability_model.get("network_config") !=
                artifact.manifest.get("network_config")) or (
                    probability_model.get("training_config") !=
                    artifact.manifest.get("training_config")) or (
                        probability_model.get("loss_config") !=
                        artifact.manifest.get("loss_config")) or not isinstance(
                            probability_model.get("members"), list) or not (
                                isinstance(artifact_members, list)) or len(
                                    probability_model["members"]) != 5 or len(
                                        artifact_members) != 5:
        raise StageBModelTestError(
            "probability report model contract differs from Q_safe artifact")
    reported_temperatures: list[float] = []
    for index, (reported, stored) in enumerate(zip(
            probability_model["members"], artifact_members, strict=True)):
        if not isinstance(reported, Mapping) or set(reported) != {
                "member_index", "seed", "epochs", "epoch_loss_f8_sha256",
                "temperature", "trajectory_bootstrap_count",
                "trajectory_bootstrap_sha256"} or not isinstance(
                    stored, Mapping):
            raise StageBModelTestError(
                f"probability report member {index} is malformed")
        epoch_loss = stored.get("epoch_loss")
        trajectories = stored.get("bootstrap_trajectories")
        if not isinstance(epoch_loss, list) or not isinstance(trajectories, list):
            raise StageBModelTestError(
                f"probability report member {index} metadata is malformed")
        member_valid = all((
            reported.get("member_index") == index,
            reported.get("seed") == stored.get("seed"),
            reported.get("epochs") == len(epoch_loss),
            reported.get("epoch_loss_f8_sha256") == hashlib.sha256(
                np.asarray(epoch_loss, dtype="<f8").tobytes(
                    order="C")).hexdigest(),
            reported.get("temperature") == stored.get("temperature"),
            reported.get("trajectory_bootstrap_count") == len(trajectories),
            reported.get("trajectory_bootstrap_sha256") == (
                _ordered_text_sha256(trajectories)),
        ))
        if not member_valid:
            raise StageBModelTestError(
                f"probability report member {index} differs from artifact")
        reported_temperatures.append(float(reported["temperature"]))
    if not isinstance(temperature_calibration, Mapping) or (
            temperature_calibration.get("member_temperatures") !=
            reported_temperatures) or temperature_calibration.get(
                "steps") != 100 or temperature_calibration.get(
                    "optimizer") != "Adam" or temperature_calibration.get(
                        "learning_rate") != 0.05 or temperature_calibration.get(
                            "log_temperature_clamp") != [-4.0, 4.0] or (
                                temperature_calibration.get("weighting") !=
                                "equal_group_then_equal_valid_K9_candidate"):
        raise StageBModelTestError(
            "probability temperature calibration differs from artifact")

    actor_binding = provenance.get("actor_bank_manifest")
    if not isinstance(actor_binding, Mapping) or set(actor_binding) != {
            "relative_path", "file_sha256", "contract_sha256",
            "identity_count"} or actor_binding.get("relative_path") != (
                _FROZEN_PATHS["actor_bank_manifest"]) or actor_binding.get(
                    "file_sha256") != hashes[
                        "actor_bank_manifest"] or actor_binding.get(
                            "contract_sha256") != actor_bank_contract_sha256 or (
                                actor_binding.get("identity_count") !=
                                actor_bank.get("identity_count")):
        raise StageBModelTestError(
            "Q_safe artifact actor-bank binding drifted")
    split_binding = provenance.get("split_disjointness_report")
    if not isinstance(split_binding, Mapping) or set(split_binding) != {
            "relative_path", "file_sha256", "report_sha256", "pairs_checked",
            "pass"} or split_binding.get("relative_path") != _FROZEN_PATHS[
                "split_disjointness_report"] or split_binding.get(
                    "file_sha256") != hashes[
                        "split_disjointness_report"] or split_binding.get(
                            "report_sha256") != identity_sha[
                                "split_disjointness_report"] or (
                                    split_binding.get("pairs_checked") != 10) or (
                                        split_binding.get("pass") is not True):
        raise StageBModelTestError(
            "Q_safe artifact split-proof binding drifted")

    frozen = provenance.get("frozen_development_artifacts")
    expected_frozen_contracts = {
        "normalization_report": identity_sha["normalization_report"],
        "probability_calibration_report": identity_sha[
            "probability_calibration_report"],
        "uncertainty_calibration_report": identity_sha[
            "uncertainty_calibration_report"],
        "selector_search_report": identity_sha["selector_search_report"],
        "recovery_selector_bundle": selector.bundle_sha256,
        "matched_random_placebo_bundle": placebo.bundle_sha256,
    }
    if not isinstance(frozen, Mapping) or set(frozen) != set(
            expected_frozen_contracts):
        raise StageBModelTestError(
            "Q_safe artifact frozen-development roster drifted")
    for name, contract_sha in expected_frozen_contracts.items():
        binding = frozen.get(name)
        if not isinstance(binding, Mapping) or set(binding) != {
                "relative_path", "file_sha256", "contract_sha256"} or (
                    binding.get("relative_path") != _FROZEN_PATHS[name]) or (
                    binding.get("file_sha256") != hashes[name]) or (
                        binding.get("contract_sha256") != contract_sha):
            raise StageBModelTestError(
                f"Q_safe artifact frozen binding {name} drifted")

    mean = np.ascontiguousarray(
        artifact.normalization.observation_mean, dtype=np.dtype("<f4"))
    std = np.ascontiguousarray(
        artifact.normalization.observation_std, dtype=np.dtype("<f4"))
    if hashlib.sha256(mean.tobytes()).hexdigest() != normalization_report.get(
            "observation_mean_f4_sha256") or hashlib.sha256(
                std.tobytes()).hexdigest() != normalization_report.get(
                    "observation_std_f4_sha256") or (
                        artifact.normalization.fit_content_sha256 !=
                        normalization_report.get("source_content_sha256")):
        raise StageBModelTestError(
            "Q_safe artifact normalization differs from its frozen report")
    if artifact.normalization.fit_split != (
            "state_dependent_recovery_v5_stage_b_fit_label"):
        raise StageBModelTestError(
            "Q_safe artifact normalization was not fitted on Stage-B fit only")
    if artifact.network_config.privileged_dim != 0 or artifact.action_view != (
            RECOVERY_PROGRAM_VIEW):
        raise StageBModelTestError(
            "Stage-B Model-Test requires deployable 82D recovery Q_safe")

    return _FrozenPrerequisites(
        hashes=hashes,
        paths=paths,
        actor_bank=actor_bank,
        reports=reports,
        report_identity_sha256=identity_sha,
        selector_bundle=selector,
        placebo_bundle=placebo,
        artifact=artifact,
        artifact_manifest_identity_sha256=artifact_manifest_identity_sha256,
    )


def _validate_model_test_dataset(
    dataset: GroupedBranchDataset,
    *,
    actor_bank: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
    generator_commit: str,
) -> np.ndarray:
    report = dataset.validate()
    expected_sources = tuple(ROLE_SOURCE_SEEDS["model_test"])
    expected_actors = tuple(ROLE_ACTOR_SEEDS["model_test"])
    try:
        trajectory_fingerprints = np.asarray(
            dataset[TRAJECTORY_FINGERPRINT_ARRAY])
    except (KeyError, TypeError) as exc:
        raise StageBModelTestError(
            "Model-Test lacks true trajectory fingerprints") from exc
    trajectory_fingerprint_gate = bool(
        trajectory_fingerprints.shape == (768,)
        and trajectory_fingerprints.dtype.kind in "US"
        and len(np.unique(trajectory_fingerprints.astype(str))) == 768
        and all(_is_sha256(value) for value in (
            trajectory_fingerprints.astype(str).tolist()))
    )
    checks = (
        dataset.group_count == 768,
        dataset.candidate_count == 9,
        dataset.replica_count == LABEL_REPLICAS["model_test"] == 64,
        dataset.horizon_steps == HORIZON_POLICY_STEPS == 96,
        dataset.manifest.get("split") == (
            "state_dependent_recovery_v5_stage_b_model_test_label"),
        dataset.manifest.get("generator_commit") == generator_commit,
        dataset.manifest.get("execution_protocol_file_sha256") == (
            EXECUTION_PROTOCOL_FILE_SHA256),
        dataset.manifest.get("execution_protocol_contract_sha256") == (
            EXECUTION_PROTOCOL_CONTRACT_SHA256),
        dataset.manifest.get("actor_bank_manifest_file_sha256") == (
            actor_bank_manifest_file_sha256),
        dataset.manifest.get("actor_bank_contract_sha256") == actor_bank.get(
            "actor_bank_contract_sha256"),
        dataset.manifest.get("stage_b_role") == "model_test",
        dataset.manifest.get("source_seed_order") == list(expected_sources),
        trajectory_fingerprint_gate,
        report.get("unique_trajectory_clusters") == 768,
        report.get("unique_source_seeds") == 12,
        np.all(np.asarray(dataset["candidate_mask"], dtype=bool)),
        np.array_equal(
            np.asarray(dataset["acceptance_probability"], dtype=np.float64),
            np.ones(768, dtype=np.float64)),
    )
    if not all(checks):
        raise StageBModelTestError("canonical Model-Test dataset gates failed")
    recovery_program = dataset.manifest.get("recovery_program")
    if not isinstance(recovery_program, Mapping) or recovery_program.get(
            "fingerprint_sha256") != RECOVERY_LIBRARY_FINGERPRINT_SHA256 or (
                not np.allclose(
                    np.asarray(dataset["command_vx"], dtype=np.float64),
                    0.30,
                    rtol=0.0,
                    atol=1e-6,
                )):
        raise StageBModelTestError(
            "Model-Test recovery-program/command binding drifted")
    collection = dataset.manifest.get("collection_protocol")
    collection_valid = isinstance(collection, Mapping) and all((
        collection.get("role") == "model_test",
        collection.get("partition") == "label",
        collection.get("label_replicas") == 64,
        collection.get("candidate_count") == 9,
        collection.get("max_groups_per_trajectory") == 1,
        collection.get("trajectory_fingerprint_array") == (
            TRAJECTORY_FINGERPRINT_ARRAY),
        collection.get("trajectory_fingerprint_contract") == (
            TRAJECTORY_FINGERPRINT_CONTRACT),
    ))
    if not collection_valid:
        raise StageBModelTestError("Model-Test collection protocol drifted")
    sources = np.asarray(dataset["source_seed"], dtype=np.int64)
    actors = np.asarray(dataset["policy_training_seed"], dtype=np.int64)
    if tuple(sorted(map(int, np.unique(sources)))) != expected_sources or (
            tuple(sorted(map(int, np.unique(actors)))) != expected_actors):
        raise StageBModelTestError("Model-Test source/actor roster drifted")
    checkpoints = np.empty(dataset.group_count, dtype=np.int64)
    policy_source = np.asarray(dataset["policy_source"]).astype(str)
    for source in expected_sources:
        selected = sources == source
        if int(np.count_nonzero(selected)) != GROUPS_PER_SOURCE["model_test"]:
            raise StageBModelTestError(
                "Model-Test must retain exactly 64 groups per source")
        assignment = assignment_for("model_test", source)
        if not np.all(actors[selected] == assignment.actor_training_seed):
            raise StageBModelTestError("Model-Test actor/source assignment drifted")
        checkpoints[selected] = assignment.checkpoint_step
        actor_identity = actor_identity_for(
            actor_bank,
            role="model_test",
            actor_seed=assignment.actor_training_seed,
            checkpoint_step=assignment.checkpoint_step,
        )
        if not np.all(policy_source[selected] == actor_identity[
                "policy_fingerprint_sha256"]):
            raise StageBModelTestError(
                "Model-Test policy fingerprint differs from actor bank")
    if tuple(sorted(map(int, np.unique(checkpoints)))) != CHECKPOINT_STEPS:
        raise StageBModelTestError("Model-Test checkpoint ages drifted")
    return checkpoints


@torch.no_grad()
def _predict_member_risk(
    artifact: LoadedQSafeArtifact,
    view: TorchGroupedView,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    artifact.require_live_integrity()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    result = np.full(
        (view.group_count, 5, 9), np.nan, dtype=np.float32)
    ensemble = artifact.ensemble.to("cpu").eval()
    for start in range(0, view.group_count, batch_size):
        indices = np.arange(
            start, min(start + batch_size, view.group_count), dtype=np.int64)
        batch = view.batch(indices, "cpu")
        prediction = ensemble.predict(
            batch.observation_history,
            batch.nominal_action,
            batch.candidate_action,
            batch.privileged_state,
        ).member_risk.detach().cpu().numpy()
        if prediction.shape != (5, len(indices), 9):
            raise StageBModelTestError("Q_safe member prediction shape drifted")
        result[indices] = np.transpose(prediction, (1, 0, 2))
    artifact.require_live_integrity()
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise StageBModelTestError("Q_safe produced invalid Model-Test risks")
    return result


def _final_report(
    *,
    evaluator_commit: str,
    generator_commit: str,
    commitment_sha256: str,
    consumed: Mapping[str, Any],
    prerequisites: _FrozenPrerequisites,
    dataset: GroupedBranchDataset,
    dataset_file_sha256: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    prerequisite_gates = {
        "identity_gate": True,
        "data_gate": True,
        "calibration_gate": True,
        "selector_gate": True,
        "placebo_gate": bool(
            prerequisites.placebo_bundle.fit_metrics.eligible),
        "hash_gate": set(prerequisites.hashes) == set(
            STAGE_B_FROZEN_INPUT_NAMES) and all(
                _is_sha256(value) for value in prerequisites.hashes.values()),
        "one_shot_consumption_gate": _is_sha256(
            consumed.get("consumed_marker_sha256")),
    }
    all_pass = bool(statistics.get("pass")) and all(
        prerequisite_gates.values())
    report: dict[str, Any] = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256),
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
        "generator_commit": generator_commit,
        "evaluator_clean_commit": evaluator_commit,
        "model_test_commitment_sha256": commitment_sha256,
        "model_test_consumed_marker_sha256": consumed[
            "consumed_marker_sha256"],
        "prerequisite_artifact_raw_file_sha256": dict(sorted(
            prerequisites.hashes.items())),
        "prerequisite_report_identity_sha256": dict(sorted(
            prerequisites.report_identity_sha256.items())),
        "qsafe_artifact_manifest_identity_sha256": (
            prerequisites.artifact_manifest_identity_sha256),
        "recovery_selector_bundle_sha256": (
            prerequisites.selector_bundle.bundle_sha256),
        "matched_random_placebo_bundle_sha256": (
            prerequisites.placebo_bundle.bundle_sha256),
        "model_test_dataset": {
            "path": _MODEL_TEST_LABEL_RELATIVE,
            "file_sha256": dataset_file_sha256,
            "content_sha256": dataset.manifest["content_sha256"],
            "groups": dataset.group_count,
            "candidates": dataset.candidate_count,
            "replicas": dataset.replica_count,
            "outcome_used_for_model_fit_calibration_or_selection": False,
        },
        "prerequisite_gates": prerequisite_gates,
        "statistics": dict(statistics),
        "stage_B_pass": all_pass,
        "decision": (
            "authorize_stage_C_only" if all_pass else "no_further_stage"),
        "stage_C_authorized": all_pass,
        "objective1_pass": False,
        "phase2_authorized": False,
        "model_or_threshold_updates_from_model_test": False,
        "report_publication": "atomic_no_clobber_report_last",
    }
    report["report_sha256"] = _canonical_object_sha256(report)
    return report


def _publish_report_no_clobber(path: Path, report: Mapping[str, Any]) -> str:
    payload = _canonical_json_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.pending-")
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise StageBModelTestError(
                "refusing to overwrite the Stage-B report") from exc
        published = True
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if not published:  # pragma: no cover - exceptions leave first.
        raise StageBModelTestError("Stage-B report publication failed")
    return hashlib.sha256(payload).hexdigest()


def evaluate_canonical_stage_b_model_test(
    *,
    expected_commitment_sha256: str,
) -> dict[str, Any]:
    """Run the canonical one-shot workflow with no statistical override."""
    if not _is_sha256(expected_commitment_sha256):
        raise StageBModelTestError(
            "expected Model-Test commitment must be lowercase SHA-256")
    execution = load_stage_b_execution_protocol()
    load_stage_b_reduced7_amendment()
    validate_stage_a_authorization(execution)
    parent = load_state_dependent_recovery_v5_protocol()
    stage_root = stage_b_artifact_root(parent)
    try:
        root_metadata = stage_root.lstat()
    except OSError as exc:
        raise StageBModelTestError("canonical Stage-B root is missing") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
            root_metadata.st_mode):
        raise StageBModelTestError(
            "canonical Stage-B root must be a real directory")
    commitment_path = stage_root / "model-test-committed.json"
    consumed_path = stage_root / "model-test-consumed.json"
    label_path = stage_root / "model-test/labels-r64-deployable.npz"
    report_path = stage_root.parent / "state-dependent-recovery-stage-b-report.json"
    if os.path.lexists(os.fspath(report_path)):
        raise StageBModelTestError("canonical Stage-B report already exists")
    if os.path.lexists(os.fspath(consumed_path)):
        raise StageBModelTestError(
            "Stage-B Model-Test has already been consumed or reserved")

    evaluator_commit = require_clean_stage_b_generator()
    committed_label_sha256, commitment_generator_commit = (
        _commitment_label_sha256(
            commitment_path, expected_commitment_sha256)
    )
    prerequisites = _load_frozen_prerequisites(
        stage_b_root=stage_root,
        commitment_sha256=expected_commitment_sha256,
        committed_model_test_label_sha256=committed_label_sha256,
    )
    generator_commit = str(
        prerequisites.actor_bank["stage_b_generator_commit"])
    _require_generator_commit_alignment(
        evaluator_commit=evaluator_commit,
        actor_bank_generator_commit=generator_commit,
        commitment_generator_commit=commitment_generator_commit,
    )

    # `label_path` remains a purely lexical Path until the context manager has
    # atomically published the consumed marker.  Scope setup then performs the
    # first commitment hash check, followed by the schema load below.
    with consume_stage_b_model_test(
        commitment_path=commitment_path,
        consumed_path=consumed_path,
        expected_commitment_sha256=expected_commitment_sha256,
        prerequisite_artifact_sha256=prerequisites.hashes,
        evaluator_clean_commit=evaluator_commit,
        evidence_paths=[label_path],
    ) as consumed:
        dataset = GroupedBranchDataset.load(label_path)
        dataset_file_sha256 = _regular_sha256(
            label_path, "consumed canonical Model-Test label array")
        if dataset_file_sha256 != committed_label_sha256:
            raise StageBModelTestError(
                "consumed Model-Test label bytes differ from commitment")
        model_test_aggregate = prerequisites.reports[
            "split_disjointness_report"]["role_aggregate_labels"][-1]
        if model_test_aggregate.get("file_sha256") != dataset_file_sha256 or (
                dataset.manifest.get("content_sha256") !=
                model_test_aggregate.get("content_sha256")):
            raise StageBModelTestError(
                "consumed Model-Test identity differs from split proof")
        checkpoints = _validate_model_test_dataset(
            dataset,
            actor_bank=prerequisites.actor_bank,
            actor_bank_manifest_file_sha256=prerequisites.hashes[
                "actor_bank_manifest"],
            generator_commit=generator_commit,
        )
        view = TorchGroupedView(
            dataset,
            prerequisites.artifact.normalization,
            action_view=RECOVERY_PROGRAM_VIEW,
            view_role="test",
        )
        member_risk = _predict_member_risk(prerequisites.artifact, view)
        statistics = evaluate_stage_b_model_test(
            member_risk=member_risk,
            fall=np.asarray(dataset["fall"]),
            candidate_requested=np.asarray(dataset["candidate_requested"]),
            candidate_executed=np.asarray(dataset["candidate_executed"]),
            candidate_q_target=np.asarray(dataset["candidate_q_target"]),
            candidate_mask=np.asarray(dataset["candidate_mask"]),
            candidate_behavior_steps=np.asarray(
                dataset["candidate_behavior_steps"]),
            actor_training_seed=np.asarray(dataset["policy_training_seed"]),
            source_seed=np.asarray(dataset["source_seed"]),
            checkpoint_step=checkpoints,
            trajectory_fingerprint_sha256=np.asarray(
                dataset[TRAJECTORY_FINGERPRINT_ARRAY]),
            group_id=np.asarray(dataset["group_id"]),
            group_fingerprint_sha256=np.asarray(dataset["state_hash"]),
            selector_bundle=prerequisites.selector_bundle,
            placebo_bundle=prerequisites.placebo_bundle,
            bootstrap_replicates=STAGE_B_MODEL_TEST_BOOTSTRAP_REPLICATES,
            bootstrap_seed=STAGE_B_MODEL_TEST_BOOTSTRAP_SEED,
            production_contract=True,
        )
        if require_clean_stage_b_generator() != evaluator_commit:
            raise StageBModelTestError(
                "worktree changed during the one-shot Stage-B evaluation")
        prerequisites.artifact.require_live_integrity()
        for name, path in prerequisites.paths.items():
            if _regular_sha256(
                    path, f"live frozen prerequisite {name}") != (
                        prerequisites.hashes[name]):
                raise StageBModelTestError(
                    f"frozen prerequisite {name} changed during evaluation")
        if _regular_sha256(
                label_path, "live consumed canonical Model-Test label array") != (
                    dataset_file_sha256):
            raise StageBModelTestError(
                "canonical Model-Test label changed during evaluation")
        report = _final_report(
            evaluator_commit=evaluator_commit,
            generator_commit=generator_commit,
            commitment_sha256=expected_commitment_sha256,
            consumed=consumed,
            prerequisites=prerequisites,
            dataset=dataset,
            dataset_file_sha256=dataset_file_sha256,
            statistics=statistics,
        )
        _publish_report_no_clobber(report_path, report)
    return report


def main() -> int:
    args = _parser().parse_args()
    report = evaluate_canonical_stage_b_model_test(
        expected_commitment_sha256=args.model_test_commitment_sha256)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
