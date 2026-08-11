"""Exact-path capabilities and one-shot Model-Test consumption for Stage B.

The controls here are a fail-closed accidental-misuse boundary.  They are not
a same-UID operating-system security boundary.  In particular, the compiler
receives one deliberately narrow API: it may read the outcome-free Model-Test
collection report and copy its already-produced file commitments, but it has
no evidence-read capability.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Iterator, Mapping, Sequence

from safety_data.paths import (
    ProtectedEvidencePathError,
    STAGE_B_EXECUTION_PROTOCOL_NAME,
    STAGE_B_FROZEN_INPUT_NAMES,
    STAGE_B_MODEL_TEST_COMMITMENT_SCHEMA,
    STAGE_B_MODEL_TEST_CONSUMED_SCHEMA,
    STAGE_B_MODEL_TEST_PRODUCER_ATTEMPT_SCHEMA,
    STAGE_B_MODEL_TEST_REPORT_SCHEMA,
    STAGE_B_PROTOCOL_NAME,
    _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS,
    _STAGE_B_ROLE_SOURCE_SEEDS,
    _canonical_control_json,
    _is_lower_hex,
    _pure_lexical_checks,
    _regular_bytes_no_symlink,
    _stage_b_path_contract,
    _validate_stage_b_model_test_commitment,
    _validate_stage_b_model_test_producer_attempt,
    require_workflow_authorized_or_safe_input,
    workflow_evidence_read_scope,
)
from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_CONTRACT_SHA256 as PARENT_PROTOCOL_CONTRACT_SHA256,
    PROTOCOL_FILE_SHA256 as PARENT_PROTOCOL_FILE_SHA256,
)
from safety_data.state_dependent_recovery_v5_stage_b import (
    EXECUTION_PROTOCOL_CONTRACT_SHA256,
    EXECUTION_PROTOCOL_FILE_SHA256,
    STAGE_A_DISPOSITION_COMMIT,
    STAGE_A_REPORT_SHA256,
    require_clean_stage_b_generator,
)


STAGE_B_SCIENTIFIC_ROLES = tuple(_STAGE_B_ROLE_SOURCE_SEEDS)
STAGE_B_EVIDENCE_KINDS = (
    "attempt_marker",
    "source_attempt_marker",
    "admission",
    "label",
    "label_privileged",
    "step_log",
    "source_step_log",
    "source_report",
    "collection_manifest",
    "completion_marker",
    "report",
)
_MODEL_TEST_REPORT_FIELDS = frozenset({
    "schema_version",
    "parent_protocol_name",
    "parent_protocol_contract_sha256",
    "parent_protocol_file_sha256",
    "execution_protocol_name",
    "execution_protocol_contract_sha256",
    "execution_protocol_file_sha256",
    "stage_a_report_sha256",
    "stage_a_disposition_commit",
    "generator_commit",
    "role",
    "source_seeds",
    "groups",
    "admission_replicas",
    "label_replicas",
    "evidence_artifacts",
    "producer_attempt_sha256",
    "status",
    "created_at_utc",
})


def _canonical_timestamp(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str) or not value:
        raise ProtectedEvidencePathError("created_at_utc must be nonempty text")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ProtectedEvidencePathError(
            "created_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProtectedEvidencePathError("created_at_utc must be UTC")
    return value


def _validate_frozen_inputs(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != (
            STAGE_B_FROZEN_INPUT_NAMES):
        raise ProtectedEvidencePathError(
            "frozen_inputs_sha256 must bind the exact Stage-B frozen inputs")
    result = {str(name): str(digest) for name, digest in value.items()}
    if any(not _is_lower_hex(digest, 64) for digest in result.values()):
        raise ProtectedEvidencePathError(
            "frozen_inputs_sha256 values must be lowercase SHA-256")
    return dict(sorted(result.items()))


def _validate_report_artifacts(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ProtectedEvidencePathError(
            "outcome-free report evidence_artifacts must be a list")
    expected_paths = sorted(_STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS)
    if len(value) != len(expected_paths):
        raise ProtectedEvidencePathError(
            "outcome-free report must commit every Model-Test evidence file")
    result: list[dict[str, str]] = []
    observed_paths: list[str] = []
    for record in value:
        if not isinstance(record, Mapping) or set(record) != {
                "kind", "path", "sha256"}:
            raise ProtectedEvidencePathError(
                "outcome-free report evidence record is malformed")
        relative = record.get("path")
        kind = record.get("kind")
        digest = record.get("sha256")
        if not isinstance(relative, str) or relative not in (
                _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS) or kind != (
                    _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS.get(relative)) or (
                        not _is_lower_hex(digest, 64)):
            raise ProtectedEvidencePathError(
                "outcome-free report evidence identity is invalid")
        observed_paths.append(relative)
        result.append({
            "kind": str(kind),
            "path": relative,
            "sha256": str(digest),
        })
    if observed_paths != expected_paths:
        raise ProtectedEvidencePathError(
            "outcome-free report evidence records must use canonical order")
    return result


def _validate_outcome_free_report(
    value: Mapping[str, object],
) -> dict[str, object]:
    if set(value) != _MODEL_TEST_REPORT_FIELDS:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test outcome-free report has extra or missing fields")
    if (
        value.get("schema_version") != STAGE_B_MODEL_TEST_REPORT_SCHEMA
        or value.get("parent_protocol_name") != STAGE_B_PROTOCOL_NAME
        or value.get("execution_protocol_name") != STAGE_B_EXECUTION_PROTOCOL_NAME
        or value.get("role") != "model_test"
        or value.get("status") != "complete_evidence_hashes_only"
    ):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test outcome-free report identity is invalid")
    for name in (
        "parent_protocol_contract_sha256", "parent_protocol_file_sha256",
        "execution_protocol_contract_sha256", "execution_protocol_file_sha256",
        "stage_a_report_sha256",
        "producer_attempt_sha256",
    ):
        if not _is_lower_hex(value.get(name), 64):
            raise ProtectedEvidencePathError(
                f"outcome-free report {name} is invalid")
    expected_hashes = {
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256),
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
    }
    for name, expected in expected_hashes.items():
        if value.get(name) != expected:
            raise ProtectedEvidencePathError(
                f"outcome-free report {name} differs from the frozen protocol")
    for name in (
        "stage_a_disposition_commit", "generator_commit",
    ):
        if not _is_lower_hex(value.get(name), 40):
            raise ProtectedEvidencePathError(
                f"outcome-free report {name} is invalid")
    if value.get("stage_a_disposition_commit") != STAGE_A_DISPOSITION_COMMIT:
        raise ProtectedEvidencePathError(
            "outcome-free report Stage-A disposition commit has drifted")
    expected_seeds = list(_STAGE_B_ROLE_SOURCE_SEEDS["model_test"])
    if value.get("source_seeds") != expected_seeds or value.get(
            "groups") != 768 or value.get("admission_replicas") != 32 or (
                value.get("label_replicas") != 64):
        raise ProtectedEvidencePathError(
            "outcome-free report Model-Test cohort is invalid")
    _canonical_timestamp(value.get("created_at_utc"))
    result = dict(value)
    result["evidence_artifacts"] = _validate_report_artifacts(
        value.get("evidence_artifacts"))
    return result


@contextmanager
def stage_b_evidence_read_scope(
    *,
    scientific_role: str,
    evidence_kind: str,
    path: str | Path,
) -> Iterator[Path]:
    """Validate and issue one exact scientific-role/kind/path capability.

    The barrier is evaluated before control reaches the caller, so code inside
    this context cannot perform a direct ``exists``/``stat``/``open``/hash on
    committed-but-unconsumed Model-Test evidence.
    """
    if scientific_role not in STAGE_B_SCIENTIFIC_ROLES:
        raise ProtectedEvidencePathError(
            f"unsupported Stage-B scientific role {scientific_role!r}")
    if evidence_kind not in STAGE_B_EVIDENCE_KINDS:
        raise ProtectedEvidencePathError(
            f"unsupported Stage-B evidence kind {evidence_kind!r}")
    capability = f"stage_b_{scientific_role}_{evidence_kind}"
    with workflow_evidence_read_scope(
        workflow=STAGE_B_PROTOCOL_NAME,
        role=capability,
        path=path,
    ):
        checked = require_workflow_authorized_or_safe_input(
            path, allowed_roles=(capability,))
        yield checked


def read_stage_b_model_test_outcome_free_report(
    path: str | Path,
) -> tuple[dict[str, object], str]:
    """Read the compiler's sole Model-Test input and return it plus file SHA."""
    lexical = _pure_lexical_checks(path)
    contract = _stage_b_path_contract(lexical)
    if contract != (
            "model_test", "report", "stage-b/model-test/report.json"):
        raise ProtectedEvidencePathError(
            "compiler input must be the exact Model-Test outcome-free report")
    with stage_b_evidence_read_scope(
        scientific_role="model_test",
        evidence_kind="report",
        path=lexical,
    ) as source:
        raw = _regular_bytes_no_symlink(
            source, "Stage-B Model-Test outcome-free report")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test outcome-free report is not valid JSON") from exc
    if not isinstance(decoded, Mapping) or raw != _canonical_control_json(decoded):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test outcome-free report must be canonical JSON")
    report = _validate_outcome_free_report(decoded)
    return report, hashlib.sha256(raw).hexdigest()


def _require_stage_b_control_path(
    path: str | Path,
    *,
    expected_name: str,
    stage_b_root: Path,
) -> Path:
    lexical = _pure_lexical_checks(path)
    expected = stage_b_root / expected_name
    if lexical != expected:
        raise ProtectedEvidencePathError(
            f"control output must be the exact {expected_name!r} path")
    return lexical


def _atomic_no_clobber_control_json(
    path: Path,
    value: Mapping[str, object],
) -> str:
    """Publish canonical JSON once; any interrupted write stays reserved."""
    raw = _canonical_control_json(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ProtectedEvidencePathError(
            f"control path already exists or cannot be created: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ProtectedEvidencePathError(
                    "new control marker is not a single-link regular file")
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # Deliberately do not unlink: a crash or short write consumes/reserves
        # the one-shot transition permanently.
        raise
    return hashlib.sha256(raw).hexdigest()


def create_stage_b_model_test_producer_attempt(
    *,
    attempt_path: str | Path,
    generator_commit: str,
    created_at_utc: str | None = None,
) -> dict[str, object]:
    """Freeze the exact pre-commit producer roster before its first read."""
    lexical = _pure_lexical_checks(attempt_path)
    if _stage_b_path_contract(lexical) != (
            "model_test", "attempt_marker",
            "stage-b/model-test/attempt-started.json"):
        raise ProtectedEvidencePathError(
            "producer attempt must use the exact Model-Test role marker path")
    stage_b_root = lexical.parent.parent
    for control in (
        lexical,
        lexical.parent / "report.json",
        stage_b_root / "model-test-committed.json",
        stage_b_root / "model-test-consumed.json",
    ):
        if os.path.lexists(os.fspath(control)):
            raise ProtectedEvidencePathError(
                "Model-Test producer attempt/report/commit control already exists")
    if not _is_lower_hex(generator_commit, 40):
        raise ProtectedEvidencePathError(
            "generator_commit must be a full lowercase Git commit")
    try:
        actual_generator = require_clean_stage_b_generator()
    except Exception as exc:
        raise ProtectedEvidencePathError(
            "Model-Test producer requires a clean generator commit") from exc
    if actual_generator != generator_commit:
        raise ProtectedEvidencePathError(
            "generator_commit differs from the current clean HEAD")
    marker: dict[str, object] = {
        "schema_version": STAGE_B_MODEL_TEST_PRODUCER_ATTEMPT_SCHEMA,
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
        "role": "model_test",
        "source_seeds": list(_STAGE_B_ROLE_SOURCE_SEEDS["model_test"]),
        "producer_read_paths": sorted(
            relative
            for relative in _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS
            if relative != "stage-b/model-test/attempt-started.json"
        ),
        "created_at_utc": _canonical_timestamp(created_at_utc),
        "status": "producer_attempt_started_before_evidence_read",
    }
    _validate_stage_b_model_test_producer_attempt(marker)
    attempt_sha256 = _atomic_no_clobber_control_json(lexical, marker)
    result = dict(marker)
    result["attempt_sha256"] = attempt_sha256
    return result


@contextmanager
def stage_b_model_test_producer_read_scope(
    *,
    attempt_path: str | Path,
    expected_attempt_sha256: str,
    evidence_paths: Sequence[str | Path],
) -> Iterator[list[Path]]:
    """Grant the blind producer exact pre-commit reads, then revoke them.

    The role report must still be absent.  Report publication or commitment
    publication makes every producer capability fail closed.  Source reports
    and identity/hash commitments are readable; this API computes no outcome
    statistic and grants no path outside the frozen roster.
    """
    if not _is_lower_hex(expected_attempt_sha256, 64):
        raise ProtectedEvidencePathError(
            "expected_attempt_sha256 must be lowercase SHA-256")
    attempt_file = _pure_lexical_checks(attempt_path)
    if _stage_b_path_contract(attempt_file) != (
            "model_test", "attempt_marker",
            "stage-b/model-test/attempt-started.json"):
        raise ProtectedEvidencePathError(
            "producer attempt path is not the exact Model-Test role marker")
    raw_attempt = _regular_bytes_no_symlink(
        attempt_file, "Stage-B Model-Test producer attempt")
    if hashlib.sha256(raw_attempt).hexdigest() != expected_attempt_sha256:
        raise ProtectedEvidencePathError(
            "Model-Test producer attempt hash differs from the expected hash")
    try:
        decoded = json.loads(raw_attempt.decode("utf-8"))
    except Exception as exc:
        raise ProtectedEvidencePathError(
            "Model-Test producer attempt is not valid JSON") from exc
    if not isinstance(decoded, Mapping) or raw_attempt != (
            _canonical_control_json(decoded)):
        raise ProtectedEvidencePathError(
            "Model-Test producer attempt must be canonical JSON")
    attempt = _validate_stage_b_model_test_producer_attempt(decoded)
    try:
        actual_generator = require_clean_stage_b_generator()
    except Exception as exc:
        raise ProtectedEvidencePathError(
            "Model-Test producer requires its clean generator commit") from exc
    if actual_generator != attempt["generator_commit"]:
        raise ProtectedEvidencePathError(
            "Model-Test producer generator commit changed after its attempt")
    stage_b_root = attempt_file.parent.parent
    for control in (
        attempt_file.parent / "report.json",
        stage_b_root / "model-test-committed.json",
        stage_b_root / "model-test-consumed.json",
    ):
        if os.path.lexists(os.fspath(control)):
            raise ProtectedEvidencePathError(
                "Model-Test producer scope was revoked by report/commit/consume")

    roster = attempt["producer_read_paths"]
    assert isinstance(roster, list)
    requested: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for supplied in evidence_paths:
        lexical = _pure_lexical_checks(supplied)
        contract = _stage_b_path_contract(lexical)
        if contract is None or contract[0] != "model_test" or contract[1] in (
                "attempt_marker", "report") or lexical.parent != (
                    attempt_file.parent) or contract[2] not in roster:
            raise ProtectedEvidencePathError(
                "producer requested evidence outside the frozen Model-Test roster")
        if lexical in seen:
            raise ProtectedEvidencePathError(
                "producer evidence paths must be unique")
        seen.add(lexical)
        requested.append((lexical, contract[1]))
    if not requested:
        raise ProtectedEvidencePathError(
            "producer requires at least one exact rostered evidence path")

    checked_paths: list[Path] = []
    with ExitStack() as stack:
        for lexical, kind in requested:
            capability = f"stage_b_model_test_producer_{kind}"
            stack.enter_context(workflow_evidence_read_scope(
                workflow=STAGE_B_PROTOCOL_NAME,
                role=capability,
                path=lexical,
            ))
            checked_paths.append(require_workflow_authorized_or_safe_input(
                lexical, allowed_roles=(capability,)))
        yield checked_paths


def compile_stage_b_model_test_commitment(
    *,
    report_path: str | Path,
    commitment_path: str | Path,
    expected_producer_attempt_sha256: str,
    created_at_utc: str | None = None,
) -> dict[str, object]:
    """Compile a no-clobber commitment while reading only the report.

    Evidence hashes are copied from the strict, outcome-free report.  This
    function never stats, opens, hashes, loads, resolves, or schema-loads any
    Model-Test evidence artifact.
    """
    report_lexical = _pure_lexical_checks(report_path)
    stage_b_root = report_lexical.parent.parent
    commitment_file = _require_stage_b_control_path(
        commitment_path,
        expected_name="model-test-committed.json",
        stage_b_root=stage_b_root,
    )
    consumed_file = stage_b_root / "model-test-consumed.json"
    if os.path.lexists(os.fspath(commitment_file)) or os.path.lexists(
            os.fspath(consumed_file)):
        raise ProtectedEvidencePathError(
            "Model-Test commitment or consumption was already reserved")
    report, report_sha256 = read_stage_b_model_test_outcome_free_report(
        report_lexical)
    if not _is_lower_hex(expected_producer_attempt_sha256, 64) or report.get(
            "producer_attempt_sha256") != expected_producer_attempt_sha256:
        raise ProtectedEvidencePathError(
            "outcome-free report producer-attempt hash differs from the "
            "expected frozen attempt")
    commitment: dict[str, object] = {
        "schema_version": STAGE_B_MODEL_TEST_COMMITMENT_SCHEMA,
        "parent_protocol_name": STAGE_B_PROTOCOL_NAME,
        "parent_protocol_contract_sha256": report[
            "parent_protocol_contract_sha256"],
        "parent_protocol_file_sha256": report[
            "parent_protocol_file_sha256"],
        "execution_protocol_name": STAGE_B_EXECUTION_PROTOCOL_NAME,
        "execution_protocol_contract_sha256": report[
            "execution_protocol_contract_sha256"],
        "execution_protocol_file_sha256": report[
            "execution_protocol_file_sha256"],
        "stage_a_report_sha256": report["stage_a_report_sha256"],
        "stage_a_disposition_commit": report["stage_a_disposition_commit"],
        "generator_commit": report["generator_commit"],
        "model_test_report_path": "stage-b/model-test/report.json",
        "model_test_report_sha256": report_sha256,
        "producer_attempt_sha256": expected_producer_attempt_sha256,
        "evidence_artifacts": report["evidence_artifacts"],
        "created_at_utc": _canonical_timestamp(created_at_utc),
    }
    _validate_stage_b_model_test_commitment(commitment)
    commitment_sha256 = _atomic_no_clobber_control_json(
        commitment_file, commitment)
    result = dict(commitment)
    result["commitment_sha256"] = commitment_sha256
    return result


@contextmanager
def consume_stage_b_model_test(
    *,
    commitment_path: str | Path,
    consumed_path: str | Path,
    expected_commitment_sha256: str,
    prerequisite_artifact_sha256: Mapping[str, str],
    evaluator_clean_commit: str,
    evidence_paths: Sequence[str | Path],
    created_at_utc: str | None = None,
) -> Iterator[dict[str, object]]:
    """Irreversibly reserve Model-Test, then grant only requested paths.

    The marker is published before any evidence leaf is probed.  It remains in
    place if scope setup, the first read, evaluation, or report publication
    crashes.
    """
    if not _is_lower_hex(expected_commitment_sha256, 64):
        raise ProtectedEvidencePathError(
            "expected_commitment_sha256 must be lowercase SHA-256")
    prerequisite = _validate_frozen_inputs(prerequisite_artifact_sha256)
    if not _is_lower_hex(evaluator_clean_commit, 40):
        raise ProtectedEvidencePathError(
            "evaluator_clean_commit must be a full lowercase Git commit")
    commitment_file = _pure_lexical_checks(commitment_path)
    if commitment_file.name != "model-test-committed.json":
        raise ProtectedEvidencePathError(
            "commitment_path must name model-test-committed.json")
    stage_b_root = commitment_file.parent
    consumed_file = _require_stage_b_control_path(
        consumed_path,
        expected_name="model-test-consumed.json",
        stage_b_root=stage_b_root,
    )
    if os.path.lexists(os.fspath(consumed_file)):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test has already been consumed or reserved")
    try:
        actual_evaluator_commit = require_clean_stage_b_generator()
    except Exception as exc:
        raise ProtectedEvidencePathError(
            "Model-Test consumption requires a clean evaluator commit") from exc
    if actual_evaluator_commit != evaluator_clean_commit:
        raise ProtectedEvidencePathError(
            "evaluator_clean_commit differs from the current clean HEAD")
    raw_commitment = _regular_bytes_no_symlink(
        commitment_file, "Stage-B Model-Test commitment")
    if hashlib.sha256(raw_commitment).hexdigest() != (
            expected_commitment_sha256):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment hash differs from the expected hash")
    try:
        decoded = json.loads(raw_commitment.decode("utf-8"))
    except Exception as exc:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment is not valid JSON") from exc
    if not isinstance(decoded, Mapping) or raw_commitment != (
            _canonical_control_json(decoded)):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment must be canonical JSON")
    commitment = _validate_stage_b_model_test_commitment(decoded)

    requested: list[tuple[Path, str, str]] = []
    seen: set[Path] = set()
    commitments = commitment["_artifact_sha256_by_path"]
    assert isinstance(commitments, Mapping)
    for supplied in evidence_paths:
        lexical = _pure_lexical_checks(supplied)
        contract = _stage_b_path_contract(lexical)
        if contract is None or contract[0] != "model_test" or contract[
                1] == "report" or lexical.parent.parent != stage_b_root or (
                    contract[2] not in commitments):
            raise ProtectedEvidencePathError(
                "requested consumption path is not committed Model-Test evidence")
        if lexical in seen:
            raise ProtectedEvidencePathError(
                "requested Model-Test evidence paths must be unique")
        seen.add(lexical)
        requested.append((lexical, contract[1], contract[2]))
    if not requested:
        raise ProtectedEvidencePathError(
            "at least one committed Model-Test evidence path is required")

    marker: dict[str, object] = {
        "schema_version": STAGE_B_MODEL_TEST_CONSUMED_SCHEMA,
        "parent_protocol_name": commitment["parent_protocol_name"],
        "parent_protocol_contract_sha256": commitment[
            "parent_protocol_contract_sha256"],
        "parent_protocol_file_sha256": commitment[
            "parent_protocol_file_sha256"],
        "execution_protocol_name": commitment["execution_protocol_name"],
        "execution_protocol_contract_sha256": commitment[
            "execution_protocol_contract_sha256"],
        "execution_protocol_file_sha256": commitment[
            "execution_protocol_file_sha256"],
        "stage_a_report_sha256": commitment["stage_a_report_sha256"],
        "stage_a_disposition_commit": commitment[
            "stage_a_disposition_commit"],
        "generator_commit": commitment["generator_commit"],
        "model_test_commitment_sha256": expected_commitment_sha256,
        "prerequisite_artifact_sha256": prerequisite,
        "evaluator_clean_commit": evaluator_clean_commit,
        "created_at_utc": _canonical_timestamp(created_at_utc),
        "status": "irreversibly_consumed_before_outcome_read",
    }
    marker_sha256 = _atomic_no_clobber_control_json(consumed_file, marker)

    with ExitStack() as stack:
        for lexical, kind, _ in requested:
            stack.enter_context(stage_b_evidence_read_scope(
                scientific_role="model_test",
                evidence_kind=kind,
                path=lexical,
            ))
        result = dict(marker)
        result["consumed_marker_sha256"] = marker_sha256
        result["evidence_paths"] = [relative for _, _, relative in requested]
        yield result


__all__ = [
    "STAGE_B_EVIDENCE_KINDS",
    "STAGE_B_SCIENTIFIC_ROLES",
    "compile_stage_b_model_test_commitment",
    "consume_stage_b_model_test",
    "create_stage_b_model_test_producer_attempt",
    "read_stage_b_model_test_outcome_free_report",
    "stage_b_evidence_read_scope",
    "stage_b_model_test_producer_read_scope",
]
