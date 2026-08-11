"""Fail-closed path checks for development-only Q_safe tooling."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterator, Mapping, Sequence


_PROTECTED_COMPONENT = re.compile(r"^(sealed|formal)", re.IGNORECASE)
_AMBIGUOUS_AUDIT_BASENAMES = frozenset({
    "audit-g384.npz",
    "audit-g384-privileged.npz",
})
_LOCKED_V3_AUDIT_BASENAMES = frozenset({
    *(f"source-{seed}.audit.npz"
      for seed in (7801, 7802, 7811, 7812, 7821, 7822)),
    *(f"source-{seed}.audit.privileged.npz"
      for seed in (7801, 7802, 7811, 7812, 7821, 7822)),
})
_LOCKED_V4_AUDIT_BASENAMES = frozenset({
    *(f"source-{seed}.audit.npz"
      for seed in (8401, 8402, 8411, 8412, 8421, 8422)),
    *(f"source-{seed}.audit.privileged.npz"
      for seed in (8401, 8402, 8411, 8412, 8421, 8422)),
})
_LOCKED_V5_AUDIT_BASENAMES = frozenset({
    *(f"source-{seed}.audit.npz"
      for seed in (8901, 8902, 8911, 8912, 8921, 8922)),
    *(f"source-{seed}.audit.privileged.npz"
      for seed in (8901, 8902, 8911, 8912, 8921, 8922)),
})
_LOCKED_AUDIT_BASENAMES = (
    _LOCKED_V3_AUDIT_BASENAMES
    | _LOCKED_V4_AUDIT_BASENAMES
    | _LOCKED_V5_AUDIT_BASENAMES)
_GENERIC_FORBIDDEN_AUDIT_COMPONENTS = (
    _LOCKED_AUDIT_BASENAMES | _AMBIGUOUS_AUDIT_BASENAMES)

_STAGE_B_ROLE_SOURCE_SEEDS = {
    "fit": (8501, 8502, 8511, 8512, 8521, 8522),
    "probability_calibration": (8601, 8611, 8621),
    "uncertainty_calibration": (8631, 8641, 8651),
    "selector_calibration": (8661, 8671, 8681),
    "model_test": (8701, 8702, 8711, 8712, 8721, 8722),
}
_STAGE_B_ROLE_DIRECTORIES = {
    role: role.replace("_", "-") for role in _STAGE_B_ROLE_SOURCE_SEEDS
}
_STAGE_B_ROLE_LABEL_REPLICAS = {
    role: (64 if role == "model_test" else 32)
    for role in _STAGE_B_ROLE_SOURCE_SEEDS
}
_STAGE_B_EVIDENCE_KINDS = (
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
_STAGE_B_CAPABILITY_CONTRACTS = {
    f"stage_b_{role}_{kind}": (role, kind)
    for role in _STAGE_B_ROLE_SOURCE_SEEDS
    for kind in _STAGE_B_EVIDENCE_KINDS
}
_STAGE_B_PRODUCER_CAPABILITY_CONTRACTS = {
    f"stage_b_model_test_producer_{kind}": ("model_test", kind)
    for kind in _STAGE_B_EVIDENCE_KINDS
    if kind not in ("attempt_marker", "report")
}
_ALL_STAGE_B_CAPABILITY_CONTRACTS = {
    **_STAGE_B_CAPABILITY_CONTRACTS,
    **_STAGE_B_PRODUCER_CAPABILITY_CONTRACTS,
}


def _stage_b_filenames(role: str, kind: str) -> frozenset[str]:
    seeds = _STAGE_B_ROLE_SOURCE_SEEDS[role]
    replicas = _STAGE_B_ROLE_LABEL_REPLICAS[role]
    if kind == "attempt_marker":
        return frozenset({"attempt-started.json"})
    if kind == "source_attempt_marker":
        return frozenset({
            *(f"source-{seed}.attempt-started.json" for seed in seeds),
        })
    if kind == "admission":
        return frozenset({
            "admission-r32.npz",
            *(f"source-{seed}.admission-r32.npz" for seed in seeds),
        })
    if kind == "label":
        return frozenset({
            f"labels-r{replicas}-deployable.npz",
            *(f"source-{seed}.labels-r{replicas}.npz" for seed in seeds),
        })
    if kind == "label_privileged":
        return frozenset({
            f"labels-r{replicas}-privileged.npz",
            *(f"source-{seed}.labels-r{replicas}.privileged.npz"
              for seed in seeds),
        })
    fixed = {
        "step_log": "steps.jsonl",
        "collection_manifest": "collection-manifest.json",
        "completion_marker": "completed.json",
        "report": "report.json",
    }
    if kind == "source_step_log":
        return frozenset({f"source-{seed}.steps.jsonl" for seed in seeds})
    if kind == "source_report":
        return frozenset({
            f"source-{seed}.collection-report.json" for seed in seeds})
    try:
        return frozenset({fixed[kind]})
    except KeyError as exc:  # pragma: no cover - constants are exhaustive.
        raise ValueError(f"unknown Stage-B evidence kind {kind!r}") from exc


_STAGE_B_RESERVED_EVIDENCE_BASENAMES = frozenset({
    filename
    for role in _STAGE_B_ROLE_SOURCE_SEEDS
    for kind in ("admission", "label", "label_privileged")
    for filename in _stage_b_filenames(role, kind)
})
_RESERVED_V3_OUTPUT_BASENAMES = frozenset({
    "cohort-lock.json",
    "admission-ledger-deployable.npz",
    "admission-ledger-privileged.npz",
    "admission-merge-report.json",
    "discovery-g384.npz",
    "discovery-g384-privileged.npz",
    "discovery-merge-report.json",
    "audit-g384.npz",
    "audit-g384-privileged.npz",
    "selection-lock.json",
    "audit-consumed.json",
    "closed-loop-recovery-triage-report.json",
    *(f"source-{seed}.{suffix}"
      for seed in (7801, 7802, 7811, 7812, 7821, 7822)
      for suffix in (
          "attempt-started.json",
          "admission.npz",
          "admission.privileged.npz",
          "discovery.npz",
          "discovery.privileged.npz",
          "audit.npz",
          "audit.privileged.npz",
          "collection-report.json",
      )),
})
_RESERVED_V4_OUTPUT_BASENAMES = frozenset({
    "state-dependent-recovery-stage-a-report.json",
    *(f"source-{seed}.{suffix}"
      for seed in (8401, 8402, 8411, 8412, 8421, 8422)
      for suffix in (
          "attempt-started.json",
          "admission.npz",
          "admission.privileged.npz",
          "discovery.npz",
          "discovery.privileged.npz",
          "audit.npz",
          "audit.privileged.npz",
          "collection-report.json",
      )),
})
_RESERVED_V5_OUTPUT_BASENAMES = frozenset({
    "cohort-lock.json",
    "admission-ledger-deployable.npz",
    "admission-ledger-privileged.npz",
    "admission-merge-report.json",
    "discovery-g384.npz",
    "discovery-g384-privileged.npz",
    "discovery-merge-report.json",
    "audit-g384.npz",
    "audit-g384-privileged.npz",
    "selection-lock.json",
    "audit-consumed.json",
    "state-dependent-recovery-stage-a-report.json",
    "actor-bank-manifest.json",
    "stage-b-split-disjointness-report.json",
    "normalization-fit-only-report.json",
    "probability-calibration-report.json",
    "uncertainty-calibration-report.json",
    "selector-search-report.json",
    "recovery-selector-bundle.json",
    "matched-random-placebo-bundle.json",
    "model-test-committed.json",
    "model-test-consumed.json",
    "state-dependent-recovery-stage-b-report.json",
    *_STAGE_B_RESERVED_EVIDENCE_BASENAMES,
    *(f"source-{seed}.{suffix}"
      for seed in (8901, 8902, 8911, 8912, 8921, 8922)
      for suffix in (
          "attempt-started.json",
          "admission.npz",
          "admission.privileged.npz",
          "discovery.npz",
          "discovery.privileged.npz",
          "audit.npz",
          "audit.privileged.npz",
          "collection-report.json",
      )),
})
_RESERVED_WORKFLOW_OUTPUT_BASENAMES = (
    _RESERVED_V3_OUTPUT_BASENAMES
    | _RESERVED_V4_OUTPUT_BASENAMES
    | _RESERVED_V5_OUTPUT_BASENAMES)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_V3 = "objective1_closed_loop_recovery_triage_v3"
_WORKFLOW_V4 = "objective1_state_dependent_recovery_qsafe_v4"
_WORKFLOW_V5 = "objective1_state_dependent_recovery_qsafe_v5"
_CANONICAL_WORKFLOW_ROOTS = {
    _WORKFLOW_V3: Path(os.path.abspath(
        _REPOSITORY_ROOT
        / "saved" / "qsafe_development" / "closed_loop_recovery_triage_v3")),
    _WORKFLOW_V4: Path(os.path.abspath(
        _REPOSITORY_ROOT
        / "saved" / "qsafe_development" / "state_dependent_recovery_v4")),
    _WORKFLOW_V5: Path(os.path.abspath(
        _REPOSITORY_ROOT
        / "saved" / "qsafe_development" / "state_dependent_recovery_v5")),
}
_AUTHORIZABLE_WORKFLOW_ROLES = frozenset({
    "admission",
    "admission_privileged",
    "discovery",
    "discovery_privileged",
    "audit",
    "audit_privileged",
    *_ALL_STAGE_B_CAPABILITY_CONTRACTS,
})
_WORKFLOW_RESERVED_BASENAMES = {
    _WORKFLOW_V3: _RESERVED_V3_OUTPUT_BASENAMES,
    _WORKFLOW_V5: _RESERVED_V5_OUTPUT_BASENAMES,
}


@dataclass(frozen=True, slots=True)
class _WorkflowReadGrant:
    """One process-local, exact-path workflow read capability."""

    workflow: str
    role: str
    path: Path


_ACTIVE_WORKFLOW_READ_GRANTS: ContextVar[tuple[_WorkflowReadGrant, ...]] = (
    ContextVar("qsafe_active_workflow_read_grants", default=()))


def _lexical_absolute(path: str | Path) -> Path:
    """Normalize a spelling without resolving or probing any component."""
    return Path(os.path.abspath(os.fspath(path)))


def _pure_lexical_checks(path: str | Path) -> Path:
    lexical = _lexical_absolute(path)
    for component in lexical.parts:
        if _PROTECTED_COMPONENT.match(component.casefold()):
            raise ProtectedEvidencePathError(
                f"protected evidence path component {component!r} in {lexical}")
    return lexical


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _workflow_for_path(path: Path) -> str | None:
    matches = [
        workflow for workflow, root in _CANONICAL_WORKFLOW_ROOTS.items()
        if _is_within(path, root)
    ]
    if len(matches) > 1:
        raise ProtectedEvidencePathError(
            "canonical workflow evidence roots must not overlap")
    return None if not matches else matches[0]


def _stage_b_path_contract(path: Path) -> tuple[str, str, str] | None:
    """Return ``(scientific_role, evidence_kind, relative_path)``.

    Stage-B grants are recognized only for a complete, exact
    ``stage-b/<role-directory>/<leaf>`` suffix.  In particular, a matching
    basename elsewhere and a role directory used as an ancestor are not
    capabilities.  The suffix rule also permits isolated temporary fixtures
    without weakening the canonical workflow-root denial.
    """
    indices = [
        index for index, component in enumerate(path.parts)
        if component == "stage-b"
    ]
    if len(indices) != 1:
        return None
    index = indices[0]
    tail = path.parts[index:]
    if len(tail) != 3:
        return None
    _, directory, filename = tail
    matching_roles = [
        role for role, expected_directory in _STAGE_B_ROLE_DIRECTORIES.items()
        if directory == expected_directory
    ]
    if len(matching_roles) != 1:
        return None
    role = matching_roles[0]
    matching_kinds = [
        kind for kind in _STAGE_B_EVIDENCE_KINDS
        if filename in _stage_b_filenames(role, kind)
    ]
    if len(matching_kinds) != 1:
        return None
    return role, matching_kinds[0], "/".join(tail)


def _validate_stage_b_capability_path(role: str, path: Path) -> None:
    scientific_role, kind = _ALL_STAGE_B_CAPABILITY_CONTRACTS[role]
    contract = _stage_b_path_contract(path)
    if contract is None or contract[:2] != (scientific_role, kind):
        raise ProtectedEvidencePathError(
            "Stage-B evidence-read grants require the exact scientific-role/"
            "artifact-kind/path tuple")


def assert_no_locked_audit_path_components(path: str | Path) -> Path:
    """Purely lexically reject every locked or ambiguous audit component.

    Generic tools call this on their complete argument set before any protocol
    read, resolution, existence check, or other filesystem operation.
    """
    lexical = _pure_lexical_checks(path)
    offenders = [
        component for component in lexical.parts
        if component in _GENERIC_FORBIDDEN_AUDIT_COMPONENTS
    ]
    if offenders:
        raise ProtectedEvidencePathError(
            "generic tools may not inspect a locked or ambiguous audit path "
            f"component: {offenders[0]!r}")
    return lexical


def assert_generic_evidence_path(path: str | Path) -> Path:
    """Purely lexically reject paths owned by a preregistered workflow."""
    lexical = assert_no_locked_audit_path_components(path)
    reserved_component = next((
        component for component in lexical.parts
        if component in _RESERVED_WORKFLOW_OUTPUT_BASENAMES
    ), None)
    if reserved_component is not None:
        raise ProtectedEvidencePathError(
            "generic tools may not inspect a reserved V3/V4/V5 workflow path "
            f"component: {reserved_component!r}")
    workflow = _workflow_for_path(lexical)
    if workflow is not None:
        raise ProtectedEvidencePathError(
            f"generic tools may not inspect the canonical {workflow} subtree")
    return lexical


def _grant_matches(
    lexical: Path,
    *,
    allowed_roles: frozenset[str],
) -> bool:
    return any(
        grant.path == lexical and grant.role in allowed_roles
        and _workflow_for_path(grant.path) in (None, grant.workflow)
        for grant in _ACTIVE_WORKFLOW_READ_GRANTS.get()
    )


def _model_test_producer_grant_matches(lexical: Path) -> bool:
    return any(
        grant.path == lexical
        and grant.role in _STAGE_B_PRODUCER_CAPABILITY_CONTRACTS
        and grant.workflow == _WORKFLOW_V5
        for grant in _ACTIVE_WORKFLOW_READ_GRANTS.get()
    )


@contextmanager
def workflow_evidence_read_scope(
    *,
    workflow: str,
    role: str,
    path: str | Path,
) -> Iterator[None]:
    """Authorize one exact V3/V5 role path for the dynamic scope only.

    This is an accidental-misuse capability, not a same-UID security boundary.
    It cannot authorize terminal V4 data or either ambiguous aggregate audit
    spelling, and it never bypasses the audit-consumption marker checks.
    """
    if workflow not in (_WORKFLOW_V3, _WORKFLOW_V5):
        raise ProtectedEvidencePathError(
            "only active V3/V5 workflows may issue evidence-read grants")
    if role not in _AUTHORIZABLE_WORKFLOW_ROLES:
        raise ProtectedEvidencePathError(
            f"unsupported workflow evidence role {role!r}")
    lexical = _pure_lexical_checks(path)
    if any(component in _AMBIGUOUS_AUDIT_BASENAMES
           for component in lexical.parts):
        raise ProtectedEvidencePathError(
            "ambiguous aggregate audit names are generic-deny-only")
    if any(component in _LOCKED_AUDIT_BASENAMES
           for component in lexical.parts[:-1]):
        raise ProtectedEvidencePathError(
            "a locked audit leaf may not be used as a path ancestor")
    canonical_root = _CANONICAL_WORKFLOW_ROOTS[workflow]
    basename_owned = lexical.name in _WORKFLOW_RESERVED_BASENAMES[workflow]
    stage_b_capability = role in _ALL_STAGE_B_CAPABILITY_CONTRACTS
    if stage_b_capability:
        if workflow != _WORKFLOW_V5:
            raise ProtectedEvidencePathError(
                "Stage-B capabilities belong only to the active V5 workflow")
        _validate_stage_b_capability_path(role, lexical)
    if not _is_within(lexical, canonical_root) and not basename_owned and not (
            stage_b_capability and _stage_b_path_contract(lexical) is not None):
        raise ProtectedEvidencePathError(
            "workflow evidence-read grants require the canonical root or an "
            "exact workflow-owned basename")
    if role in ("audit", "audit_privileged"):
        expected = (
            _LOCKED_V3_AUDIT_BASENAMES
            if workflow == _WORKFLOW_V3 else _LOCKED_V5_AUDIT_BASENAMES)
        if lexical.name not in expected:
            raise ProtectedEvidencePathError(
                "audit evidence-read grants require a workflow-specific "
                "physical audit leaf")
    grant = _WorkflowReadGrant(workflow=workflow, role=role, path=lexical)
    active = _ACTIVE_WORKFLOW_READ_GRANTS.get()
    token = _ACTIVE_WORKFLOW_READ_GRANTS.set((*active, grant))
    try:
        yield
    finally:
        _ACTIVE_WORKFLOW_READ_GRANTS.reset(token)


def _audit_workflow_contract(basename: str) -> tuple[str, str, str]:
    """Return the consumed-marker contract for a locked audit basename."""
    if basename in _LOCKED_V5_AUDIT_BASENAMES:
        return (
            "qsafe.state_dependent_recovery_v5.audit_consumed.v1",
            "objective1_state_dependent_recovery_qsafe_v5",
            "V5",
        )
    if basename in _LOCKED_V4_AUDIT_BASENAMES:
        return (
            "qsafe.state_dependent_recovery_v4.audit_consumed.v1",
            "objective1_state_dependent_recovery_qsafe_v4",
            "V4",
        )
    if basename in _LOCKED_V3_AUDIT_BASENAMES:
        return (
            "qsafe.closed_loop_recovery_triage.audit_consumed.v1",
            "objective1_closed_loop_recovery_triage_v3",
            "v3",
        )
    raise ValueError("audit workflow contract requested for an unlocked basename")


class ProtectedEvidencePathError(PermissionError):
    """Raised before development tooling can open a protected evidence path."""


STAGE_B_MODEL_TEST_COMMITMENT_SCHEMA = (
    "qsafe.state_dependent_recovery_v5.stage_b_model_test_commitment.v1")
STAGE_B_MODEL_TEST_CONSUMED_SCHEMA = (
    "qsafe.state_dependent_recovery_v5.stage_b_model_test_consumed.v1")
STAGE_B_MODEL_TEST_REPORT_SCHEMA = (
    "qsafe.state_dependent_recovery_v5.stage_b.model_test_outcome_free_report.v1")
STAGE_B_MODEL_TEST_PRODUCER_ATTEMPT_SCHEMA = (
    "qsafe.state_dependent_recovery_v5.stage_b_model_test_producer_attempt.v1")
STAGE_B_PROTOCOL_NAME = "objective1_state_dependent_recovery_qsafe_v5"
STAGE_B_EXECUTION_PROTOCOL_NAME = (
    "objective1_state_dependent_recovery_qsafe_v5_stage_b_execution")
STAGE_B_FROZEN_INPUT_NAMES = frozenset({
    "actor_bank_manifest",
    "split_disjointness_report",
    "normalization_report",
    "qsafe_artifact",
    "probability_calibration_report",
    "uncertainty_calibration_report",
    "selector_search_report",
    "recovery_selector_bundle",
    "matched_random_placebo_bundle",
})
_STAGE_B_MODEL_TEST_COMMITMENT_FIELDS = frozenset({
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
    "model_test_report_path",
    "model_test_report_sha256",
    "producer_attempt_sha256",
    "evidence_artifacts",
    "created_at_utc",
})
_STAGE_B_MODEL_TEST_CONSUMED_FIELDS = frozenset({
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
    "model_test_commitment_sha256",
    "prerequisite_artifact_sha256",
    "evaluator_clean_commit",
    "created_at_utc",
    "status",
})


def _is_lower_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        character in "0123456789abcdef" for character in value)


def _canonical_control_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _regular_bytes_no_symlink(path: Path, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ProtectedEvidencePathError(
                    f"{name} must be a regular file")
            if metadata.st_nlink != 1:
                raise ProtectedEvidencePathError(
                    f"{name} must have exactly one filesystem link")
            return stream.read()
    except ProtectedEvidencePathError:
        raise
    except OSError as exc:
        raise ProtectedEvidencePathError(
            f"{name} is missing, unreadable, or a symlink") from exc


def _regular_sha256_no_symlink(path: Path, name: str) -> str:
    """Hash a regular single-link file without following its final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ProtectedEvidencePathError(
                    f"{name} must be a regular file")
            if metadata.st_nlink != 1:
                raise ProtectedEvidencePathError(
                    f"{name} must have exactly one filesystem link")
            digest = hashlib.sha256()
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            return digest.hexdigest()
    except ProtectedEvidencePathError:
        raise
    except OSError as exc:
        raise ProtectedEvidencePathError(
            f"{name} is missing, unreadable, or a symlink") from exc


def _read_canonical_control_json(path: Path, name: str) -> dict[str, object]:
    raw = _regular_bytes_no_symlink(path, name)
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ProtectedEvidencePathError(f"{name} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ProtectedEvidencePathError(f"{name} must be a JSON object")
    result = dict(value)
    if raw != _canonical_control_json(result):
        raise ProtectedEvidencePathError(f"{name} must use canonical JSON")
    return result


def _stage_b_expected_model_test_artifacts() -> dict[str, str]:
    """Map every pre-report Model-Test evidence path to its exact kind."""
    result: dict[str, str] = {}
    directory = _STAGE_B_ROLE_DIRECTORIES["model_test"]
    for kind in _STAGE_B_EVIDENCE_KINDS:
        if kind == "report":
            continue
        for filename in _stage_b_filenames("model_test", kind):
            result[f"stage-b/{directory}/{filename}"] = kind
    return result


_STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS = (
    _stage_b_expected_model_test_artifacts())
_STAGE_B_MODEL_TEST_PRODUCER_ATTEMPT_FIELDS = frozenset({
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
    "producer_read_paths",
    "created_at_utc",
    "status",
})


def _validate_stage_b_model_test_producer_attempt(
    value: Mapping[str, object],
) -> dict[str, object]:
    if set(value) != _STAGE_B_MODEL_TEST_PRODUCER_ATTEMPT_FIELDS:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test producer attempt has extra or missing fields")
    if (
        value.get("schema_version") != STAGE_B_MODEL_TEST_PRODUCER_ATTEMPT_SCHEMA
        or value.get("parent_protocol_name") != STAGE_B_PROTOCOL_NAME
        or value.get("execution_protocol_name") != STAGE_B_EXECUTION_PROTOCOL_NAME
        or value.get("role") != "model_test"
        or value.get("status") !=
        "producer_attempt_started_before_evidence_read"
    ):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test producer attempt identity is invalid")
    try:
        from safety_data.state_dependent_recovery_v5 import (
            PROTOCOL_CONTRACT_SHA256 as parent_contract_sha256,
            PROTOCOL_FILE_SHA256 as parent_file_sha256,
        )
        from safety_data.state_dependent_recovery_v5_stage_b import (
            EXECUTION_PROTOCOL_CONTRACT_SHA256 as execution_contract_sha256,
            EXECUTION_PROTOCOL_FILE_SHA256 as execution_file_sha256,
            STAGE_A_DISPOSITION_COMMIT as stage_a_disposition_commit,
            STAGE_A_REPORT_SHA256 as stage_a_report_sha256,
        )
    except Exception as exc:  # pragma: no cover - import failure is fail-closed.
        raise ProtectedEvidencePathError(
            "could not bind Stage-B producer attempt to frozen protocols") from exc
    expected_identity = {
        "parent_protocol_contract_sha256": parent_contract_sha256,
        "parent_protocol_file_sha256": parent_file_sha256,
        "execution_protocol_contract_sha256": execution_contract_sha256,
        "execution_protocol_file_sha256": execution_file_sha256,
        "stage_a_report_sha256": stage_a_report_sha256,
        "stage_a_disposition_commit": stage_a_disposition_commit,
    }
    for name, expected in expected_identity.items():
        if value.get(name) != expected:
            raise ProtectedEvidencePathError(
                f"Stage-B Model-Test producer attempt {name} has drifted")
    if not _is_lower_hex(value.get("generator_commit"), 40):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test producer generator commit is invalid")
    if value.get("source_seeds") != list(
            _STAGE_B_ROLE_SOURCE_SEEDS["model_test"]):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test producer source roster is invalid")
    expected_paths = sorted(
        relative
        for relative in _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS
        if relative != "stage-b/model-test/attempt-started.json"
    )
    if value.get("producer_read_paths") != expected_paths:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test producer evidence roster is invalid")
    if not isinstance(value.get("created_at_utc"), str) or not value[
            "created_at_utc"]:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test producer timestamp is invalid")
    return dict(value)


def _validate_stage_b_model_test_commitment(
    value: Mapping[str, object],
) -> dict[str, object]:
    if set(value) != _STAGE_B_MODEL_TEST_COMMITMENT_FIELDS:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment has extra or missing fields")
    if value.get("schema_version") != STAGE_B_MODEL_TEST_COMMITMENT_SCHEMA or (
            value.get("parent_protocol_name") != STAGE_B_PROTOCOL_NAME) or (
                value.get("execution_protocol_name") !=
                STAGE_B_EXECUTION_PROTOCOL_NAME):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment identity is invalid")
    for name in (
        "parent_protocol_contract_sha256", "parent_protocol_file_sha256",
        "execution_protocol_contract_sha256", "execution_protocol_file_sha256",
        "stage_a_report_sha256", "model_test_report_sha256",
        "producer_attempt_sha256",
    ):
        if not _is_lower_hex(value.get(name), 64):
            raise ProtectedEvidencePathError(
                f"Stage-B Model-Test commitment {name} is invalid")
    for name in (
        "stage_a_disposition_commit", "generator_commit",
    ):
        if not _is_lower_hex(value.get(name), 40):
            raise ProtectedEvidencePathError(
                f"Stage-B Model-Test commitment {name} is invalid")
    if value.get("model_test_report_path") != "stage-b/model-test/report.json":
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment report path is invalid")
    if not isinstance(value.get("created_at_utc"), str) or not value[
            "created_at_utc"]:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment timestamp is invalid")

    raw_records = value.get("evidence_artifacts")
    if not isinstance(raw_records, list):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment evidence_artifacts must be a list")
    expected_paths = sorted(_STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS)
    if len(raw_records) != len(expected_paths):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment is not exhaustive")
    records: dict[str, str] = {}
    observed_order: list[str] = []
    for record in raw_records:
        if not isinstance(record, Mapping) or set(record) != {
                "kind", "path", "sha256"}:
            raise ProtectedEvidencePathError(
                "Stage-B Model-Test evidence commitment is malformed")
        relative = record.get("path")
        kind = record.get("kind")
        digest = record.get("sha256")
        if not isinstance(relative, str) or relative not in (
                _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS) or kind != (
                    _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS.get(relative)) or (
                        not _is_lower_hex(digest, 64)):
            raise ProtectedEvidencePathError(
                "Stage-B Model-Test evidence commitment identity is invalid")
        if relative in records:
            raise ProtectedEvidencePathError(
                "Stage-B Model-Test evidence commitment paths are duplicated")
        observed_order.append(relative)
        records[relative] = str(digest)
    if observed_order != expected_paths:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test evidence commitments must use canonical order")
    result = dict(value)
    result["_artifact_sha256_by_path"] = records
    return result


def _validate_stage_b_model_test_consumed(
    value: Mapping[str, object],
    *,
    commitment: Mapping[str, object],
    commitment_sha256: str,
) -> dict[str, object]:
    if set(value) != _STAGE_B_MODEL_TEST_CONSUMED_FIELDS:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test consumed marker has extra or missing fields")
    if (
        value.get("schema_version") != STAGE_B_MODEL_TEST_CONSUMED_SCHEMA
        or value.get("parent_protocol_name") != STAGE_B_PROTOCOL_NAME
        or value.get("execution_protocol_name") != STAGE_B_EXECUTION_PROTOCOL_NAME
        or value.get("status") != "irreversibly_consumed_before_outcome_read"
        or value.get("model_test_commitment_sha256") != commitment_sha256
    ):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test consumed marker identity is invalid")
    for name in (
        "parent_protocol_name", "parent_protocol_contract_sha256",
        "parent_protocol_file_sha256", "execution_protocol_name",
        "execution_protocol_contract_sha256", "execution_protocol_file_sha256",
        "stage_a_report_sha256", "stage_a_disposition_commit",
        "generator_commit",
    ):
        if value.get(name) != commitment.get(name):
            raise ProtectedEvidencePathError(
                "Stage-B Model-Test consumed marker is not bound to its "
                "commitment")
    prerequisite = value.get("prerequisite_artifact_sha256")
    if not isinstance(prerequisite, Mapping) or set(prerequisite) != (
            STAGE_B_FROZEN_INPUT_NAMES) or any(
                not _is_lower_hex(item, 64) for item in prerequisite.values()):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test consumed marker prerequisite hashes are invalid")
    if not _is_lower_hex(value.get("evaluator_clean_commit"), 40):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test evaluator clean commit is invalid")
    if not isinstance(value.get("created_at_utc"), str) or not value[
            "created_at_utc"]:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test consumed marker timestamp is invalid")
    return dict(value)


def _stage_b_model_test_control_paths(evidence_path: Path) -> tuple[Path, Path]:
    contract = _stage_b_path_contract(evidence_path)
    if contract is None or contract[0] != "model_test":
        raise ProtectedEvidencePathError(
            "path is not exact Stage-B Model-Test evidence")
    stage_b_root = evidence_path.parent.parent
    return (
        stage_b_root / "model-test-committed.json",
        stage_b_root / "model-test-consumed.json",
    )


def _require_stage_b_model_test_consumed_if_committed(path: Path) -> Path:
    """Enforce the Model-Test embargo before probing the evidence leaf.

    Collection reads remain possible before the commitment is published. As
    soon as either control marker exists, a missing/invalid pair fails closed.
    Once both controls validate, the evidence is hashed against the commitment
    before its path can reach a public schema loader.
    """
    contract = _stage_b_path_contract(path)
    if contract is None or contract[0] != "model_test" or contract[1] == (
            "report"):
        return path
    producer_grant = _model_test_producer_grant_matches(path)
    commitment_path, consumed_path = _stage_b_model_test_control_paths(path)
    commitment_exists = os.path.lexists(os.fspath(commitment_path))
    consumed_exists = os.path.lexists(os.fspath(consumed_path))
    if not commitment_exists and not consumed_exists:
        if not producer_grant:
            raise ProtectedEvidencePathError(
                "pre-commit Stage-B Model-Test evidence requires the dedicated "
                "producer capability")
        if os.path.lexists(os.fspath(path.parent / "report.json")):
            raise ProtectedEvidencePathError(
                "Stage-B Model-Test producer capability ended when its report "
                "was published")
        attempt_path = path.parent / "attempt-started.json"
        attempt = _read_canonical_control_json(
            attempt_path, "Stage-B Model-Test producer attempt")
        validated_attempt = _validate_stage_b_model_test_producer_attempt(attempt)
        roster = validated_attempt["producer_read_paths"]
        if not isinstance(roster, list) or contract[2] not in roster:
            raise ProtectedEvidencePathError(
                "Stage-B Model-Test producer attempt does not roster this path")
        return path
    if not commitment_exists or not consumed_exists:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test evidence is embargoed between commitment and "
            "irreversible consumption")
    if producer_grant:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test producer capability was revoked by commitment")

    raw_commitment = _regular_bytes_no_symlink(
        commitment_path, "Stage-B Model-Test commitment")
    try:
        decoded_commitment = json.loads(raw_commitment.decode("utf-8"))
    except Exception as exc:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment is not valid JSON") from exc
    if not isinstance(decoded_commitment, Mapping) or raw_commitment != (
            _canonical_control_json(decoded_commitment)):
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment must be canonical JSON")
    commitment = _validate_stage_b_model_test_commitment(decoded_commitment)
    commitment_sha256 = hashlib.sha256(raw_commitment).hexdigest()
    consumed = _read_canonical_control_json(
        consumed_path, "Stage-B Model-Test consumed marker")
    _validate_stage_b_model_test_consumed(
        consumed,
        commitment=commitment,
        commitment_sha256=commitment_sha256,
    )
    relative = contract[2]
    commitments = commitment["_artifact_sha256_by_path"]
    assert isinstance(commitments, Mapping)  # validated above
    expected_sha256 = commitments.get(relative)
    if expected_sha256 is None:
        raise ProtectedEvidencePathError(
            "Stage-B Model-Test commitment does not bind this evidence path")
    if _regular_sha256_no_symlink(
            path, "consumed Stage-B Model-Test evidence") != expected_sha256:
        raise ProtectedEvidencePathError(
            "consumed Stage-B Model-Test evidence differs from its commitment")
    return path


def _lexical_safe_path(path: str | Path) -> Path:
    lexical = _pure_lexical_checks(path)
    # A locked audit filename used as an ancestor is never a meaningful
    # evidence path.  Reject that spelling purely lexically before lstat could
    # reveal whether the audit artifact exists (or what type it is).
    for component in lexical.parts[:-1]:
        if component in _GENERIC_FORBIDDEN_AUDIT_COMPONENTS:
            raise ProtectedEvidencePathError(
                "locked v3/V4/V5 audit artifact may not be used as a path "
                "ancestor")
    current = Path(lexical.anchor)
    for component in lexical.parts[1:-1]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ProtectedEvidencePathError(
                f"evidence path has symlinked ancestor {current}")
    return lexical


def assert_safe_evidence_output(path: str | Path) -> Path:
    """Reject protected paths, symlink aliases, and locked v3 audit names."""
    lexical = _lexical_safe_path(path)
    if lexical.name in _RESERVED_WORKFLOW_OUTPUT_BASENAMES:
        raise ProtectedEvidencePathError(
            "generic tools may not write a reserved v3/V4/V5 workflow "
            "artifact name")
    try:
        metadata = lexical.lstat()
    except OSError:
        return lexical
    if stat.S_ISLNK(metadata.st_mode):
        raise ProtectedEvidencePathError(
            "evidence outputs may not follow a final symlink")
    return lexical


def require_v3_audit_consumed_or_safe_input(path: str | Path) -> Path:
    """Authorize a read without allowing a pre-marker v3 audit alias.

    Existing non-audit file inputs may not be symlinks or multiply linked;
    directories remain available for root checks.  A canonical audit basename
    additionally requires the exact
    sibling consumed marker to bind a regular selection lock that authorizes
    and commits that shard.  The audit path itself is not stat'ed until this
    control binding succeeds.
    """
    lexical = _pure_lexical_checks(path)
    if any(component in _AMBIGUOUS_AUDIT_BASENAMES
           for component in lexical.parts):
        raise ProtectedEvidencePathError(
            "ambiguous aggregate audit artifacts are generic-deny-only")
    lexical = _lexical_safe_path(lexical)
    # Reject every existing symlinked ancestor using lstat only.  This closes
    # aliases into protected or locked directories without ever resolving the
    # audit final component before its consumed marker.
    if lexical.name not in _LOCKED_AUDIT_BASENAMES:
        try:
            metadata = lexical.lstat()
        except OSError:
            return lexical
        if stat.S_ISLNK(metadata.st_mode):
            raise ProtectedEvidencePathError(
                "evidence loaders refuse symlink inputs")
        if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1:
            raise ProtectedEvidencePathError(
                "evidence loaders refuse hard-linked file inputs")
        return lexical

    marker = lexical.parent / "audit-consumed.json"
    try:
        marker_bytes = _regular_bytes_no_symlink(
            marker, "v3 audit-consumed marker")
        contract = json.loads(marker_bytes.decode("utf-8"))
    except ProtectedEvidencePathError:
        raise
    except Exception as exc:
        raise ProtectedEvidencePathError(
            "v3 audit-consumed marker is unreadable") from exc
    marker_fields = {
        "schema_version", "protocol_name", "protocol_contract_sha256",
        "protocol_file_sha256", "selection_lock_sha256", "audit_identifier",
        "created_at_utc", "status",
    }

    def hex64(value: object) -> bool:
        return isinstance(value, str) and len(value) == 64 and all(
            char in "0123456789abcdef" for char in value)

    (
        expected_marker_schema,
        expected_protocol_name,
        workflow_label,
    ) = _audit_workflow_contract(lexical.name)
    if not isinstance(contract, Mapping) or set(contract) != marker_fields or (
            contract.get("schema_version") != expected_marker_schema) or (
                contract.get("protocol_name") != expected_protocol_name) or (
                    contract.get("status") !=
                    "irreversibly_consumed_before_outcome_read") or any(
                        not hex64(contract.get(name)) for name in (
                            "protocol_contract_sha256", "protocol_file_sha256",
                            "selection_lock_sha256", "audit_identifier")) or not (
                                isinstance(contract.get("created_at_utc"), str)
                                and contract["created_at_utc"]):
        raise ProtectedEvidencePathError(
            f"{workflow_label} audit-consumed marker contract is invalid")

    selection_lock = lexical.parent / "selection-lock.json"
    try:
        lock_bytes = _regular_bytes_no_symlink(
            selection_lock, "v3 selection lock")
        lock = json.loads(lock_bytes.decode("utf-8"))
    except ProtectedEvidencePathError:
        raise
    except Exception as exc:
        raise ProtectedEvidencePathError(
            "v3 selection lock is unreadable") from exc
    if not isinstance(lock, Mapping) or hashlib.sha256(lock_bytes).hexdigest() != (
            contract["selection_lock_sha256"]) or any(
                lock.get(name) != contract.get(name) for name in (
                    "protocol_name", "protocol_contract_sha256",
                    "protocol_file_sha256", "audit_identifier")) or lock.get(
                        "audit_authorized") is not True:
        raise ProtectedEvidencePathError(
            f"{workflow_label} audit-consumed marker is not bound to the "
            "selection lock")
    if lexical.name.startswith("source-"):
        readiness = lock.get("collection_readiness_manifest")
        roles = readiness.get("role_commitments") if isinstance(
            readiness, Mapping) else None
        role = (
            "audit_privileged"
            if lexical.name.endswith(".audit.privileged.npz") else "audit")
        records = roles.get(role) if isinstance(roles, Mapping) else None
        matching = [
            record for record in records or []
            if isinstance(record, Mapping)
            and Path(os.path.abspath(str(record.get("path", "")))) == lexical
        ] if isinstance(records, list) else []
        if len(matching) != 1:
            raise ProtectedEvidencePathError(
                "v3 selection lock does not commit this audit shard")
        expected_file_sha256 = matching[0].get("file_sha256")
        if not hex64(expected_file_sha256):
            raise ProtectedEvidencePathError(
                "v3 selection lock has an invalid audit file commitment")
    try:
        audit_metadata = lexical.lstat()
    except OSError as exc:
        raise ProtectedEvidencePathError(
            "consumed v3 audit artifact is missing") from exc
    if stat.S_ISLNK(audit_metadata.st_mode) or not stat.S_ISREG(
            audit_metadata.st_mode):
        raise ProtectedEvidencePathError(
            "consumed v3 audit artifact must be a regular non-symlink file")
    if lexical.name.startswith("source-"):
        audit_bytes = _regular_bytes_no_symlink(
            lexical, "consumed v3 audit artifact")
        if hashlib.sha256(audit_bytes).hexdigest() != expected_file_sha256:
            raise ProtectedEvidencePathError(
                "consumed v3 audit artifact differs from the selection lock")
    return lexical


def require_workflow_authorized_or_safe_input(
    path: str | Path,
    *,
    allowed_roles: Sequence[str],
) -> Path:
    """Guard a public schema load, requiring a scoped grant when reserved.

    The reserved decision is purely lexical and precedes every filesystem
    operation.  A matching grant only authorizes the workflow/role/path tuple;
    the ordinary symlink, hard-link, and consumed-audit checks still run.
    """
    lexical = _pure_lexical_checks(path)
    if any(component in _AMBIGUOUS_AUDIT_BASENAMES
           for component in lexical.parts):
        raise ProtectedEvidencePathError(
            "ambiguous aggregate audit artifacts are generic-deny-only")
    if any(component in _LOCKED_AUDIT_BASENAMES
           for component in lexical.parts[:-1]):
        raise ProtectedEvidencePathError(
            "locked v3/V4/V5 audit artifact may not be used as a path ancestor")
    roles = frozenset(allowed_roles)
    if not roles or not roles.issubset(_AUTHORIZABLE_WORKFLOW_ROLES):
        raise ProtectedEvidencePathError(
            "public schema loader supplied an invalid workflow role contract")
    reserved_component = any(
        component in _RESERVED_WORKFLOW_OUTPUT_BASENAMES
        for component in lexical.parts)
    workflow_owned = _workflow_for_path(lexical) is not None
    stage_b_owned = _stage_b_path_contract(lexical) is not None
    if (reserved_component or workflow_owned or stage_b_owned) and not _grant_matches(
            lexical, allowed_roles=roles):
        raise ProtectedEvidencePathError(
            "public schema loaders require a scoped workflow/role/exact-path "
            "authorization for reserved evidence")
    # This check is deliberately before `_lexical_safe_path`, lstat of the
    # final component, path resolution, hashing, or a schema loader's np.load.
    # It probes only the two outcome-free control markers until consumption.
    _require_stage_b_model_test_consumed_if_committed(lexical)
    # Never let authorization bypass the irreversible audit marker or the
    # ordinary no-symlink/no-hard-link checks.
    return require_v3_audit_consumed_or_safe_input(lexical)


def _checked_paths(path: str | Path) -> tuple[Path, ...]:
    supplied = Path(path).expanduser()
    # Resolve existing symlinks as well as the lexical spelling.  For a new
    # output, strict=False still resolves every existing parent component.
    return supplied.absolute(), supplied.resolve(strict=False)


def assert_development_path(path: str | Path) -> Path:
    """Return a resolved path or reject protected Formal Test components.

    There is intentionally no ``allow_protected`` escape hatch in this
    package.  Confirmation data must be evaluated by a separately reviewed
    one-shot harness, not by development/audit commands.
    """
    for candidate in _checked_paths(path):
        for component in candidate.parts:
            if _PROTECTED_COMPONENT.match(component.casefold()):
                raise ProtectedEvidencePathError(
                    f"protected evidence path component {component!r} in {candidate}")
    return Path(path).expanduser().resolve(strict=False)


__all__ = [
    "ProtectedEvidencePathError",
    "assert_generic_evidence_path",
    "assert_no_locked_audit_path_components",
    "assert_development_path",
    "assert_safe_evidence_output",
    "require_v3_audit_consumed_or_safe_input",
    "require_workflow_authorized_or_safe_input",
    "workflow_evidence_read_scope",
]
