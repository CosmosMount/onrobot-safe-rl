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
    if not _is_within(lexical, canonical_root) and not basename_owned:
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
    if (reserved_component or workflow_owned) and not _grant_matches(
            lexical, allowed_roles=roles):
        raise ProtectedEvidencePathError(
            "public schema loaders require a scoped workflow/role/exact-path "
            "authorization for reserved evidence")
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
