"""Fail-closed path checks for development-only Q_safe tooling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Mapping


_PROTECTED_COMPONENT = re.compile(r"^(sealed|formal)", re.IGNORECASE)
_LOCKED_V3_AUDIT_BASENAMES = frozenset({
    *(f"source-{seed}.audit.npz"
      for seed in (7801, 7802, 7811, 7812, 7821, 7822)),
    *(f"source-{seed}.audit.privileged.npz"
      for seed in (7801, 7802, 7811, 7812, 7821, 7822)),
    "audit-g384.npz",
    "audit-g384-privileged.npz",
})
_LOCKED_V4_AUDIT_BASENAMES = frozenset({
    *(f"source-{seed}.audit.npz"
      for seed in (8401, 8402, 8411, 8412, 8421, 8422)),
    *(f"source-{seed}.audit.privileged.npz"
      for seed in (8401, 8402, 8411, 8412, 8421, 8422)),
})
_LOCKED_AUDIT_BASENAMES = (
    _LOCKED_V3_AUDIT_BASENAMES | _LOCKED_V4_AUDIT_BASENAMES)
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
_RESERVED_WORKFLOW_OUTPUT_BASENAMES = (
    _RESERVED_V3_OUTPUT_BASENAMES | _RESERVED_V4_OUTPUT_BASENAMES)


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
    lexical = Path(os.path.abspath(os.fspath(path)))
    for component in lexical.parts:
        if _PROTECTED_COMPONENT.match(component.casefold()):
            raise ProtectedEvidencePathError(
                f"protected evidence path component {component!r} in {lexical}")
    # A locked audit filename used as an ancestor is never a meaningful
    # evidence path.  Reject that spelling purely lexically before lstat could
    # reveal whether the audit artifact exists (or what type it is).
    for component in lexical.parts[:-1]:
        if component in _LOCKED_AUDIT_BASENAMES:
            raise ProtectedEvidencePathError(
                "locked v3 audit artifact may not be used as a path ancestor")
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
            "generic tools may not write a reserved v3 workflow artifact name")
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
    lexical = _lexical_safe_path(path)
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

    is_v4 = lexical.name in _LOCKED_V4_AUDIT_BASENAMES
    expected_marker_schema = (
        "qsafe.state_dependent_recovery_v4.audit_consumed.v1"
        if is_v4 else
        "qsafe.closed_loop_recovery_triage.audit_consumed.v1")
    expected_protocol_name = (
        "objective1_state_dependent_recovery_qsafe_v4"
        if is_v4 else
        "objective1_closed_loop_recovery_triage_v3")
    workflow_label = "V4" if is_v4 else "v3"
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
    "assert_development_path",
    "assert_safe_evidence_output",
    "require_v3_audit_consumed_or_safe_input",
]
