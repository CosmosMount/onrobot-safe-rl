"""No-clobber five-role collection workflow for V5 Stage B.

This module owns the operational envelope around the single-label collector:
role/source attempt markers are durable before outcomes, source shards are
published report-last, and role merges are published report-last without
computing or serializing any candidate-outcome statistic.  Model-Test uses the
separate blind-producer capability until its outcome-free report is published.
"""

from __future__ import annotations

import copy
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from safety_data.closed_loop_recovery_collector import (
    AdmissionLedger,
    load_admission_ledger_blind,
    merge_admission_ledgers,
    merge_admission_ledgers_blind,
    save_admission_ledger_blind,
)
from safety_data.merge import (
    load_grouped_shard_blind,
    load_privileged_shard_blind,
    merge_grouped_shards,
    merge_grouped_shards_blind,
    merge_privileged_shards,
    merge_privileged_shards_blind,
    save_grouped_shard_blind,
    save_privileged_shard_blind,
)
from safety_data.paths import (
    STAGE_B_EXECUTION_PROTOCOL_NAME,
    STAGE_B_MODEL_TEST_REPORT_SCHEMA,
    STAGE_B_PROTOCOL_NAME,
    _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS,
    _stage_b_path_contract,
    assert_development_path,
)
from safety_data.schema import (
    PRIVILEGED_SCHEMA_VERSION,
    REQUIRED_ARRAYS,
    REQUIRED_MANIFEST_KEYS,
    SCHEMA_VERSION,
    GroupedBranchDataset,
    PrivilegedBranchView,
)
from safety_data.stage_b_paths import (
    create_stage_b_model_test_producer_attempt,
    stage_b_evidence_read_scope,
    stage_b_model_test_producer_read_scope,
)
from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_CONTRACT_SHA256 as PARENT_PROTOCOL_CONTRACT_SHA256,
    PROTOCOL_FILE_SHA256 as PARENT_PROTOCOL_FILE_SHA256,
)
from safety_data.state_dependent_recovery_v5_stage_b import (
    ADMISSION_REPLICAS,
    CANDIDATES,
    CHECKPOINT_STEPS,
    EXECUTION_PROTOCOL_CONTRACT_SHA256,
    EXECUTION_PROTOCOL_FILE_SHA256,
    GROUPS_PER_SOURCE,
    HORIZON_POLICY_STEPS,
    LABEL_REPLICAS,
    REDUCED7_AMENDMENT_CONTRACT_SHA256,
    REDUCED7_AMENDMENT_FILE_SHA256,
    ROLE_ORDER,
    ROLE_ACTOR_SEEDS,
    ROLE_SOURCE_SEEDS,
    StageBAdmissionIdentityView,
    STAGE_A_DISPOSITION_COMMIT,
    STAGE_A_REPORT_SHA256,
    StageBSplitIdentityView,
    StageBExecutionError,
    TRAJECTORY_FINGERPRINT_ARRAY,
    TRAJECTORY_FINGERPRINT_CONTRACT,
    _actor_identities_by_role,
    assignment_for,
    canonical_sha256,
    compile_partition_rng_disjointness,
    compile_split_disjointness,
    make_admission_identity_view,
    make_split_identity_view,
    load_admission_identity_view,
    load_split_identity_view,
)
from safety_data.state_dependent_recovery_v5_stage_b_collector import (
    StageBRoleCollectionResult,
)
from train.state_dependent_recovery_v5_stage_b_actor_bank import (
    REDUCED7_ACTOR_BANK_SCHEMA_VERSION,
    actor_identity_for,
)


ROLE_ATTEMPT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_role_attempt.v1")
SOURCE_ATTEMPT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_source_attempt.v1")
SOURCE_REPORT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_source_collection_report.v1")
COLLECTION_MANIFEST_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_collection_manifest.v1")
COMPLETION_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_role_completion.v1")
ROLE_REPORT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_role_outcome_free_report.v1")
SPLIT_REPORT_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_split_disjointness_bound.v3")

_SPLIT_IDENTITY_ARRAY_NAMES = (
    "group_id",
    "source_seed",
    "policy_training_seed",
    "policy_source",
    "state_hash",
    TRAJECTORY_FINGERPRINT_ARRAY,
    "crn_id",
    "rollout_seed",
    "perturbation_seed",
    "candidate_seed",
)
_CANDIDATE_OUTCOME_ARRAY_NAMES = frozenset({
    "fall", "first_failure_step", "max_tilt_rad", "min_height_m",
})

_HEX40 = frozenset("0123456789abcdef")


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and set(value).issubset(_HEX40)
    )


def _utc_timestamp(value: str | None = None) -> str:
    result = datetime.now(timezone.utc).isoformat() if value is None else value
    if not isinstance(result, str) or not result:
        raise StageBExecutionError("created_at_utc must be nonempty text")
    try:
        parsed = datetime.fromisoformat(result)
    except ValueError as exc:
        raise StageBExecutionError(
            "created_at_utc must be ISO-8601") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise StageBExecutionError("created_at_utc must use UTC")
    return result


def _canonical_json(value: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StageBExecutionError(
            "Stage-B control value is not canonical JSON") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _regular_canonical_json(path: Path, name: str) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageBExecutionError(
                    f"{name} must be a single-link regular file")
            raw = stream.read()
    except OSError as exc:
        raise StageBExecutionError(f"cannot read {name}") from exc
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StageBExecutionError(f"{name} is not valid JSON") from exc
    if not isinstance(decoded, dict) or raw != _canonical_json(decoded):
        raise StageBExecutionError(f"{name} must be canonical JSON")
    return decoded, _bytes_sha256(raw)


def _atomic_no_clobber_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise StageBExecutionError(
            f"refusing to clobber Stage-B control: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StageBExecutionError(
                    "new Stage-B control is not a regular file")
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # A partial attempt marker remains deliberately consumed.
        raise
    return _bytes_sha256(payload)


def _atomic_no_clobber_json(
    path: Path, value: Mapping[str, object]
) -> str:
    return _atomic_no_clobber_bytes(path, _canonical_json(value))


def _staging_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.pending-",
        suffix=destination.suffix,
    )
    os.close(descriptor)
    staging = Path(name)
    staging.unlink()
    return staging


def _publish_staged(
    staged: Sequence[tuple[Path, Path]],
    *,
    terminal_last: bool = False,
) -> None:
    """Hard-link a bundle in order and roll back links from this invocation."""
    published: list[tuple[Path, Path]] = []
    terminal_published = False
    try:
        for index, (source, destination) in enumerate(staged):
            os.link(source, destination)
            published.append((source, destination))
            if terminal_last and index == len(staged) - 1:
                terminal_published = True
        directory_fd = os.open(
            staged[0][1].parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # A published terminal role report revokes Model-Test producer access
        # immediately.  Never probe or roll back any role evidence after that
        # boundary, even when the following directory fsync reports failure.
        if not terminal_published:
            for source, destination in reversed(published):
                try:
                    if os.path.samefile(source, destination):
                        destination.unlink()
                except FileNotFoundError:
                    pass
        raise
    finally:
        for source, _ in staged:
            source.unlink(missing_ok=True)


def _role_directory(role: str) -> str:
    if role not in ROLE_ORDER:
        raise StageBExecutionError(f"unknown Stage-B role {role!r}")
    return role.replace("_", "-")


def _checked_root(stage_b_root: str | Path) -> Path:
    root = assert_development_path(stage_b_root)
    if root.name != "stage-b":
        raise StageBExecutionError("Stage-B artifact root must end in stage-b")
    return root


def stage_b_role_paths(
    stage_b_root: str | Path,
    role: str,
) -> dict[str, Any]:
    """Return every exact frozen path for one scientific role."""
    root = _checked_root(stage_b_root)
    directory = root / _role_directory(role)
    replicas = LABEL_REPLICAS[role]
    source: dict[int, dict[str, Path]] = {}
    for seed in ROLE_SOURCE_SEEDS[role]:
        source[seed] = {
            "attempt_marker": directory / f"source-{seed}.attempt-started.json",
            "admission": directory / f"source-{seed}.admission-r32.npz",
            "label": directory / f"source-{seed}.labels-r{replicas}.npz",
            "label_privileged": (
                directory / f"source-{seed}.labels-r{replicas}.privileged.npz"
            ),
            "source_step_log": directory / f"source-{seed}.steps.jsonl",
            "source_report": (
                directory / f"source-{seed}.collection-report.json"
            ),
        }
    return {
        "directory": directory,
        "attempt_marker": directory / "attempt-started.json",
        "sources": source,
        "admission": directory / "admission-r32.npz",
        "label": directory / f"labels-r{replicas}-deployable.npz",
        "label_privileged": (
            directory / f"labels-r{replicas}-privileged.npz"
        ),
        "step_log": directory / "steps.jsonl",
        "collection_manifest": directory / "collection-manifest.json",
        "completion_marker": directory / "completed.json",
        "report": directory / "report.json",
    }


def _all_role_evidence_paths(paths: Mapping[str, Any]) -> list[Path]:
    result = [Path(paths["attempt_marker"])]
    for source in paths["sources"].values():
        result.extend(Path(value) for value in source.values())
    result.extend(Path(paths[name]) for name in (
        "admission",
        "label",
        "label_privileged",
        "step_log",
        "collection_manifest",
        "completion_marker",
    ))
    return result


def _relative_stage_b(path: Path) -> str:
    indices = [
        index for index, component in enumerate(path.parts)
        if component == "stage-b"
    ]
    if len(indices) != 1:
        raise StageBExecutionError("evidence path has ambiguous stage-b root")
    return "/".join(path.parts[indices[0]:])


def _artifact_record(path: Path, sha256: str) -> dict[str, str]:
    contract = _stage_b_path_contract(path)
    if contract is None:
        raise StageBExecutionError(f"path is not frozen Stage-B evidence: {path}")
    return {
        "kind": contract[1],
        "path": contract[2],
        "sha256": sha256,
    }


def _validate_actor_bank(
    actor_bank_manifest: Mapping[str, Any],
    *,
    actor_bank_manifest_file_sha256: str,
    generator_commit: str,
) -> dict[str, Any]:
    if actor_bank_manifest.get("schema_version") != (
        REDUCED7_ACTOR_BANK_SCHEMA_VERSION
    ):
        raise StageBExecutionError("actor-bank manifest schema has drifted")
    if actor_bank_manifest.get("stage_b_generator_commit") != generator_commit:
        raise StageBExecutionError(
            "actor-bank amendment generator differs from Stage-B generator")
    actor_source_commit = actor_bank_manifest.get("generator_commit")
    contract_sha256 = actor_bank_manifest.get("actor_bank_contract_sha256")
    if not _is_lower_hex(contract_sha256, 64) or not _is_lower_hex(
        actor_bank_manifest_file_sha256, 64
    ):
        raise StageBExecutionError("actor-bank hash binding is malformed")
    if not _is_lower_hex(generator_commit, 40):
        raise StageBExecutionError("generator_commit must be lowercase Git SHA")
    required_fields = {
        "schema_version", "protocol_binding", "execution_supplement_binding",
        "roster_amendment_binding",
        "stage_a_report_binding", "training_config_binding",
        "actor_bank_attempt_binding", "generator_commit",
        "stage_b_generator_commit", "upstream_actor_training_seeds",
        "actor_training_seeds", "checkpoint_steps", "checkpoint_semantics",
        "identity_count", "expected_identity_count", "actor_inclusion_rule",
        "return_or_fall_filtering", "checkpoint_substitution",
        "nearby_checkpoint_substitution", "policy_only", "identities",
        "actor_bank_contract_sha256",
    }
    if set(actor_bank_manifest) != required_fields:
        raise StageBExecutionError(
            "actor-bank manifest has extra or missing fields")
    hash_basis = dict(actor_bank_manifest)
    observed_contract_sha256 = hash_basis.pop("actor_bank_contract_sha256")
    if canonical_sha256(hash_basis) != observed_contract_sha256:
        raise StageBExecutionError("actor-bank self-hash is invalid")
    identities_by_role = _actor_identities_by_role(actor_bank_manifest)
    identities = actor_bank_manifest.get("identities")
    expected_identity_count = sum(
        len(values) for values in ROLE_ACTOR_SEEDS.values()
    ) * len(CHECKPOINT_STEPS)
    if not isinstance(identities, list) or sum(
        len(values) for values in identities_by_role.values()
    ) != expected_identity_count:
        raise StageBExecutionError(
            "actor-bank must bind the complete amended identities")
    expected_identity_order = [
        (role, actor_seed, checkpoint_step)
        for role in ROLE_ORDER
        for actor_seed in ROLE_ACTOR_SEEDS[role]
        for checkpoint_step in CHECKPOINT_STEPS
    ]
    observed_identity_order = [
        (
            identity.get("role"),
            identity.get("actor_training_seed"),
            identity.get("checkpoint_step"),
        )
        for identity in identities
    ]
    if observed_identity_order != expected_identity_order:
        raise StageBExecutionError(
            "actor-bank identities are not in the exact frozen order")
    for index, identity in enumerate(identities):
        if identity.get("generator_commit") != actor_source_commit:
            raise StageBExecutionError(
                f"actor-bank identity {index} changes actor source commit")
        for name in (
            "actor_checkpoint_sha256", "actor_sha256",
            "actor_state_dict_sha256", "policy_fingerprint_sha256",
            "checkpoint_fingerprint_sha256", "policy_config_sha256",
            "run_contract_sha256", "snapshot_manifest_file_sha256",
        ):
            if not _is_lower_hex(identity.get(name), 64):
                raise StageBExecutionError(
                    f"actor-bank identity {index} {name} is malformed")
        if identity.get("actor_checkpoint_sha256") != identity.get(
            "actor_sha256"):
            raise StageBExecutionError(
                f"actor-bank identity {index} actor hashes differ")
    # actor_identity_for performs exact tuple lookup; touch every frozen source
    # assignment here so no role attempt can precede an incomplete roster.
    for role in ROLE_ORDER:
        for source_seed in ROLE_SOURCE_SEEDS[role]:
            assignment = assignment_for(role, source_seed)
            actor_identity_for(
                actor_bank_manifest,
                role=role,
                actor_seed=assignment.actor_training_seed,
                checkpoint_step=assignment.checkpoint_step,
            )
    return json.loads(json.dumps(actor_bank_manifest))


def _frozen_identity(generator_commit: str) -> dict[str, object]:
    return {
        "parent_protocol_name": STAGE_B_PROTOCOL_NAME,
        "parent_protocol_contract_sha256": PARENT_PROTOCOL_CONTRACT_SHA256,
        "parent_protocol_file_sha256": PARENT_PROTOCOL_FILE_SHA256,
        "execution_protocol_name": STAGE_B_EXECUTION_PROTOCOL_NAME,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256
        ),
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "roster_amendment_contract_sha256": (
            REDUCED7_AMENDMENT_CONTRACT_SHA256
        ),
        "roster_amendment_file_sha256": REDUCED7_AMENDMENT_FILE_SHA256,
        "stage_a_report_sha256": STAGE_A_REPORT_SHA256,
        "stage_a_disposition_commit": STAGE_A_DISPOSITION_COMMIT,
        "generator_commit": generator_commit,
    }


def _path_absent(path: Path, name: str) -> None:
    if os.path.lexists(os.fspath(path)):
        raise StageBExecutionError(f"{name} already exists: {path}")


def prepare_stage_b_role(
    *,
    stage_b_root: str | Path,
    role: str,
    generator_commit: str,
    actor_bank_manifest: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Publish the irreversible role attempt before its first outcome."""
    actor_bank = _validate_actor_bank(
        actor_bank_manifest,
        actor_bank_manifest_file_sha256=actor_bank_manifest_file_sha256,
        generator_commit=generator_commit,
    )
    paths = stage_b_role_paths(stage_b_root, role)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    timestamp = _utc_timestamp(created_at_utc)
    if role == "model_test":
        # Model-Test outcome paths are never probed without the dedicated
        # producer capability.  Publish its attempt first, then prove every
        # later rostered leaf is absent; a stale leaf consumes the attempt and
        # fails closed.
        marker = create_stage_b_model_test_producer_attempt(
            attempt_path=paths["attempt_marker"],
            generator_commit=generator_commit,
            created_at_utc=timestamp,
        )
        evidence = _all_role_evidence_paths(paths)[1:]
        with stage_b_model_test_producer_read_scope(
            attempt_path=paths["attempt_marker"],
            expected_attempt_sha256=str(marker["attempt_sha256"]),
            evidence_paths=evidence,
        ):
            for path in evidence:
                _path_absent(path, "Model-Test role evidence")
        return dict(marker)

    with ExitStack() as stack:
        for path in (*_all_role_evidence_paths(paths), Path(paths["report"])):
            contract = _stage_b_path_contract(path)
            if contract is None:
                raise StageBExecutionError("role path roster has drifted")
            stack.enter_context(stage_b_evidence_read_scope(
                scientific_role=role,
                evidence_kind=contract[1],
                path=path,
            ))
            _path_absent(path, "role evidence")

    marker: dict[str, object] = {
        "schema_version": ROLE_ATTEMPT_SCHEMA_VERSION,
        **_frozen_identity(generator_commit),
        "role": role,
        "source_seeds": list(ROLE_SOURCE_SEEDS[role]),
        "groups": len(ROLE_SOURCE_SEEDS[role]) * GROUPS_PER_SOURCE[role],
        "groups_per_source": GROUPS_PER_SOURCE[role],
        "admission_replicas": ADMISSION_REPLICAS,
        "label_replicas": LABEL_REPLICAS[role],
        "actor_bank_manifest_file_sha256": actor_bank_manifest_file_sha256,
        "actor_bank_contract_sha256": actor_bank[
            "actor_bank_contract_sha256"
        ],
        "created_at_utc": timestamp,
        "status": "role_attempt_started_before_first_candidate_outcome",
    }
    attempt_sha256 = _atomic_no_clobber_json(paths["attempt_marker"], marker)
    return dict(marker) | {"attempt_sha256": attempt_sha256}


def _validate_role_attempt(
    marker: Mapping[str, Any],
    *,
    role: str,
    generator_commit: str,
    actor_bank_manifest: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
) -> None:
    if role == "model_test":
        # The dedicated producer API validates its stricter exact schema.
        return
    expected = {
        "schema_version": ROLE_ATTEMPT_SCHEMA_VERSION,
        **_frozen_identity(generator_commit),
        "role": role,
        "source_seeds": list(ROLE_SOURCE_SEEDS[role]),
        "groups": len(ROLE_SOURCE_SEEDS[role]) * GROUPS_PER_SOURCE[role],
        "groups_per_source": GROUPS_PER_SOURCE[role],
        "admission_replicas": ADMISSION_REPLICAS,
        "label_replicas": LABEL_REPLICAS[role],
        "actor_bank_manifest_file_sha256": actor_bank_manifest_file_sha256,
        "actor_bank_contract_sha256": actor_bank_manifest[
            "actor_bank_contract_sha256"
        ],
        "status": "role_attempt_started_before_first_candidate_outcome",
    }
    for name, value in expected.items():
        if marker.get(name) != value:
            raise StageBExecutionError(f"role attempt {name} has drifted")
    _utc_timestamp(marker.get("created_at_utc"))
    if set(marker) != {*expected, "created_at_utc"}:
        raise StageBExecutionError("role attempt has extra or missing fields")


@contextmanager
def _source_scope(
    *,
    paths: Mapping[str, Any],
    role: str,
    expected_role_attempt_sha256: str,
) -> Iterator[None]:
    source_paths = [Path(value) for value in paths.values()]
    role_attempt = Path(paths["attempt_marker"]) if "attempt_marker" in paths else None
    del role_attempt  # source mappings never carry the role marker.
    if role == "model_test":
        attempt = Path(next(iter(paths.values()))).parent / "attempt-started.json"
        with stage_b_model_test_producer_read_scope(
            attempt_path=attempt,
            expected_attempt_sha256=expected_role_attempt_sha256,
            evidence_paths=source_paths,
        ):
            yield
        return
    with ExitStack() as stack:
        for path in source_paths:
            contract = _stage_b_path_contract(path)
            if contract is None or contract[0] != role:
                raise StageBExecutionError("source path role contract has drifted")
            stack.enter_context(stage_b_evidence_read_scope(
                scientific_role=role,
                evidence_kind=contract[1],
                path=path,
            ))
        yield


def _source_attempt(
    *,
    role: str,
    source_seed: int,
    generator_commit: str,
    actor_identity: Mapping[str, Any],
    actor_bank_manifest: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
    role_attempt_sha256: str,
    simulator_fingerprint: Mapping[str, Any],
    recovery_library_fingerprint_sha256: str,
    created_at_utc: str | None,
) -> dict[str, object]:
    assignment = assignment_for(role, source_seed)
    required_actor = {
        "role": role,
        "actor_training_seed": assignment.actor_training_seed,
        "checkpoint_step": assignment.checkpoint_step,
    }
    if any(actor_identity.get(name) != expected
           for name, expected in required_actor.items()):
        raise StageBExecutionError("source actor identity differs from assignment")
    actor_hashes = {}
    for name in (
        "actor_checkpoint_sha256",
        "actor_state_dict_sha256",
        "policy_fingerprint_sha256",
        "checkpoint_fingerprint_sha256",
    ):
        value = actor_identity.get(name)
        if not _is_lower_hex(value, 64):
            raise StageBExecutionError(f"actor identity {name} is malformed")
        actor_hashes[name] = value
    if not _is_lower_hex(recovery_library_fingerprint_sha256, 64):
        raise StageBExecutionError("recovery library fingerprint is malformed")
    return {
        "schema_version": SOURCE_ATTEMPT_SCHEMA_VERSION,
        **_frozen_identity(generator_commit),
        "role": role,
        "source_seed": source_seed,
        "actor_training_seed": assignment.actor_training_seed,
        "checkpoint_step": assignment.checkpoint_step,
        **actor_hashes,
        "groups": GROUPS_PER_SOURCE[role],
        "admission_replicas": ADMISSION_REPLICAS,
        "label_replicas": LABEL_REPLICAS[role],
        "role_attempt_sha256": role_attempt_sha256,
        "actor_bank_manifest_file_sha256": actor_bank_manifest_file_sha256,
        "actor_bank_contract_sha256": actor_bank_manifest[
            "actor_bank_contract_sha256"
        ],
        "simulator_fingerprint": json.loads(json.dumps(simulator_fingerprint)),
        "recovery_library_fingerprint_sha256": (
            recovery_library_fingerprint_sha256
        ),
        "created_at_utc": _utc_timestamp(created_at_utc),
        "status": "source_attempt_started_before_candidate_outcomes",
    }


def _validate_source_result(
    result: StageBRoleCollectionResult,
    *,
    role: str,
    source_seed: int,
    actor_identity: Mapping[str, Any],
    actor_bank_manifest: Mapping[str, Any],
) -> None:
    if not isinstance(result, StageBRoleCollectionResult) or result.role != role:
        raise StageBExecutionError("collector returned the wrong Stage-B role")
    admission_report = result.admission.validate()
    label_report = result.labels.validate()
    result.labels_privileged.validate(result.labels)
    result.admission_privileged.validate(result.admission)
    if label_report["groups"] != GROUPS_PER_SOURCE[role] or (
        result.labels.replica_count != LABEL_REPLICAS[role]
    ) or int(admission_report["accepted"]) != GROUPS_PER_SOURCE[role]:
        raise StageBExecutionError("source result dimensions differ from protocol")
    admission_contract = {
        "admission_replicas": ADMISSION_REPLICAS,
        "horizon_steps": HORIZON_POLICY_STEPS,
        "accept_min_falls_inclusive": 6,
        "accept_max_falls_inclusive": 26,
        "stage_b_role": role,
    }
    for name, expected in admission_contract.items():
        if result.admission.manifest.get(name) != expected:
            raise StageBExecutionError(
                f"source admission {name} differs from protocol")
    if result.labels.candidate_count != CANDIDATES or (
        int(result.labels.manifest.get("horizon_steps", -1))
        != HORIZON_POLICY_STEPS
    ) or result.labels.manifest.get("collection_protocol", {}).get(
        "role"
    ) != role:
        raise StageBExecutionError("source label candidate/horizon/role drifted")
    if set(map(int, np.asarray(result.labels["source_seed"]))) != {source_seed}:
        raise StageBExecutionError("label shard contains another source seed")
    assignment = assignment_for(role, source_seed)
    if set(map(int, np.asarray(
        result.labels["policy_training_seed"]))) != {
            assignment.actor_training_seed
        }:
        raise StageBExecutionError("label shard contains another actor seed")
    expected_policy = str(actor_identity["policy_fingerprint_sha256"])
    if set(np.asarray(result.labels["policy_source"]).astype(str)) != {
        expected_policy
    }:
        raise StageBExecutionError("label shard policy fingerprint has drifted")
    for name in ("source_policy", "continuation_policy"):
        if result.labels.manifest.get(name) != actor_bank_manifest:
            raise StageBExecutionError(
                f"label shard {name} must be the complete actor-bank manifest")


def collect_stage_b_source_once(
    *,
    stage_b_root: str | Path,
    role: str,
    source_seed: int,
    generator_commit: str,
    actor_identity: Mapping[str, Any],
    actor_bank_manifest: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
    expected_role_attempt_sha256: str,
    simulator_fingerprint: Mapping[str, Any],
    recovery_library_fingerprint_sha256: str,
    collect: Callable[
        [Callable[[Mapping[str, Any]], None]], StageBRoleCollectionResult
    ],
    created_at_utc: str | None = None,
    progress_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Consume and publish exactly one source after its durable attempt."""
    actor_bank = _validate_actor_bank(
        actor_bank_manifest,
        actor_bank_manifest_file_sha256=actor_bank_manifest_file_sha256,
        generator_commit=generator_commit,
    )
    if source_seed not in ROLE_SOURCE_SEEDS.get(role, ()):
        raise StageBExecutionError("source seed is outside the frozen role roster")
    if not _is_lower_hex(expected_role_attempt_sha256, 64):
        raise StageBExecutionError("expected role attempt SHA-256 is malformed")
    all_paths = stage_b_role_paths(stage_b_root, role)
    source_paths = all_paths["sources"][source_seed]
    all_paths["directory"].mkdir(parents=True, exist_ok=True)

    role_attempt_path = Path(all_paths["attempt_marker"])
    if role != "model_test":
        with stage_b_evidence_read_scope(
            scientific_role=role,
            evidence_kind="attempt_marker",
            path=role_attempt_path,
        ):
            role_attempt, attempt_sha = _regular_canonical_json(
                role_attempt_path, "Stage-B role attempt")
        if attempt_sha != expected_role_attempt_sha256:
            raise StageBExecutionError("role attempt hash differs from expected")
        _validate_role_attempt(
            role_attempt,
            role=role,
            generator_commit=generator_commit,
            actor_bank_manifest=actor_bank,
            actor_bank_manifest_file_sha256=actor_bank_manifest_file_sha256,
        )

    with _source_scope(
        paths=source_paths,
        role=role,
        expected_role_attempt_sha256=expected_role_attempt_sha256,
    ):
        for name, path in source_paths.items():
            _path_absent(path, f"source {name}")
        attempt = _source_attempt(
            role=role,
            source_seed=source_seed,
            generator_commit=generator_commit,
            actor_identity=actor_identity,
            actor_bank_manifest=actor_bank,
            actor_bank_manifest_file_sha256=actor_bank_manifest_file_sha256,
            role_attempt_sha256=expected_role_attempt_sha256,
            simulator_fingerprint=simulator_fingerprint,
            recovery_library_fingerprint_sha256=(
                recovery_library_fingerprint_sha256
            ),
            created_at_utc=created_at_utc,
        )
        source_attempt_sha256 = _atomic_no_clobber_json(
            source_paths["attempt_marker"], attempt)

        destinations = {
            name: Path(source_paths[name])
            for name in (
                "admission", "label", "label_privileged",
                "source_step_log", "source_report",
            )
        }
        staging = {
            name: _staging_path(path) for name, path in destinations.items()
        }
        sequence = 0
        try:
            with staging["source_step_log"].open("wb") as step_stream:
                def progress(values: Mapping[str, Any]) -> None:
                    nonlocal sequence
                    record = {
                        "schema_version": (
                            "qsafe.state_dependent_recovery_v5."
                            "stage_b_source_progress.v1"
                        ),
                        "role": role,
                        "source_seed": source_seed,
                        "sequence": sequence,
                        **{str(name): value for name, value in values.items()},
                    }
                    step_stream.write(_canonical_json(record))
                    step_stream.flush()
                    sequence += 1
                    if progress_sink is not None:
                        progress_sink(record)

                result = collect(progress)
                _validate_source_result(
                    result,
                    role=role,
                    source_seed=source_seed,
                    actor_identity=actor_identity,
                    actor_bank_manifest=actor_bank,
                )
                final_progress = {
                    "event": "source_collection_complete",
                    "groups": result.labels.group_count,
                    "target_groups": GROUPS_PER_SOURCE[role],
                    "proposals": int(result.proposals),
                    "source_steps": int(result.source_steps),
                    "trajectories": int(result.trajectories),
                }
                progress(final_progress)
                os.fsync(step_stream.fileno())

            result.admission.save(staging["admission"])
            result.labels.save(staging["label"])
            # Saving deployable refreshes its on-disk content hash; re-link the
            # privileged manifest to that exact hash before its physical save.
            label_hash = result.labels.validate()["content_sha256"]
            result.labels_privileged.manifest[
                "deployable_content_sha256"
            ] = label_hash
            result.labels_privileged.save(staging["label_privileged"])
            result.labels_privileged.validate(result.labels)

            artifact_sha256 = {
                "attempt_marker": source_attempt_sha256,
                "admission": _file_sha256(staging["admission"]),
                "label": _file_sha256(staging["label"]),
                "label_privileged": _file_sha256(
                    staging["label_privileged"]
                ),
                "source_step_log": _file_sha256(
                    staging["source_step_log"]
                ),
            }
            report: dict[str, object] = {
                "schema_version": SOURCE_REPORT_SCHEMA_VERSION,
                **_frozen_identity(generator_commit),
                "role": role,
                "source_seed": source_seed,
                "actor_training_seed": assignment_for(
                    role, source_seed
                ).actor_training_seed,
                "checkpoint_step": assignment_for(
                    role, source_seed
                ).checkpoint_step,
                "groups": GROUPS_PER_SOURCE[role],
                "admission_replicas": ADMISSION_REPLICAS,
                "label_replicas": LABEL_REPLICAS[role],
                "source_steps": int(result.source_steps),
                "trajectories": int(result.trajectories),
                "proposals": int(result.proposals),
                "role_attempt_sha256": expected_role_attempt_sha256,
                "source_attempt_sha256": source_attempt_sha256,
                "actor_bank_manifest_file_sha256": (
                    actor_bank_manifest_file_sha256
                ),
                "actor_bank_contract_sha256": actor_bank[
                    "actor_bank_contract_sha256"
                ],
                "evidence_artifacts": sorted([
                    _artifact_record(
                        Path(source_paths[name]), digest
                    )
                    for name, digest in artifact_sha256.items()
                ], key=lambda item: item["path"]),
                "candidate_outcome_summary": "forbidden",
                "created_at_utc": _utc_timestamp(created_at_utc),
                "status": "complete_evidence_hashes_and_operational_counts_only",
            }
            staging["source_report"].write_bytes(_canonical_json(report))
            report_sha256 = _file_sha256(staging["source_report"])
            _publish_staged([
                (staging["admission"], destinations["admission"]),
                (staging["label"], destinations["label"]),
                (
                    staging["label_privileged"],
                    destinations["label_privileged"],
                ),
                (
                    staging["source_step_log"],
                    destinations["source_step_log"],
                ),
                (staging["source_report"], destinations["source_report"]),
            ])
        finally:
            for path in staging.values():
                path.unlink(missing_ok=True)
    return dict(report) | {"report_sha256": report_sha256}


def _source_report_expected_fields() -> frozenset[str]:
    return frozenset({
        "schema_version",
        *list(_frozen_identity("0" * 40)),
        "role",
        "source_seed",
        "actor_training_seed",
        "checkpoint_step",
        "groups",
        "admission_replicas",
        "label_replicas",
        "source_steps",
        "trajectories",
        "proposals",
        "role_attempt_sha256",
        "source_attempt_sha256",
        "actor_bank_manifest_file_sha256",
        "actor_bank_contract_sha256",
        "evidence_artifacts",
        "candidate_outcome_summary",
        "created_at_utc",
        "status",
    })


def _validate_source_report(
    report: Mapping[str, Any],
    *,
    role: str,
    source_seed: int,
    generator_commit: str,
    role_attempt_sha256: str,
    actor_bank_manifest: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
) -> dict[str, str]:
    if set(report) != _source_report_expected_fields():
        raise StageBExecutionError("source report has extra or missing fields")
    assignment = assignment_for(role, source_seed)
    expected = {
        "schema_version": SOURCE_REPORT_SCHEMA_VERSION,
        **_frozen_identity(generator_commit),
        "role": role,
        "source_seed": source_seed,
        "actor_training_seed": assignment.actor_training_seed,
        "checkpoint_step": assignment.checkpoint_step,
        "groups": GROUPS_PER_SOURCE[role],
        "admission_replicas": ADMISSION_REPLICAS,
        "label_replicas": LABEL_REPLICAS[role],
        "role_attempt_sha256": role_attempt_sha256,
        "actor_bank_manifest_file_sha256": actor_bank_manifest_file_sha256,
        "actor_bank_contract_sha256": actor_bank_manifest[
            "actor_bank_contract_sha256"
        ],
        "candidate_outcome_summary": "forbidden",
        "status": "complete_evidence_hashes_and_operational_counts_only",
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise StageBExecutionError(f"source report {name} has drifted")
    _utc_timestamp(report.get("created_at_utc"))
    for name in ("source_steps", "trajectories", "proposals"):
        if isinstance(report.get(name), bool) or not isinstance(
            report.get(name), int
        ) or int(report[name]) < 0:
            raise StageBExecutionError(f"source report {name} is invalid")
    artifacts = report.get("evidence_artifacts")
    if not isinstance(artifacts, list) or artifacts != sorted(
        artifacts, key=lambda item: item.get("path", "")
    ):
        raise StageBExecutionError("source report artifacts are not ordered")
    result: dict[str, str] = {}
    expected_kind_by_key = {
        "attempt_marker": "source_attempt_marker",
        "admission": "admission",
        "label": "label",
        "label_privileged": "label_privileged",
        "source_step_log": "source_step_log",
    }
    if len(artifacts) != len(expected_kind_by_key):
        raise StageBExecutionError("source report artifact roster is incomplete")
    seen_kinds: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {
            "kind", "path", "sha256"
        }:
            raise StageBExecutionError("source report artifact is malformed")
        kind = item.get("kind")
        matching_keys = [
            key for key, expected_kind in expected_kind_by_key.items()
            if kind == expected_kind
        ]
        if len(matching_keys) != 1 or str(kind) in seen_kinds or not _is_lower_hex(
            item.get("sha256"), 64
        ):
            raise StageBExecutionError("source report artifact identity is invalid")
        seen_kinds.add(str(kind))
        key = matching_keys[0]
        expected_path = stage_b_role_paths(
            Path.cwd() / "stage-b", role
        )["sources"][source_seed][key]
        if item.get("path") != _relative_stage_b(expected_path):
            raise StageBExecutionError("source report artifact path has drifted")
        result[key] = str(item["sha256"])
    if set(result) != set(expected_kind_by_key) or result[
        "attempt_marker"
    ] != report[
        "source_attempt_sha256"
    ]:
        raise StageBExecutionError("source report artifact hashes are incomplete")
    return result


def _load_development_split_inputs(
    *,
    stage_b_root: Path,
    generator_commit: str,
) -> tuple[
    dict[str, StageBSplitIdentityView],
    dict[str, StageBAdmissionIdentityView],
    dict[str, dict[str, object]],
]:
    """Load the four completed development aggregates under exact scopes."""
    datasets: dict[str, StageBSplitIdentityView] = {}
    admissions: dict[str, StageBAdmissionIdentityView] = {}
    bindings: dict[str, dict[str, object]] = {}
    for development_role in ROLE_ORDER[:-1]:
        role_paths = stage_b_role_paths(stage_b_root, development_role)
        label_path = Path(role_paths["label"])
        admission_path = Path(role_paths["admission"])
        report_path = Path(role_paths["report"])
        with ExitStack() as stack:
            stack.enter_context(stage_b_evidence_read_scope(
                scientific_role=development_role,
                evidence_kind="label",
                path=label_path,
            ))
            stack.enter_context(stage_b_evidence_read_scope(
                scientific_role=development_role,
                evidence_kind="admission",
                path=admission_path,
            ))
            stack.enter_context(stage_b_evidence_read_scope(
                scientific_role=development_role,
                evidence_kind="report",
                path=report_path,
            ))
            dataset = load_split_identity_view(label_path)
            label_file_sha256 = _file_sha256(label_path)
            admission_file_sha256 = _file_sha256(admission_path)
            admission_identity = load_admission_identity_view(admission_path)
            report, report_file_sha256 = _regular_canonical_json(
                report_path, f"{development_role} outcome-free role report")
        if report.get("schema_version") != ROLE_REPORT_SCHEMA_VERSION or (
            report.get("role") != development_role) or (
                report.get("generator_commit") != generator_commit) or (
                    report.get("status") != "complete_evidence_hashes_only"):
            raise StageBExecutionError(
                f"{development_role} role report is incomplete or drifted")
        artifacts = report.get("evidence_artifacts")
        if not isinstance(artifacts, list):
            raise StageBExecutionError(
                f"{development_role} role report artifacts are missing")
        label_relative = _relative_stage_b(label_path)
        label_records = [
            item for item in artifacts
            if isinstance(item, Mapping) and item.get("path") == label_relative
        ]
        if len(label_records) != 1 or label_records[0].get(
            "kind"
        ) != "label" or label_records[0].get("sha256") != label_file_sha256:
            raise StageBExecutionError(
                f"{development_role} role report does not bind its aggregate")
        admission_relative = _relative_stage_b(admission_path)
        admission_records = [
            item for item in artifacts
            if isinstance(item, Mapping) and item.get("path") == admission_relative
        ]
        if len(admission_records) != 1 or admission_records[0].get(
            "kind") != "admission" or admission_records[0].get(
                "sha256") != admission_file_sha256:
            raise StageBExecutionError(
                f"{development_role} role report does not bind admission aggregate")
        datasets[development_role] = dataset
        admissions[development_role] = admission_identity
        bindings[development_role] = {
            "path": label_relative,
            "file_sha256": label_file_sha256,
            "content_sha256": datasets[development_role].content_sha256,
            "groups": datasets[development_role].group_count,
            "role_report_file_sha256": report_file_sha256,
            "admission_path": admission_relative,
            "admission_file_sha256": admission_file_sha256,
            "admission_content_sha256": admissions[
                development_role].content_sha256,
            "admission_proposals": admissions[
                development_role].proposal_count,
        }
    return datasets, admissions, bindings


@contextmanager
def _finalize_scope(
    *,
    paths: Mapping[str, Any],
    role: str,
    expected_role_attempt_sha256: str,
) -> Iterator[None]:
    evidence = _all_role_evidence_paths(paths)
    if role == "model_test":
        # The attempt is validated internally and the report is deliberately
        # excluded.  Publishing report.json revokes every producer grant.
        with stage_b_model_test_producer_read_scope(
            attempt_path=paths["attempt_marker"],
            expected_attempt_sha256=expected_role_attempt_sha256,
            evidence_paths=evidence[1:],
        ):
            yield
        return
    with ExitStack() as stack:
        for path in (*evidence, Path(paths["report"])):
            contract = _stage_b_path_contract(path)
            if contract is None or contract[0] != role:
                raise StageBExecutionError("finalize path contract has drifted")
            stack.enter_context(stage_b_evidence_read_scope(
                scientific_role=role,
                evidence_kind=contract[1],
                path=path,
            ))
        yield


def finalize_stage_b_role(
    *,
    stage_b_root: str | Path,
    role: str,
    generator_commit: str,
    actor_bank_manifest: Mapping[str, Any],
    actor_bank_manifest_file_sha256: str,
    expected_role_attempt_sha256: str,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Merge an exact role roster and publish an outcome-free report last."""
    actor_bank = _validate_actor_bank(
        actor_bank_manifest,
        actor_bank_manifest_file_sha256=actor_bank_manifest_file_sha256,
        generator_commit=generator_commit,
    )
    if role not in ROLE_ORDER or not _is_lower_hex(
        expected_role_attempt_sha256, 64
    ):
        raise StageBExecutionError("role or attempt hash is invalid")
    paths = stage_b_role_paths(stage_b_root, role)
    final_names = (
        "admission", "label", "label_privileged", "step_log",
        "collection_manifest", "completion_marker", "report",
    )
    split_output = (
        Path(paths["directory"]).parent
        / "stage-b-split-disjointness-report.json"
    )

    with _finalize_scope(
        paths=paths,
        role=role,
        expected_role_attempt_sha256=expected_role_attempt_sha256,
    ):
        if role != "model_test":
            role_attempt, attempt_sha = _regular_canonical_json(
                paths["attempt_marker"], "Stage-B role attempt")
            if attempt_sha != expected_role_attempt_sha256:
                raise StageBExecutionError("role attempt hash differs")
            _validate_role_attempt(
                role_attempt,
                role=role,
                generator_commit=generator_commit,
                actor_bank_manifest=actor_bank,
                actor_bank_manifest_file_sha256=(
                    actor_bank_manifest_file_sha256
                ),
            )
        else:
            attempt_sha = expected_role_attempt_sha256
        for name in final_names:
            _path_absent(paths[name], f"role {name}")
        if role == "model_test":
            _path_absent(split_output, "Stage-B split-disjointness report")

        admissions: list[AdmissionLedger] = []
        labels: list[GroupedBranchDataset] = []
        privileged: list[PrivilegedBranchView] = []
        source_records: list[dict[str, Any]] = []
        existing_hashes: dict[str, str] = {
            _relative_stage_b(paths["attempt_marker"]): attempt_sha
        }
        combined_steps = bytearray()
        for ordinal, source_seed in enumerate(ROLE_SOURCE_SEEDS[role]):
            source = paths["sources"][source_seed]
            report, report_sha256 = _regular_canonical_json(
                source["source_report"], "Stage-B source report")
            artifact_hashes = _validate_source_report(
                report,
                role=role,
                source_seed=source_seed,
                generator_commit=generator_commit,
                role_attempt_sha256=attempt_sha,
                actor_bank_manifest=actor_bank,
                actor_bank_manifest_file_sha256=(
                    actor_bank_manifest_file_sha256
                ),
            )
            for kind, expected_hash in artifact_hashes.items():
                actual = _file_sha256(source[kind])
                if actual != expected_hash:
                    raise StageBExecutionError(
                        f"source {source_seed} {kind} bytes changed")
                existing_hashes[_relative_stage_b(source[kind])] = actual
            existing_hashes[
                _relative_stage_b(source["source_report"])
            ] = report_sha256

            admission = (
                load_admission_ledger_blind(source["admission"])
                if role == "model_test"
                else AdmissionLedger.load(source["admission"])
            )
            label = (
                load_grouped_shard_blind(source["label"])
                if role == "model_test"
                else GroupedBranchDataset.load(source["label"])
            )
            view = (
                load_privileged_shard_blind(source["label_privileged"])
                if role == "model_test"
                else PrivilegedBranchView.load(
                    source["label_privileged"], deployable=label)
            )
            if admission.manifest.get("source_seed") != source_seed or (
                label.group_count != GROUPS_PER_SOURCE[role]
            ):
                raise StageBExecutionError("source shard identity/count drifted")
            for manifest_name in ("source_policy", "continuation_policy"):
                if admission.manifest.get(manifest_name) != actor_bank or (
                    label.manifest.get(manifest_name) != actor_bank
                ):
                    raise StageBExecutionError(
                        f"source shard {manifest_name} is not the complete "
                        "actor-bank manifest")
            if label.replica_count != LABEL_REPLICAS[role] or (
                label.candidate_count != CANDIDATES
            ) or label.horizon_steps != HORIZON_POLICY_STEPS or (
                label.manifest.get("collection_protocol", {}).get("role")
                != role
            ) or (
                role != "model_test"
                and int(admission.validate()["accepted"]) != GROUPS_PER_SOURCE[role]
            ):
                raise StageBExecutionError(
                    "source shard replicas/candidates/horizon/admission drifted")
            if role != "model_test":
                accepted = np.asarray(admission["accepted"], dtype=bool)
                if not np.array_equal(
                    np.asarray(admission["proposal_id"])[accepted].astype(str),
                    np.asarray(label["group_id"]).astype(str),
                ):
                    raise StageBExecutionError(
                        "accepted admission order differs from label groups")
            else:
                proposal_ids = set(np.asarray(admission["proposal_id"]).astype(str))
                label_ids = np.asarray(label["group_id"]).astype(str)
                if not set(label_ids).issubset(proposal_ids):
                    raise StageBExecutionError(
                        "model-test label identities are absent from admission")
            assignment = assignment_for(role, source_seed)
            if set(map(int, np.asarray(label["policy_training_seed"]))) != {
                assignment.actor_training_seed
            } or set(map(int, np.asarray(label["source_seed"]))) != {
                source_seed
            }:
                raise StageBExecutionError("source actor mapping drifted")
            identity = actor_identity_for(
                actor_bank,
                role=role,
                actor_seed=assignment.actor_training_seed,
                checkpoint_step=assignment.checkpoint_step,
            )
            if set(np.asarray(label["policy_source"]).astype(str)) != {
                identity["policy_fingerprint_sha256"]
            }:
                raise StageBExecutionError("source policy fingerprint drifted")
            admissions.append(admission)
            labels.append(label)
            privileged.append(view)
            step_bytes = source["source_step_log"].read_bytes()
            if step_bytes and not step_bytes.endswith(b"\n"):
                raise StageBExecutionError("source step log lacks final newline")
            combined_steps.extend(step_bytes)
            source_records.append({
                "ordinal": ordinal,
                "source_seed": source_seed,
                "actor_training_seed": assignment.actor_training_seed,
                "checkpoint_step": assignment.checkpoint_step,
                "groups": GROUPS_PER_SOURCE[role],
                "source_report_sha256": report_sha256,
                "source_attempt_sha256": report["source_attempt_sha256"],
                "artifact_sha256": dict(sorted(artifact_hashes.items())),
            })

        merged_admission = (
            merge_admission_ledgers_blind(admissions)
            if role == "model_test"
            else merge_admission_ledgers(admissions)
        )
        merged_labels = (
            merge_grouped_shards_blind(labels)
            if role == "model_test"
            else merge_grouped_shards(labels)
        )
        merged_labels.manifest["stage_b_role"] = role
        merged_labels.manifest["source_seed_order"] = list(
            ROLE_SOURCE_SEEDS[role])
        merged_labels.manifest["actor_bank_manifest_file_sha256"] = (
            actor_bank_manifest_file_sha256)
        merged_labels.manifest["actor_bank_contract_sha256"] = actor_bank[
            "actor_bank_contract_sha256"]
        if role != "model_test":
            merged_labels.validate(verify_hash=False)
        merged_privileged = (
            merge_privileged_shards_blind(privileged, labels, merged_labels)
            if role == "model_test"
            else merge_privileged_shards(privileged, labels, merged_labels)
        )

        expected_groups = len(ROLE_SOURCE_SEEDS[role]) * GROUPS_PER_SOURCE[role]
        if merged_labels.group_count != expected_groups or (
            role != "model_test"
            and int(merged_admission.validate()["accepted"]) != expected_groups
        ):
            raise StageBExecutionError("merged role group count has drifted")
        if tuple(int(value) for value in merged_labels["source_seed"]) != tuple(
            source_seed
            for source_seed in ROLE_SOURCE_SEEDS[role]
            for _ in range(GROUPS_PER_SOURCE[role])
        ):
            raise StageBExecutionError("merged source order has drifted")

        staging = {name: _staging_path(paths[name]) for name in final_names}
        if role == "model_test":
            staging["split_disjointness_report"] = _staging_path(
                split_output)
        try:
            if role == "model_test":
                save_admission_ledger_blind(merged_admission, staging["admission"])
            else:
                merged_admission.save(staging["admission"])
            if role == "model_test":
                save_grouped_shard_blind(merged_labels, staging["label"])
            else:
                merged_labels.save(staging["label"])
            merged_privileged.manifest["deployable_content_sha256"] = str(
                merged_labels.manifest["content_sha256"])
            if role == "model_test":
                save_privileged_shard_blind(
                    merged_privileged, staging["label_privileged"])
            else:
                merged_privileged.save(staging["label_privileged"])
            staging["step_log"].write_bytes(bytes(combined_steps))

            merged_hashes = {
                name: _file_sha256(staging[name])
                for name in (
                    "admission", "label", "label_privileged", "step_log"
                )
            }
            split_report_sha256: str | None = None
            if role == "model_test":
                development_datasets, development_admissions, aggregate_bindings = (
                    _load_development_split_inputs(
                        stage_b_root=Path(stage_b_root),
                        generator_commit=generator_commit,
                    )
                )
                ordered_role_datasets = {
                    **development_datasets,
                    "model_test": make_split_identity_view(
                        merged_labels,
                        content_sha256=merged_labels.manifest.get(
                            "content_sha256"),
                    ),
                }
                if tuple(ordered_role_datasets) != ROLE_ORDER:
                    raise StageBExecutionError(
                        "split proof role order differs from protocol")
                identity_proof = compile_split_disjointness(
                    role_datasets=ordered_role_datasets,
                    actor_bank_manifest=actor_bank,
                )
                ordered_role_admissions = {
                    **development_admissions,
                    "model_test": make_admission_identity_view(
                        merged_admission,
                        content_sha256=merged_admission.manifest.get(
                            "content_sha256"),
                    ),
                }
                partition_rng_proof = compile_partition_rng_disjointness(
                    role_admissions=ordered_role_admissions,
                    role_labels=ordered_role_datasets,
                )
                aggregate_bindings["model_test"] = {
                    "path": _relative_stage_b(paths["label"]),
                    "file_sha256": merged_hashes["label"],
                    "content_sha256": ordered_role_datasets[
                        "model_test"
                    ].content_sha256,
                    "groups": merged_labels.group_count,
                    "role_report_file_sha256": None,
                    "admission_path": _relative_stage_b(paths["admission"]),
                    "admission_file_sha256": merged_hashes["admission"],
                    "admission_content_sha256": ordered_role_admissions[
                        "model_test"
                    ].content_sha256,
                    "admission_proposals": ordered_role_admissions[
                        "model_test"
                    ].proposal_count,
                }
                split_basis: dict[str, object] = {
                    "schema_version": SPLIT_REPORT_SCHEMA_VERSION,
                    **_frozen_identity(generator_commit),
                    "actor_bank_manifest_file_sha256": (
                        actor_bank_manifest_file_sha256
                    ),
                    "actor_bank_contract_sha256": actor_bank[
                        "actor_bank_contract_sha256"
                    ],
                    "role_order": list(ROLE_ORDER),
                    "role_aggregate_labels": [
                        {"role": split_role, **{
                            key: aggregate_bindings[split_role][key]
                            for key in (
                                "path", "file_sha256", "content_sha256",
                                "groups", "role_report_file_sha256",
                            )
                        }}
                        for split_role in ROLE_ORDER
                    ],
                    "role_aggregate_admissions": [
                        {"role": split_role, **{
                            key: aggregate_bindings[split_role][key]
                            for key in (
                                "admission_path", "admission_file_sha256",
                                "admission_content_sha256",
                                "admission_proposals",
                            )
                        }}
                        for split_role in ROLE_ORDER
                    ],
                    "identity_proof": identity_proof,
                    "partition_rng_proof": partition_rng_proof,
                    "model_test_source": (
                        "in_memory_merged_dataset_and_staged_label_bytes_"
                        "before_role_report"
                    ),
                    "outcome_columns_read": False,
                    "pass": True,
                }
                split_report = dict(split_basis) | {
                    "report_sha256": canonical_sha256(split_basis)
                }
                staging["split_disjointness_report"].write_bytes(
                    _canonical_json(split_report))
                split_report_sha256 = _file_sha256(
                    staging["split_disjointness_report"])
            manifest: dict[str, object] = {
                "schema_version": COLLECTION_MANIFEST_SCHEMA_VERSION,
                **_frozen_identity(generator_commit),
                "role": role,
                "source_seed_order": list(ROLE_SOURCE_SEEDS[role]),
                "groups_per_source": GROUPS_PER_SOURCE[role],
                "groups": expected_groups,
                "admission_replicas": ADMISSION_REPLICAS,
                "label_replicas": LABEL_REPLICAS[role],
                "actor_bank_manifest_file_sha256": (
                    actor_bank_manifest_file_sha256
                ),
                "actor_bank_contract_sha256": actor_bank[
                    "actor_bank_contract_sha256"
                ],
                "role_attempt_sha256": attempt_sha,
                "sources": source_records,
                "merged_artifact_sha256": dict(sorted(merged_hashes.items())),
                "source_policy_and_continuation_manifest": (
                    "complete_actor_bank_manifest_exact"
                ),
                "candidate_outcomes_summarized": False,
                "created_at_utc": _utc_timestamp(created_at_utc),
                "status": "complete_fixed_roster_merge",
            }
            if split_report_sha256 is not None:
                manifest["split_disjointness_report"] = {
                    "path": "stage-b/stage-b-split-disjointness-report.json",
                    "file_sha256": split_report_sha256,
                    "report_sha256": split_report["report_sha256"],
                    "published_before_model_test_role_report": True,
                }
            staging["collection_manifest"].write_bytes(
                _canonical_json(manifest))
            manifest_sha256 = _file_sha256(staging["collection_manifest"])

            precompletion_hashes = dict(existing_hashes)
            for name, digest in merged_hashes.items():
                precompletion_hashes[_relative_stage_b(paths[name])] = digest
            precompletion_hashes[
                _relative_stage_b(paths["collection_manifest"])
            ] = manifest_sha256
            if split_report_sha256 is not None:
                precompletion_hashes[
                    "stage-b/stage-b-split-disjointness-report.json"
                ] = split_report_sha256
            completion: dict[str, object] = {
                "schema_version": COMPLETION_SCHEMA_VERSION,
                **_frozen_identity(generator_commit),
                "role": role,
                "source_seeds": list(ROLE_SOURCE_SEEDS[role]),
                "groups": expected_groups,
                "role_attempt_sha256": attempt_sha,
                "artifact_sha256_by_path": dict(
                    sorted(precompletion_hashes.items())
                ),
                "every_precompletion_file_bound_by_sha256": True,
                "candidate_outcomes_summarized": False,
                "created_at_utc": _utc_timestamp(created_at_utc),
                "status": "role_complete_report_pending",
            }
            staging["completion_marker"].write_bytes(
                _canonical_json(completion))
            completion_sha256 = _file_sha256(staging["completion_marker"])

            all_hashes = dict(precompletion_hashes)
            all_hashes[
                _relative_stage_b(paths["completion_marker"])
            ] = completion_sha256
            evidence_artifacts = [
                _artifact_record(path, all_hashes[_relative_stage_b(path)])
                for path in _all_role_evidence_paths(paths)
            ]
            evidence_artifacts.sort(key=lambda item: item["path"])
            if role == "model_test":
                expected_roster = sorted(
                    _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS)
                if [item["path"] for item in evidence_artifacts] != (
                    expected_roster
                ):
                    raise StageBExecutionError(
                        "Model-Test evidence roster differs from firewall")
                for item in evidence_artifacts:
                    if item["kind"] != _STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS[
                        item["path"]
                    ]:
                        raise StageBExecutionError(
                            "Model-Test evidence kind differs from firewall")
            report: dict[str, object] = {
                "schema_version": (
                    STAGE_B_MODEL_TEST_REPORT_SCHEMA
                    if role == "model_test"
                    else ROLE_REPORT_SCHEMA_VERSION
                ),
                **_frozen_identity(generator_commit),
                "role": role,
                "source_seeds": list(ROLE_SOURCE_SEEDS[role]),
                "groups": expected_groups,
                "admission_replicas": ADMISSION_REPLICAS,
                "label_replicas": LABEL_REPLICAS[role],
                "evidence_artifacts": evidence_artifacts,
                "producer_attempt_sha256": attempt_sha,
                "status": "complete_evidence_hashes_only",
                "created_at_utc": _utc_timestamp(created_at_utc),
            }
            staging["report"].write_bytes(_canonical_json(report))
            report_sha256 = _file_sha256(staging["report"])
            publication = [
                (staging["admission"], paths["admission"]),
                (staging["label"], paths["label"]),
                (
                    staging["label_privileged"],
                    paths["label_privileged"],
                ),
                (staging["step_log"], paths["step_log"]),
                (
                    staging["collection_manifest"],
                    paths["collection_manifest"],
                ),
            ]
            if role == "model_test":
                publication.append((
                    staging["split_disjointness_report"], split_output))
            publication.extend([
                (staging["completion_marker"], paths["completion_marker"]),
                # Report is the irreversible producer-revocation boundary.
                (staging["report"], paths["report"]),
            ])
            _publish_staged(publication, terminal_last=True)
        finally:
            for path in staging.values():
                path.unlink(missing_ok=True)
    # Do not touch any role evidence here: Model-Test producer access ended
    # the instant report.json was published.
    return dict(report) | {"report_sha256": report_sha256}


__all__ = [
    "COLLECTION_MANIFEST_SCHEMA_VERSION",
    "COMPLETION_SCHEMA_VERSION",
    "ROLE_ATTEMPT_SCHEMA_VERSION",
    "ROLE_REPORT_SCHEMA_VERSION",
    "SOURCE_ATTEMPT_SCHEMA_VERSION",
    "SOURCE_REPORT_SCHEMA_VERSION",
    "collect_stage_b_source_once",
    "finalize_stage_b_role",
    "prepare_stage_b_role",
    "stage_b_role_paths",
]
