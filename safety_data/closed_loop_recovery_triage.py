"""One-way discovery/audit firewall for the v3 closed-loop recovery triage.

The module consumes the collector's native, physically separate artifacts:

* a merged :class:`AdmissionLedger` with admission-only nominal outcomes,
* a merged discovery :class:`GroupedBranchDataset` carrying preassigned audit
  RNG identities, and
* six physical audit :class:`GroupedBranchDataset` source shards.

Discovery may create exactly one no-clobber selection lock.  Audit consumption
creates a no-clobber marker *before* opening the first audit shard, so malformed
files, exceptions, and interrupted analysis all permanently consume the audit.
All effect estimates macro-average groups within source seed and then average
the six source seeds, as preregistered by the v3 protocol.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from safety_data.closed_loop_recovery_collector import AdmissionLedger
from safety_data.paths import (
    ProtectedEvidencePathError,
    require_v3_audit_consumed_or_safe_input,
)
from safety_data.schema import GroupedBranchDataset


SELECTION_LOCK_SCHEMA_VERSION = (
    "qsafe.closed_loop_recovery_triage.selection_lock.v1")
AUDIT_CONSUMED_SCHEMA_VERSION = (
    "qsafe.closed_loop_recovery_triage.audit_consumed.v1")
REPORT_SCHEMA_VERSION = "qsafe.closed_loop_recovery_triage.report.v1"
COLLECTION_READINESS_SCHEMA_VERSION = (
    "qsafe.closed_loop_recovery_triage.collection_readiness.v1")

V3_CANDIDATE_NAMES = (
    "nominal",
    "mature_actor_L10",
    "mature_actor_L25",
    "mature_actor_L50",
    "joint_brake_L10",
    "halfway_neutral_L10",
    "halfway_neutral_L25",
    "ramp_neutral_L25",
    "ramp_crouch_L25",
)
V3_BEHAVIOR_STEPS = (0, 10, 25, 50, 10, 10, 25, 25, 25)
V3_SOURCE_SEEDS = (7801, 7802, 7811, 7812, 7821, 7822)

_SHARD_OUTPUT_TEMPLATE_KEYS = {
    "admission": "admission_shard_filename_template",
    "admission_privileged": "admission_privileged_shard_filename_template",
    "discovery": "discovery_shard_filename_template",
    "discovery_privileged": "discovery_privileged_shard_filename_template",
    "audit": "audit_shard_filename_template",
    "audit_privileged": "audit_privileged_shard_filename_template",
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_HARD_DENIED_PREFIXES = ("formal", "sealed")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# These are private implementation constants, not runtime knobs.  Production
# validation rejects any scale drift from the preregistered v3 cohort.
_V3_GROUPS_PER_SEED = 64
_V3_ADMISSION_REPLICAS = 32
_V3_ADMISSION_MIN_FALLS = 6
_V3_ADMISSION_MAX_FALLS = 26
_V3_DISCOVERY_REPLICAS = 64
_V3_AUDIT_REPLICAS = 64
_V3_HORIZON_STEPS = 96
_V3_PROTOCOL_CONTRACT_SHA256 = (
    "07f530c582df38a1ff685fa0f8c0546f01eebb8cb9ec9573911e6f6076a59c3b")


class ClosedLoopRecoveryTriageError(ValueError):
    """The v3 firewall or statistical contract failed closed."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClosedLoopRecoveryTriageError(f"{name} must be a mapping")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ClosedLoopRecoveryTriageError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ClosedLoopRecoveryTriageError(
            f"{name} must be at least {minimum}")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ClosedLoopRecoveryTriageError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ClosedLoopRecoveryTriageError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ClosedLoopRecoveryTriageError(f"{name} must be finite")
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ClosedLoopRecoveryTriageError(
            "value is not canonical JSON") from exc
    return (rendered + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the v3 canonical-JSON SHA-256 used for protocol subcontracts."""
    return hashlib.sha256(_canonical_json_bytes(value)[:-1]).hexdigest()


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(
            value, allow_nan=False, ensure_ascii=True, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ClosedLoopRecoveryTriageError(
            f"{name} must be JSON serializable") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes_once(path: Path, name: str) -> bytes:
    """Read one control file from one no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ClosedLoopRecoveryTriageError(
                    f"{name} must be a regular file")
            if metadata.st_nlink != 1:
                raise ClosedLoopRecoveryTriageError(
                    f"{name} must have exactly one filesystem link")
            return stream.read()
    except ClosedLoopRecoveryTriageError:
        raise
    except OSError as exc:
        raise ClosedLoopRecoveryTriageError(
            f"{name} is missing, unreadable, or a symlink") from exc


def _atomic_no_clobber_json(path: Path, value: Mapping[str, Any]) -> str:
    """Publish complete JSON with an atomic, no-replace hard-link operation."""
    payload = _canonical_json_bytes(value)
    parent = path.parent
    if not parent.is_dir():
        raise ClosedLoopRecoveryTriageError(
            f"artifact directory does not exist: {parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{path.name}.pending-")
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
            raise ClosedLoopRecoveryTriageError(
                f"refusing to clobber existing artifact: {path.name}") from exc
        published = True
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not published:  # pragma: no cover - defensive; exceptions leave earlier.
        raise ClosedLoopRecoveryTriageError(
            f"failed to publish artifact: {path.name}")
    return hashlib.sha256(payload).hexdigest()


def _denied_prefixes(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    protection = _mapping(protocol.get("protection"), "protection")
    configured = protection.get("denied_path_component_prefixes")
    if not isinstance(configured, list) or any(
            not isinstance(item, str) or not item for item in configured):
        raise ClosedLoopRecoveryTriageError(
            "protection.denied_path_component_prefixes must be text")
    lowered = {item.casefold() for item in configured}
    if not set(_HARD_DENIED_PREFIXES).issubset(lowered):
        raise ClosedLoopRecoveryTriageError(
            "v3 protection must deny formal* and sealed* path components")
    return tuple(sorted(lowered | set(_HARD_DENIED_PREFIXES)))


def _reject_protected_components(
    path: Path,
    prefixes: Sequence[str],
    name: str,
) -> None:
    for component in path.parts:
        lowered = component.casefold()
        if any(lowered.startswith(prefix) for prefix in prefixes):
            raise ClosedLoopRecoveryTriageError(
                f"{name} contains a protected path component")


def _absolute_artifact_root(protocol: Mapping[str, Any]) -> Path:
    """Resolve YAML-relative artifact roots against the repository, not cwd."""
    collection = _mapping(protocol.get("collection"), "collection")
    raw = Path(str(collection.get("artifact_root")))
    anchored = raw if raw.is_absolute() else _REPOSITORY_ROOT / raw
    return Path(os.path.abspath(os.fspath(anchored)))


def _absolute_repo_path(value: str | os.PathLike[str]) -> Path:
    raw = Path(value)
    anchored = raw if raw.is_absolute() else _REPOSITORY_ROOT / raw
    return Path(os.path.abspath(os.fspath(anchored)))


def _artifact_path(
    value: str | os.PathLike[str],
    *,
    protocol: Mapping[str, Any],
    expected_filename: str,
    name: str,
) -> Path:
    path = _absolute_repo_path(value)
    prefixes = _denied_prefixes(protocol)
    _reject_protected_components(path, prefixes, name)
    if path.name != expected_filename:
        raise ClosedLoopRecoveryTriageError(
            f"{name} filename must be {expected_filename!r}")
    collection = _mapping(protocol.get("collection"), "collection")
    root = _absolute_artifact_root(protocol)
    _reject_protected_components(root, prefixes, "collection.artifact_root")
    safe_root = _authorize_evidence_path(root, "collection.artifact_root")
    safe_path = _authorize_evidence_path(path, name)
    # Both values are absolute lexical paths, and the shared guard has already
    # rejected every existing symlinked ancestor.  Parent equality therefore
    # proves containment without resolving either final component.
    if safe_path.parent != safe_root:
        raise ClosedLoopRecoveryTriageError(
            f"{name} must be directly inside the locked artifact root")
    return safe_path


def _authorize_evidence_path(path: Path, name: str) -> Path:
    """Translate the shared no-follow firewall into this module's error type."""
    try:
        return require_v3_audit_consumed_or_safe_input(path)
    except ProtectedEvidencePathError as exc:
        raise ClosedLoopRecoveryTriageError(f"unsafe {name}: {exc}") from exc


def _text_vector(array: np.ndarray, size: int, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != (size,) or value.dtype.kind not in "US":
        raise ClosedLoopRecoveryTriageError(
            f"{name} must be a text [{size}] vector")
    result = value.astype(str, copy=False)
    if np.any(result == ""):
        raise ClosedLoopRecoveryTriageError(f"{name} contains empty text")
    return result


def _int_vector(array: np.ndarray, size: int, name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != (size,) or value.dtype.kind not in "iu":
        raise ClosedLoopRecoveryTriageError(
            f"{name} must be an integer [{size}] vector")
    return value.astype(np.int64, copy=False)


def _binary_array(array: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != shape or value.dtype.kind not in "biu" or not np.all(
            np.isin(value, (0, 1, False, True))):
        raise ClosedLoopRecoveryTriageError(
            f"{name} must be a binary array with shape {shape}")
    return value.astype(bool, copy=False)


def _seed_matrix(
    array: np.ndarray,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != shape or value.dtype.kind not in "iu" or np.any(value < 0):
        raise ClosedLoopRecoveryTriageError(
            f"{name} must be a nonnegative integer array with shape {shape}")
    result = value.astype(np.int64, copy=False)
    if len(np.unique(result)) != result.size:
        raise ClosedLoopRecoveryTriageError(
            f"{name} seeds must be globally unique")
    return result


def _validate_fingerprints(values: np.ndarray, name: str) -> None:
    if any(_HEX64.fullmatch(str(item)) is None for item in values):
        raise ClosedLoopRecoveryTriageError(
            f"{name} values must be lowercase SHA-256 strings")


def _validate_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _mapping(protocol, "protocol")
    if protocol.get("protocol_schema_version") != 1 or protocol.get(
            "protocol_name") != "objective1_closed_loop_recovery_triage_v3":
        raise ClosedLoopRecoveryTriageError("unexpected v3 protocol identity")
    if protocol.get("scope") != (
            "conditional_development_mechanism_triage_only") or protocol.get(
                "claim_eligible") is not False:
        raise ClosedLoopRecoveryTriageError(
            "v3 protocol must remain development-only and claim-ineligible")
    production_shape = (
        _V3_GROUPS_PER_SEED,
        _V3_ADMISSION_REPLICAS,
        _V3_ADMISSION_MIN_FALLS,
        _V3_ADMISSION_MAX_FALLS,
        _V3_DISCOVERY_REPLICAS,
        _V3_AUDIT_REPLICAS,
        _V3_HORIZON_STEPS,
    ) == (64, 32, 6, 26, 64, 64, 96)
    if production_shape and canonical_sha256(protocol) != (
            _V3_PROTOCOL_CONTRACT_SHA256):
        raise ClosedLoopRecoveryTriageError(
            "parsed protocol differs from the complete canonical v3 contract")
    protection = _mapping(protocol.get("protection"), "protection")
    required_protection = {
        "assignment_before_candidate_outcomes": True,
        "optional_stopping": "forbidden",
        "sample_top_up": "forbidden",
    }
    for key, expected in required_protection.items():
        if protection.get(key) != expected:
            raise ClosedLoopRecoveryTriageError(
                f"protection.{key} has drifted")
    _denied_prefixes(protocol)

    collection = _mapping(protocol.get("collection"), "collection")
    candidates = _mapping(collection.get("candidates"), "collection.candidates")
    if candidates.get("protocol_version") != (
            "qsafe.closed_loop_recovery_behaviors.v3"):
        raise ClosedLoopRecoveryTriageError("candidate protocol version drifted")
    if candidates.get("count") != 9 or candidates.get("nominal_index") != 0:
        raise ClosedLoopRecoveryTriageError("v3 requires K9 with nominal at zero")
    if tuple(candidates.get("ordered_names", ())) != V3_CANDIDATE_NAMES or tuple(
            candidates.get("behavior_override_steps", ())) != V3_BEHAVIOR_STEPS:
        raise ClosedLoopRecoveryTriageError(
            "candidate names, order, or behavior duration drifted")
    firewall = _mapping(protocol.get("firewall"), "firewall")
    firewall_expected = {
        "selection_uses": ["admission_ledger", "discovery"],
        "selection_forbidden": ["audit", "parent_iteration_outcomes"],
        "selection_lock_no_clobber": True,
        "selected_global_candidate": (
            "minimum_equal_group_equal_seed_discovery_risk"),
        "global_exact_tie_break": "locked_candidate_order",
        "per_group_ties": "uniform_expectation_all_discovery_minima",
        "audit_requires_exact_selection_lock_hash": True,
        "audit_consumed_marker_created_before_outcome_read": True,
        "interrupted_audit_remains_consumed": True,
        "audit_runner_up_policy": "forbidden",
    }
    for key, expected in firewall_expected.items():
        if firewall.get(key) != expected:
            raise ClosedLoopRecoveryTriageError(f"firewall.{key} has drifted")

    triage_gates = _mapping(protocol.get("triage_gates"), "triage_gates")
    data_gate = _mapping(triage_gates.get("data"), "triage_gates.data")
    required_seeds = tuple(_integer(
        seed, "required_source_seed", minimum=0)
        for seed in data_gate.get("required_source_seeds", ()))
    if required_seeds != V3_SOURCE_SEEDS:
        raise ClosedLoopRecoveryTriageError(
            "v3 requires the canonical six source seeds in locked order")
    if tuple(collection.get("audit_analysis_input_order", ())) != (
            required_seeds):
        raise ClosedLoopRecoveryTriageError(
            "audit shard input order differs from the canonical seed order")
    expected_templates = {
        "cohort_lock_filename": "cohort-lock.json",
        "attempt_shard_filename_template": (
            "source-{source_seed}.attempt-started.json"),
        "admission_shard_filename_template": (
            "source-{source_seed}.admission.npz"),
        "admission_privileged_shard_filename_template": (
            "source-{source_seed}.admission.privileged.npz"),
        "discovery_shard_filename_template": (
            "source-{source_seed}.discovery.npz"),
        "discovery_privileged_shard_filename_template": (
            "source-{source_seed}.discovery.privileged.npz"),
        "audit_shard_filename_template": "source-{source_seed}.audit.npz",
        "audit_privileged_shard_filename_template": (
            "source-{source_seed}.audit.privileged.npz"),
        "collection_report_shard_filename_template": (
            "source-{source_seed}.collection-report.json"),
    }
    for field, expected in expected_templates.items():
        if collection.get(field) != expected:
            raise ClosedLoopRecoveryTriageError(
                f"collection.{field} differs from the canonical v3 template")
    if data_gate.get("independent_groups_exact") != collection.get(
            "total_groups") or data_gate.get(
                "groups_per_required_source_seed_exact") != collection.get(
                    "groups_per_source_seed"):
        raise ClosedLoopRecoveryTriageError(
            "collection and data-gate group counts disagree")
    if int(collection["total_groups"]) != (
            len(required_seeds) * int(collection["groups_per_source_seed"])):
        raise ClosedLoopRecoveryTriageError(
            "total_groups must equal six times groups_per_source_seed")
    if int(collection["groups_per_source_seed"]) != _V3_GROUPS_PER_SEED or int(
            collection["total_groups"]) != len(required_seeds) * (
                _V3_GROUPS_PER_SEED):
        raise ClosedLoopRecoveryTriageError(
            "v3 requires exactly 64 groups/seed and 384 groups overall")
    for key, collection_key in (
        ("candidates_per_group_exact", "count"),
    ):
        if data_gate.get(key) != candidates.get(collection_key):
            raise ClosedLoopRecoveryTriageError(f"data gate {key} drifted")
    partition = _mapping(
        collection.get("replica_partition"), "collection.replica_partition")
    if partition.get("schema_version") != (
            "qsafe.physically_separate_replica_partition.v3") or partition.get(
                "assignment_timing") != "before_candidate_outcomes" or any(
                    partition.get(key) is not True for key in (
                        "exhaustive", "disjoint_seed_domains", "physical_files")):
        raise ClosedLoopRecoveryTriageError(
            "replica partition must remain physical, exhaustive, and disjoint")
    discovery_replicas = _integer(
        partition.get("discovery_replicas"),
        "discovery_replicas", minimum=1)
    audit_replicas = _integer(
        partition.get("audit_replicas"), "audit_replicas", minimum=1)
    if discovery_replicas != _V3_DISCOVERY_REPLICAS or audit_replicas != (
            _V3_AUDIT_REPLICAS):
        raise ClosedLoopRecoveryTriageError(
            "v3 requires exactly 64 discovery and 64 audit replicas")
    if data_gate.get("discovery_replicas_exact") != discovery_replicas or (
            data_gate.get("audit_replicas_exact") != audit_replicas):
        raise ClosedLoopRecoveryTriageError(
            "replica counts disagree between collection and data gate")
    discovery_range = _mapping(
        partition.get("discovery_indices"), "replica_partition.discovery_indices")
    audit_range = _mapping(
        partition.get("audit_indices"), "replica_partition.audit_indices")
    if discovery_range != {
            "start_inclusive": 0,
            "stop_exclusive": discovery_replicas,
    } or audit_range != {
            "start_inclusive": discovery_replicas,
            "stop_exclusive": discovery_replicas + audit_replicas,
    } or collection.get("total_candidate_replicas") != (
            discovery_replicas + audit_replicas):
        raise ClosedLoopRecoveryTriageError(
            "replica index ranges must be contiguous, exhaustive, and preassigned")
    admission = _mapping(collection.get("admission"), "collection.admission")
    admission_replicas = _integer(
        admission.get("replicas"), "admission.replicas", minimum=1)
    if admission_replicas != _V3_ADMISSION_REPLICAS:
        raise ClosedLoopRecoveryTriageError(
            "v3 requires exactly 32 admission replicas")
    if data_gate.get("admission_replicas_exact") != admission_replicas:
        raise ClosedLoopRecoveryTriageError(
            "admission replica count disagrees with data gate")
    admission_bounds = tuple(data_gate.get("admission_falls_inclusive", ()))
    if admission_bounds != (
            admission.get("accept_min_falls_inclusive"),
            admission.get("accept_max_falls_inclusive"),
    ):
        raise ClosedLoopRecoveryTriageError(
            "admission bounds disagree between collection and data gate")
    if admission_bounds != (
            _V3_ADMISSION_MIN_FALLS, _V3_ADMISSION_MAX_FALLS):
        raise ClosedLoopRecoveryTriageError(
            "v3 admission bounds must remain inclusive 6..26 of 32")
    if admission.get("labels_used_in_effect_estimation") is not False:
        raise ClosedLoopRecoveryTriageError(
            "admission outcomes may not enter effect estimation")
    if admission.get("all_proposals_recorded") is not True:
        raise ClosedLoopRecoveryTriageError(
            "the admission ledger must retain every proposed state")
    horizon = _integer(
        data_gate.get("horizon_policy_steps_exact"),
        "horizon_policy_steps_exact", minimum=1)
    if horizon != collection.get("admission", {}).get(
            "horizon_policy_steps") or horizon != protocol.get("target", {}).get(
                "horizon_policy_steps"):
        raise ClosedLoopRecoveryTriageError("H96 horizon fields disagree")
    if horizon != _V3_HORIZON_STEPS:
        raise ClosedLoopRecoveryTriageError("v3 requires exactly H96")
    expected_data_gate = {
        "independent_groups_exact": len(required_seeds) * _V3_GROUPS_PER_SEED,
        "unique_source_trajectories_exact": (
            len(required_seeds) * _V3_GROUPS_PER_SEED),
        "groups_per_required_source_seed_exact": _V3_GROUPS_PER_SEED,
        "required_source_seeds": list(V3_SOURCE_SEEDS),
        "candidates_per_group_exact": 9,
        "discovery_replicas_exact": _V3_DISCOVERY_REPLICAS,
        "audit_replicas_exact": _V3_AUDIT_REPLICAS,
        "horizon_policy_steps_exact": _V3_HORIZON_STEPS,
        "admission_replicas_exact": _V3_ADMISSION_REPLICAS,
        "admission_falls_inclusive": [
            _V3_ADMISSION_MIN_FALLS, _V3_ADMISSION_MAX_FALLS],
        "min_discovery_nominal_risk": 0.15,
        "max_discovery_nominal_risk": 0.75,
        "min_each_policy_age_discovery_nominal_risk": 0.10,
        "max_each_policy_age_discovery_nominal_risk": 0.90,
        "discovery_informativeness_checked_before_audit_open": True,
        "zero_duplicate_state_fingerprints": True,
        "zero_duplicate_trajectory_fingerprints": True,
    }
    if dict(data_gate) != expected_data_gate:
        raise ClosedLoopRecoveryTriageError(
            "triage_gates.data differs from the exact v3 gate")

    statistics = _mapping(protocol.get("statistics"), "statistics")
    if statistics.get("group_weighting") != (
            "equal_groups_within_seed_then_equal_six_source_seeds"):
        raise ClosedLoopRecoveryTriageError("group-weighting estimand drifted")
    if statistics.get("natural_frequency_weighting") != "forbidden":
        raise ClosedLoopRecoveryTriageError(
            "natural-frequency weighting is forbidden for this enriched cohort")
    pair_agreement = _mapping(
        statistics.get("pair_agreement"), "statistics.pair_agreement")
    pair_expected = {
        "candidate_scope": "all_K9_unordered_pairs",
        "pairs_per_group": 36,
        "tie_condition": "discovery_or_audit_pair_risk_equal",
        "tie_score": 0.5,
        "non_tie_score": (
            "one_if_risk_difference_signs_agree_else_zero"),
        "within_group_aggregation": "mean_over_36_pairs",
        "cohort_aggregation": (
            "equal_groups_within_seed_then_equal_six_source_seeds"),
        "bootstrap_unit": "complete_group_score",
    }
    if dict(pair_agreement) != pair_expected or statistics.get(
            "pair_tie_score") != 0.5:
        raise ClosedLoopRecoveryTriageError(
            "statistics.pair_agreement definition drifted")
    bootstrap = _mapping(statistics.get("bootstrap"), "statistics.bootstrap")
    bootstrap_expected = {
        "kind": "hierarchical_policy_age_seed_then_trajectory_group",
        "replicates": 50_000,
        "seed": 20_260_809,
        "rng_bit_generator": "numpy_PCG64",
        "chunk_size": 512,
        "draw_order": (
            "chunk_then_sorted_age_then_slot_seed_vector_C_then_"
            "group_matrix_C_order"),
        "quantile_method": "linear",
        "confidence": "one_sided_0.95",
        "resample_policy_age_strata": False,
        "resample_source_seeds_within_stratum": True,
        "resample_complete_groups_within_seed": True,
        "resample_candidates": False,
        "resample_replicas": False,
    }
    if dict(bootstrap) != bootstrap_expected:
        raise ClosedLoopRecoveryTriageError(
            "statistics.bootstrap differs from the exact v3 algorithm")

    strata_raw = _mapping(
        statistics.get("policy_age_strata"), "statistics.policy_age_strata")
    strata: dict[int, tuple[int, ...]] = {}
    for age_text, seeds in strata_raw.items():
        try:
            age = int(age_text)
        except (TypeError, ValueError) as exc:
            raise ClosedLoopRecoveryTriageError(
                "policy-age stratum keys must be integers encoded as text") from exc
        if not isinstance(seeds, list) or not seeds:
            raise ClosedLoopRecoveryTriageError(
                "each policy-age stratum requires source seeds")
        strata[age] = tuple(int(seed) for seed in seeds)
    flattened = tuple(seed for seeds in strata.values() for seed in seeds)
    if sorted(flattened) != sorted(required_seeds) or len(flattened) != len(
            set(flattened)):
        raise ClosedLoopRecoveryTriageError(
            "policy-age strata must partition the six required seeds")
    if strata != {
        25_438: (7801, 7802),
        50_030: (7811, 7812),
        100_359: (7821, 7822),
    }:
        raise ClosedLoopRecoveryTriageError(
            "v3 policy-age strata or source allocation drifted")
    early = protocol.get("early_task_policies")
    if not isinstance(early, list):
        raise ClosedLoopRecoveryTriageError("early_task_policies must be a list")
    policy_seed_age: dict[int, int] = {}
    for entry_value in early:
        entry = _mapping(entry_value, "early_task_policy")
        age = _integer(entry.get("training_step"), "training_step", minimum=1)
        seeds = entry.get("source_seeds")
        if not isinstance(seeds, list):
            raise ClosedLoopRecoveryTriageError(
                "early_task_policy.source_seeds must be a list")
        for seed in seeds:
            seed_int = int(seed)
            if seed_int in policy_seed_age:
                raise ClosedLoopRecoveryTriageError(
                    "source seed appears under multiple policy ages")
            policy_seed_age[seed_int] = age
    expected_seed_age = {
        seed: age for age, seeds in strata.items() for seed in seeds}
    if policy_seed_age != expected_seed_age:
        raise ClosedLoopRecoveryTriageError(
            "policy identities and statistical age strata disagree")

    primary = _mapping(
        triage_gates.get("primary_global_backup"),
        "triage_gates.primary_global_backup",
    )
    primary_expected = {
        "min_audit_absolute_reduction": 0.05,
        "min_one_sided_95_lcb_reduction": 0.03,
        "require_each_policy_age_positive": True,
        "min_positive_source_seeds": 6,
    }
    if dict(primary) != primary_expected:
        raise ClosedLoopRecoveryTriageError(
            "primary global-backup thresholds or directions drifted")
    conditional = _mapping(
        triage_gates.get("conditional_state_dependent"),
        "triage_gates.conditional_state_dependent",
    )
    conditional_expected = {
        "tested_only_if_primary_global_passes": True,
        "min_audit_absolute_reduction": 0.05,
        "min_one_sided_95_lcb_reduction": 0.03,
        "min_incremental_reduction_over_global": 0.02,
        "min_incremental_one_sided_95_lcb": 0.0,
        "incremental_lcb_strictly_greater": True,
        "min_discovery_to_audit_pair_agreement": 0.58,
        "min_pair_agreement_one_sided_95_lcb": 0.55,
        "max_discovery_minimizer_tie_fraction": 0.50,
        "max_mean_winner_count": 2.0,
        "require_each_policy_age_increment_positive": True,
        "min_increment_positive_source_seeds": 6,
    }
    if dict(conditional) != conditional_expected:
        raise ClosedLoopRecoveryTriageError(
            "conditional-state gate thresholds or directions drifted")
    no_headroom = _mapping(
        triage_gates.get("no_headroom"), "triage_gates.no_headroom")
    no_headroom_expected = {
        "simultaneous_band": (
            "hierarchical_bootstrap_max_centered_error_nonstudentized"),
        "confidence": "one_sided_0.95",
        "joint_effect_vector": (
            "eight_fixed_candidates_plus_discovery_locked_per_state_rule"),
        "preserve_k9_crn_correlation": True,
        "critical_value_formula": (
            "quantile_0.95_of_max_k_theta_hat_k_minus_theta_star_bk"),
        "candidate_ucb_formula": "theta_hat_k_plus_common_critical_value",
        "min_audit_nominal_risk_one_sided_95_lcb": 0.10,
        "require_every_simultaneous_effect_ucb_below": 0.03,
        "scope_limit": (
            "locked_K9_fixed_behaviors_and_locked_per_state_rule_only"),
        "decision": "redesign_recovery_library_no_model_training",
    }
    if dict(no_headroom) != no_headroom_expected:
        raise ClosedLoopRecoveryTriageError(
            "no-headroom thresholds, directions, or semantics drifted")
    inconclusive = _mapping(
        triage_gates.get("inconclusive"), "triage_gates.inconclusive")
    if dict(inconclusive) != {
        "condition": "primary_fails_and_no_headroom_does_not_fire",
        "decision": "no_model_training_no_topup_no_audit_reuse",
    }:
        raise ClosedLoopRecoveryTriageError(
            "inconclusive decision semantics drifted")

    protocol_copy = _json_copy(protocol, "protocol")
    return {
        "protocol": protocol_copy,
        "protocol_contract_sha256": canonical_sha256(protocol_copy),
        "collection": collection,
        "data_gate": data_gate,
        "required_seeds": required_seeds,
        "seed_age": expected_seed_age,
        "age_strata": strata,
        "groups": int(collection["total_groups"]),
        "groups_per_seed": int(collection["groups_per_source_seed"]),
        "admission_replicas": admission_replicas,
        "discovery_replicas": discovery_replicas,
        "audit_replicas": audit_replicas,
        "horizon": horizon,
    }


def validate_closed_loop_recovery_protocol(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete production v3 protocol without touching files.

    Collection and merge entrypoints can call this before creating an attempt
    marker or opening any outcome.  The returned record is deliberately small
    and contains no artifact-derived information.
    """
    spec = _validate_protocol(protocol)
    return {
        "protocol_name": spec["protocol"]["protocol_name"],
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "required_source_seeds": list(spec["required_seeds"]),
        "groups": spec["groups"],
        "groups_per_source_seed": spec["groups_per_seed"],
        "candidates": len(V3_CANDIDATE_NAMES),
        "admission_replicas": spec["admission_replicas"],
        "discovery_replicas": spec["discovery_replicas"],
        "audit_replicas": spec["audit_replicas"],
        "horizon_policy_steps": spec["horizon"],
        "bootstrap_replicates": int(
            spec["protocol"]["statistics"]["bootstrap"]["replicates"]),
        "bootstrap_seed": int(
            spec["protocol"]["statistics"]["bootstrap"]["seed"]),
    }


def _validate_commit(value: str, name: str) -> None:
    if _GIT_COMMIT.fullmatch(value) is None:
        raise ClosedLoopRecoveryTriageError(
            f"{name} must be a lowercase hexadecimal Git commit")


def _hash_text(value: object, name: str) -> str:
    result = str(value)
    if _HEX64.fullmatch(result) is None:
        raise ClosedLoopRecoveryTriageError(
            f"{name} must be a lowercase SHA-256 string")
    return result


def _require_regular_nonsymlink(path: Path, name: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ClosedLoopRecoveryTriageError(f"{name} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ClosedLoopRecoveryTriageError(f"{name} may not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ClosedLoopRecoveryTriageError(f"{name} must be a regular file")


def _expected_fall_definition(spec: Mapping[str, Any]) -> dict[str, Any]:
    failure = _mapping(
        spec["protocol"].get("target", {}).get("failure"), "target.failure")
    fields = (
        "max_abs_roll_pitch_rad",
        "min_base_height_m",
        "tilt_comparator",
        "height_comparator",
        "height_reference",
        "sampling_cadence",
        "within_policy_hold_crossings",
        "first_failure_step_semantics",
    )
    if any(field not in failure for field in fields):
        raise ClosedLoopRecoveryTriageError(
            "target.failure is missing the exact artifact label contract")
    return {field: failure[field] for field in fields}


def _validate_artifact_fall_definition(
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    role: str,
) -> None:
    observed = manifest.get("fall_definition")
    if not isinstance(observed, Mapping) or dict(observed) != (
            _expected_fall_definition(spec)):
        raise ClosedLoopRecoveryTriageError(
            f"{role} fall-definition thresholds/semantics drifted")


def _expected_policy_bundle(spec: Mapping[str, Any]) -> dict[str, Any]:
    protocol = spec["protocol"]
    config = protocol["policy_config"]
    return {
        "type": "locked_early_sac_policy_age_set_v3",
        "policy_training_seed": int(config["policy_training_seed"]),
        "config_sha256": str(config["config_sha256"]),
        "policies": [
            {
                name: policy[name]
                for name in (
                    "training_step", "source_seeds", "actor_sha256",
                    "actor_state_dict_sha256", "policy_fingerprint_sha256",
                    "checkpoint_fingerprint_sha256",
                )
            }
            for policy in protocol["early_task_policies"]
        ],
    }


def _expected_policy_fingerprint_by_seed(
    spec: Mapping[str, Any],
) -> dict[int, str]:
    return {
        int(seed): str(policy["policy_fingerprint_sha256"])
        for policy in spec["protocol"]["early_task_policies"]
        for seed in policy["source_seeds"]
    }


def _validate_outcome_runtime_manifest(
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    role: str,
) -> None:
    """Independently bind simulator, action projection, and policy identity."""
    target = _mapping(spec["protocol"].get("target"), "target")
    simulator = _mapping(
        manifest.get("simulator_fingerprint"),
        f"{role}.manifest.simulator_fingerprint",
    )
    candidates = _mapping(
        spec["collection"].get("candidates"), "collection.candidates")
    exact_scalars = {
        "backend": "mujoco",
        "model_path": target["model_mjcf"],
        "mjcf_xml_sha256": target["model_mjcf_dependency_sha256"],
        "timestep_s": 1.0 / float(target["low_level_hz"]),
        "policy_frequency_hz": float(target["policy_hz"]),
        "substeps": int(round(
            float(target["low_level_hz"]) / float(target["policy_hz"]))),
        "action_filter": None,
        "max_joint_delta": None,
    }
    if any(simulator.get(name) != expected
           for name, expected in exact_scalars.items()):
        raise ClosedLoopRecoveryTriageError(
            f"{role} simulator model/timing/projection fingerprint drifted")
    for name, expected in (
        ("kp", float(candidates["kp"])),
        ("kd", float(candidates["kd"])),
    ):
        values = np.asarray(simulator.get(name), dtype=np.float64)
        if values.shape != (12,) or not np.all(values == expected):
            raise ClosedLoopRecoveryTriageError(
                f"{role} simulator {name} gains drifted")
    failure_measurement = _mapping(
        simulator.get("failure_measurement"),
        f"{role}.simulator_fingerprint.failure_measurement",
    )
    if dict(failure_measurement) != {
        "height_reference": "base_link_body_origin_world_z",
        "cadence": "post_policy_step_after_all_low_level_substeps",
        "low_level_substeps_per_policy_step": 10,
    }:
        raise ClosedLoopRecoveryTriageError(
            f"{role} simulator failure-measurement fingerprint drifted")
    if manifest.get("action_application_contract") != target.get(
            "action_application_contract"):
        raise ClosedLoopRecoveryTriageError(
            f"{role} action-application contract drifted")
    expected_policy = _expected_policy_bundle(spec)
    if manifest.get("source_policy") != expected_policy or manifest.get(
            "continuation_policy") != expected_policy:
        raise ClosedLoopRecoveryTriageError(
            f"{role} early-policy bundle differs from the v3 protocol")
    if role != "admission":
        mature = spec["protocol"]["mature_recovery_policy"]
        action_contract = target["action_application_contract"]
        expected_program_manifest = {
            "candidate_protocol": dict(candidates),
            "candidate_protocol_sha256": canonical_sha256(candidates),
            "mature_policy_identity": {
                "training_step": int(mature["training_step"]),
                "config_sha256": str(
                    spec["protocol"]["policy_config"]["config_sha256"]),
                "actor_sha256": str(mature["actor_sha256"]),
                "actor_state_dict_sha256": str(
                    mature["actor_state_dict_sha256"]),
                "policy_fingerprint_sha256": str(
                    mature["policy_fingerprint_sha256"]),
                "checkpoint_fingerprint_sha256": str(
                    mature["checkpoint_fingerprint_sha256"]),
                "observation_dim": 46,
                "actor_observation_dim": 46,
                "action_dim": 12,
            },
            "action_projection": {
                name: action_contract[name]
                for name in (
                    "init_qpos", "action_offset", "joint_min", "joint_max")
            } | {
                "max_joint_delta": None,
                "use_action_filter": False,
            },
            "input_boundary": "corrected_deployable_5x46_only",
            "privileged_inputs": "forbidden",
        }
        expected_program = {
            "manifest": expected_program_manifest,
            "fingerprint_sha256": canonical_sha256(expected_program_manifest),
        }
        if manifest.get("recovery_program") != expected_program:
            raise ClosedLoopRecoveryTriageError(
                f"{role} mature recovery-program identity drifted")


def _admission_leaf_content_hashes(
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> list[str]:
    """Validate merged-ledger provenance in canonical source-shard order."""
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != len(
            spec["required_seeds"]):
        raise ClosedLoopRecoveryTriageError(
            "admission input must be a six-leaf merged ledger")
    if manifest.get("source_seeds") != list(spec["required_seeds"]) or (
            manifest.get("policy_training_steps") != [
                spec["seed_age"][seed] for seed in spec["required_seeds"]]):
        raise ClosedLoopRecoveryTriageError(
            "admission merged-leaf source order or policy ages drifted")
    hashes: list[str] = []
    for ordinal, (raw, seed) in enumerate(zip(
            shards, spec["required_seeds"], strict=True)):
        shard = _mapping(raw, f"admission.manifest.shards[{ordinal}]")
        if shard.get("ordinal") != ordinal or shard.get(
                "source_seed") != int(seed) or shard.get(
                    "policy_training_step") != spec["seed_age"][seed] or (
                    shard.get("accepted") != spec["groups_per_seed"]):
            raise ClosedLoopRecoveryTriageError(
                "admission merged-leaf provenance/order drifted")
        proposals = _integer(
            shard.get("proposals"), "admission leaf proposals", minimum=1)
        if proposals < spec["groups_per_seed"] or proposals > int(
                spec["collection"]["max_proposals_per_source_seed"]):
            raise ClosedLoopRecoveryTriageError(
                "admission leaf proposal count is outside the v3 contract")
        hashes.append(_hash_text(
            shard.get("content_sha256"),
            f"admission.manifest.shards[{ordinal}].content_sha256",
        ))
    return hashes


def _discovery_leaf_content_hashes(
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    generator_commit: str,
) -> list[str]:
    """Validate merged discovery provenance without reopening leaf files."""
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != len(
            spec["required_seeds"]):
        raise ClosedLoopRecoveryTriageError(
            "discovery input must be a six-leaf merged dataset")
    hashes: list[str] = []
    for ordinal, (raw, seed) in enumerate(zip(
            shards, spec["required_seeds"], strict=True)):
        shard = _mapping(raw, f"discovery.manifest.shards[{ordinal}]")
        if shard.get("ordinal") != ordinal or shard.get(
                "source_seeds") != [int(seed)] or shard.get(
                    "groups") != spec["groups_per_seed"] or shard.get(
                    "generator_commit") != generator_commit:
            raise ClosedLoopRecoveryTriageError(
                "discovery merged-leaf provenance/order drifted")
        hashes.append(_hash_text(
            shard.get("content_sha256"),
            f"discovery.manifest.shards[{ordinal}].content_sha256",
        ))
    return hashes


def _load_admission(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    path = _authorize_evidence_path(path, "merged admission artifact")
    _require_regular_nonsymlink(path, "merged admission artifact")
    try:
        ledger = AdmissionLedger.load(path)
        validation = ledger.validate()
    except Exception as exc:
        raise ClosedLoopRecoveryTriageError(
            "could not load the collector AdmissionLedger") from exc
    arrays = ledger.arrays
    manifest = ledger.manifest
    _validate_artifact_fall_definition(manifest, spec, "admission")
    _validate_outcome_runtime_manifest(manifest, spec, "admission")
    protocol_sha = str(manifest.get("protocol_sha256", ""))
    if _HEX64.fullmatch(protocol_sha) is None:
        raise ClosedLoopRecoveryTriageError(
            "admission manifest requires the protocol-file SHA-256")
    if manifest.get("protocol_contract_sha256") != spec[
            "protocol_contract_sha256"]:
        raise ClosedLoopRecoveryTriageError(
            "admission protocol-contract SHA-256 drifted")
    commit = str(manifest.get("generator_commit", ""))
    _validate_commit(commit, "generator_commit")
    leaf_content_sha256 = _admission_leaf_content_hashes(manifest, spec)
    proposals = int(validation["proposals"])
    group_id = _text_vector(arrays["proposal_id"], proposals, "proposal_id")
    state = _text_vector(arrays["state_hash"], proposals, "state_hash")
    trajectory = _text_vector(
        arrays["trajectory_id"], proposals, "trajectory_id")
    _validate_fingerprints(state, "state_hash")
    source_seed = _int_vector(arrays["source_seed"], proposals, "source_seed")
    recorded_age = _int_vector(
        arrays["policy_training_step"], proposals, "policy_training_step")
    policy_source = _text_vector(
        arrays["policy_source"], proposals, "policy_source")
    policy_age = np.asarray([
        spec["seed_age"].get(int(seed), -1) for seed in source_seed
    ], dtype=np.int64)
    expected_fingerprints = _expected_policy_fingerprint_by_seed(spec)
    expected_policy_source = np.asarray([
        expected_fingerprints.get(int(seed), "") for seed in source_seed
    ], dtype=str)
    if np.any(policy_age < 0) or not np.array_equal(
            recorded_age, policy_age) or not np.array_equal(
                policy_source, expected_policy_source):
        raise ClosedLoopRecoveryTriageError(
            "admission source seed/age/actor fingerprint binding drifted")
    accepted = _binary_array(arrays["accepted"], (proposals,), "accepted")
    raw_fall = _binary_array(
        arrays["fall"], (proposals, spec["admission_replicas"]),
        "admission fall")
    falls = np.sum(raw_fall, axis=1, dtype=np.int64)
    replicas = int(manifest.get("admission_replicas", -1))
    if replicas != spec["admission_replicas"] or np.any(falls < 0) or np.any(
            falls > replicas):
        raise ClosedLoopRecoveryTriageError(
            "admission replica count or fall counts drifted")
    lower, upper = map(int, spec["data_gate"]["admission_falls_inclusive"])
    expected_accepted = (falls >= lower) & (falls <= upper)
    if not np.array_equal(accepted, expected_accepted):
        raise ClosedLoopRecoveryTriageError(
            "accepted flags do not exactly implement locked admission bounds")
    horizon = int(manifest.get("horizon_steps", -1))
    if horizon != spec["horizon"]:
        raise ClosedLoopRecoveryTriageError("admission horizon drifted")
    seed_matrices = {
        name: _seed_matrix(
            arrays[name], (proposals, replicas), name)
        for name in (
            "admission_crn_id",
            "admission_rollout_seed",
            "admission_perturbation_seed",
        )
    }
    selected = np.flatnonzero(accepted)
    if len(selected) != spec["groups"]:
        raise ClosedLoopRecoveryTriageError(
            "admission accepted-group count failed the exact data gate")
    required = set(spec["required_seeds"])
    if set(map(int, source_seed[selected])) != required:
        raise ClosedLoopRecoveryTriageError(
            "accepted admission source seeds do not match the protocol")
    for seed in spec["required_seeds"]:
        if int(np.count_nonzero(source_seed[selected] == seed)) != spec[
                "groups_per_seed"]:
            raise ClosedLoopRecoveryTriageError(
                f"accepted source seed {seed} has the wrong group count")
    if len(np.unique(state[selected])) != len(selected):
        raise ClosedLoopRecoveryTriageError(
            "accepted state fingerprints must be unique")
    if len(np.unique(trajectory[selected])) != len(selected):
        raise ClosedLoopRecoveryTriageError(
            "accepted trajectories must be unique")
    return {
        "file_sha256": _sha256_file(path),
        "content_sha256": str(validation["content_sha256"]),
        "protocol_file_sha256": protocol_sha,
        "generator_commit": commit,
        "leaf_content_sha256": leaf_content_sha256,
        "validation": _json_copy(validation, "admission validation"),
        "selected": selected,
        "group_id": group_id[selected],
        "state_hash": state[selected],
        "trajectory_id": trajectory[selected],
        "source_seed": source_seed[selected],
        "policy_age": policy_age[selected],
        "admission_falls": falls[selected],
        "admission_crn_id": seed_matrices["admission_crn_id"][selected],
        "admission_rollout_seed": seed_matrices[
            "admission_rollout_seed"][selected],
        "admission_perturbation_seed": seed_matrices[
            "admission_perturbation_seed"][selected],
        "proposal_count": proposals,
    }


def _load_outcome_npz(
    path: Path,
    role: str,
    spec: Mapping[str, Any],
    *,
    expected_groups: int | None = None,
    expected_source_seed: int | None = None,
) -> dict[str, Any]:
    if role not in ("discovery", "audit"):
        raise ClosedLoopRecoveryTriageError("outcome role must be discovery/audit")
    path = _authorize_evidence_path(path, f"{role} outcome artifact")
    _require_regular_nonsymlink(path, f"{role} outcome artifact")
    try:
        dataset = GroupedBranchDataset.load(path)
        validation = dataset.validate()
    except Exception as exc:
        raise ClosedLoopRecoveryTriageError(
            f"could not load collector {role} GroupedBranchDataset") from exc
    arrays = dataset.arrays
    manifest = dataset.manifest
    _validate_artifact_fall_definition(manifest, spec, role)
    _validate_outcome_runtime_manifest(manifest, spec, role)
    collection_protocol = _mapping(
        manifest.get("collection_protocol"),
        f"{role}.manifest.collection_protocol")
    if collection_protocol.get("role") != role or collection_protocol.get(
            "physical_replica_role_files") is not True:
        raise ClosedLoopRecoveryTriageError(
            f"{role} dataset has the wrong physical outcome role")
    protocol_sha = str(collection_protocol.get("protocol_sha256", ""))
    if _HEX64.fullmatch(protocol_sha) is None:
        raise ClosedLoopRecoveryTriageError(
            f"{role} manifest requires the protocol-file SHA-256")
    if collection_protocol.get("protocol_contract_sha256") != spec[
            "protocol_contract_sha256"]:
        raise ClosedLoopRecoveryTriageError(
            f"{role} protocol-contract SHA-256 drifted")
    commit = str(manifest.get("generator_commit", ""))
    _validate_commit(commit, "generator_commit")
    leaf_content_sha256: list[str] | None = None
    if role == "discovery" and expected_groups is None:
        leaf_content_sha256 = _discovery_leaf_content_hashes(
            manifest, spec, commit)
    groups = dataset.group_count
    required_groups = spec["groups"] if expected_groups is None else int(
        expected_groups)
    if groups != required_groups:
        raise ClosedLoopRecoveryTriageError(
            f"{role} has {groups} groups, expected {required_groups}")
    candidates = len(V3_CANDIDATE_NAMES)
    replicas = (
        spec["discovery_replicas"] if role == "discovery"
        else spec["audit_replicas"])
    if dataset.replica_count != replicas or dataset.horizon_steps != spec["horizon"]:
        raise ClosedLoopRecoveryTriageError(
            f"{role} replica count or horizon drifted")
    group_id = _text_vector(arrays["group_id"], groups, "group_id")
    state = _text_vector(arrays["state_hash"], groups, "state_hash")
    trajectory = _text_vector(arrays["trajectory_id"], groups, "trajectory_id")
    if any(len(np.unique(value)) != groups for value in (
            group_id, state, trajectory)):
        raise ClosedLoopRecoveryTriageError(
            f"{role} group/state/trajectory identities must be unique")
    _validate_fingerprints(state, "state_hash")
    source_seed = _int_vector(arrays["source_seed"], groups, "source_seed")
    policy_source = _text_vector(
        arrays["policy_source"], groups, "policy_source")
    policy_age = np.asarray([
        spec["seed_age"].get(int(seed), -1) for seed in source_seed
    ], dtype=np.int64)
    if expected_source_seed is None:
        if set(map(int, source_seed)) != set(spec["required_seeds"]) or any(
                np.count_nonzero(source_seed == seed) != spec["groups_per_seed"]
                for seed in spec["required_seeds"]):
            raise ClosedLoopRecoveryTriageError(f"{role} source seeds drifted")
        if role == "discovery" and any(not np.all(source_seed[
                ordinal * spec["groups_per_seed"]:
                (ordinal + 1) * spec["groups_per_seed"]] == seed)
                for ordinal, seed in enumerate(spec["required_seeds"])):
            raise ClosedLoopRecoveryTriageError(
                "discovery rows do not follow merged source-shard order")
    elif not np.all(source_seed == expected_source_seed):
        raise ClosedLoopRecoveryTriageError(
            f"{role} shard source seed differs from its locked path order")
    if np.any(policy_age < 0):
        raise ClosedLoopRecoveryTriageError(
            f"{role} source seed/policy age binding drifted")
    expected_fingerprints = _expected_policy_fingerprint_by_seed(spec)
    expected_policy_source = np.asarray([
        expected_fingerprints.get(int(seed), "") for seed in source_seed
    ], dtype=str)
    if not np.array_equal(policy_source, expected_policy_source):
        raise ClosedLoopRecoveryTriageError(
            f"{role} row-level early actor fingerprint drifted")
    kinds = np.asarray(arrays["candidate_kind"]).astype(str)
    if kinds.shape != (groups, candidates) or not np.all(
            kinds == np.asarray(V3_CANDIDATE_NAMES)[None, :]):
        raise ClosedLoopRecoveryTriageError(f"{role} candidate order drifted")
    steps = np.asarray(arrays["candidate_behavior_steps"])
    if steps.shape != (groups, candidates) or not np.all(
            steps == np.asarray(V3_BEHAVIOR_STEPS)[None, :]):
        raise ClosedLoopRecoveryTriageError(
            f"{role} candidate behavior steps drifted")
    if manifest.get("candidate_protocol") != spec["collection"]["candidates"]:
        raise ClosedLoopRecoveryTriageError(
            f"{role} candidate manifest differs from the v3 protocol")
    valid = _binary_array(
        arrays["candidate_mask"], (groups, candidates), "candidate_mask")
    if not np.all(valid):
        raise ClosedLoopRecoveryTriageError(
            f"{role} requires complete K9 candidate validity")
    seed_matrices = {
        name: _seed_matrix(arrays[name], (groups, replicas), name)
        for name in ("crn_id", "rollout_seed", "perturbation_seed")
    }
    candidate_seed = _int_vector(
        arrays["candidate_seed"], groups, "candidate_seed")
    if len(np.unique(candidate_seed)) != groups:
        raise ClosedLoopRecoveryTriageError(
            f"{role} candidate_seed must be globally unique")
    fall = _binary_array(
        arrays["fall"], (groups, candidates, replicas), "fall")
    result = {
        "file_sha256": _sha256_file(path),
        "content_sha256": str(validation["content_sha256"]),
        "protocol_file_sha256": protocol_sha,
        "generator_commit": commit,
        "group_id": group_id,
        "state_hash": state,
        "trajectory_id": trajectory,
        "source_seed": source_seed,
        "policy_age": policy_age,
        "crn_id": seed_matrices["crn_id"],
        "rollout_seed": seed_matrices["rollout_seed"],
        "perturbation_seed": seed_matrices["perturbation_seed"],
        "candidate_seed": candidate_seed,
        "fall": fall.astype(np.float64, copy=False),
        "validation": _json_copy(validation, f"{role} validation"),
    }
    if leaf_content_sha256 is not None:
        result["leaf_content_sha256"] = leaf_content_sha256
    if role == "discovery":
        for name in (
            "preassigned_audit_crn_id",
            "preassigned_audit_rollout_seed",
            "preassigned_audit_perturbation_seed",
        ):
            result[name] = _seed_matrix(
                arrays[name], (groups, spec["audit_replicas"]), name)
        result["preassigned_audit_candidate_seed"] = _int_vector(
            arrays["preassigned_audit_candidate_seed"],
            groups,
            "preassigned_audit_candidate_seed",
        )
        if len(np.unique(result["preassigned_audit_candidate_seed"])) != groups:
            raise ClosedLoopRecoveryTriageError(
                "preassigned audit candidate seeds must be globally unique")
    return result


def _same_identities(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(np.array_equal(left[name], right[name]) for name in (
        "group_id",
        "state_hash",
        "trajectory_id",
        "source_seed",
        "policy_age",
    ))


def _disjoint_seed_domains(*matrices: np.ndarray) -> bool:
    domains = [set(map(int, np.asarray(value).reshape(-1))) for value in matrices]
    return all(domains[left].isdisjoint(domains[right])
               for left in range(len(domains))
               for right in range(left + 1, len(domains)))


def _expected_audit_shard_paths(
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> list[Path]:
    """Construct locked audit names lexically, without stat/hash/open."""
    prefixes = _denied_prefixes(protocol)
    root = _absolute_artifact_root(protocol)
    _reject_protected_components(root, prefixes, "collection.artifact_root")
    root_absolute = root
    _reject_protected_components(
        root_absolute, prefixes, "absolute collection.artifact_root")
    result = [
        root_absolute / f"source-{seed}.audit.npz"
        for seed in spec["required_seeds"]
    ]
    for path in result:
        _reject_protected_components(path, prefixes, "expected audit shard")
    return result


def _canonical_embedded_path(
    value: object,
    *,
    root: Path,
    expected_filename: str,
    prefixes: Sequence[str],
    name: str,
) -> Path:
    """Validate a report-carried path lexically, without touching its target."""
    if not isinstance(value, str) or not value:
        raise ClosedLoopRecoveryTriageError(f"{name} must be a path string")
    lexical = Path(value)
    _reject_protected_components(lexical, prefixes, name)
    absolute = _absolute_repo_path(lexical)
    _reject_protected_components(absolute, prefixes, f"absolute {name}")
    expected = root / expected_filename
    if absolute != expected:
        raise ClosedLoopRecoveryTriageError(
            f"{name} differs from the canonical source-shard template")
    return absolute


def _verify_bound_json_marker(
    path: Path,
    *,
    resolved_root: Path,
    prefixes: Sequence[str],
    expected_file_sha256: str,
    expected_contract: Mapping[str, Any],
    name: str,
) -> None:
    """Read a safe control JSON and bind its bytes/content to the report."""
    path = _authorize_evidence_path(path, name)
    if path.parent != resolved_root:
        raise ClosedLoopRecoveryTriageError(
            f"{name} is outside the canonical artifact root")
    payload = _read_regular_bytes_once(path, name)
    if hashlib.sha256(payload).hexdigest() != expected_file_sha256:
        raise ClosedLoopRecoveryTriageError(
            f"{name} file hash differs from the completion report")
    try:
        observed = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ClosedLoopRecoveryTriageError(
            f"could not read {name}") from exc
    if not isinstance(observed, Mapping) or dict(observed) != dict(
            expected_contract):
        raise ClosedLoopRecoveryTriageError(
            f"{name} content differs from the embedded contract")


def _structural_validations(
    raw: object,
    *,
    outputs: Mapping[str, Mapping[str, str]],
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Accept only non-outcome summaries in report-last D/A validations."""
    validations = _mapping(raw, "collection report validations")
    if set(validations) != set(_SHARD_OUTPUT_TEMPLATE_KEYS):
        raise ClosedLoopRecoveryTriageError(
            "collection report validations must contain exactly six roles")
    expected_fields = {
        "admission": {"proposals", "accepted", "content_sha256"},
        "admission_privileged": {"proposals", "content_sha256"},
        "discovery": {
            "groups", "max_candidates", "replicas", "horizon_steps",
            "content_sha256",
        },
        "discovery_privileged": {"groups", "content_sha256"},
        "audit": {
            "groups", "max_candidates", "replicas", "horizon_steps",
            "content_sha256",
        },
        "audit_privileged": {"groups", "content_sha256"},
    }
    result: dict[str, dict[str, Any]] = {}
    for role in _SHARD_OUTPUT_TEMPLATE_KEYS:
        value = _mapping(validations.get(role), f"validations.{role}")
        if set(value) != expected_fields[role]:
            raise ClosedLoopRecoveryTriageError(
                f"validations.{role} contains nonstructural or missing fields")
        content_sha = _hash_text(
            value.get("content_sha256"),
            f"validations.{role}.content_sha256",
        )
        if content_sha != outputs[role]["content_sha256"]:
            raise ClosedLoopRecoveryTriageError(
                f"validations.{role} content hash differs from outputs")
        sanitized: dict[str, Any] = {"content_sha256": content_sha}
        if role == "admission":
            proposals = _integer(
                value.get("proposals"), "validations.admission.proposals",
                minimum=1)
            accepted = _integer(
                value.get("accepted"), "validations.admission.accepted",
                minimum=0)
            if accepted != spec["groups_per_seed"] or proposals < accepted or (
                    proposals > int(spec["collection"][
                        "max_proposals_per_source_seed"])):
                raise ClosedLoopRecoveryTriageError(
                    "admission validation count differs from the v3 shard gate")
            sanitized.update({"proposals": proposals, "accepted": accepted})
        elif role == "admission_privileged":
            sanitized["proposals"] = _integer(
                value.get("proposals"),
                "validations.admission_privileged.proposals",
                minimum=1,
            )
        elif role in ("discovery", "audit"):
            expected_replicas = (
                spec["discovery_replicas"] if role == "discovery"
                else spec["audit_replicas"])
            structural = {
                "groups": spec["groups_per_seed"],
                "max_candidates": len(V3_CANDIDATE_NAMES),
                "replicas": expected_replicas,
                "horizon_steps": spec["horizon"],
            }
            for field, expected in structural.items():
                if _integer(
                        value.get(field), f"validations.{role}.{field}",
                        minimum=1) != expected:
                    raise ClosedLoopRecoveryTriageError(
                        f"validations.{role} does not encode exact G/K/R/H")
            sanitized.update(structural)
        else:
            groups = _integer(
                value.get("groups"), f"validations.{role}.groups", minimum=1)
            if groups != spec["groups_per_seed"]:
                raise ClosedLoopRecoveryTriageError(
                    f"validations.{role}.groups differs from the shard gate")
            sanitized["groups"] = groups
        result[role] = sanitized
    if result["admission_privileged"]["proposals"] != result[
            "admission"]["proposals"]:
        raise ClosedLoopRecoveryTriageError(
            "admission privileged/deployable proposal counts differ")
    return result


def _collection_readiness(
    values: Sequence[str | os.PathLike[str]],
    *,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate report-last records without opening/stat'ing any role NPZ."""
    if isinstance(values, (str, bytes, os.PathLike)):
        raise ClosedLoopRecoveryTriageError(
            "collection_report_paths must be the ordered six-path sequence")
    paths = list(values)
    if len(paths) != len(spec["required_seeds"]):
        raise ClosedLoopRecoveryTriageError(
            "six canonical collection completion reports are required")
    prefixes = _denied_prefixes(protocol)
    root = _authorize_evidence_path(
        _absolute_artifact_root(protocol), "collection.artifact_root")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ClosedLoopRecoveryTriageError(
            "collection artifact root does not exist") from exc
    _reject_protected_components(
        resolved_root, prefixes, "resolved collection.artifact_root")

    records: list[dict[str, Any]] = []
    role_commitments: dict[str, list[dict[str, Any]]] = {
        role: [] for role in _SHARD_OUTPUT_TEMPLATE_KEYS}
    expected_cohort_path = root / str(
        spec["collection"]["cohort_lock_filename"])
    expected_seed_age = {
        str(seed): spec["seed_age"][seed] for seed in spec["required_seeds"]}
    for ordinal, (raw_path, seed) in enumerate(zip(
            paths, spec["required_seeds"], strict=True)):
        age = spec["seed_age"][seed]
        lexical = Path(raw_path)
        _reject_protected_components(
            lexical, prefixes, f"collection_report_paths[{ordinal}]")
        report_path = _absolute_repo_path(lexical)
        report_filename = str(spec["collection"][
            "collection_report_shard_filename_template"]).format(
                source_seed=seed)
        expected_report = root / report_filename
        if report_path != expected_report:
            raise ClosedLoopRecoveryTriageError(
                "collection reports must follow locked source-seed path order")
        report_path = _authorize_evidence_path(
            report_path, f"collection_report_paths[{ordinal}]")
        if report_path.parent != resolved_root or report_path.name != (
                expected_report.name):
            raise ClosedLoopRecoveryTriageError(
                "collection completion report is outside artifact_root")
        report_bytes = _read_regular_bytes_once(
            report_path, "collection completion report")
        try:
            report = json.loads(report_bytes.decode("utf-8"))
        except Exception as exc:
            raise ClosedLoopRecoveryTriageError(
                "could not read canonical collection completion report") from exc
        if not isinstance(report, Mapping) or report.get("schema_version") != (
                "qsafe.closed_loop_recovery_collection_report.v3") or report.get(
                    "protocol_name") != spec["protocol"]["protocol_name"]:
            raise ClosedLoopRecoveryTriageError(
                "collection completion report identity drifted")
        expected_report_fields = {
            "schema_version",
            "protocol_name",
            "protocol_path",
            "protocol_file_sha256",
            "protocol_contract_sha256",
            "cohort_lock",
            "cohort_lock_sha256",
            "cohort_contract",
            "attempt_marker",
            "attempt_marker_sha256",
            "attempt_contract",
            "development_only",
            "claim_eligible",
            "source_seed",
            "policy_training_step",
            "generator_commit",
            "generator_worktree_clean",
            "outputs",
            "validations",
            "source_steps",
            "trajectories",
            "proposals",
            "candidate_outcomes_summarized",
            "selection_lock_created",
            "audit_opened_for_analysis",
            "model_training_authorized",
            "phase2_authorized",
        }
        if set(report) != expected_report_fields or not isinstance(
                report.get("protocol_path"), str) or not report.get(
                    "protocol_path"):
            raise ClosedLoopRecoveryTriageError(
                "collection completion report contains extra or missing fields")
        protocol_file_sha = _hash_text(
            report.get("protocol_file_sha256"),
            "collection report protocol_file_sha256",
        )
        if report.get("protocol_contract_sha256") != spec[
                "protocol_contract_sha256"]:
            raise ClosedLoopRecoveryTriageError(
                "collection report protocol-contract SHA-256 drifted")
        generator_commit = str(report.get("generator_commit", ""))
        _validate_commit(generator_commit, "collection report generator_commit")
        if report.get("source_seed") != seed or report.get(
                "policy_training_step") != age or report.get(
                    "generator_worktree_clean") is not True or report.get(
                        "development_only") is not True or report.get(
                            "claim_eligible") is not False:
            raise ClosedLoopRecoveryTriageError(
                "collection completion report provenance drifted")
        if any(report.get(field) is not False for field in (
                "candidate_outcomes_summarized",
                "selection_lock_created",
                "audit_opened_for_analysis",
                "model_training_authorized",
                "phase2_authorized",
        )):
            raise ClosedLoopRecoveryTriageError(
                "collection report is not a pre-selection completion marker")

        cohort_path = _canonical_embedded_path(
            report.get("cohort_lock"),
            root=root,
            expected_filename=expected_cohort_path.name,
            prefixes=prefixes,
            name="collection report cohort_lock",
        )
        cohort_sha = _hash_text(
            report.get("cohort_lock_sha256"), "cohort_lock_sha256")
        cohort = _mapping(report.get("cohort_contract"), "cohort_contract")
        cohort_expected = {
            "schema_version": "qsafe.closed_loop_recovery.cohort_lock.v3",
            "protocol_name": spec["protocol"]["protocol_name"],
            "protocol_file_sha256": protocol_file_sha,
            "protocol_contract_sha256": spec["protocol_contract_sha256"],
            "generator_commit": generator_commit,
            "source_seed_policy_step": expected_seed_age,
            "outcome_state": "no_analysis_before_all_shards_complete",
        }
        if dict(cohort) != cohort_expected:
            raise ClosedLoopRecoveryTriageError(
                "embedded cohort lock contract drifted")
        _verify_bound_json_marker(
            cohort_path,
            resolved_root=resolved_root,
            prefixes=prefixes,
            expected_file_sha256=cohort_sha,
            expected_contract=cohort_expected,
            name="cohort lock",
        )
        attempt_filename = str(spec["collection"][
            "attempt_shard_filename_template"]).format(source_seed=seed)
        attempt_path = _canonical_embedded_path(
            report.get("attempt_marker"),
            root=root,
            expected_filename=attempt_filename,
            prefixes=prefixes,
            name="collection report attempt_marker",
        )
        attempt_sha = _hash_text(
            report.get("attempt_marker_sha256"), "attempt_marker_sha256")
        attempt = _mapping(report.get("attempt_contract"), "attempt_contract")
        expected_attempt_fields = {
            "schema_version",
            "protocol_name",
            "protocol_file_sha256",
            "protocol_contract_sha256",
            "generator_commit",
            "cohort_lock_sha256",
            "source_seed",
            "policy_training_step",
            "started_at_unix_ns",
            "state",
            "restart_authorized",
            "candidate_outcomes_summarized",
        }
        if set(attempt) != expected_attempt_fields or attempt.get(
                "schema_version") != (
                    "qsafe.closed_loop_recovery.attempt.v3") or attempt.get(
                        "protocol_name") != spec["protocol"][
                            "protocol_name"] or attempt.get(
                                "protocol_file_sha256") != protocol_file_sha or (
                                    attempt.get("protocol_contract_sha256") !=
                                    spec["protocol_contract_sha256"]) or (
                                    attempt.get("generator_commit") !=
                                    generator_commit) or attempt.get(
                                        "cohort_lock_sha256") != cohort_sha or (
                                            attempt.get("source_seed") != seed) or (
                                                attempt.get(
                                                    "policy_training_step") != age) or (
                                                        attempt.get("state") !=
                                                        "started_outcome_may_have_been_generated") or (
                                                            attempt.get(
                                                                "restart_authorized") is not False) or (
                                                                    attempt.get(
                                                                        "candidate_outcomes_summarized") is not False):
            raise ClosedLoopRecoveryTriageError(
                "embedded attempt marker is not bound to cohort/source/protocol")
        _integer(
            attempt.get("started_at_unix_ns"), "attempt.started_at_unix_ns",
            minimum=1)
        _verify_bound_json_marker(
            attempt_path,
            resolved_root=resolved_root,
            prefixes=prefixes,
            expected_file_sha256=attempt_sha,
            expected_contract=attempt,
            name=f"source-{seed} attempt marker",
        )

        outputs_raw = _mapping(
            report.get("outputs"), "collection report outputs")
        if set(outputs_raw) != set(_SHARD_OUTPUT_TEMPLATE_KEYS):
            raise ClosedLoopRecoveryTriageError(
                "collection report outputs must contain exactly six roles")
        outputs: dict[str, dict[str, str]] = {}
        for role, template_key in _SHARD_OUTPUT_TEMPLATE_KEYS.items():
            output = _mapping(outputs_raw.get(role), f"outputs.{role}")
            if set(output) != {"path", "file_sha256", "content_sha256"}:
                raise ClosedLoopRecoveryTriageError(
                    f"outputs.{role} must contain path/file/content commitments")
            filename = str(spec["collection"][template_key]).format(
                source_seed=seed)
            artifact_path = _canonical_embedded_path(
                output.get("path"),
                root=root,
                expected_filename=filename,
                prefixes=prefixes,
                name=f"outputs.{role}.path",
            )
            outputs[role] = {
                "path": str(artifact_path),
                "file_sha256": _hash_text(
                    output.get("file_sha256"),
                    f"outputs.{role}.file_sha256",
                ),
                "content_sha256": _hash_text(
                    output.get("content_sha256"),
                    f"outputs.{role}.content_sha256",
                ),
            }
        validations = _structural_validations(
            report.get("validations"), outputs=outputs, spec=spec)
        if report.get("proposals") != validations["admission"]["proposals"]:
            raise ClosedLoopRecoveryTriageError(
                "report proposal count differs from admission validation")
        _integer(report.get("source_steps"), "report.source_steps", minimum=0)
        _integer(report.get("trajectories"), "report.trajectories", minimum=0)

        for role, output in outputs.items():
            role_commitments[role].append({
                "ordinal": ordinal,
                "source_seed": int(seed),
                "policy_training_step": int(age),
                **output,
            })
        records.append({
            "ordinal": ordinal,
            "source_seed": int(seed),
            "policy_training_step": int(age),
            "protocol_file_sha256": protocol_file_sha,
            "protocol_contract_sha256": spec["protocol_contract_sha256"],
            "generator_commit": generator_commit,
            "collection_report_path": str(report_path),
            "collection_report_file_sha256": _sha256_file(report_path),
            "cohort_lock": {
                "path": str(cohort_path),
                "file_sha256": cohort_sha,
                "contract_sha256": canonical_sha256(cohort_expected),
            },
            "attempt_marker": {
                "path": str(attempt_path),
                "file_sha256": attempt_sha,
                "contract_sha256": canonical_sha256(dict(attempt)),
            },
            "outputs": outputs,
            "validations": validations,
        })

    if len({record["protocol_file_sha256"] for record in records}) != 1 or len({
            record["generator_commit"] for record in records}) != 1 or len({
                record["cohort_lock"]["file_sha256"]
                for record in records}) != 1 or len({
                    record["cohort_lock"]["contract_sha256"]
                    for record in records}) != 1:
        raise ClosedLoopRecoveryTriageError(
            "six collection reports disagree on protocol/commit/cohort lock")
    readiness_manifest = {
        "schema_version": COLLECTION_READINESS_SCHEMA_VERSION,
        "protocol_name": spec["protocol"]["protocol_name"],
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": records[0]["protocol_file_sha256"],
        "generator_commit": records[0]["generator_commit"],
        "artifact_root": str(root),
        "required_source_seeds": list(spec["required_seeds"]),
        "source_records": records,
        "role_commitments": role_commitments,
    }
    return {
        "manifest": readiness_manifest,
        "readiness_sha256": canonical_sha256(readiness_manifest),
        "role_commitments": role_commitments,
        "protocol_file_sha256": records[0]["protocol_file_sha256"],
        "generator_commit": records[0]["generator_commit"],
    }


def validate_collection_readiness(
    *,
    protocol: Mapping[str, Any],
    collection_report_paths: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Return a no-outcome readiness manifest/digest and role commitments.

    Only the six canonical report-last JSON files and their bound cohort/attempt
    control JSON are opened.  The six role artifacts named by the reports,
    including audit shards, are neither stat'ed nor resolved nor hashed.
    """
    spec = _validate_protocol(protocol)
    return _json_copy(_collection_readiness(
        collection_report_paths, protocol=protocol, spec=spec),
        "collection readiness",
    )


def _read_canonical_control_json(
    path: Path,
    *,
    resolved_root: Path,
    prefixes: Sequence[str],
    name: str,
) -> tuple[dict[str, Any], str]:
    """Open one direct-child control JSON with symlink rejection."""
    path = _authorize_evidence_path(path, name)
    if path.parent != resolved_root:
        raise ClosedLoopRecoveryTriageError(
            f"{name} is outside the canonical artifact root")
    payload = _read_regular_bytes_once(path, name)
    try:
        value = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ClosedLoopRecoveryTriageError(f"could not read {name}") from exc
    if not isinstance(value, dict):
        raise ClosedLoopRecoveryTriageError(f"{name} must contain a mapping")
    return value, hashlib.sha256(payload).hexdigest()


def _validate_merge_completion_reports(
    *,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    readiness: Mapping[str, Any],
    admission_path: Path,
    discovery_path: Path,
    admission: Mapping[str, Any] | None,
    discovery: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind canonical report-last merge records to leaves and merged inputs."""
    prefixes = _denied_prefixes(protocol)
    root = _authorize_evidence_path(
        _absolute_artifact_root(protocol), "collection.artifact_root")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ClosedLoopRecoveryTriageError(
            "collection artifact root does not exist") from exc
    _reject_protected_components(
        resolved_root, prefixes, "resolved collection.artifact_root")
    admission_report_path = root / str(
        spec["collection"]["admission_merge_report_filename"])
    discovery_report_path = root / str(
        spec["collection"]["discovery_merge_report_filename"])
    admission_report, admission_report_sha = _read_canonical_control_json(
        admission_report_path,
        resolved_root=resolved_root,
        prefixes=prefixes,
        name="admission merge completion report",
    )
    discovery_report, discovery_report_sha = _read_canonical_control_json(
        discovery_report_path,
        resolved_root=resolved_root,
        prefixes=prefixes,
        name="discovery merge completion report",
    )

    admission_fields = {
        "schema_version",
        "protocol_file_sha256",
        "protocol_contract_sha256",
        "merge_commit",
        "collection_readiness_sha256",
        "source_seed_order",
        "inputs",
        "output",
        "output_file_sha256",
        "output_content_sha256",
        "privileged_output",
        "privileged_file_sha256",
        "privileged_content_sha256",
        "validation",
        "privileged_validation",
        "candidate_outcomes_opened",
        "audit_opened",
        "model_training_authorized",
        "phase2_authorized",
    }
    admission_identity_checks = (
        set(admission_report) == admission_fields,
        admission_report.get("schema_version") == (
            "qsafe.closed_loop_admission_merge_report.v3"),
        admission_report.get("protocol_file_sha256") == readiness[
            "protocol_file_sha256"],
        admission_report.get("protocol_contract_sha256") == spec[
            "protocol_contract_sha256"],
        admission_report.get("merge_commit") == readiness["generator_commit"],
        admission_report.get("collection_readiness_sha256") == readiness[
            "readiness_sha256"],
        admission_report.get("source_seed_order") == list(
            spec["required_seeds"]),
    )
    if not all(admission_identity_checks):
        raise ClosedLoopRecoveryTriageError(
            "admission merge completion identity/provenance drifted")
    if any(admission_report.get(field) is not False for field in (
            "candidate_outcomes_opened",
            "audit_opened",
            "model_training_authorized",
            "phase2_authorized",
    )):
        raise ClosedLoopRecoveryTriageError(
            "admission merge report authorizes or opened forbidden outcomes")
    admission_inputs = admission_report.get("inputs")
    admission_commitments = readiness["role_commitments"]["admission"]
    source_records = readiness["manifest"]["source_records"]
    if not isinstance(admission_inputs, list) or len(admission_inputs) != len(
            admission_commitments):
        raise ClosedLoopRecoveryTriageError(
            "admission merge report requires six input commitments")
    for ordinal, (raw, commitment, source_record) in enumerate(zip(
            admission_inputs,
            admission_commitments,
            source_records,
            strict=True,
    )):
        value = _mapping(raw, f"admission merge inputs[{ordinal}]")
        validation = source_record["validations"]["admission"]
        input_path = _canonical_embedded_path(
            value.get("path"),
            root=root,
            expected_filename=Path(commitment["path"]).name,
            prefixes=prefixes,
            name=f"admission merge inputs[{ordinal}].path",
        )
        if set(value) != {
                "path", "file_sha256", "content_sha256", "proposals",
                "accepted",
        } or str(input_path) != commitment["path"] or value.get(
                "file_sha256") != commitment["file_sha256"] or value.get(
                    "content_sha256") != commitment[
                        "content_sha256"] or value.get(
                            "proposals") != validation["proposals"] or value.get(
                                "accepted") != validation["accepted"]:
            raise ClosedLoopRecoveryTriageError(
                "admission merge input differs from source completion records")
    expected_admission_output = _canonical_embedded_path(
        admission_report.get("output"),
        root=root,
        expected_filename=admission_path.name,
        prefixes=prefixes,
        name="admission merge output",
    )
    admission_output_file_sha = _hash_text(
        admission_report.get("output_file_sha256"),
        "admission merge output_file_sha256",
    )
    admission_output_content_sha = _hash_text(
        admission_report.get("output_content_sha256"),
        "admission merge output_content_sha256",
    )
    admission_validation = _mapping(
        admission_report.get("validation"), "admission merge validation")
    expected_proposals = sum(int(record["validations"]["admission"][
        "proposals"]) for record in source_records)
    if expected_admission_output != admission_path or set(
            admission_validation) != {
                "proposals", "accepted", "content_sha256",
            } or admission_validation.get("proposals") != expected_proposals or (
                admission_validation.get("accepted") != spec["groups"]) or (
                    admission_validation.get("content_sha256") !=
                    admission_output_content_sha):
        raise ClosedLoopRecoveryTriageError(
            "admission merge output commitment differs from merged ledger")
    if admission is not None and (
            admission_output_file_sha != admission["file_sha256"] or
            admission_output_content_sha != admission["content_sha256"] or
            dict(admission_validation) != admission["validation"]):
        raise ClosedLoopRecoveryTriageError(
            "admission merge output commitment differs from merged ledger")
    _canonical_embedded_path(
        admission_report.get("privileged_output"),
        root=root,
        expected_filename=str(
            spec["collection"]["admission_privileged_filename"]),
        prefixes=prefixes,
        name="admission privileged merge output",
    )
    if any(_HEX64.fullmatch(str(admission_report.get(field, ""))) is None
           for field in (
               "privileged_file_sha256", "privileged_content_sha256")):
        raise ClosedLoopRecoveryTriageError(
            "admission privileged merge hashes are invalid")
    privileged_validation = _mapping(
        admission_report.get("privileged_validation"),
        "admission merge privileged_validation",
    )
    if set(privileged_validation) != {"proposals", "content_sha256"} or (
            privileged_validation.get("proposals") !=
            expected_proposals) or privileged_validation.get(
                "content_sha256") != admission_report.get(
                    "privileged_content_sha256"):
        raise ClosedLoopRecoveryTriageError(
            "admission privileged merge validation drifted")

    discovery_fields = {
        "schema_version",
        "development_only",
        "publication_contract",
        "merge_tool_commit",
        "merge_tool_worktree_clean",
        "merge_tool_commit_stable",
        "output",
        "output_sha256",
        "output_content_sha256",
        "privileged_output",
        "privileged_sha256",
        "privileged_content_sha256",
        "input_shards",
        "input_privileged_shards",
        "validation",
        "data_gate_role",
        "collection_data_gate",
        "phase1_data_gate",
        "phase2_authorized",
        "collection_readiness_sha256",
    }
    if set(discovery_report) != discovery_fields or discovery_report.get(
            "schema_version") != "qsafe.grouped_merge_report.v3" or (
                discovery_report.get("development_only") is not True) or (
                    discovery_report.get("publication_contract") !=
                    "atomic_no_clobber_report_last_v1") or discovery_report.get(
                        "merge_tool_commit") != readiness[
                            "generator_commit"] or discovery_report.get(
                                "merge_tool_worktree_clean") is not True or (
                                    discovery_report.get(
                                        "merge_tool_commit_stable") is not True) or (
                                            discovery_report.get(
                                                "collection_readiness_sha256") !=
                                            readiness["readiness_sha256"]) or (
                                                discovery_report.get(
                                                    "phase2_authorized") is not False):
        raise ClosedLoopRecoveryTriageError(
            "discovery merge completion identity/provenance drifted")
    discovery_inputs = discovery_report.get("input_shards")
    discovery_commitments = readiness["role_commitments"]["discovery"]
    if not isinstance(discovery_inputs, list) or len(discovery_inputs) != len(
            discovery_commitments):
        raise ClosedLoopRecoveryTriageError(
            "discovery merge report requires six input commitments")
    for ordinal, (raw, commitment, seed) in enumerate(zip(
            discovery_inputs,
            discovery_commitments,
            spec["required_seeds"],
            strict=True,
    )):
        value = _mapping(raw, f"discovery merge input_shards[{ordinal}]")
        input_path = _canonical_embedded_path(
            value.get("path"),
            root=root,
            expected_filename=Path(commitment["path"]).name,
            prefixes=prefixes,
            name=f"discovery merge input_shards[{ordinal}].path",
        )
        if set(value) != {
                "path", "file_sha256", "content_sha256", "generator_commit",
                "groups", "source_seeds",
        } or str(input_path) != commitment["path"] or value.get(
                "file_sha256") != commitment["file_sha256"] or value.get(
                    "content_sha256") != commitment[
                        "content_sha256"] or value.get(
                            "generator_commit") != readiness[
                                "generator_commit"] or value.get(
                                    "groups") != spec[
                                        "groups_per_seed"] or value.get(
                                            "source_seeds") != [int(seed)]:
            raise ClosedLoopRecoveryTriageError(
                "discovery merge input differs from source completion records")
    privileged_inputs = discovery_report.get("input_privileged_shards")
    privileged_commitments = readiness[
        "role_commitments"]["discovery_privileged"]
    if not isinstance(privileged_inputs, list) or len(privileged_inputs) != len(
            privileged_commitments):
        raise ClosedLoopRecoveryTriageError(
            "discovery merge report requires six privileged inputs")
    for ordinal, (raw, privileged, deployable) in enumerate(zip(
            privileged_inputs,
            privileged_commitments,
            discovery_commitments,
            strict=True,
    )):
        value = _mapping(
            raw, f"discovery merge input_privileged_shards[{ordinal}]")
        input_path = _canonical_embedded_path(
            value.get("path"),
            root=root,
            expected_filename=Path(privileged["path"]).name,
            prefixes=prefixes,
            name=(
                f"discovery merge input_privileged_shards[{ordinal}].path"),
        )
        if set(value) != {
                "path", "file_sha256", "content_sha256", "generator_commit",
                "deployable_content_sha256",
        } or str(input_path) != privileged["path"] or value.get(
                "file_sha256") != privileged["file_sha256"] or value.get(
                    "content_sha256") != privileged[
                        "content_sha256"] or value.get(
                            "generator_commit") != readiness[
                                "generator_commit"] or value.get(
                                    "deployable_content_sha256") != deployable[
                                        "content_sha256"]:
            raise ClosedLoopRecoveryTriageError(
                "discovery privileged input differs from completion records")
    expected_discovery_output = _canonical_embedded_path(
        discovery_report.get("output"),
        root=root,
        expected_filename=discovery_path.name,
        prefixes=prefixes,
        name="discovery merge output",
    )
    discovery_output_file_sha = _hash_text(
        discovery_report.get("output_sha256"),
        "discovery merge output_sha256",
    )
    discovery_output_content_sha = _hash_text(
        discovery_report.get("output_content_sha256"),
        "discovery merge output_content_sha256",
    )
    discovery_validation = _mapping(
        discovery_report.get("validation"), "discovery merge validation")
    expected_validation_fields = {
        "schema_version",
        "groups",
        "max_candidates",
        "replicas",
        "valid_candidates",
        "min_valid_candidates_per_group",
        "replicas_per_candidate",
        "replica_partition",
        "unique_trajectory_clusters",
        "unique_source_seeds",
        "duplicate_state_fraction",
        "mixed_outcome_fraction",
        "content_sha256",
    }
    if expected_discovery_output != discovery_path or set(
            discovery_validation) != expected_validation_fields or (
                discovery_validation.get("groups") != spec["groups"]) or (
                    discovery_validation.get("max_candidates") !=
                    len(V3_CANDIDATE_NAMES)) or discovery_validation.get(
                        "replicas") != spec[
                            "discovery_replicas"] or (
                                discovery_validation.get("content_sha256") !=
                                discovery_output_content_sha):
        raise ClosedLoopRecoveryTriageError(
            "discovery merge output commitment differs from merged dataset")
    if discovery is not None and (
            discovery_output_file_sha != discovery["file_sha256"] or
            discovery_output_content_sha != discovery["content_sha256"] or
            dict(discovery_validation) != discovery["validation"]):
        raise ClosedLoopRecoveryTriageError(
            "discovery merge output commitment differs from merged dataset")
    _canonical_embedded_path(
        discovery_report.get("privileged_output"),
        root=root,
        expected_filename=str(
            spec["collection"]["discovery_privileged_filename"]),
        prefixes=prefixes,
        name="discovery privileged merge output",
    )
    if any(_HEX64.fullmatch(str(discovery_report.get(field, ""))) is None
           for field in ("privileged_sha256", "privileged_content_sha256")):
        raise ClosedLoopRecoveryTriageError(
            "discovery privileged merge hashes are invalid")
    expected_gate = {
        "pass": True,
        "checks": {
            "physical_role_discovery": True,
            "independent_groups_exact": True,
            "trajectory_clusters_exact": True,
            "source_seed_order_and_counts_exact": True,
            "candidates_exact": True,
            "candidate_kind_exact": True,
            "candidate_behavior_steps_exact": True,
            "candidate_protocol_exact": True,
            "discovery_replicas_exact": True,
            "horizon_exact": True,
            "discovery_seed_shape_exact": True,
            "audit_seed_preassignment_shape_exact": True,
            "audit_seed_preassignment_unique": True,
            "discovery_audit_seed_domains_disjoint": True,
            "audit_merge_forbidden": True,
        },
    }
    if discovery_report.get("data_gate_role") != (
            "closed_loop_recovery_triage") or discovery_report.get(
                "collection_data_gate") != expected_gate or discovery_report.get(
                    "phase1_data_gate") != expected_gate:
        raise ClosedLoopRecoveryTriageError(
            "discovery merge exact data gate did not pass")

    manifest = {
        "schema_version": (
            "qsafe.closed_loop_recovery_triage.merge_readiness.v1"),
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "collection_readiness_sha256": readiness["readiness_sha256"],
        "admission_merge_report": {
            "path": str(admission_report_path),
            "file_sha256": admission_report_sha,
            "output_file_sha256": admission_output_file_sha,
            "output_content_sha256": admission_output_content_sha,
        },
        "discovery_merge_report": {
            "path": str(discovery_report_path),
            "file_sha256": discovery_report_sha,
            "output_file_sha256": discovery_output_file_sha,
            "output_content_sha256": discovery_output_content_sha,
        },
    }
    return {
        "manifest": manifest,
        "merge_readiness_sha256": canonical_sha256(manifest),
    }


def _locked_audit_paths_before_consumption(
    values: Sequence[str | os.PathLike[str]],
    *,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> list[Path]:
    """Check audit path strings without touching their filesystem entries."""
    if isinstance(values, (str, bytes, os.PathLike)):
        raise ClosedLoopRecoveryTriageError(
            "audit_paths must be the ordered six-path sequence")
    paths = list(values)
    if len(paths) != len(spec["required_seeds"]):
        raise ClosedLoopRecoveryTriageError(
            "audit_paths must contain exactly six physical source shards")
    prefixes = _denied_prefixes(protocol)
    normalized: list[Path] = []
    for index, value in enumerate(paths):
        lexical = Path(value)
        _reject_protected_components(
            lexical, prefixes, f"audit_paths[{index}]")
        absolute = _absolute_repo_path(lexical)
        _reject_protected_components(
            absolute, prefixes, f"absolute audit_paths[{index}]")
        normalized.append(absolute)
    locked = lock.get("expected_audit_shards")
    if not isinstance(locked, list) or len(locked) != len(normalized):
        raise ClosedLoopRecoveryTriageError(
            "selection lock has no exhaustive audit-shard path binding")
    expected = _expected_audit_shard_paths(protocol, spec)
    for ordinal, (path, locked_value, expected_path, seed) in enumerate(zip(
            normalized, locked, expected, spec["required_seeds"], strict=True)):
        record = _mapping(locked_value, f"expected_audit_shards[{ordinal}]")
        if record.get("ordinal") != ordinal or record.get(
                "source_seed") != int(seed) or record.get("path") != str(
                    expected_path) or path != expected_path or any(
                        _HEX64.fullmatch(str(record.get(field, ""))) is None
                        for field in ("file_sha256", "content_sha256")):
            raise ClosedLoopRecoveryTriageError(
                "audit shard paths/order differ from the selection lock")
    return normalized


def _equal_seed_mean(
    values: np.ndarray,
    source_seed: np.ndarray,
    required_seeds: Sequence[int],
) -> float:
    group_values = np.asarray(values, dtype=np.float64)
    if group_values.shape != source_seed.shape or not np.all(
            np.isfinite(group_values)):
        raise ClosedLoopRecoveryTriageError(
            "equal-seed metric must be a finite group vector")
    return float(np.mean([
        np.mean(group_values[source_seed == seed], dtype=np.float64)
        for seed in required_seeds
    ], dtype=np.float64))


def _seed_effects(
    values: np.ndarray,
    source_seed: np.ndarray,
    required_seeds: Sequence[int],
) -> dict[str, float]:
    return {
        str(seed): float(np.mean(values[source_seed == seed], dtype=np.float64))
        for seed in required_seeds
    }


def _age_effects(
    values: np.ndarray,
    source_seed: np.ndarray,
    age_strata: Mapping[int, Sequence[int]],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for age in sorted(age_strata):
        seeds = age_strata[age]
        per_seed = [
            float(np.mean(values[source_seed == seed], dtype=np.float64))
            for seed in seeds
        ]
        result[str(age)] = float(np.mean(per_seed, dtype=np.float64))
    return result


def _discovery_informativeness(
    nominal_group_risk: np.ndarray,
    source_seed: np.ndarray,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    gate = spec["data_gate"]
    overall = _equal_seed_mean(
        nominal_group_risk, source_seed, spec["required_seeds"])
    by_age = _age_effects(
        nominal_group_risk, source_seed, spec["age_strata"])
    min_overall = _finite(
        gate["min_discovery_nominal_risk"],
        "min_discovery_nominal_risk")
    max_overall = _finite(
        gate["max_discovery_nominal_risk"],
        "max_discovery_nominal_risk")
    min_age = _finite(
        gate["min_each_policy_age_discovery_nominal_risk"],
        "min_each_policy_age_discovery_nominal_risk")
    max_age = _finite(
        gate["max_each_policy_age_discovery_nominal_risk"],
        "max_each_policy_age_discovery_nominal_risk")
    checks = {
        "overall_nominal_risk_inclusive": min_overall <= overall <= max_overall,
        "each_policy_age_nominal_risk_inclusive": all(
            min_age <= value <= max_age for value in by_age.values()),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "overall_equal_seed_nominal_risk": overall,
        "policy_age_equal_seed_nominal_risk": by_age,
        "locked_overall_interval_inclusive": [min_overall, max_overall],
        "locked_each_policy_age_interval_inclusive": [min_age, max_age],
    }


def _global_discovery_selection(
    discovery_risk: np.ndarray,
    source_seed: np.ndarray,
    required_seeds: Sequence[int],
) -> tuple[int, list[dict[str, Any]]]:
    scores = [
        _equal_seed_mean(discovery_risk[:, candidate], source_seed,
                         required_seeds)
        for candidate in range(1, len(V3_CANDIDATE_NAMES))
    ]
    # np.argmin returns the first exact minimizer, which is the locked order.
    selected = 1 + int(np.argmin(np.asarray(scores, dtype=np.float64)))
    table = [{
        "candidate_index": candidate,
        "candidate_name": V3_CANDIDATE_NAMES[candidate],
        "equal_seed_discovery_risk": float(scores[candidate - 1]),
    } for candidate in range(1, len(V3_CANDIDATE_NAMES))]
    return selected, table


def create_selection_lock(
    *,
    protocol: Mapping[str, Any],
    admission_path: str | os.PathLike[str],
    discovery_path: str | os.PathLike[str],
    collection_report_paths: Sequence[str | os.PathLike[str]],
    selection_lock_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read only admission/discovery artifacts and publish the one-way lock.

    Structural contract failures raise without creating a lock.  A structurally
    valid but uninformative discovery cohort is still locked permanently with
    ``audit_authorized=false``; the audit function then refuses to create a
    consumed marker or open audit outcomes.
    """
    spec = _validate_protocol(protocol)
    collection = spec["collection"]
    admission_file = _artifact_path(
        admission_path,
        protocol=protocol,
        expected_filename=str(collection["admission_deployable_filename"]),
        name="admission_path",
    )
    discovery_file = _artifact_path(
        discovery_path,
        protocol=protocol,
        expected_filename=str(collection["discovery_filename"]),
        name="discovery_path",
    )
    lock_file = _artifact_path(
        selection_lock_path,
        protocol=protocol,
        expected_filename=str(collection["selection_lock_filename"]),
        name="selection_lock_path",
    )
    if os.path.lexists(lock_file):
        raise ClosedLoopRecoveryTriageError(
            "refusing to reuse or clobber an existing selection lock")

    readiness = _collection_readiness(
        collection_report_paths, protocol=protocol, spec=spec)
    # Both report-last merge markers are fully validated before the merged
    # discovery outcome is opened.  A second pass below binds their output
    # commitments to the loaded native artifacts.
    preopen_merge_readiness = _validate_merge_completion_reports(
        protocol=protocol,
        spec=spec,
        readiness=readiness,
        admission_path=admission_file,
        discovery_path=discovery_file,
        admission=None,
        discovery=None,
    )
    admission = _load_admission(admission_file, spec)
    discovery = _load_outcome_npz(discovery_file, "discovery", spec)
    if admission["generator_commit"] != discovery["generator_commit"]:
        raise ClosedLoopRecoveryTriageError(
            "admission and discovery generator commits differ")
    if readiness["generator_commit"] != admission["generator_commit"]:
        raise ClosedLoopRecoveryTriageError(
            "completion reports and merged inputs use different commits")
    if admission["protocol_file_sha256"] != discovery[
            "protocol_file_sha256"]:
        raise ClosedLoopRecoveryTriageError(
            "admission and discovery protocol-file hashes differ")
    if readiness["protocol_file_sha256"] != admission["protocol_file_sha256"]:
        raise ClosedLoopRecoveryTriageError(
            "completion reports and merged inputs use different protocols")
    admission_report_hashes = [
        record["content_sha256"]
        for record in readiness["role_commitments"]["admission"]
    ]
    discovery_report_hashes = [
        record["content_sha256"]
        for record in readiness["role_commitments"]["discovery"]
    ]
    if admission["leaf_content_sha256"] != admission_report_hashes:
        raise ClosedLoopRecoveryTriageError(
            "merged admission leaves differ from completion-report commitments")
    if discovery["leaf_content_sha256"] != discovery_report_hashes:
        raise ClosedLoopRecoveryTriageError(
            "merged discovery leaves differ from completion-report commitments")
    merge_readiness = _validate_merge_completion_reports(
        protocol=protocol,
        spec=spec,
        readiness=readiness,
        admission_path=admission_file,
        discovery_path=discovery_file,
        admission=admission,
        discovery=discovery,
    )
    if merge_readiness != preopen_merge_readiness:
        raise ClosedLoopRecoveryTriageError(
            "merge completion records changed while selection was opening inputs")
    if not _same_identities(admission, discovery):
        raise ClosedLoopRecoveryTriageError(
            "admission and discovery group identities/order differ")
    if not _disjoint_seed_domains(
            admission["admission_crn_id"],
            admission["admission_rollout_seed"],
            admission["admission_perturbation_seed"],
            discovery["crn_id"],
            discovery["rollout_seed"],
            discovery["perturbation_seed"],
            discovery["candidate_seed"],
            discovery["preassigned_audit_crn_id"],
            discovery["preassigned_audit_rollout_seed"],
            discovery["preassigned_audit_perturbation_seed"],
            discovery["preassigned_audit_candidate_seed"]):
        raise ClosedLoopRecoveryTriageError(
            "admission/discovery/audit seed domains are not disjoint")

    discovery_risk = np.mean(discovery["fall"], axis=2, dtype=np.float64)
    informativeness = _discovery_informativeness(
        discovery_risk[:, 0], discovery["source_seed"], spec)
    global_index, global_table = _global_discovery_selection(
        discovery_risk, discovery["source_seed"], spec["required_seeds"])

    group_selection: list[dict[str, Any]] = []
    replica_partition: list[dict[str, Any]] = []
    for group in range(spec["groups"]):
        risks = discovery_risk[group]
        winners = np.flatnonzero(risks == np.min(risks)).astype(int).tolist()
        weight = 1.0 / len(winners)
        group_selection.append({
            "group_index": group,
            "group_id": str(discovery["group_id"][group]),
            "state_hash": str(discovery["state_hash"][group]),
            "trajectory_id": str(discovery["trajectory_id"][group]),
            "source_seed": int(discovery["source_seed"][group]),
            "policy_age": int(discovery["policy_age"][group]),
            "admission_falls": int(admission["admission_falls"][group]),
            "discovery_candidate_risk": risks.astype(float).tolist(),
            "discovery_minimizer_indices": winners,
            "discovery_minimizer_names": [
                V3_CANDIDATE_NAMES[index] for index in winners],
            "uniform_weights": [weight] * len(winners),
        })
        replica_partition.append({
            "group_index": group,
            "group_id": str(discovery["group_id"][group]),
            "admission_crn_ids": admission[
                "admission_crn_id"][group].astype(int).tolist(),
            "admission_rollout_seeds": admission[
                "admission_rollout_seed"][group].astype(int).tolist(),
            "admission_perturbation_seeds": admission[
                "admission_perturbation_seed"][group].astype(int).tolist(),
            "discovery_crn_ids": discovery[
                "crn_id"][group].astype(int).tolist(),
            "discovery_rollout_seeds": discovery[
                "rollout_seed"][group].astype(int).tolist(),
            "discovery_perturbation_seeds": discovery[
                "perturbation_seed"][group].astype(int).tolist(),
            "discovery_candidate_seed": int(discovery[
                "candidate_seed"][group]),
            "audit_crn_ids": discovery[
                "preassigned_audit_crn_id"][group].astype(int).tolist(),
            "audit_rollout_seeds": discovery[
                "preassigned_audit_rollout_seed"][group].astype(int).tolist(),
            "audit_perturbation_seeds": discovery[
                "preassigned_audit_perturbation_seed"][group].astype(int).tolist(),
            "audit_candidate_seed": int(discovery[
                "preassigned_audit_candidate_seed"][group]),
        })

    candidate_contract = spec["collection"]["candidates"]
    policy_contract = {
        "policy_config": spec["protocol"]["policy_config"],
        "early_task_policies": spec["protocol"]["early_task_policies"],
        "mature_recovery_policy": spec["protocol"]["mature_recovery_policy"],
    }
    lock: dict[str, Any] = {
        "schema_version": SELECTION_LOCK_SCHEMA_VERSION,
        "protocol_name": spec["protocol"]["protocol_name"],
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": admission["protocol_file_sha256"],
        "generator_commit": admission["generator_commit"],
        "candidate_library_sha256": canonical_sha256(candidate_contract),
        "policy_bundle_sha256": canonical_sha256(policy_contract),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_artifacts": {
            "admission": {
                "filename": admission_file.name,
                "file_sha256": admission["file_sha256"],
                "content_sha256": admission["content_sha256"],
                "proposal_count": int(admission["proposal_count"]),
            },
            "discovery": {
                "filename": discovery_file.name,
                "file_sha256": discovery["file_sha256"],
                "content_sha256": discovery["content_sha256"],
            },
        },
        "candidate_order": list(V3_CANDIDATE_NAMES),
        "selected_global_candidate": {
            "candidate_index": global_index,
            "candidate_name": V3_CANDIDATE_NAMES[global_index],
            "selection_scope": "eight_nonnominal_candidates",
            "exact_tie_break": "locked_candidate_order",
            "discovery_candidate_table": global_table,
        },
        "group_selection": group_selection,
        "group_selection_sha256": canonical_sha256(group_selection),
        "replica_partition": replica_partition,
        "replica_partition_sha256": canonical_sha256(replica_partition),
        "collection_readiness_sha256": readiness["readiness_sha256"],
        "collection_readiness_manifest": readiness["manifest"],
        "merge_readiness_sha256": merge_readiness[
            "merge_readiness_sha256"],
        "merge_readiness_manifest": merge_readiness["manifest"],
        "expected_audit_shards": readiness[
            "role_commitments"]["audit"],
        "data_gate": {
            "structural_contract_pass": True,
            "independent_groups": spec["groups"],
            "unique_state_fingerprints": spec["groups"],
            "unique_trajectory_fingerprints": spec["groups"],
            "groups_per_source_seed": spec["groups_per_seed"],
            "required_source_seeds": list(spec["required_seeds"]),
            "candidates": len(V3_CANDIDATE_NAMES),
            "admission_replicas": spec["admission_replicas"],
            "discovery_replicas": spec["discovery_replicas"],
            "audit_replicas_preassigned": spec["audit_replicas"],
            "horizon_policy_steps": spec["horizon"],
            "discovery_informativeness": informativeness,
            "pass": bool(informativeness["pass"]),
        },
        "bootstrap": _json_copy(
            spec["protocol"]["statistics"]["bootstrap"], "bootstrap"),
        "triage_gates": _json_copy(
            spec["protocol"]["triage_gates"], "triage_gates"),
        "audit_identifier": hashlib.sha256(os.urandom(32)).hexdigest(),
        "audit_authorized": bool(informativeness["pass"]),
        "audit_runner_up_policy": "forbidden",
    }
    lock_sha256 = _atomic_no_clobber_json(lock_file, lock)
    result = _json_copy(lock, "selection lock")
    result["selection_lock_sha256"] = lock_sha256
    return result


def _read_selection_lock(
    path: Path,
    expected_sha256: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    if _HEX64.fullmatch(expected_sha256) is None:
        raise ClosedLoopRecoveryTriageError(
            "expected_selection_lock_sha256 must be lowercase SHA-256")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ClosedLoopRecoveryTriageError(
                    "selection lock must be a regular file")
            if metadata.st_nlink != 1:
                raise ClosedLoopRecoveryTriageError(
                    "selection lock must have exactly one filesystem link")
            payload = stream.read()
    except ClosedLoopRecoveryTriageError:
        raise
    except OSError as exc:
        raise ClosedLoopRecoveryTriageError(
            "selection lock is missing, unreadable, or a symlink") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ClosedLoopRecoveryTriageError(
            "selection-lock file hash differs from the required hash")
    try:
        lock = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ClosedLoopRecoveryTriageError(
            "could not parse selection lock") from exc
    if not isinstance(lock, dict):
        raise ClosedLoopRecoveryTriageError("selection lock must be a mapping")
    if lock.get("schema_version") != SELECTION_LOCK_SCHEMA_VERSION or lock.get(
            "protocol_name") != spec["protocol"]["protocol_name"] or lock.get(
                "protocol_contract_sha256") != spec[
                    "protocol_contract_sha256"] or _HEX64.fullmatch(str(
                        lock.get("protocol_file_sha256", ""))) is None:
        raise ClosedLoopRecoveryTriageError("selection-lock identity mismatch")
    _validate_commit(str(lock.get("generator_commit", "")), "generator_commit")
    expected_candidate_hash = canonical_sha256(spec["collection"]["candidates"])
    expected_policy_hash = canonical_sha256({
        "policy_config": spec["protocol"]["policy_config"],
        "early_task_policies": spec["protocol"]["early_task_policies"],
        "mature_recovery_policy": spec["protocol"]["mature_recovery_policy"],
    })
    if lock.get("candidate_library_sha256") != expected_candidate_hash or (
            lock.get("policy_bundle_sha256") != expected_policy_hash):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock candidate/policy hash mismatch")
    if lock.get("candidate_order") != list(V3_CANDIDATE_NAMES):
        raise ClosedLoopRecoveryTriageError("selection-lock candidate order drifted")
    if lock.get("bootstrap") != spec["protocol"]["statistics"]["bootstrap"] or (
            lock.get("triage_gates") != spec["protocol"]["triage_gates"]):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock statistics or gates drifted")
    data_gate = _mapping(lock.get("data_gate"), "selection_lock.data_gate")
    if data_gate.get("structural_contract_pass") is not True or data_gate.get(
            "pass") is not True or lock.get("audit_authorized") is not True:
        raise ClosedLoopRecoveryTriageError(
            "selection lock did not authorize opening audit outcomes")
    expected_data_values = {
        "independent_groups": spec["groups"],
        "unique_state_fingerprints": spec["groups"],
        "unique_trajectory_fingerprints": spec["groups"],
        "groups_per_source_seed": spec["groups_per_seed"],
        "required_source_seeds": list(spec["required_seeds"]),
        "candidates": len(V3_CANDIDATE_NAMES),
        "admission_replicas": spec["admission_replicas"],
        "discovery_replicas": spec["discovery_replicas"],
        "audit_replicas_preassigned": spec["audit_replicas"],
        "horizon_policy_steps": spec["horizon"],
    }
    if any(data_gate.get(key) != value
           for key, value in expected_data_values.items()):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock exact data-gate values drifted")
    if lock.get("audit_runner_up_policy") != "forbidden":
        raise ClosedLoopRecoveryTriageError(
            "selection-lock runner-up policy drifted")
    inputs = _mapping(lock.get("input_artifacts"), "input_artifacts")
    for role, filename in (
        ("admission", spec["collection"]["admission_deployable_filename"]),
        ("discovery", spec["collection"]["discovery_filename"]),
    ):
        artifact = _mapping(inputs.get(role), f"input_artifacts.{role}")
        if artifact.get("filename") != filename or any(
                _HEX64.fullmatch(str(artifact.get(field, ""))) is None
                for field in ("file_sha256", "content_sha256")):
            raise ClosedLoopRecoveryTriageError(
                f"selection-lock {role} artifact binding is invalid")
    global_choice = _mapping(
        lock.get("selected_global_candidate"), "selected_global_candidate")
    global_index = _integer(
        global_choice.get("candidate_index"), "global candidate index", minimum=1)
    if global_index >= len(V3_CANDIDATE_NAMES) or global_choice.get(
            "candidate_name") != V3_CANDIDATE_NAMES[global_index]:
        raise ClosedLoopRecoveryTriageError(
            "selection-lock global candidate is invalid")
    if global_choice.get("selection_scope") != (
            "eight_nonnominal_candidates") or global_choice.get(
                "exact_tie_break") != "locked_candidate_order":
        raise ClosedLoopRecoveryTriageError(
            "selection-lock global selection rule drifted")
    groups = lock.get("group_selection")
    partition = lock.get("replica_partition")
    if not isinstance(groups, list) or len(groups) != spec["groups"] or not (
            isinstance(partition, list) and len(partition) == spec["groups"]):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock group/replica records have the wrong length")
    if canonical_sha256(groups) != lock.get("group_selection_sha256") or (
            canonical_sha256(partition) != lock.get("replica_partition_sha256")):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock group or replica digest mismatch")
    readiness_manifest = _mapping(
        lock.get("collection_readiness_manifest"),
        "selection_lock.collection_readiness_manifest",
    )
    if readiness_manifest.get("schema_version") != (
            COLLECTION_READINESS_SCHEMA_VERSION) or readiness_manifest.get(
                "protocol_name") != spec["protocol"]["protocol_name"] or (
                    readiness_manifest.get("protocol_contract_sha256") !=
                    spec["protocol_contract_sha256"]) or canonical_sha256(
                        readiness_manifest) != lock.get(
                            "collection_readiness_sha256"):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock collection readiness digest mismatch")
    if readiness_manifest.get("protocol_file_sha256") != lock.get(
            "protocol_file_sha256") or readiness_manifest.get(
                "generator_commit") != lock.get("generator_commit") or (
                    readiness_manifest.get("required_source_seeds") !=
                    list(spec["required_seeds"])):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock readiness provenance differs from the lock")
    roles = _mapping(
        readiness_manifest.get("role_commitments"),
        "collection_readiness_manifest.role_commitments",
    )
    if set(roles) != set(_SHARD_OUTPUT_TEMPLATE_KEYS):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock readiness role commitments are incomplete")
    expected_audit = lock.get("expected_audit_shards")
    if not isinstance(expected_audit, list) or expected_audit != roles.get(
            "audit"):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock audit commitments differ from readiness manifest")
    merge_manifest = _mapping(
        lock.get("merge_readiness_manifest"),
        "selection_lock.merge_readiness_manifest",
    )
    if merge_manifest.get("schema_version") != (
            "qsafe.closed_loop_recovery_triage.merge_readiness.v1") or (
                merge_manifest.get("protocol_contract_sha256") !=
                spec["protocol_contract_sha256"]) or merge_manifest.get(
                    "collection_readiness_sha256") != lock.get(
                        "collection_readiness_sha256") or canonical_sha256(
                            merge_manifest) != lock.get(
                                "merge_readiness_sha256"):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock merge readiness digest mismatch")
    for role, key in (
        ("admission", "admission_merge_report"),
        ("discovery", "discovery_merge_report"),
    ):
        merge_record = _mapping(
            merge_manifest.get(key), f"merge_readiness_manifest.{key}")
        artifact = _mapping(inputs.get(role), f"input_artifacts.{role}")
        if merge_record.get("output_file_sha256") != artifact.get(
                "file_sha256") or merge_record.get(
                    "output_content_sha256") != artifact.get(
                        "content_sha256"):
            raise ClosedLoopRecoveryTriageError(
                f"selection-lock {role} merge/output binding drifted")
    if not isinstance(lock.get("audit_identifier"), str) or _HEX64.fullmatch(
            lock["audit_identifier"]) is None:
        raise ClosedLoopRecoveryTriageError("invalid one-shot audit identifier")

    risks = np.empty((spec["groups"], len(V3_CANDIDATE_NAMES)), dtype=np.float64)
    source_seed = np.empty(spec["groups"], dtype=np.int64)
    group_id: list[str] = []
    state: list[str] = []
    trajectory: list[str] = []
    policy_age = np.empty(spec["groups"], dtype=np.int64)
    winners: list[np.ndarray] = []
    seed_arrays = {
        key: np.empty((spec["groups"], count), dtype=np.int64)
        for key, count in (
            ("admission_crn_ids", spec["admission_replicas"]),
            ("admission_rollout_seeds", spec["admission_replicas"]),
            ("admission_perturbation_seeds", spec["admission_replicas"]),
            ("discovery_crn_ids", spec["discovery_replicas"]),
            ("discovery_rollout_seeds", spec["discovery_replicas"]),
            ("discovery_perturbation_seeds", spec["discovery_replicas"]),
            ("audit_crn_ids", spec["audit_replicas"]),
            ("audit_rollout_seeds", spec["audit_replicas"]),
            ("audit_perturbation_seeds", spec["audit_replicas"]),
        )
    }
    candidate_seed_arrays = {
        "discovery_candidate_seed": np.empty(spec["groups"], dtype=np.int64),
        "audit_candidate_seed": np.empty(spec["groups"], dtype=np.int64),
    }
    admission_lower, admission_upper = map(
        int, spec["data_gate"]["admission_falls_inclusive"])
    for index, (group_value, partition_value) in enumerate(zip(groups, partition)):
        group = _mapping(group_value, f"group_selection[{index}]")
        seed_record = _mapping(partition_value, f"replica_partition[{index}]")
        if group.get("group_index") != index or seed_record.get(
                "group_index") != index or group.get("group_id") != seed_record.get(
                    "group_id"):
            raise ClosedLoopRecoveryTriageError(
                "selection-lock group indices/order drifted")
        group_id.append(str(group.get("group_id")))
        state.append(str(group.get("state_hash")))
        trajectory.append(str(group.get("trajectory_id")))
        source_seed[index] = _integer(
            group.get("source_seed"), "source_seed", minimum=0)
        policy_age[index] = _integer(
            group.get("policy_age"), "policy_age", minimum=1)
        admission_falls = _integer(
            group.get("admission_falls"), "admission_falls", minimum=0)
        if not admission_lower <= admission_falls <= admission_upper:
            raise ClosedLoopRecoveryTriageError(
                "selection-lock group falls outside admission bounds")
        row = np.asarray(group.get("discovery_candidate_risk"), dtype=np.float64)
        if row.shape != (len(V3_CANDIDATE_NAMES),) or not np.all(
                np.isfinite(row)) or np.any(row < 0.0) or np.any(row > 1.0):
            raise ClosedLoopRecoveryTriageError(
                "selection-lock discovery risks are invalid")
        risks[index] = row
        raw_winners = group.get("discovery_minimizer_indices")
        if not isinstance(raw_winners, list) or not raw_winners:
            raise ClosedLoopRecoveryTriageError(
                "selection-lock winner set is empty")
        winner = np.asarray(raw_winners, dtype=np.int64)
        expected_winner = np.flatnonzero(row == np.min(row))
        if not np.array_equal(winner, expected_winner):
            raise ClosedLoopRecoveryTriageError(
                "selection-lock winner set disagrees with discovery risks")
        if group.get("discovery_minimizer_names") != [
                V3_CANDIDATE_NAMES[item] for item in winner]:
            raise ClosedLoopRecoveryTriageError(
                "selection-lock winner names disagree with indices")
        weights = np.asarray(group.get("uniform_weights"), dtype=np.float64)
        if weights.shape != winner.shape or not np.allclose(
                weights, np.full(len(winner), 1.0 / len(winner)),
                rtol=0.0, atol=1e-15):
            raise ClosedLoopRecoveryTriageError(
                "selection-lock ties are not uniformly weighted")
        winners.append(winner)
        for key, target in seed_arrays.items():
            count = target.shape[1]
            values = np.asarray(seed_record.get(key))
            if values.shape != (count,) or values.dtype.kind not in "iu" or np.any(
                    values < 0) or len(np.unique(values)) != count:
                raise ClosedLoopRecoveryTriageError(
                    f"selection-lock {key} is invalid")
            target[index] = values
        for key, target in candidate_seed_arrays.items():
            target[index] = _integer(
                seed_record.get(key), f"selection-lock {key}", minimum=0)
    text_state = np.asarray(state, dtype=str)
    text_trajectory = np.asarray(trajectory, dtype=str)
    if any(len(set(values)) != spec["groups"] for values in (
            group_id, state, trajectory)):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock identities are not unique")
    _validate_fingerprints(text_state, "state_hash")
    if set(map(int, source_seed)) != set(spec["required_seeds"]) or any(
            np.count_nonzero(source_seed == seed) != spec["groups_per_seed"]
            for seed in spec["required_seeds"]):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock source-seed composition drifted")
    if any(policy_age[index] != spec["seed_age"][int(source_seed[index])]
           for index in range(spec["groups"])):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock source seed/policy age binding drifted")
    if any(len(np.unique(value)) != spec["groups"]
           for value in candidate_seed_arrays.values()) or not (
               _disjoint_seed_domains(
                   *seed_arrays.values(), *candidate_seed_arrays.values())):
        raise ClosedLoopRecoveryTriageError(
            "selection-lock seed domains overlap")
    recomputed_global, _ = _global_discovery_selection(
        risks, source_seed, spec["required_seeds"])
    if recomputed_global != global_index:
        raise ClosedLoopRecoveryTriageError(
            "selection-lock global choice disagrees with discovery risks")
    informativeness = _discovery_informativeness(
        risks[:, 0], source_seed, spec)
    if not informativeness["pass"]:
        raise ClosedLoopRecoveryTriageError(
            "selection-lock discovery informativeness no longer passes")
    if data_gate.get("discovery_informativeness") != informativeness:
        raise ClosedLoopRecoveryTriageError(
            "selection-lock discovery informativeness record drifted")
    lock["_validated"] = {
        "global_index": global_index,
        "group_id": np.asarray(group_id, dtype=str),
        "state_hash": text_state,
        "trajectory_id": text_trajectory,
        "source_seed": source_seed,
        "policy_age": policy_age,
        "discovery_risk": risks,
        "winners": winners,
        "audit_crn_id": seed_arrays["audit_crn_ids"],
        "audit_rollout_seed": seed_arrays["audit_rollout_seeds"],
        "audit_perturbation_seed": seed_arrays[
            "audit_perturbation_seeds"],
        "audit_candidate_seed": candidate_seed_arrays[
            "audit_candidate_seed"],
    }
    return lock


def _hierarchical_bootstrap(
    group_metrics: np.ndarray,
    source_seed: np.ndarray,
    age_strata: Mapping[int, Sequence[int]],
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 512,
) -> np.ndarray:
    """Preserve each complete group metric vector in hierarchical draws."""
    metrics = np.asarray(group_metrics, dtype=np.float64)
    if metrics.ndim != 2 or metrics.shape[0] != len(source_seed) or not np.all(
            np.isfinite(metrics)):
        raise ClosedLoopRecoveryTriageError(
            "bootstrap metrics must be finite [G,M]")
    replicates = _integer(replicates, "bootstrap_replicates", minimum=1)
    seed = _integer(seed, "bootstrap_seed", minimum=0)
    chunk_size = _integer(chunk_size, "bootstrap_chunk_size", minimum=1)
    seed_groups = {
        int(source): np.flatnonzero(source_seed == source)
        for seeds in age_strata.values() for source in seeds
    }
    group_counts = {len(indices) for indices in seed_groups.values()}
    if len(group_counts) != 1 or next(iter(group_counts)) == 0:
        raise ClosedLoopRecoveryTriageError(
            "hierarchical bootstrap requires equal nonempty groups per seed")
    groups_per_seed = next(iter(group_counts))
    source_slots = sum(len(seeds) for seeds in age_strata.values())
    draws = np.empty((replicates, metrics.shape[1]), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        count = stop - start
        chunk = np.zeros((count, metrics.shape[1]), dtype=np.float64)
        for age in sorted(age_strata):
            seeds_value = age_strata[age]
            seeds = np.asarray(seeds_value, dtype=np.int64)
            # Locked RNG call order: chunk, increasing policy age, source-seed
            # slot, then that slot's complete group-index matrix.  Drawing one
            # slot at a time avoids an implicit ndarray flattening convention.
            for slot in range(len(seeds)):
                selected_seeds = seeds[rng.integers(
                    0, len(seeds), size=count)]
                group_positions = rng.integers(
                    0, groups_per_seed, size=(count, groups_per_seed))
                absolute = np.empty_like(group_positions)
                for row, selected_seed in enumerate(selected_seeds):
                    absolute[row] = seed_groups[int(selected_seed)][
                        group_positions[row]]
                # Indexing one complete [M] vector preserves all K9 CRN
                # correlations needed by the simultaneous max-error band.
                chunk += np.mean(metrics[absolute], axis=1, dtype=np.float64)
        draws[start:stop] = chunk / float(source_slots)
    return draws


def _pair_agreement_groups(
    discovery_risk: np.ndarray,
    audit_risk: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    groups, candidates = discovery_risk.shape
    scores = np.zeros(groups, dtype=np.float64)
    comparisons = 0
    tie_comparisons = 0
    for left in range(candidates):
        for right in range(left + 1, candidates):
            discovery_delta = discovery_risk[:, left] - discovery_risk[:, right]
            audit_delta = audit_risk[:, left] - audit_risk[:, right]
            tied = (discovery_delta == 0.0) | (audit_delta == 0.0)
            pair_score = np.where(
                tied,
                0.5,
                (np.sign(discovery_delta) == np.sign(audit_delta)).astype(
                    np.float64),
            )
            scores += pair_score
            comparisons += groups
            tie_comparisons += int(np.count_nonzero(tied))
    pairs_per_group = candidates * (candidates - 1) // 2
    return scores / float(pairs_per_group), comparisons, tie_comparisons


def _load_audit_shards_after_consumption(
    paths: Sequence[Path],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and splice six already-consumed physical audit outcome shards."""
    shards = [
        _load_outcome_npz(
            path,
            "audit",
            spec,
            expected_groups=spec["groups_per_seed"],
            expected_source_seed=int(seed),
        )
        for path, seed in zip(paths, spec["required_seeds"], strict=True)
    ]
    if len({shard["generator_commit"] for shard in shards}) != 1 or len({
            shard["protocol_file_sha256"] for shard in shards}) != 1:
        raise ClosedLoopRecoveryTriageError(
            "audit shards disagree on generator commit or protocol file hash")
    vector_names = (
        "group_id", "state_hash", "trajectory_id", "source_seed", "policy_age")
    matrix_names = (
        "crn_id", "rollout_seed", "perturbation_seed", "candidate_seed", "fall")
    result = {
        name: np.concatenate([shard[name] for shard in shards], axis=0)
        for name in (*vector_names, *matrix_names)
    }
    result.update({
        "generator_commit": shards[0]["generator_commit"],
        "protocol_file_sha256": shards[0]["protocol_file_sha256"],
        "file_sha256": [shard["file_sha256"] for shard in shards],
        "content_sha256": [shard["content_sha256"] for shard in shards],
    })
    return result


def consume_and_evaluate_audit(
    *,
    protocol: Mapping[str, Any],
    selection_lock_path: str | os.PathLike[str],
    expected_selection_lock_sha256: str,
    audit_paths: Sequence[str | os.PathLike[str]],
    audit_consumed_path: str | os.PathLike[str],
    expected_generator_commit: str | None = None,
    expected_protocol_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Consume audit once, then compute the preregistered v3 decision.

    There is no runtime bootstrap override surface: formal analysis always uses
    the protocol-locked B=50000, PCG64 seed, chunk size, and quantile method.
    """
    spec = _validate_protocol(protocol)
    collection = spec["collection"]
    lock_file = _artifact_path(
        selection_lock_path,
        protocol=protocol,
        expected_filename=str(collection["selection_lock_filename"]),
        name="selection_lock_path",
    )
    # The exact externally supplied hash is checked before any marker or audit
    # access.  Merely choosing a different lock cannot silently choose a new
    # candidate after discovery.
    lock = _read_selection_lock(
        lock_file, expected_selection_lock_sha256, spec)
    if expected_generator_commit is not None:
        _validate_commit(expected_generator_commit, "expected_generator_commit")
        if lock.get("generator_commit") != expected_generator_commit:
            raise ClosedLoopRecoveryTriageError(
                "selection lock differs from the current clean commit")
    if expected_protocol_file_sha256 is not None:
        if _HEX64.fullmatch(expected_protocol_file_sha256) is None or lock.get(
                "protocol_file_sha256") != expected_protocol_file_sha256:
            raise ClosedLoopRecoveryTriageError(
                "selection lock differs from the canonical protocol file")
    # Pure lexical comparison only: no audit shard is stat'ed, hashed, opened,
    # or merged before the irreversible marker below.
    audit_files = _locked_audit_paths_before_consumption(
        audit_paths, protocol=protocol, spec=spec, lock=lock)
    consumed_file = _artifact_path(
        audit_consumed_path,
        protocol=protocol,
        expected_filename=str(collection["audit_consumed_filename"]),
        name="audit_consumed_path",
    )
    if os.path.lexists(consumed_file):
        raise ClosedLoopRecoveryTriageError(
            "refusing to reuse or clobber an existing audit-consumed marker")
    marker = {
        "schema_version": AUDIT_CONSUMED_SCHEMA_VERSION,
        "protocol_name": spec["protocol"]["protocol_name"],
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": lock["protocol_file_sha256"],
        "selection_lock_sha256": expected_selection_lock_sha256,
        "audit_identifier": lock["audit_identifier"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "irreversibly_consumed_before_outcome_read",
    }
    # This is the one-way boundary.  No open/hash/load of any audit shard occurs
    # above this line.  O_EXCL-equivalent hard-link publication leaves the
    # marker in place after all subsequent failures and interruptions.
    marker_sha256 = _atomic_no_clobber_json(consumed_file, marker)

    audit = _load_audit_shards_after_consumption(audit_files, spec)
    validated = lock["_validated"]
    if audit["file_sha256"] != [
            record["file_sha256"] for record in lock["expected_audit_shards"]
    ] or audit["content_sha256"] != [
            record["content_sha256"] for record in lock["expected_audit_shards"]
    ]:
        raise ClosedLoopRecoveryTriageError(
            "audit shard hash differs from the report-last selection commitment")
    if audit["generator_commit"] != lock["generator_commit"]:
        raise ClosedLoopRecoveryTriageError(
            "audit generator commit differs from the selection lock")
    if audit["protocol_file_sha256"] != lock["protocol_file_sha256"]:
        raise ClosedLoopRecoveryTriageError(
            "audit protocol-file hash differs from the selection lock")
    if not _same_identities(validated, audit):
        raise ClosedLoopRecoveryTriageError(
            "audit group identities/order differ from the selection lock")
    if not all(np.array_equal(audit[observed], validated[locked])
               for observed, locked in (
                   ("crn_id", "audit_crn_id"),
                   ("rollout_seed", "audit_rollout_seed"),
                   ("perturbation_seed", "audit_perturbation_seed"),
                   ("candidate_seed", "audit_candidate_seed"),
               )):
        raise ClosedLoopRecoveryTriageError(
            "audit CRN/rollout/perturbation seeds differ from the preassigned lock")

    audit_risk = np.mean(audit["fall"], axis=2, dtype=np.float64)
    discovery_risk = validated["discovery_risk"]
    source_seed = audit["source_seed"]
    global_index = int(validated["global_index"])
    nominal_group = audit_risk[:, 0]
    fixed_effect_groups = np.stack([
        nominal_group - audit_risk[:, candidate]
        for candidate in range(1, len(V3_CANDIDATE_NAMES))
    ], axis=1)
    conditional_risk = np.asarray([
        float(np.mean(audit_risk[group, winners], dtype=np.float64))
        for group, winners in enumerate(validated["winners"])
    ], dtype=np.float64)
    conditional_effect_group = nominal_group - conditional_risk
    global_effect_group = fixed_effect_groups[:, global_index - 1]
    incremental_group = conditional_effect_group - global_effect_group
    pair_group, pair_comparisons, pair_ties = _pair_agreement_groups(
        discovery_risk, audit_risk)
    winner_count = np.asarray(
        [len(winners) for winners in validated["winners"]], dtype=np.float64)
    tie_indicator = (winner_count > 1.0).astype(np.float64)

    metric_names = [
        "audit_nominal_risk",
        *[f"fixed_effect_{candidate}" for candidate in range(1, 9)],
        "conditional_effect",
        "conditional_increment_over_global",
        "pair_agreement",
    ]
    metric_matrix = np.column_stack((
        nominal_group,
        fixed_effect_groups,
        conditional_effect_group,
        incremental_group,
        pair_group,
    ))
    bootstrap_protocol = spec["protocol"]["statistics"]["bootstrap"]
    protocol_replicates = int(bootstrap_protocol["replicates"])
    protocol_seed = int(bootstrap_protocol["seed"])
    protocol_chunk_size = int(bootstrap_protocol["chunk_size"])
    quantile_method = str(bootstrap_protocol["quantile_method"])
    draws = _hierarchical_bootstrap(
        metric_matrix,
        source_seed,
        spec["age_strata"],
        replicates=protocol_replicates,
        seed=protocol_seed,
        chunk_size=protocol_chunk_size,
    )
    estimates = np.asarray([
        _equal_seed_mean(metric_matrix[:, metric], source_seed,
                         spec["required_seeds"])
        for metric in range(metric_matrix.shape[1])
    ], dtype=np.float64)
    lower = np.quantile(
        draws, 0.05, axis=0, method=quantile_method)
    global_metric = global_index  # metric 0 is nominal; fixed k is metric k.
    conditional_metric = 9
    incremental_metric = 10
    pair_metric = 11

    gates = spec["protocol"]["triage_gates"]
    primary_gate = gates["primary_global_backup"]
    global_seed_effects = _seed_effects(
        global_effect_group, source_seed, spec["required_seeds"])
    global_age_effects = _age_effects(
        global_effect_group, source_seed, spec["age_strata"])
    primary_checks = {
        "audit_absolute_reduction": bool(
            estimates[global_metric]
            >= float(primary_gate["min_audit_absolute_reduction"])),
        "one_sided_95_lcb_reduction": bool(
            lower[global_metric]
            >= float(primary_gate["min_one_sided_95_lcb_reduction"])),
        "each_policy_age_positive": bool(
            not primary_gate["require_each_policy_age_positive"] or all(
                value > 0.0 for value in global_age_effects.values())),
        "positive_source_seeds": bool(sum(
            value > 0.0 for value in global_seed_effects.values())
            >= int(primary_gate["min_positive_source_seeds"])),
    }
    primary_pass = bool(all(primary_checks.values()))

    conditional_gate = gates["conditional_state_dependent"]
    conditional_seed_increment = _seed_effects(
        incremental_group, source_seed, spec["required_seeds"])
    conditional_age_increment = _age_effects(
        incremental_group, source_seed, spec["age_strata"])
    tie_fraction = _equal_seed_mean(
        tie_indicator, source_seed, spec["required_seeds"])
    mean_winners = _equal_seed_mean(
        winner_count, source_seed, spec["required_seeds"])
    conditional_tested = bool(primary_pass)
    conditional_checks: dict[str, bool] | None = None
    conditional_pass: bool | None = None
    if conditional_tested:
        incremental_lcb_threshold = float(
            conditional_gate["min_incremental_one_sided_95_lcb"])
        if conditional_gate["incremental_lcb_strictly_greater"]:
            incremental_lcb_check = lower[incremental_metric] > (
                incremental_lcb_threshold)
        else:
            incremental_lcb_check = lower[incremental_metric] >= (
                incremental_lcb_threshold)
        conditional_checks = {
            "audit_absolute_reduction": bool(
                estimates[conditional_metric]
                >= float(conditional_gate["min_audit_absolute_reduction"])),
            "one_sided_95_lcb_reduction": bool(
                lower[conditional_metric]
                >= float(conditional_gate["min_one_sided_95_lcb_reduction"])),
            "incremental_reduction_over_global": bool(
                estimates[incremental_metric] >= float(
                    conditional_gate[
                        "min_incremental_reduction_over_global"])),
            "incremental_one_sided_95_lcb": bool(incremental_lcb_check),
            "discovery_to_audit_pair_agreement": bool(
                estimates[pair_metric] >= float(conditional_gate[
                    "min_discovery_to_audit_pair_agreement"])),
            "pair_agreement_one_sided_95_lcb": bool(
                lower[pair_metric] >= float(conditional_gate[
                    "min_pair_agreement_one_sided_95_lcb"])),
            "discovery_minimizer_tie_fraction": bool(
                tie_fraction <= float(conditional_gate[
                    "max_discovery_minimizer_tie_fraction"])),
            "mean_winner_count": bool(
                mean_winners <= float(conditional_gate[
                    "max_mean_winner_count"])),
            "each_policy_age_increment_positive": bool(
                not conditional_gate[
                    "require_each_policy_age_increment_positive"] or all(
                        value > 0.0
                        for value in conditional_age_increment.values())),
            "increment_positive_source_seeds": bool(sum(
                value > 0.0 for value in conditional_seed_increment.values())
                >= int(conditional_gate[
                    "min_increment_positive_source_seeds"])),
        }
        conditional_pass = bool(all(conditional_checks.values()))

    # Joint vector: eight fixed recovery candidates followed by the locked
    # per-state discovery rule.  The same bootstrap group indices produced all
    # columns, preserving K9 CRN dependence in the max-centered-error band.
    joint_estimates = np.concatenate((estimates[1:9], [
        estimates[conditional_metric]]))
    joint_draws = np.column_stack((draws[:, 1:9], draws[:, conditional_metric]))
    max_centered_error = np.max(
        joint_estimates[None, :] - joint_draws, axis=1)
    common_critical_value = float(np.quantile(
        max_centered_error, 0.95, method=quantile_method))
    simultaneous_ucb = joint_estimates + common_critical_value
    no_headroom_gate = gates["no_headroom"]
    nominal_opportunity = bool(
        lower[0]
        >= float(no_headroom_gate[
            "min_audit_nominal_risk_one_sided_95_lcb"]))
    all_ucb_below = bool(np.all(
        simultaneous_ucb
        < float(no_headroom_gate["require_every_simultaneous_effect_ucb_below"])))
    no_headroom_criterion = bool(nominal_opportunity and all_ucb_below)
    no_headroom_fires = bool(not primary_pass and no_headroom_criterion)

    if primary_pass and conditional_pass:
        decision = "preregister_fresh_option_ranking_qsafe_protocol"
    elif primary_pass:
        decision = "preregister_fresh_state_risk_qsafe_plus_fixed_backup_protocol"
    elif no_headroom_fires:
        decision = "redesign_recovery_library_before_model_training"
    else:
        decision = "report_inconclusive_no_model_training"

    fixed_results = []
    for candidate in range(1, len(V3_CANDIDATE_NAMES)):
        fixed_results.append({
            "candidate_index": candidate,
            "candidate_name": V3_CANDIDATE_NAMES[candidate],
            "audit_absolute_reduction": float(estimates[candidate]),
            "one_sided_95_lcb": float(lower[candidate]),
            "source_seed_effects": _seed_effects(
                fixed_effect_groups[:, candidate - 1], source_seed,
                spec["required_seeds"]),
            "policy_age_effects": _age_effects(
                fixed_effect_groups[:, candidate - 1], source_seed,
                spec["age_strata"]),
        })
    joint_labels = [*V3_CANDIDATE_NAMES[1:], "locked_per_state_rule"]
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_name": spec["protocol"]["protocol_name"],
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": lock["protocol_file_sha256"],
        "selection_lock_sha256": expected_selection_lock_sha256,
        "audit_identifier": lock["audit_identifier"],
        "audit_consumed_marker_sha256": marker_sha256,
        "audit_shard_file_sha256": audit["file_sha256"],
        "audit_shard_content_sha256": audit["content_sha256"],
        "estimand": (
            "equal_groups_within_source_seed_then_equal_six_source_seeds; "
            "conditional_on_admission_positive_fixed_states"),
        "pair_agreement_contract": _json_copy(
            spec["protocol"]["statistics"]["pair_agreement"],
            "pair_agreement_contract",
        ),
        "data_gate": {
            "pass": True,
            "groups": spec["groups"],
            "groups_per_source_seed": spec["groups_per_seed"],
            "required_source_seeds": list(spec["required_seeds"]),
            "candidates": len(V3_CANDIDATE_NAMES),
            "audit_replicas": spec["audit_replicas"],
            "horizon_policy_steps": spec["horizon"],
            "discovery_informativeness": lock["data_gate"][
                "discovery_informativeness"],
        },
        "bootstrap": {
            "kind": bootstrap_protocol["kind"],
            "protocol_replicates": protocol_replicates,
            "protocol_seed": protocol_seed,
            "rng_bit_generator": bootstrap_protocol["rng_bit_generator"],
            "chunk_size": protocol_chunk_size,
            "draw_order": bootstrap_protocol["draw_order"],
            "quantile_method": quantile_method,
            "replicates_used": protocol_replicates,
            "seed_used": protocol_seed,
            "override_used": False,
            "confidence": "one_sided_0.95",
            "metric_columns": metric_names,
        },
        "audit_nominal_risk": {
            "equal_seed_estimate": float(estimates[0]),
            "one_sided_95_lcb": float(lower[0]),
            "source_seed_risks": _seed_effects(
                nominal_group, source_seed, spec["required_seeds"]),
            "policy_age_risks": _age_effects(
                nominal_group, source_seed, spec["age_strata"]),
        },
        "selected_global_candidate": {
            "candidate_index": global_index,
            "candidate_name": V3_CANDIDATE_NAMES[global_index],
            "audit_runner_up_policy": "forbidden",
        },
        "fixed_candidate_effects": fixed_results,
        "primary_global_backup": {
            "audit_absolute_reduction": float(estimates[global_metric]),
            "one_sided_95_lcb": float(lower[global_metric]),
            "source_seed_effects": global_seed_effects,
            "policy_age_effects": global_age_effects,
            "checks": primary_checks,
            "pass": primary_pass,
        },
        "conditional_state_dependent": {
            "tested": conditional_tested,
            "audit_absolute_reduction": float(estimates[conditional_metric]),
            "one_sided_95_lcb": float(lower[conditional_metric]),
            "incremental_reduction_over_global": float(
                estimates[incremental_metric]),
            "incremental_one_sided_95_lcb": float(lower[incremental_metric]),
            "pair_agreement": float(estimates[pair_metric]),
            "pair_agreement_one_sided_95_lcb": float(lower[pair_metric]),
            "pair_comparisons": pair_comparisons,
            "pair_tie_comparisons": pair_ties,
            "discovery_minimizer_tie_fraction": tie_fraction,
            "mean_winner_count": mean_winners,
            "increment_source_seed_effects": conditional_seed_increment,
            "increment_policy_age_effects": conditional_age_increment,
            "checks": conditional_checks,
            "pass": conditional_pass if conditional_tested else None,
        },
        "no_headroom": {
            "joint_effect_labels": joint_labels,
            "joint_effect_estimates": joint_estimates.astype(float).tolist(),
            "common_critical_value": common_critical_value,
            "simultaneous_one_sided_95_ucb": simultaneous_ucb.astype(
                float).tolist(),
            "candidate_ucb": {
                label: float(value)
                for label, value in zip(joint_labels, simultaneous_ucb)
            },
            "audit_nominal_risk_one_sided_95_lcb": float(lower[0]),
            "checks": {
                "nominal_opportunity_lcb": nominal_opportunity,
                "every_simultaneous_effect_ucb_strictly_below_threshold": (
                    all_ucb_below),
            },
            "criterion_met": no_headroom_criterion,
            "fires": no_headroom_fires,
            "scope_limit": no_headroom_gate["scope_limit"],
        },
        "decision": decision,
        "model_training_authorized": False,
        "selector_calibration_authorized": False,
        "paired_closed_loop_authorized": False,
        "online_training_authorized": False,
        "phase2_authorized": False,
        "authorization_note": (
            "A triage pass authorizes only preregistration on wholly fresh data; "
            "this consumed development audit is not an Objective-1 result."),
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


__all__ = [
    "AUDIT_CONSUMED_SCHEMA_VERSION",
    "COLLECTION_READINESS_SCHEMA_VERSION",
    "ClosedLoopRecoveryTriageError",
    "REPORT_SCHEMA_VERSION",
    "SELECTION_LOCK_SCHEMA_VERSION",
    "V3_BEHAVIOR_STEPS",
    "V3_CANDIDATE_NAMES",
    "V3_SOURCE_SEEDS",
    "canonical_sha256",
    "consume_and_evaluate_audit",
    "create_selection_lock",
    "validate_closed_loop_recovery_protocol",
    "validate_collection_readiness",
]
