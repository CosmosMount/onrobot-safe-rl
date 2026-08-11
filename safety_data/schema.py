"""Versioned group-centric schema for same-state Q_safe branches."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from safety_data.paths import (
    ProtectedEvidencePathError,
    assert_development_path,
    assert_safe_evidence_output,
    require_workflow_authorized_or_safe_input,
)


SCHEMA_VERSION = "qsafe.grouped.v1"
PRIVILEGED_SCHEMA_VERSION = "qsafe.privileged.v1"
OBSERVATION_FRAMES = 5
OBSERVATION_DIM = 46
ACTION_DIM = 12
INDEPENDENT_REPLICA_PARTITION_VERSION = (
    "qsafe.independent_replica_partition.v2")
RECOVERY_OPTION_CANDIDATE_PROTOCOL_VERSION = (
    "qsafe.recovery_option_candidates.v2")
RECOVERY_OPTION_CANDIDATE_COUNT = 29
CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION = (
    "qsafe.closed_loop_recovery_behaviors.v3")
CLOSED_LOOP_RECOVERY_CANDIDATE_COUNT = 9
CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS = (
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
CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS = (0, 10, 25, 50, 10, 10, 25, 25, 25)

REQUIRED_ARRAYS = (
    "group_id",
    "state_hash",
    "trajectory_id",
    "episode_id",
    "episode_step",
    "policy_training_seed",
    "source_seed",
    "policy_source",
    "command_vx",
    "acceptance_probability",
    "obs_history",
    "q_send_history",
    "nominal_action_requested",
    "candidate_requested",
    "candidate_executed",
    "candidate_q_target",
    "candidate_kind",
    "candidate_mask",
    "fall",
    "first_failure_step",
    "max_tilt_rad",
    "min_height_m",
    "crn_id",
    "rollout_seed",
    "perturbation_seed",
)

REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "split",
    "feature_view",
    "horizon_steps",
    "generator_commit",
    "simulator_fingerprint",
    "source_policy",
    "continuation_policy",
    "candidate_protocol",
    "fall_definition",
    "observation_contract",
    "action_application_contract",
    "state_hash_contract",
)


class DatasetValidationError(ValueError):
    """The grouped dataset cannot be used without violating its contract."""


def _authorize_evidence_input(
    path: str | Path,
    *,
    allowed_roles: tuple[str, ...],
) -> Path:
    try:
        return require_workflow_authorized_or_safe_input(
            path, allowed_roles=allowed_roles)
    except ProtectedEvidencePathError as exc:
        raise DatasetValidationError(str(exc)) from exc


def _validate_optional_replica_partition(
    manifest: Mapping[str, Any],
    *,
    replica_count: int,
) -> dict[str, Any] | None:
    """Validate a pre-outcome discovery/audit replica-axis contract.

    Legacy grouped datasets intentionally remain loadable without a partition,
    but an artifact that declares one is checked at the schema boundary.  The
    label-reliability gate separately requires this contract and therefore
    refuses legacy datasets.
    """
    collection = manifest.get("collection_protocol")
    if collection is None:
        return None
    if not isinstance(collection, Mapping):
        raise DatasetValidationError("collection_protocol must be a mapping")
    partition = collection.get("replica_partition")
    if partition is None:
        return None
    expected_keys = {
        "schema_version", "assignment_timing", "axis", "ordering",
        "discovery_indices", "audit_indices", "discovery_replicas",
        "audit_replicas", "exhaustive",
    }
    if not isinstance(partition, Mapping) or set(partition) != expected_keys:
        raise DatasetValidationError(
            "replica_partition must contain exactly the v2 contract fields")
    expected_literals = {
        "schema_version": INDEPENDENT_REPLICA_PARTITION_VERSION,
        "assignment_timing": "before_candidate_outcomes",
        "axis": "replica",
        "ordering": "discovery_then_audit",
    }
    for name, expected in expected_literals.items():
        if partition[name] != expected:
            raise DatasetValidationError(
                f"replica_partition.{name}={partition[name]!r}, "
                f"expected {expected!r}")
    if partition["exhaustive"] is not True:
        raise DatasetValidationError("replica_partition.exhaustive must be true")

    def indices(name: str) -> list[int]:
        raw = partition[name]
        if not isinstance(raw, list) or not raw or any(
                isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)) for value in raw):
            raise DatasetValidationError(
                f"replica_partition.{name} must be a nonempty integer list")
        result = [int(value) for value in raw]
        if result != sorted(set(result)) or result[0] < 0:
            raise DatasetValidationError(
                f"replica_partition.{name} must be unique and sorted")
        return result

    discovery = indices("discovery_indices")
    audit = indices("audit_indices")
    counts: dict[str, int] = {}
    for name in ("discovery_replicas", "audit_replicas"):
        raw = partition[name]
        if isinstance(raw, (bool, np.bool_)) or not isinstance(
                raw, (int, np.integer)) or int(raw) <= 0:
            raise DatasetValidationError(
                f"replica_partition.{name} must be a positive integer")
        counts[name] = int(raw)
    if counts["discovery_replicas"] != len(discovery) or (
            counts["audit_replicas"] != len(audit)):
        raise DatasetValidationError(
            "replica_partition counts do not match their index lists")
    expected_discovery = list(range(counts["discovery_replicas"]))
    expected_audit = list(range(
        counts["discovery_replicas"],
        counts["discovery_replicas"] + counts["audit_replicas"],
    ))
    if discovery != expected_discovery or audit != expected_audit or (
            len(discovery) + len(audit) != replica_count):
        raise DatasetValidationError(
            "replica_partition must exhaustively order discovery then audit "
            "over the replica axis")
    return dict(partition)


def _as_text(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "US":
        raise DatasetValidationError("identifier arrays must use non-object text dtype")
    return array.astype(str, copy=False)


def _finite(name: str, value: np.ndarray, mask: np.ndarray | None = None) -> None:
    selected = np.asarray(value) if mask is None else np.asarray(value)[mask]
    if not np.all(np.isfinite(selected)):
        raise DatasetValidationError(f"{name} contains non-finite values")


def _canonical_manifest(manifest: Mapping[str, Any]) -> bytes:
    content = dict(manifest)
    content.pop("content_sha256", None)
    return json.dumps(
        content, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")


def _content_hash(manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(_canonical_manifest(manifest))
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        if value.dtype.hasobject:
            raise DatasetValidationError(f"{name} uses forbidden object dtype")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _require_content_hash(manifest: Mapping[str, Any]) -> str:
    value = manifest.get("content_sha256")
    if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value):
        raise DatasetValidationError(
            "on-disk dataset manifest requires a lowercase SHA-256 content hash")
    return value


def _mixed_outcome_fraction(
    fall_bool: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    """Compute the optional candidate-outcome diagnostic.

    Integrity-only writers and Stage-B evidence plumbing deliberately skip
    this helper.  Keeping it separate makes that no-summary boundary both
    explicit and regression-testable.
    """
    return float(np.mean([
        len(np.unique(np.mean(fall_bool[g, candidate_mask[g]], axis=1))) > 1
        for g in range(len(fall_bool))
    ]))


@dataclass
class GroupedBranchDataset:
    """A deployable-view dataset whose independent unit is one state group."""

    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.manifest = dict(self.manifest)
        self.arrays = {
            str(name): np.asarray(value)
            for name, value in self.arrays.items()
        }

    @property
    def group_count(self) -> int:
        return int(np.asarray(self.arrays["group_id"]).shape[0])

    @property
    def candidate_count(self) -> int:
        return int(np.asarray(self.arrays["candidate_mask"]).shape[1])

    @property
    def replica_count(self) -> int:
        return int(np.asarray(self.arrays["fall"]).shape[2])

    @property
    def horizon_steps(self) -> int:
        return int(self.manifest["horizon_steps"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def validate(
        self,
        *,
        verify_hash: bool = True,
        summarize_outcomes: bool = True,
    ) -> dict[str, Any]:
        missing_manifest = sorted(set(REQUIRED_MANIFEST_KEYS) - self.manifest.keys())
        if missing_manifest:
            raise DatasetValidationError(
                f"manifest missing required keys: {missing_manifest}")
        if self.manifest["schema_version"] != SCHEMA_VERSION:
            raise DatasetValidationError(
                f"schema version {self.manifest['schema_version']!r}, "
                f"expected {SCHEMA_VERSION!r}")
        if self.manifest["feature_view"] != "deployable":
            raise DatasetValidationError(
                "grouped branch loader accepts only the physically separate deployable view")
        raw_horizon = self.manifest["horizon_steps"]
        if isinstance(raw_horizon, bool) or not isinstance(raw_horizon, int):
            raise DatasetValidationError("horizon_steps must be an integer")
        horizon = raw_horizon
        if horizon <= 0:
            raise DatasetValidationError("horizon_steps must be positive")
        for name in (
            "split", "generator_commit", "source_policy",
            "continuation_policy", "candidate_protocol", "fall_definition",
            "simulator_fingerprint", "observation_contract",
            "action_application_contract",
            "state_hash_contract",
        ):
            value = self.manifest[name]
            if value is None or value == "" or value == {}:
                raise DatasetValidationError(f"manifest field {name} is empty")
        contract = self.manifest["observation_contract"]
        expected_contract = {
            "frames": OBSERVATION_FRAMES,
            "dimension": OBSERVATION_DIM,
            "tail_semantic": "previous_absolute_action_q_target",
        }
        for name, expected in expected_contract.items():
            if contract.get(name) != expected:
                raise DatasetValidationError(
                    f"observation_contract.{name}={contract.get(name)!r}, "
                    f"expected {expected!r}")
        if self.manifest["state_hash_contract"] != "sha256_compound_snapshot_v1":
            raise DatasetValidationError(
                "state_hash_contract must be 'sha256_compound_snapshot_v1'")
        action_contract = self.manifest["action_application_contract"]
        if action_contract.get("q_target_semantic") != "absolute_joint_position_sent":
            raise DatasetValidationError(
                "action_application_contract.q_target_semantic must be "
                "'absolute_joint_position_sent'")
        action_vectors: dict[str, np.ndarray] = {}
        for name in ("init_qpos", "action_offset", "joint_min", "joint_max"):
            value = np.asarray(action_contract.get(name), dtype=np.float64)
            if value.shape != (ACTION_DIM,) or not np.all(np.isfinite(value)):
                raise DatasetValidationError(
                    f"action_application_contract.{name} must be a finite 12-vector")
            action_vectors[name] = value
        if np.any(action_vectors["action_offset"] <= 0.0):
            raise DatasetValidationError("action offsets must be positive")
        if np.any(action_vectors["joint_min"] >= action_vectors["joint_max"]):
            raise DatasetValidationError("joint_min must be below joint_max")
        if np.any(action_vectors["init_qpos"] < action_vectors["joint_min"]) or np.any(
                action_vectors["init_qpos"] > action_vectors["joint_max"]):
            raise DatasetValidationError("init_qpos must lie inside physical joint bounds")

        missing_arrays = sorted(set(REQUIRED_ARRAYS) - self.arrays.keys())
        if missing_arrays:
            raise DatasetValidationError(
                f"dataset missing required arrays: {missing_arrays}")
        forbidden = sorted(name for name in self.arrays if name.startswith("privileged"))
        if forbidden:
            raise DatasetValidationError(
                "privileged features must be stored in a physically separate view: "
                f"{forbidden}")
        for name, value in self.arrays.items():
            if value.dtype.hasobject:
                raise DatasetValidationError(f"{name} uses forbidden object dtype")

        group_id = _as_text(self.arrays["group_id"])
        if group_id.ndim != 1 or group_id.size == 0:
            raise DatasetValidationError("group_id must be a nonempty vector")
        group_count = len(group_id)
        if np.any(group_id == "") or len(np.unique(group_id)) != group_count:
            raise DatasetValidationError("group_id values must be nonempty and unique")

        vector_fields = (
            "state_hash", "trajectory_id", "episode_id", "episode_step",
            "policy_training_seed", "source_seed", "policy_source", "command_vx",
            "acceptance_probability",
        )
        for name in vector_fields:
            if self.arrays[name].shape != (group_count,):
                raise DatasetValidationError(
                    f"{name} shape {self.arrays[name].shape}, expected {(group_count,)}")
        for name in ("state_hash", "trajectory_id", "policy_source"):
            text = _as_text(self.arrays[name])
            if np.any(text == ""):
                raise DatasetValidationError(f"{name} values must be nonempty")
        state_hash = _as_text(self.arrays["state_hash"])
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in state_hash
        ):
            raise DatasetValidationError(
                "state_hash must contain lowercase compound-snapshot SHA-256 values")
        if len(np.unique(state_hash)) != group_count:
            raise DatasetValidationError("duplicate state_hash values violate independence")
        for name in (
            "episode_id", "episode_step", "policy_training_seed", "source_seed"):
            value = np.asarray(self.arrays[name])
            if value.dtype.kind not in "iu" or np.any(value < 0):
                raise DatasetValidationError(
                    f"{name} must be a nonnegative integer vector")

        trajectory_id = _as_text(self.arrays["trajectory_id"])
        source_seed = np.asarray(self.arrays["source_seed"])
        episode_id = np.asarray(self.arrays["episode_id"])
        episode_step = np.asarray(self.arrays["episode_step"])
        policy_training_seed = np.asarray(self.arrays["policy_training_seed"])
        policy_source = _as_text(self.arrays["policy_source"])
        command_vx = np.asarray(self.arrays["command_vx"])

        def require_functional_identity(
            label: str,
            keys: Iterable[Any],
            values: Iterable[Any],
        ) -> None:
            mapping: dict[Any, Any] = {}
            for key, value in zip(keys, values, strict=True):
                previous = mapping.setdefault(key, value)
                if previous != value:
                    raise DatasetValidationError(
                        f"{label} is not identity-isolated: {key!r} maps to "
                        f"both {previous!r} and {value!r}")

        require_functional_identity(
            "source_seed",
            map(int, source_seed),
            (
                (int(training_seed), str(policy), float(speed))
                for training_seed, policy, speed in zip(
                    policy_training_seed, policy_source, command_vx, strict=True)
            ),
        )
        require_functional_identity(
            "trajectory_id",
            map(str, trajectory_id),
            (
                (int(seed), int(episode))
                for seed, episode in zip(source_seed, episode_id, strict=True)
            ),
        )
        require_functional_identity(
            "(source_seed, episode_id)",
            (
                (int(seed), int(episode))
                for seed, episode in zip(source_seed, episode_id, strict=True)
            ),
            map(str, trajectory_id),
        )
        trajectory_steps = list(zip(
            map(str, trajectory_id), map(int, episode_step), strict=True))
        if len(set(trajectory_steps)) != group_count:
            raise DatasetValidationError(
                "(trajectory_id, episode_step) must uniquely identify a group")

        trajectory_fingerprint = self.arrays.get(
            "trajectory_fingerprint_sha256")
        trajectory_contract = self.manifest.get(
            "collection_protocol", {}).get("trajectory_fingerprint_contract")
        expected_trajectory_contract = (
            "sha256_compound_post_settle_pre_source_trajectory_snapshot_v1"
        )
        if trajectory_fingerprint is not None or trajectory_contract is not None:
            if trajectory_contract != expected_trajectory_contract or (
                self.manifest.get("collection_protocol", {}).get(
                    "trajectory_fingerprint_array")
                != "trajectory_fingerprint_sha256"
            ):
                raise DatasetValidationError(
                    "trajectory fingerprint contract has drifted")
            fingerprint = np.asarray(trajectory_fingerprint)
            if fingerprint.shape != (group_count,) or fingerprint.dtype.kind not in (
                "US"
            ):
                raise DatasetValidationError(
                    "trajectory_fingerprint_sha256 must be text [G]")
            fingerprint_text = fingerprint.astype(str)
            if any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in fingerprint_text
            ):
                raise DatasetValidationError(
                    "trajectory_fingerprint_sha256 must contain lowercase SHA-256")
            if len(np.unique(fingerprint_text)) != group_count:
                raise DatasetValidationError(
                    "trajectory fingerprints must be unique within a dataset")

        obs_history = np.asarray(self.arrays["obs_history"])
        q_send_history = np.asarray(self.arrays["q_send_history"])
        if obs_history.shape != (
                group_count, OBSERVATION_FRAMES, OBSERVATION_DIM):
            raise DatasetValidationError(
                f"obs_history shape {obs_history.shape}, expected "
                f"{(group_count, OBSERVATION_FRAMES, OBSERVATION_DIM)}")
        if q_send_history.shape != (
                group_count, OBSERVATION_FRAMES, ACTION_DIM):
            raise DatasetValidationError(
                f"q_send_history shape {q_send_history.shape}, expected "
                f"{(group_count, OBSERVATION_FRAMES, ACTION_DIM)}")
        _finite("obs_history", obs_history)
        _finite("q_send_history", q_send_history)
        if not np.allclose(
                obs_history[..., -ACTION_DIM:], q_send_history,
                rtol=0.0, atol=1e-6):
            error = float(np.max(np.abs(
                obs_history[..., -ACTION_DIM:] - q_send_history)))
            raise DatasetValidationError(
                "corrected observation tail is not absolute q_send history; "
                f"max error={error}")
        if np.any(q_send_history < action_vectors["joint_min"] - 1e-6) or np.any(
                q_send_history > action_vectors["joint_max"] + 1e-6):
            raise DatasetValidationError(
                "q_send_history is outside physical joint bounds; normalized "
                "previous actions were likely stored as absolute targets")

        raw_candidate_mask = np.asarray(self.arrays["candidate_mask"])
        if raw_candidate_mask.dtype.kind not in "biu" or not np.all(
                np.isin(raw_candidate_mask, (0, 1, False, True))):
            raise DatasetValidationError("candidate_mask must be binary")
        candidate_mask = raw_candidate_mask.astype(bool)
        if candidate_mask.ndim != 2 or candidate_mask.shape[0] != group_count:
            raise DatasetValidationError("candidate_mask must have shape [G, K]")
        candidate_count = candidate_mask.shape[1]
        if candidate_count < 2 or np.any(~candidate_mask[:, 0]):
            raise DatasetValidationError("candidate 0 must be valid and K must be at least 2")
        if np.any(candidate_mask.sum(axis=1) < 2):
            raise DatasetValidationError("every group must contain at least two candidates")
        candidate_kind = _as_text(self.arrays["candidate_kind"])
        if candidate_kind.shape != (group_count, candidate_count):
            raise DatasetValidationError("candidate_kind must have shape [G, K]")
        if np.any(candidate_kind[:, 0] != "nominal"):
            raise DatasetValidationError("candidate 0 must be named 'nominal'")

        candidate_protocol = self.manifest["candidate_protocol"]
        if not isinstance(candidate_protocol, Mapping):
            raise DatasetValidationError("candidate_protocol must be a mapping")
        option_protocol = candidate_protocol.get("protocol_version")
        has_option_steps = "candidate_option_steps" in self.arrays
        has_behavior_steps = "candidate_behavior_steps" in self.arrays
        if has_option_steps and has_behavior_steps:
            raise DatasetValidationError(
                "candidate_option_steps and candidate_behavior_steps are "
                "mutually exclusive")
        if option_protocol == RECOVERY_OPTION_CANDIDATE_PROTOCOL_VERSION:
            if candidate_protocol.get(
                    "option_steps_array") != "candidate_option_steps" or (
                    candidate_protocol.get("option_semantics")
                    != "linear_decay_actor_residual_v1"):
                raise DatasetValidationError(
                    "recovery-option candidate protocol does not bind its "
                    "duration array and execution semantics")
            if candidate_count != RECOVERY_OPTION_CANDIDATE_COUNT or (
                    candidate_protocol.get("count")
                    != RECOVERY_OPTION_CANDIDATE_COUNT):
                raise DatasetValidationError(
                    "recovery-option v2 requires exactly K=29")
            if not has_option_steps:
                raise DatasetValidationError(
                    "recovery-option dataset is missing candidate_option_steps")
            ordered_kinds = candidate_protocol.get("ordered_kinds")
            if not isinstance(ordered_kinds, list) or len(
                    ordered_kinds) != RECOVERY_OPTION_CANDIDATE_COUNT or (
                    any(not isinstance(value, str) or not value
                        for value in ordered_kinds)) or not np.all(
                            candidate_kind == np.asarray(
                                ordered_kinds, dtype=str)[None, :]):
                raise DatasetValidationError(
                    "candidate_kind differs from recovery-option ordered_kinds")
        elif has_option_steps:
            raise DatasetValidationError(
                "candidate_option_steps requires the recovery-option v2 protocol")
        if has_option_steps:
            option_steps = np.asarray(self.arrays["candidate_option_steps"])
            if option_steps.shape != (group_count, candidate_count) or (
                    option_steps.dtype.kind not in "iu"):
                raise DatasetValidationError(
                    "candidate_option_steps must be an integer [G,K] array")
            if np.any(option_steps[:, 0] != 1) or np.any(option_steps < 1) or (
                    np.any(option_steps > 4)):
                raise DatasetValidationError(
                    "candidate_option_steps must lock nominal to 1 and lie in [1,4]")
            if option_protocol == RECOVERY_OPTION_CANDIDATE_PROTOCOL_VERSION:
                expected_option_steps = np.asarray(
                    [1, *([1, 2, 3, 4] * 7)], dtype=np.int64)
                if not np.all(option_steps == expected_option_steps[None, :]):
                    raise DatasetValidationError(
                        "candidate_option_steps differs from locked template-major order")
        if option_protocol == CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION:
            if candidate_protocol.get(
                    "behavior_steps_array") != "candidate_behavior_steps":
                raise DatasetValidationError(
                    "closed-loop recovery protocol does not bind its behavior "
                    "duration array")
            if candidate_count != CLOSED_LOOP_RECOVERY_CANDIDATE_COUNT or (
                    candidate_protocol.get("count")
                    != CLOSED_LOOP_RECOVERY_CANDIDATE_COUNT):
                raise DatasetValidationError(
                    "closed-loop recovery v3 requires exactly K=9")
            if not has_behavior_steps:
                raise DatasetValidationError(
                    "closed-loop recovery dataset is missing "
                    "candidate_behavior_steps")
            ordered_names = candidate_protocol.get("ordered_names")
            if ordered_names != list(CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS) or (
                    not np.all(candidate_kind == np.asarray(
                        CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS,
                        dtype=str)[None, :])):
                raise DatasetValidationError(
                    "candidate_kind differs from locked closed-loop recovery "
                    "order")
            if candidate_protocol.get("behavior_override_steps") != list(
                    CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS):
                raise DatasetValidationError(
                    "closed-loop recovery manifest durations differ from the "
                    "locked K9 order")
        elif has_behavior_steps:
            raise DatasetValidationError(
                "candidate_behavior_steps requires the closed-loop recovery "
                "v3 protocol")
        if has_behavior_steps:
            behavior_steps = np.asarray(self.arrays[
                "candidate_behavior_steps"])
            if behavior_steps.shape != (group_count, candidate_count) or (
                    behavior_steps.dtype.kind not in "iu"):
                raise DatasetValidationError(
                    "candidate_behavior_steps must be an integer [G,K] array")
            if np.any(behavior_steps[:, 0] != 0) or np.any(
                    behavior_steps[:, 1:] < 1) or np.any(
                        behavior_steps > horizon):
                raise DatasetValidationError(
                    "candidate_behavior_steps must lock nominal to 0 and "
                    "recovery behaviors to [1,H]")
            if option_protocol == CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION:
                expected_behavior_steps = np.asarray(
                    CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS, dtype=np.int64)
                if not np.all(
                        behavior_steps == expected_behavior_steps[None, :]):
                    raise DatasetValidationError(
                        "candidate_behavior_steps differs from locked K9 order")

        nominal = np.asarray(self.arrays["nominal_action_requested"])
        if nominal.shape != (group_count, ACTION_DIM):
            raise DatasetValidationError("nominal_action_requested must have shape [G, 12]")
        for name in (
            "candidate_requested", "candidate_executed", "candidate_q_target"):
            value = np.asarray(self.arrays[name])
            expected = (group_count, candidate_count, ACTION_DIM)
            if value.shape != expected:
                raise DatasetValidationError(
                    f"{name} shape {value.shape}, expected {expected}")
            _finite(name, value, np.repeat(candidate_mask[..., None], ACTION_DIM, axis=2))
        for name in ("candidate_requested", "candidate_executed"):
            value = np.asarray(self.arrays[name])
            selected = value[np.repeat(candidate_mask[..., None], ACTION_DIM, axis=2)]
            if np.any(selected < -1.0 - 1e-6) or np.any(selected > 1.0 + 1e-6):
                raise DatasetValidationError(f"{name} must lie in normalized [-1, 1]")
        q_target_values = np.asarray(self.arrays["candidate_q_target"])
        q_target_valid = np.repeat(candidate_mask[..., None], ACTION_DIM, axis=2)
        lower = np.broadcast_to(
            action_vectors["joint_min"], q_target_values.shape)[q_target_valid]
        upper = np.broadcast_to(
            action_vectors["joint_max"], q_target_values.shape)[q_target_valid]
        selected_q_target = q_target_values[q_target_valid]
        if np.any(selected_q_target < lower - 1e-6) or np.any(
                selected_q_target > upper + 1e-6):
            raise DatasetValidationError(
                "candidate_q_target is outside physical joint bounds")
        expected_executed = np.clip(
            (q_target_values - action_vectors["init_qpos"])
            / action_vectors["action_offset"],
            -1.0,
            1.0,
        )
        if not np.allclose(
                np.asarray(self.arrays["candidate_executed"])[q_target_valid],
                expected_executed[q_target_valid], rtol=0.0, atol=1e-6):
            raise DatasetValidationError(
                "candidate_executed does not round-trip from candidate_q_target")
        if not np.allclose(
                self.arrays["candidate_requested"][:, 0], nominal,
                rtol=0.0, atol=1e-7):
            raise DatasetValidationError(
                "candidate_requested[:, 0] must equal nominal_action_requested")

        fall = np.asarray(self.arrays["fall"])
        if fall.ndim != 3 or fall.shape[:2] != candidate_mask.shape:
            raise DatasetValidationError("fall must have shape [G, K, R]")
        replica_count = fall.shape[2]
        if replica_count < 1:
            raise DatasetValidationError("at least one CRN replica is required")
        replica_partition = _validate_optional_replica_partition(
            self.manifest, replica_count=replica_count)
        outcome_shape = (group_count, candidate_count, replica_count)
        for name in ("first_failure_step", "max_tilt_rad", "min_height_m"):
            if self.arrays[name].shape != outcome_shape:
                raise DatasetValidationError(
                    f"{name} shape {self.arrays[name].shape}, expected {outcome_shape}")
        valid_outcomes = np.repeat(candidate_mask[..., None], replica_count, axis=2)
        if not np.all(np.isin(fall[valid_outcomes], (0, 1, False, True))):
            raise DatasetValidationError("fall labels must be binary")
        fall_bool = fall.astype(bool)
        failure_step = np.asarray(self.arrays["first_failure_step"])
        if failure_step.dtype.kind not in "iu":
            raise DatasetValidationError("first_failure_step must use integer dtype")
        if horizon + 1 > np.iinfo(failure_step.dtype).max:
            raise DatasetValidationError(
                "first_failure_step dtype cannot represent horizon+1")
        if np.any(failure_step[valid_outcomes & fall_bool] < 1) or np.any(
                failure_step[valid_outcomes & fall_bool] > horizon):
            raise DatasetValidationError("falling branches need first_failure_step in [1, H]")
        if np.any(failure_step[valid_outcomes & ~fall_bool] != horizon + 1):
            raise DatasetValidationError("non-falls must use first_failure_step=H+1")
        _finite("max_tilt_rad", self.arrays["max_tilt_rad"], valid_outcomes)
        _finite("min_height_m", self.arrays["min_height_m"], valid_outcomes)
        fall_definition = self.manifest["fall_definition"]
        try:
            tilt_threshold = float(fall_definition["max_abs_roll_pitch_rad"])
            height_threshold = float(fall_definition["min_base_height_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetValidationError(
                "fall_definition requires numeric max_abs_roll_pitch_rad and "
                "min_base_height_m") from exc
        if not np.isfinite(tilt_threshold) or tilt_threshold <= 0.0:
            raise DatasetValidationError(
                "fall tilt threshold must be finite and positive")
        if not np.isfinite(height_threshold) or height_threshold <= 0.0:
            raise DatasetValidationError(
                "fall height threshold must be finite and positive")
        max_tilt = np.asarray(self.arrays["max_tilt_rad"])
        min_height = np.asarray(self.arrays["min_height_m"])
        predicate = (
            (max_tilt >= tilt_threshold)
            | (min_height < height_threshold))
        if np.any(predicate[valid_outcomes] != fall_bool[valid_outcomes]):
            raise DatasetValidationError(
                "fall labels disagree with manifest tilt/height predicate")

        crn_shape = (group_count, replica_count)
        seed_namespaces: dict[str, np.ndarray] = {}
        for name in ("crn_id", "rollout_seed", "perturbation_seed"):
            if self.arrays[name].shape != crn_shape:
                raise DatasetValidationError(
                    f"{name} shape {self.arrays[name].shape}, expected {crn_shape}")
            if self.arrays[name].dtype.kind not in "iu" or np.any(
                    self.arrays[name] < 0):
                raise DatasetValidationError(
                    f"{name} must contain nonnegative integers")
            for row in np.asarray(self.arrays[name]):
                if len(np.unique(row)) != replica_count:
                    raise DatasetValidationError(
                        f"{name} must be unique across replicas within each group")
            flattened = np.asarray(self.arrays[name]).reshape(-1)
            if len(np.unique(flattened)) != flattened.size:
                raise DatasetValidationError(
                    f"{name} must be globally unique across dataset groups")
            seed_namespaces[name] = flattened
        if "candidate_seed" in self.arrays:
            candidate_seed = np.asarray(self.arrays["candidate_seed"])
            if candidate_seed.shape != (group_count,):
                raise DatasetValidationError(
                    "candidate_seed must have shape [G]")
            if candidate_seed.dtype.kind not in "iu" or np.any(candidate_seed < 0):
                raise DatasetValidationError(
                    "candidate_seed must contain nonnegative integers")
            if len(np.unique(candidate_seed)) != group_count:
                raise DatasetValidationError(
                    "candidate_seed must be globally unique across dataset groups")
            seed_namespaces["candidate_seed"] = candidate_seed.reshape(-1)
        # Each namespace is already internally unique, so a duplicate after
        # concatenation can only be a cross-namespace collision.  Keep this
        # vectorized: Python integer dictionaries are prohibitively large at
        # the preregistered 100k-group scale.
        all_seeds = np.concatenate([
            values.astype(np.uint64, copy=False)
            for values in seed_namespaces.values()
        ])
        unique_seeds, seed_counts = np.unique(all_seeds, return_counts=True)
        collisions = unique_seeds[seed_counts > 1]
        if collisions.size:
            collision = int(collisions[0])
            owners = [
                name for name, values in seed_namespaces.items()
                if np.any(values.astype(np.uint64, copy=False) == collision)
            ]
            raise DatasetValidationError(
                f"seed value {collision} is reused across RNG namespaces "
                f"{owners}")

        acceptance = np.asarray(self.arrays["acceptance_probability"], dtype=float)
        _finite("acceptance_probability", acceptance)
        if np.any(acceptance <= 0.0) or np.any(acceptance > 1.0):
            raise DatasetValidationError("acceptance_probability must lie in (0, 1]")
        _finite("command_vx", self.arrays["command_vx"])
        if np.any(np.asarray(self.arrays["episode_step"]) < 0):
            raise DatasetValidationError("episode_step must be nonnegative")

        expected_hash = self.manifest.get("content_sha256")
        actual_hash = _content_hash(self.manifest, self.arrays)
        if verify_hash and expected_hash is not None and expected_hash != actual_hash:
            raise DatasetValidationError(
                f"content hash mismatch: manifest={expected_hash}, actual={actual_hash}")
        report = {
            "schema_version": SCHEMA_VERSION,
            "groups": group_count,
            "max_candidates": candidate_count,
            "replicas": replica_count,
            "valid_candidates": int(candidate_mask.sum()),
            "min_valid_candidates_per_group": int(
                candidate_mask.sum(axis=1).min()),
            "replicas_per_candidate": replica_count,
            "replica_partition": replica_partition,
            "unique_trajectory_clusters": int(len(np.unique(
                _as_text(self.arrays["trajectory_id"])))),
            "unique_source_seeds": int(len(np.unique(
                self.arrays["source_seed"]))),
            "duplicate_state_fraction": 0.0,
            "content_sha256": actual_hash,
        }
        if summarize_outcomes:
            report["mixed_outcome_fraction"] = _mixed_outcome_fraction(
                fall_bool, candidate_mask)
        return report

    def save(self, path: str | Path) -> Path:
        output = assert_development_path(assert_safe_evidence_output(path))
        if output.suffix != ".npz":
            raise DatasetValidationError("grouped datasets must use a .npz path")
        report = self.validate(
            verify_hash=False, summarize_outcomes=False)
        manifest = dict(self.manifest)
        manifest["content_sha256"] = report["content_sha256"]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            manifest_json=np.asarray(json.dumps(
                manifest, sort_keys=True, separators=(",", ":"))),
            **self.arrays,
        )
        self.manifest = manifest
        self.path = output
        return output

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        summarize_outcomes: bool = True,
    ) -> "GroupedBranchDataset":
        source = assert_development_path(_authorize_evidence_input(
            path, allowed_roles=(
                "discovery",
                "audit",
                "stage_b_fit_label",
                "stage_b_probability_calibration_label",
                "stage_b_uncertainty_calibration_label",
                "stage_b_selector_calibration_label",
                "stage_b_model_test_label",
                "stage_b_model_test_producer_label",
            )))
        with np.load(source, allow_pickle=False) as payload:
            if "manifest_json" not in payload.files:
                raise DatasetValidationError("dataset has no manifest_json")
            try:
                manifest = json.loads(str(payload["manifest_json"].item()))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise DatasetValidationError("invalid manifest_json") from exc
            arrays = {
                name: payload[name].copy()
                for name in payload.files if name != "manifest_json"
            }
        _require_content_hash(manifest)
        dataset = cls(manifest=manifest, arrays=arrays, path=source)
        dataset.validate(
            verify_hash=True, summarize_outcomes=summarize_outcomes)
        return dataset


@dataclass
class PrivilegedBranchView:
    """Simulator-only diagnostic features stored outside deployable datasets."""

    manifest: dict[str, Any]
    group_id: np.ndarray
    state_hash: np.ndarray
    features: np.ndarray
    feature_names: np.ndarray
    path: Path | None = field(default=None, repr=False)

    def validate(
        self, deployable: GroupedBranchDataset | None = None,
        *, verify_hash: bool = True,
    ) -> dict[str, Any]:
        if self.manifest.get("schema_version") != PRIVILEGED_SCHEMA_VERSION:
            raise DatasetValidationError("invalid privileged schema version")
        if self.manifest.get("feature_view") != "privileged_diagnostic_only":
            raise DatasetValidationError("privileged view is not marked diagnostic-only")
        group_id = _as_text(self.group_id)
        state_hash = _as_text(self.state_hash)
        features = np.asarray(self.features)
        feature_names = _as_text(self.feature_names)
        if group_id.ndim != 1 or state_hash.shape != group_id.shape:
            raise DatasetValidationError("privileged identities must have shape [G]")
        if features.ndim != 2 or features.shape[0] != len(group_id):
            raise DatasetValidationError("privileged features must have shape [G, P]")
        if feature_names.shape != (features.shape[1],):
            raise DatasetValidationError("feature_names must identify every privileged column")
        if np.any(feature_names == "") or len(np.unique(feature_names)) != len(
                feature_names):
            raise DatasetValidationError(
                "privileged feature_names must be unique and nonempty")
        if len(np.unique(group_id)) != len(group_id) or np.any(group_id == ""):
            raise DatasetValidationError("privileged group_id values must be unique/nonempty")
        if len(np.unique(state_hash)) != len(state_hash) or np.any(state_hash == ""):
            raise DatasetValidationError("privileged state_hash values must be unique/nonempty")
        _finite("privileged features", features)
        arrays = {
            "group_id": group_id,
            "state_hash": state_hash,
            "features": features,
            "feature_names": feature_names,
        }
        expected_hash = self.manifest.get("content_sha256")
        actual_hash = _content_hash(self.manifest, arrays)
        if verify_hash and expected_hash is not None and expected_hash != actual_hash:
            raise DatasetValidationError("privileged content hash mismatch")
        if deployable is not None:
            deployable_report = deployable.validate(summarize_outcomes=False)
            for name in ("split", "generator_commit"):
                if self.manifest.get(name) != deployable.manifest.get(name):
                    raise DatasetValidationError(
                        f"privileged {name} does not match deployable view")
            if self.manifest.get("deployable_content_sha256") != (
                    deployable_report["content_sha256"]):
                raise DatasetValidationError(
                    "privileged deployable_content_sha256 does not match "
                    "deployable view")
            if not np.array_equal(group_id, deployable["group_id"].astype(str)):
                raise DatasetValidationError(
                    "privileged group_id order does not match deployable view")
            if not np.array_equal(state_hash, deployable["state_hash"].astype(str)):
                raise DatasetValidationError(
                    "privileged state_hash does not match deployable view")
        return {
            "schema_version": PRIVILEGED_SCHEMA_VERSION,
            "groups": len(group_id),
            "features": features.shape[1],
            "content_sha256": actual_hash,
        }

    def save(self, path: str | Path) -> Path:
        output = assert_development_path(assert_safe_evidence_output(path))
        if output.suffix != ".npz":
            raise DatasetValidationError("privileged views must use a .npz path")
        report = self.validate(verify_hash=False)
        manifest = dict(self.manifest)
        manifest["content_sha256"] = report["content_sha256"]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            manifest_json=np.asarray(json.dumps(
                manifest, sort_keys=True, separators=(",", ":"))),
            group_id=np.asarray(self.group_id),
            state_hash=np.asarray(self.state_hash),
            features=np.asarray(self.features),
            feature_names=np.asarray(self.feature_names),
        )
        self.manifest = manifest
        self.path = output
        return output

    @classmethod
    def load(
        cls, path: str | Path, *,
        deployable: GroupedBranchDataset | None = None,
    ) -> "PrivilegedBranchView":
        source = assert_development_path(_authorize_evidence_input(
            path,
            allowed_roles=(
                "discovery_privileged",
                "audit_privileged",
                "stage_b_fit_label_privileged",
                "stage_b_probability_calibration_label_privileged",
                "stage_b_uncertainty_calibration_label_privileged",
                "stage_b_selector_calibration_label_privileged",
                "stage_b_model_test_label_privileged",
                "stage_b_model_test_producer_label_privileged",
            ),
        ))
        with np.load(source, allow_pickle=False) as payload:
            required = {
                "manifest_json", "group_id", "state_hash", "features", "feature_names"}
            missing = required - set(payload.files)
            if missing:
                raise DatasetValidationError(
                    f"privileged view missing fields: {sorted(missing)}")
            manifest = json.loads(str(payload["manifest_json"].item()))
            view = cls(
                manifest=manifest,
                group_id=payload["group_id"].copy(),
                state_hash=payload["state_hash"].copy(),
                features=payload["features"].copy(),
                feature_names=payload["feature_names"].copy(),
                path=source,
            )
        _require_content_hash(manifest)
        view.validate(deployable)
        return view


def audit_split_disjointness(
    datasets: Iterable[GroupedBranchDataset],
) -> dict[str, Any]:
    """Fail if any identity/seed namespace leaks across named splits."""
    items = list(datasets)
    if len(items) < 2:
        return {"splits": len(items), "pairs_checked": 0}
    for dataset in items:
        dataset.validate()
    checks = {
        "group_id": lambda data: set(_as_text(data["group_id"])),
        "state_hash": lambda data: set(_as_text(data["state_hash"])),
        "trajectory_id": lambda data: set(_as_text(data["trajectory_id"])),
        "source_seed": lambda data: set(map(int, data["source_seed"])),
        "crn_id": lambda data: set(map(int, np.asarray(data["crn_id"]).reshape(-1))),
        "rollout_seed": lambda data: set(map(
            int, np.asarray(data["rollout_seed"]).reshape(-1))),
        "perturbation_seed": lambda data: set(map(
            int, np.asarray(data["perturbation_seed"]).reshape(-1))),
    }
    pairs_checked = 0
    for left_index, left in enumerate(items):
        for right in items[left_index + 1:]:
            left_split = str(left.manifest["split"])
            right_split = str(right.manifest["split"])
            if left_split == right_split:
                raise DatasetValidationError(
                    f"split audit received duplicate split name {left_split!r}")
            for name, getter in checks.items():
                overlap = getter(left) & getter(right)
                if overlap:
                    examples = sorted(map(str, overlap))[:3]
                    raise DatasetValidationError(
                        f"{name} leaks across {left_split!r}/{right_split!r}: {examples}")
            pairs_checked += 1
    return {
        "splits": len(items),
        "pairs_checked": pairs_checked,
        "split_names": [str(item.manifest["split"]) for item in items],
    }
