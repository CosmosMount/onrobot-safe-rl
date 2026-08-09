"""Assembly primitives for native grouped Q_safe branch collection."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from safety_data.native import NativeGroupEvaluation, ReplicaSeedBundle
from safety_data.paths import assert_development_path
from safety_data.schema import (
    GroupedBranchDataset,
    PRIVILEGED_SCHEMA_VERSION,
    PrivilegedBranchView,
    SCHEMA_VERSION,
)


@dataclass(frozen=True)
class GroupIdentity:
    group_id: str
    state_hash: str
    trajectory_id: str
    episode_id: int
    episode_step: int
    policy_training_seed: int
    source_seed: int
    policy_source: str
    command_vx: float
    acceptance_probability: float
    sampling_stratum: str = "unspecified"


@dataclass(frozen=True)
class GroupRandomness:
    crn_id: np.ndarray
    rollout_seed: np.ndarray
    perturbation_seed: np.ndarray
    candidate_seed: int

    def __post_init__(self) -> None:
        arrays = []
        for name in ("crn_id", "rollout_seed", "perturbation_seed"):
            raw = np.asarray(getattr(self, name))
            if raw.dtype.kind not in "iu" or raw.ndim != 1 or len(raw) == 0 or (
                    np.any(raw < 0)) or len(np.unique(raw)) != len(raw):
                raise ValueError(
                    f"{name} must be a nonempty unique nonnegative integer vector")
            value = raw.astype(np.uint64, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
            arrays.append(value)
        if any(value.shape != arrays[0].shape for value in arrays[1:]):
            raise ValueError("group seed vectors must have equal replica shape")
        if isinstance(self.candidate_seed, (bool, np.bool_)) or not isinstance(
                self.candidate_seed, (int, np.integer)) or not (
                    0 <= int(self.candidate_seed) < 2**64):
            raise ValueError("candidate_seed must be an unsigned 64-bit integer")


@dataclass(frozen=True)
class CollectedGroup:
    identity: GroupIdentity
    observation_history: np.ndarray
    candidate_kind: np.ndarray
    candidate_mask: np.ndarray
    evaluation: NativeGroupEvaluation
    randomness: GroupRandomness
    candidate_option_steps: np.ndarray | None = None
    privileged_features: np.ndarray | None = None


class GroupedBranchAssembler:
    """Build one rectangular, schema-validated dataset without row flattening."""

    def __init__(
        self,
        *,
        split: str,
        horizon_steps: int,
        generator_commit: str,
        simulator_fingerprint: Mapping[str, Any],
        source_policy: Mapping[str, Any],
        continuation_policy: Mapping[str, Any],
        candidate_protocol: Mapping[str, Any],
        fall_definition: Mapping[str, Any],
        action_application_contract: Mapping[str, Any],
        collection_protocol: Mapping[str, Any],
        privileged_feature_names: Sequence[str] | None = None,
    ):
        if not isinstance(split, str) or not split.strip():
            raise ValueError("split must be a nonempty string")
        # A split name is metadata rather than a file path, but applying the
        # same component guard prevents development collectors from smuggling
        # a protected evaluation label into an otherwise innocuous output.
        assert_development_path(Path(split))
        if not isinstance(generator_commit, str) or not generator_commit.strip():
            raise ValueError("generator_commit must be a nonempty string")
        if isinstance(horizon_steps, bool) or not isinstance(
                horizon_steps, (int, np.integer)) or not (
                    0 < int(horizon_steps) <= np.iinfo(np.int16).max - 1):
            raise ValueError("horizon_steps must be an int16-representable positive integer")
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "split": str(split),
            "feature_view": "deployable",
            "horizon_steps": int(horizon_steps),
            "generator_commit": str(generator_commit),
            "simulator_fingerprint": copy.deepcopy(dict(simulator_fingerprint)),
            "source_policy": copy.deepcopy(dict(source_policy)),
            "continuation_policy": copy.deepcopy(dict(continuation_policy)),
            "candidate_protocol": copy.deepcopy(dict(candidate_protocol)),
            "fall_definition": copy.deepcopy(dict(fall_definition)),
            "observation_contract": {
                "frames": 5,
                "dimension": 46,
                "tail_semantic": "previous_absolute_action_q_target",
            },
            "action_application_contract": copy.deepcopy(
                dict(action_application_contract)),
            "state_hash_contract": "sha256_compound_snapshot_v1",
            "collection_protocol": copy.deepcopy(dict(collection_protocol)),
        }
        self.privileged_feature_names = (
            None if privileged_feature_names is None
            else np.asarray(list(privileged_feature_names), dtype=str))
        if self.privileged_feature_names is not None and (
                self.privileged_feature_names.ndim != 1 or
                len(self.privileged_feature_names) == 0 or
                np.any(self.privileged_feature_names == "")):
            raise ValueError("privileged feature names must be a nonempty vector")
        self._groups: list[CollectedGroup] = []
        self._group_ids: set[str] = set()
        self._state_hashes: set[str] = set()
        self._shape: tuple[int, int] | None = None
        self._uses_option_steps: bool | None = None

    def add(self, group: CollectedGroup) -> None:
        identity = group.identity
        if not all((identity.group_id, identity.state_hash,
                    identity.trajectory_id, identity.policy_source,
                    identity.sampling_stratum)):
            raise ValueError("group identity text fields must be nonempty")
        for name in (
            "episode_id", "episode_step", "policy_training_seed", "source_seed",
        ):
            value = getattr(identity, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)) or int(value) < 0:
                raise ValueError(f"identity {name} must be a nonnegative integer")
        if not np.isfinite(identity.command_vx):
            raise ValueError("identity command_vx must be finite")
        if not np.isfinite(identity.acceptance_probability) or not (
                0.0 < float(identity.acceptance_probability) <= 1.0):
            raise ValueError("identity acceptance_probability must lie in (0,1]")
        sampling_strata = self.manifest["collection_protocol"].get(
            "sampling_strata")
        if sampling_strata is not None:
            if identity.sampling_stratum not in sampling_strata:
                raise ValueError(
                    f"unknown sampling stratum {identity.sampling_stratum!r}")
            try:
                expected_probability = float(
                    sampling_strata[identity.sampling_stratum]
                    ["acceptance_probability"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "sampling stratum must declare acceptance_probability") from exc
            if not math.isclose(
                    float(identity.acceptance_probability), expected_probability,
                    rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(
                    "recorded acceptance_probability disagrees with sampling "
                    f"stratum {identity.sampling_stratum!r}")
        if identity.group_id in self._group_ids:
            raise ValueError(f"duplicate group_id {identity.group_id!r}")
        if identity.state_hash in self._state_hashes:
            raise ValueError("duplicate compound state hash")
        history = np.asarray(group.observation_history, dtype=np.float32)
        kinds = np.asarray(group.candidate_kind)
        raw_mask = np.asarray(group.candidate_mask)
        if raw_mask.dtype.kind not in "biu" or not np.all(
                np.isin(raw_mask, (0, 1, False, True))):
            raise ValueError("candidate_mask must be binary")
        mask = raw_mask.astype(bool)
        evaluation = group.evaluation
        candidate_shape = evaluation.candidate_requested.shape
        if history.shape != (5, 46):
            raise ValueError("observation_history must have shape [5,46]")
        if not np.all(np.isfinite(history)):
            raise ValueError("observation_history must be finite")
        if candidate_shape[1:] != (12,) or kinds.shape != candidate_shape[:1] or (
                mask.shape != candidate_shape[:1]):
            raise ValueError("candidate actions, kinds and mask do not align")
        candidate_count = candidate_shape[0]
        replica_count = group.randomness.crn_id.shape[0]
        if candidate_count < 2 or not bool(mask[0]) or kinds[0] != "nominal":
            raise ValueError("candidate zero must be a valid nominal")
        uses_option_steps = group.candidate_option_steps is not None
        if self._uses_option_steps is None:
            self._uses_option_steps = uses_option_steps
        elif uses_option_steps != self._uses_option_steps:
            raise ValueError(
                "all groups must consistently include candidate_option_steps")
        option_steps = None
        if uses_option_steps:
            raw_option_steps = np.asarray(group.candidate_option_steps)
            if raw_option_steps.shape != (candidate_count,) or (
                    raw_option_steps.dtype.kind not in "iu"):
                raise ValueError(
                    "candidate_option_steps must be an integer [K] array")
            if raw_option_steps[0] != 1 or np.any(raw_option_steps < 1) or (
                    np.any(raw_option_steps > 4)):
                raise ValueError(
                    "candidate_option_steps must lock nominal to 1 and lie in [1,4]")
            option_steps = raw_option_steps.astype(np.int8, copy=True)
        for name in (
            "candidate_executed", "candidate_q_target",
        ):
            if np.asarray(getattr(evaluation, name)).shape != candidate_shape:
                raise ValueError(f"{name} does not match candidate shape")
        outcome_shape = (candidate_count, replica_count)
        for name in (
            "fall", "first_failure_step", "max_tilt_rad", "min_height_m",
        ):
            if np.asarray(getattr(evaluation, name)).shape != outcome_shape:
                raise ValueError(f"{name} does not match [K,R]")
        if evaluation.seed_contract != "explicit_three_stream_v1":
            raise ValueError(
                "evidence assembly requires the explicit three-stream seed contract")
        for name in ("crn_id", "rollout_seed", "perturbation_seed"):
            if not np.array_equal(
                    np.asarray(getattr(evaluation, name)),
                    np.asarray(getattr(group.randomness, name))):
                raise ValueError(
                    f"executed {name} does not match recorded group randomness")
        shape = (candidate_count, replica_count)
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise ValueError(
                f"all groups must share [K,R]={self._shape}, got {shape}")
        privileged = group.privileged_features
        if (privileged is None) != (self.privileged_feature_names is None):
            raise ValueError("privileged features and names must be supplied together")
        if privileged is not None:
            privileged = np.asarray(privileged, dtype=np.float32).reshape(-1)
            assert self.privileged_feature_names is not None
            if privileged.shape != self.privileged_feature_names.shape or not (
                    np.all(np.isfinite(privileged))):
                raise ValueError("privileged feature vector is invalid")
        # NativeGroupEvaluation is a frozen dataclass but contains mutable
        # ndarrays.  Copy every field so a caller cannot silently alter an
        # already accepted evidence group before finalize().
        frozen_evaluation = NativeGroupEvaluation(
            candidate_requested=np.asarray(
                evaluation.candidate_requested).copy(),
            candidate_executed=np.asarray(evaluation.candidate_executed).copy(),
            candidate_q_target=np.asarray(evaluation.candidate_q_target).copy(),
            fall=np.asarray(evaluation.fall).copy(),
            first_failure_step=np.asarray(evaluation.first_failure_step).copy(),
            max_tilt_rad=np.asarray(evaluation.max_tilt_rad).copy(),
            min_height_m=np.asarray(evaluation.min_height_m).copy(),
            crn_id=np.asarray(evaluation.crn_id).copy(),
            rollout_seed=np.asarray(evaluation.rollout_seed).copy(),
            perturbation_seed=np.asarray(evaluation.perturbation_seed).copy(),
            seed_contract=str(evaluation.seed_contract),
        )
        self._groups.append(CollectedGroup(
            identity=identity,
            observation_history=history.copy(),
            candidate_kind=kinds.astype(str, copy=True),
            candidate_mask=mask.copy(),
            evaluation=frozen_evaluation,
            randomness=group.randomness,
            candidate_option_steps=(
                None if option_steps is None else option_steps.copy()),
            privileged_features=(
                None if privileged is None else privileged.copy()),
        ))
        self._group_ids.add(identity.group_id)
        self._state_hashes.add(identity.state_hash)

    @property
    def group_count(self) -> int:
        return len(self._groups)

    def finalize(
        self,
    ) -> tuple[GroupedBranchDataset, PrivilegedBranchView | None]:
        if not self._groups:
            raise ValueError("cannot finalize an empty grouped dataset")
        identities = [group.identity for group in self._groups]
        def identity(name: str, dtype: Any | None = None) -> np.ndarray:
            values = [getattr(value, name) for value in identities]
            return np.asarray(values, dtype=dtype)

        def stack(name: str, dtype: Any | None = None) -> np.ndarray:
            value = np.stack([
                np.asarray(getattr(group.evaluation, name))
                for group in self._groups])
            return value.astype(dtype, copy=False) if dtype is not None else value

        requested = stack("candidate_requested", np.float32)
        arrays = {
            "group_id": identity("group_id", str),
            "state_hash": identity("state_hash", str),
            "trajectory_id": identity("trajectory_id", str),
            "episode_id": identity("episode_id", np.int64),
            "episode_step": identity("episode_step", np.int32),
            "policy_training_seed": identity("policy_training_seed", np.int64),
            "source_seed": identity("source_seed", np.int64),
            "policy_source": identity("policy_source", str),
            "command_vx": identity("command_vx", np.float32),
            "acceptance_probability": identity(
                "acceptance_probability", np.float64),
            "sampling_stratum": identity("sampling_stratum", str),
            "obs_history": np.stack([
                group.observation_history for group in self._groups
            ]).astype(np.float32),
            "q_send_history": np.stack([
                group.observation_history[:, -12:] for group in self._groups
            ]).astype(np.float32),
            "nominal_action_requested": requested[:, 0].copy(),
            "candidate_requested": requested,
            "candidate_executed": stack("candidate_executed", np.float32),
            "candidate_q_target": stack("candidate_q_target", np.float32),
            "candidate_kind": np.stack([
                group.candidate_kind for group in self._groups]),
            "candidate_mask": np.stack([
                group.candidate_mask for group in self._groups]),
            "fall": stack("fall", bool),
            "first_failure_step": stack("first_failure_step", np.int16),
            "max_tilt_rad": stack("max_tilt_rad", np.float32),
            "min_height_m": stack("min_height_m", np.float32),
            "crn_id": np.stack([
                group.randomness.crn_id for group in self._groups]),
            "rollout_seed": np.stack([
                group.randomness.rollout_seed for group in self._groups]),
            "perturbation_seed": np.stack([
                group.randomness.perturbation_seed for group in self._groups]),
            "candidate_seed": np.asarray([
                group.randomness.candidate_seed for group in self._groups
            ], dtype=np.uint64),
        }
        if self._uses_option_steps:
            arrays["candidate_option_steps"] = np.stack([
                group.candidate_option_steps for group in self._groups
            ]).astype(np.int8)
        dataset = GroupedBranchDataset(dict(self.manifest), arrays)
        dataset.validate(verify_hash=False)
        privileged_view = None
        if self.privileged_feature_names is not None:
            privileged_view = PrivilegedBranchView(
                manifest={
                    "schema_version": PRIVILEGED_SCHEMA_VERSION,
                    "feature_view": "privileged_diagnostic_only",
                    "split": self.manifest["split"],
                    "generator_commit": self.manifest["generator_commit"],
                    "deployable_content_sha256": dataset.validate(
                        verify_hash=False)["content_sha256"],
                },
                group_id=arrays["group_id"].copy(),
                state_hash=arrays["state_hash"].copy(),
                features=np.stack([
                    group.privileged_features for group in self._groups
                ]).astype(np.float32),
                feature_names=self.privileged_feature_names.copy(),
            )
            privileged_view.validate(dataset, verify_hash=False)
        return dataset, privileged_view


@dataclass(frozen=True)
class GaussianImpulseSchedule:
    """Reproducible future velocity impulses for branch replicas."""

    policy_steps: tuple[int, ...] = (8, 16)
    linear_std_mps: float = 0.25
    angular_std_radps: float = 1.0

    def __post_init__(self) -> None:
        if not self.policy_steps or any(
                isinstance(step, bool) or not isinstance(step, (int, np.integer))
                or step < 0 for step in self.policy_steps):
            raise ValueError("impulse policy steps must be nonnegative integers")
        if tuple(sorted(set(self.policy_steps))) != self.policy_steps:
            raise ValueError("impulse policy steps must be unique and sorted")
        for name in ("linear_std_mps", "angular_std_radps"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def __call__(
        self,
        env: Any,
        step: int,
        rng: np.random.Generator,
    ) -> None:
        if step not in self.policy_steps:
            return
        env.apply_base_velocity_impulse(
            linear_velocity_delta=rng.normal(
                0.0, self.linear_std_mps, size=3),
            angular_velocity_delta=rng.normal(
                0.0, self.angular_std_radps, size=3),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "type": "gaussian_base_velocity_impulse_v1",
            "policy_steps": list(self.policy_steps),
            "linear_std_mps": float(self.linear_std_mps),
            "angular_std_radps": float(self.angular_std_radps),
        }


def _derived_seed(*parts: int) -> int:
    digest = hashlib.sha256(b"qsafe_native_seed_v1\0")
    for part in parts:
        if isinstance(part, bool) or not isinstance(part, (int, np.integer)) or (
                int(part) < 0):
            raise ValueError("seed derivation inputs must be nonnegative integers")
        digest.update(int(part).to_bytes(16, byteorder="little", signed=False))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def group_randomness(
    *,
    source_seed: int,
    group_index: int,
    replicas: int,
) -> tuple[ReplicaSeedBundle, GroupRandomness]:
    """Create disjoint explicit seed namespaces for one collected group."""
    if isinstance(replicas, bool) or not isinstance(replicas, (int, np.integer)) or (
            replicas <= 0):
        raise ValueError("replicas must be a positive integer")
    crn_id = np.asarray([
        _derived_seed(source_seed, group_index, 0, replica)
        for replica in range(int(replicas))
    ], dtype=np.uint64)
    rollout_seed = np.asarray([
        _derived_seed(source_seed, group_index, 1, replica)
        for replica in range(int(replicas))
    ], dtype=np.uint64)
    perturbation_seed = np.asarray([
        _derived_seed(source_seed, group_index, 2, replica)
        for replica in range(int(replicas))
    ], dtype=np.uint64)
    candidate_seed = _derived_seed(source_seed, group_index, 3)
    bundle = ReplicaSeedBundle(
        crn_id=crn_id,
        rollout_seed=rollout_seed,
        perturbation_seed=perturbation_seed,
    )
    randomness = GroupRandomness(
        crn_id=crn_id,
        rollout_seed=rollout_seed,
        perturbation_seed=perturbation_seed,
        candidate_seed=candidate_seed,
    )
    return bundle, randomness


PRIVILEGED_FEATURE_NAMES = tuple(
    [f"base_position_{axis}" for axis in "xyz"]
    + [f"base_quaternion_{axis}" for axis in "wxyz"]
    + [f"base_linear_velocity_{axis}" for axis in "xyz"]
    + [f"base_angular_velocity_{axis}" for axis in "xyz"]
    + [f"joint_position_{index:02d}" for index in range(12)]
    + [f"joint_velocity_{index:02d}" for index in range(12)]
    + ["contact_count"]
)

NATIVE_POC_PROFILE_NAME = "native_poc_v1"
NATIVE_POC_PROFILE_SCOPE = "development_boundary_mechanism_only"
NATIVE_POC_DEFAULTS = {
    "natural_acceptance_probability": 0.50,
    "settle_seconds": 0.05,
    "source_linear_std_mps": 1.0,
    "source_angular_std_radps": 4.0,
    "branch_policy_steps": [8, 16],
    "branch_linear_std_mps": 1.0,
    "branch_angular_std_radps": 4.0,
    "actor_sample_max_delta_rms": 0.50,
    "perturbation_radius_rms": 0.25,
}


def privileged_features(env: Any) -> np.ndarray:
    """Extract simulator-only diagnostics without entering deployable data."""
    value = np.concatenate([
        np.asarray(env.data.qpos[:7], dtype=np.float32),
        np.asarray(env.data.qvel[:6], dtype=np.float32),
        np.asarray(env.data.qpos[env.qpos_addresses], dtype=np.float32),
        np.asarray(env.data.qvel[env.qvel_addresses], dtype=np.float32),
        np.asarray([env.data.ncon], dtype=np.float32),
    ])
    if value.shape != (len(PRIVILEGED_FEATURE_NAMES),) or not np.all(
            np.isfinite(value)):
        raise ValueError("invalid privileged simulator feature vector")
    return value


@dataclass(frozen=True)
class NativeCollectionConfig:
    split: str
    target_groups: int
    source_seed: int
    policy_training_seed: int
    horizon_steps: int = 32
    replicas: int = 8
    natural_acceptance_probability: float = 0.50
    max_episode_steps: int = 100
    max_groups_per_trajectory: int = 5
    max_source_steps: int = 100_000
    settle_seconds: float = 0.05
    source_impulse_interval_steps: int = 10
    source_linear_std_mps: float = 1.0
    source_angular_std_radps: float = 4.0
    discovery_replicas: int | None = None
    audit_replicas: int | None = None
    profile_name: str = NATIVE_POC_PROFILE_NAME
    profile_scope: str = NATIVE_POC_PROFILE_SCOPE
    evidence_limit: str = (
        "strong-impulse boundary/mechanism development data only; does not "
        "replace natural closed-loop paired evaluation or independent online "
        "fall-reduction gates")

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("collection split must be nonempty")
        assert_development_path(Path(self.split))
        for name in (
            "target_groups", "source_seed", "policy_training_seed",
            "horizon_steps", "replicas", "max_episode_steps",
            "max_groups_per_trajectory", "max_source_steps",
            "source_impulse_interval_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            minimum = 0 if name in ("source_seed", "policy_training_seed") else 1
            if int(value) < minimum:
                raise ValueError(f"{name} must be >= {minimum}")
        if self.max_groups_per_trajectory > self.max_episode_steps:
            raise ValueError(
                "max_groups_per_trajectory cannot exceed max_episode_steps")
        if self.target_groups > self.max_source_steps:
            raise ValueError("max_source_steps cannot be below target_groups")
        if self.horizon_steps > np.iinfo(np.int16).max - 1:
            raise ValueError(
                "horizon_steps must leave room for H+1 in first_failure_step")
        for name in (
            "natural_acceptance_probability", "settle_seconds",
            "source_linear_std_mps", "source_angular_std_radps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.natural_acceptance_probability <= 1.0:
            raise ValueError("natural acceptance probability must lie in (0,1]")
        if self.settle_seconds < 0.0 or self.source_linear_std_mps < 0.0 or (
                self.source_angular_std_radps < 0.0):
            raise ValueError("settle time and source impulse scales must be nonnegative")
        partition_counts = (self.discovery_replicas, self.audit_replicas)
        if (partition_counts[0] is None) != (partition_counts[1] is None):
            raise ValueError(
                "discovery_replicas and audit_replicas must be supplied together")
        if partition_counts[0] is not None:
            for name, value in zip(
                    ("discovery_replicas", "audit_replicas"),
                    partition_counts, strict=True):
                if isinstance(value, bool) or not isinstance(
                        value, (int, np.integer)) or int(value) <= 0:
                    raise ValueError(f"{name} must be a positive integer")
            if int(partition_counts[0]) + int(partition_counts[1]) != self.replicas:
                raise ValueError(
                    "discovery plus audit replicas must equal total replicas")
        for name in ("profile_name", "profile_scope", "evidence_limit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be nonempty text")
        assert_development_path(Path(self.profile_name))

    def replica_partition_manifest(self) -> dict[str, Any] | None:
        """Return the pre-outcome replica role contract, if requested."""
        if self.discovery_replicas is None:
            return None
        assert self.audit_replicas is not None
        discovery = int(self.discovery_replicas)
        audit = int(self.audit_replicas)
        return {
            "schema_version": "qsafe.independent_replica_partition.v2",
            "assignment_timing": "before_candidate_outcomes",
            "axis": "replica",
            "ordering": "discovery_then_audit",
            "discovery_indices": list(range(discovery)),
            "audit_indices": list(range(discovery, discovery + audit)),
            "discovery_replicas": discovery,
            "audit_replicas": audit,
            "exhaustive": True,
        }


@dataclass(frozen=True)
class NativeCollectionResult:
    dataset: GroupedBranchDataset
    privileged: PrivilegedBranchView
    source_steps: int
    episodes: int
    near_failure_groups: int
    randomly_accepted_groups: int
    skipped_candidate_support_groups: int = 0


def _rng_for(*parts: int) -> np.random.Generator:
    return np.random.default_rng(_derived_seed(*parts))


def _action_application_contract(env: Any) -> dict[str, Any]:
    applier = env.action_applier
    return {
        "q_target_semantic": "absolute_joint_position_sent",
        "init_qpos": np.asarray(applier.init_qpos, dtype=float).tolist(),
        "action_offset": np.asarray(applier.action_offset, dtype=float).tolist(),
        "joint_min": np.asarray(applier.joint_min, dtype=float).tolist(),
        "joint_max": np.asarray(applier.joint_max, dtype=float).tolist(),
        "projection": "clip_normalized_then_joint_bounds_then_slew_then_filter",
    }


def _policy_identity(policy: Any, role: str) -> tuple[dict[str, Any], str]:
    """Return one stable policy manifest/fingerprint pair or fail closed."""
    manifest_method = getattr(policy, "manifest", None)
    fingerprint_method = getattr(policy, "fingerprint", None)
    if not callable(manifest_method) or not callable(fingerprint_method):
        raise TypeError(
            f"{role} policy must expose manifest() and fingerprint()")
    manifest = copy.deepcopy(dict(manifest_method()))
    fingerprint = fingerprint_method()
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"{role} policy fingerprint must be nonempty text")
    recorded = manifest.get("policy_fingerprint_sha256")
    if recorded is not None and recorded != fingerprint:
        raise ValueError(
            f"{role} policy manifest fingerprint disagrees with fingerprint()")
    return manifest, fingerprint


def _profile_parameters(
    config: NativeCollectionConfig,
    candidate_config: Any,
    branch_disturbance: GaussianImpulseSchedule,
) -> dict[str, Any]:
    return {
        "natural_acceptance_probability": float(
            config.natural_acceptance_probability),
        "settle_seconds": float(config.settle_seconds),
        "source_linear_std_mps": float(config.source_linear_std_mps),
        "source_angular_std_radps": float(config.source_angular_std_radps),
        "branch_policy_steps": list(branch_disturbance.policy_steps),
        "branch_linear_std_mps": float(branch_disturbance.linear_std_mps),
        "branch_angular_std_radps": float(branch_disturbance.angular_std_radps),
        "actor_sample_max_delta_rms": float(
            candidate_config.actor_sample_max_delta_rms),
        "perturbation_radius_rms": float(
            candidate_config.perturbation_radius_rms),
    }


def collect_native_groups(
    *,
    env: Any,
    source_policy: Any,
    continuation_policy: Any,
    candidate_config: Any,
    branch_disturbance: GaussianImpulseSchedule,
    config: NativeCollectionConfig,
    generator_commit: str,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> NativeCollectionResult:
    """Collect pre-outcome-selected native groups from a frozen SAC policy."""
    from safety_data.candidates import (
        ACTOR_SAMPLE_COUNT,
        InsufficientCandidateSupportError,
        build_evidence_candidates,
    )
    from safety_data.native import evaluate_same_state_group

    candidate_protocol = candidate_config.manifest_protocol()
    source_manifest, source_fingerprint = _policy_identity(
        source_policy, "source")
    continuation_manifest, continuation_fingerprint = _policy_identity(
        continuation_policy, "continuation")
    couple_source_and_nominal = (
        source_fingerprint == continuation_fingerprint)
    profile_parameters = _profile_parameters(
        config, candidate_config, branch_disturbance)
    assembler = GroupedBranchAssembler(
        split=config.split,
        horizon_steps=config.horizon_steps,
        generator_commit=generator_commit,
        simulator_fingerprint=env.simulator_fingerprint(),
        source_policy=source_manifest,
        continuation_policy=continuation_manifest,
        candidate_protocol=candidate_protocol,
        fall_definition={
            "max_abs_roll_pitch_rad": float(env.cfg.fallen_orientation_rad),
            "min_base_height_m": 0.18,
        },
        action_application_contract=_action_application_contract(env),
        collection_protocol={
            "version": (
                "qsafe.native_collection.v3"
                if config.replica_partition_manifest() is not None
                else "qsafe.native_collection.v2"),
            "profile_name": config.profile_name,
            "scope": config.profile_scope,
            "profile_parameters": profile_parameters,
            "profile_default_parameters": copy.deepcopy(NATIVE_POC_DEFAULTS),
            "profile_default_parameters_match": (
                profile_parameters == NATIVE_POC_DEFAULTS),
            "evidence_limit": config.evidence_limit,
            "selection_timing": "before_candidate_outcomes",
            "boundary_rule": "physical_near_failure_or_random_accept",
            "natural_acceptance_probability": float(
                config.natural_acceptance_probability),
            "sampling_stratum_array": "sampling_stratum",
            "sampling_strata": {
                "physical_near_failure": {
                    "predicate": (
                        "not_failure_and_(tilt>=fallen_risk_rad_or_height<0.25)"
                    ),
                    "acceptance_probability": 1.0,
                },
                "random_accept": {
                    "predicate": "not_physical_near_failure_and_uniform_draw_lt_p",
                    "acceptance_probability": float(
                        config.natural_acceptance_probability),
                },
            },
            "ipw_estimand": (
                "declared impulse-driven source-policy state stream under "
                "the declared adaptive trajectory termination; not a natural "
                "no-impulse closed-loop state distribution"
            ),
            "trajectory_termination": {
                "failure": True,
                "max_episode_steps": int(config.max_episode_steps),
                "accepted_group_cap": int(config.max_groups_per_trajectory),
                "accepted_group_cap_is_adaptive": True,
            },
            "source_impulse": {
                "interval_policy_steps": config.source_impulse_interval_steps,
                "linear_std_mps": config.source_linear_std_mps,
                "angular_std_radps": config.source_angular_std_radps,
            },
            "branch_disturbance": branch_disturbance.manifest(),
            "replica_seed_contract": "explicit_three_stream_v1",
            "source_policy_action": "externally_seeded_stochastic_frozen_actor",
            "candidate_nominal_action": (
                "externally_seeded_stochastic_continuation_actor"),
            "source_nominal_coupling": (
                "same action iff source and continuation fingerprints match"),
            "source_and_continuation_fingerprints_match": (
                couple_source_and_nominal),
            "continuation": (
                "externally_seeded_stochastic_frozen_actor_repeated_each_policy_step"),
            "outcome_conditioned_rejection": False,
            **(
                {"replica_partition": config.replica_partition_manifest()}
                if config.replica_partition_manifest() is not None else {}),
        },
        privileged_feature_names=PRIVILEGED_FEATURE_NAMES,
    )
    source_steps = 0
    episode_number = 0
    episode_step = 0
    episode_groups = 0
    near_failure_groups = 0
    random_groups = 0
    skipped_candidate_support_groups = 0

    def reset() -> None:
        nonlocal episode_number, episode_step, episode_groups
        env.reset_standing(
            settle_seconds=config.settle_seconds,
            rng=_rng_for(config.source_seed, episode_number, 20),
        )
        episode_step = 0
        episode_groups = 0

    reset()
    while assembler.group_count < config.target_groups:
        if source_steps >= config.max_source_steps:
            raise RuntimeError(
                "native collector exhausted max_source_steps before reaching "
                f"target groups ({assembler.group_count}/{config.target_groups})")
        if episode_step > 0 and (
                episode_step % config.source_impulse_interval_steps == 0):
            impulse_rng = _rng_for(
                config.source_seed, episode_number, episode_step, 21)
            env.apply_base_velocity_impulse(
                linear_velocity_delta=impulse_rng.normal(
                    0.0, config.source_linear_std_mps, size=3),
                angular_velocity_delta=impulse_rng.normal(
                    0.0, config.source_angular_std_radps, size=3),
            )
        history = env.record_observation()
        observation = history[-1]
        nominal = None
        if couple_source_and_nominal:
            nominal = continuation_policy.sample_action(
                observation,
                _rng_for(config.source_seed, episode_number, episode_step, 22),
            )
            source_action = nominal.copy()
        else:
            source_action = source_policy.sample_action(
                observation,
                _rng_for(config.source_seed, episode_number, episode_step, 23),
            )
        measurement = env.measurement()
        random_draw = float(_rng_for(
            config.source_seed, episode_number, episode_step, 24).random())
        accepted_near = bool(measurement.near_failure)
        accepted_random = bool(
            not accepted_near
            and random_draw < config.natural_acceptance_probability)
        if accepted_near or accepted_random:
            snapshot = env.capture()
            privileged_at_snapshot = privileged_features(env)
            accepted_index = assembler.group_count
            seed_bundle, randomness = group_randomness(
                source_seed=config.source_seed,
                group_index=accepted_index,
                replicas=config.replicas,
            )
            if nominal is None:
                nominal = continuation_policy.sample_action(
                    observation,
                    _rng_for(
                        config.source_seed, episode_number, episode_step, 22),
                )
            deterministic = continuation_policy.deterministic_action(observation)
            actor_samples = np.stack([
                continuation_policy.sample_action(
                    observation,
                    _rng_for(
                        config.source_seed, episode_number, episode_step,
                        25, sample_index),
                )
                for sample_index in range(ACTOR_SAMPLE_COUNT)
            ])
            try:
                candidate_arguments = {
                    "nominal": nominal,
                    "deterministic_mean": deterministic,
                    "previous_requested": env.previous_action_requested,
                    "actor_samples": actor_samples,
                    "action_applier": env.action_applier,
                    "current_qpos": np.asarray(
                        env.data.qpos[env.qpos_addresses], dtype=np.float32),
                    "candidate_seed": randomness.candidate_seed,
                }
                custom_builder = getattr(
                    candidate_config, "build_candidates", None)
                candidates = (
                    custom_builder(**candidate_arguments)
                    if callable(custom_builder)
                    else build_evidence_candidates(
                        **candidate_arguments, config=candidate_config)
                )
            except InsufficientCandidateSupportError:
                # Candidate support is known before any branch rollout.  Keep
                # advancing the source trajectory, but do not admit this state
                # as a collected group or condition on any branch outcome.
                skipped_candidate_support_groups += 1
            else:
                evaluation = evaluate_same_state_group(
                    env,
                    snapshot,
                    candidates.requested,
                    seed_bundle,
                    horizon_steps=config.horizon_steps,
                    continuation_policy=continuation_policy,
                    disturbance_program=branch_disturbance,
                    option_steps=getattr(candidates, "option_steps", None),
                )
                for name in ("candidate_requested", "candidate_executed",
                             "candidate_q_target"):
                    expected_name = {
                        "candidate_requested": "requested",
                        "candidate_executed": "executed",
                        "candidate_q_target": "q_target",
                    }[name]
                    if not np.array_equal(
                            np.asarray(getattr(evaluation, name)),
                            np.asarray(getattr(candidates, expected_name))):
                        raise RuntimeError(
                            f"branch execution disagrees with previewed {name}")
                trajectory_id = (
                    f"{config.split}:source-{config.source_seed}:"
                    f"episode-{episode_number}")
                group_id = f"{trajectory_id}:step-{episode_step}"
                assembler.add(CollectedGroup(
                    identity=GroupIdentity(
                        group_id=group_id,
                        state_hash=snapshot.compound_sha256(),
                        trajectory_id=trajectory_id,
                        episode_id=_derived_seed(
                            config.source_seed, episode_number, 26),
                        episode_step=episode_step,
                        policy_training_seed=config.policy_training_seed,
                        source_seed=config.source_seed,
                        policy_source=source_fingerprint,
                        command_vx=float(env.cfg.move_speed),
                        acceptance_probability=(
                            1.0 if accepted_near
                            else config.natural_acceptance_probability),
                        sampling_stratum=(
                            "physical_near_failure"
                            if accepted_near else "random_accept"),
                    ),
                    observation_history=history,
                    candidate_kind=candidates.kind,
                    candidate_mask=candidates.mask,
                    evaluation=evaluation,
                    randomness=randomness,
                    candidate_option_steps=getattr(
                        candidates, "option_steps", None),
                    privileged_features=privileged_at_snapshot,
                ))
                near_failure_groups += int(accepted_near)
                random_groups += int(accepted_random)
                episode_groups += 1
                if progress is not None:
                    progress({
                        "groups": assembler.group_count,
                        "target_groups": config.target_groups,
                        "source_steps": source_steps + 1,
                        "episode": episode_number,
                        "near_failure_groups": near_failure_groups,
                        "randomly_accepted_groups": random_groups,
                        "skipped_candidate_support_groups": (
                            skipped_candidate_support_groups),
                        "valid_candidates": candidates.valid_count,
                        "group_fall_fraction": float(np.mean(
                            evaluation.fall[candidates.mask])),
                    })
        step_result = env.step(source_action)
        source_steps += 1
        episode_step += 1
        should_reset = bool(
            step_result.failure
            or episode_step >= config.max_episode_steps
            or episode_groups >= config.max_groups_per_trajectory)
        if should_reset and assembler.group_count < config.target_groups:
            episode_number += 1
            reset()
    dataset, privileged = assembler.finalize()
    assert privileged is not None
    return NativeCollectionResult(
        dataset=dataset,
        privileged=privileged,
        source_steps=source_steps,
        episodes=episode_number + 1,
        near_failure_groups=near_failure_groups,
        randomly_accepted_groups=random_groups,
        skipped_candidate_support_groups=skipped_candidate_support_groups,
    )


__all__ = [
    "CollectedGroup",
    "GaussianImpulseSchedule",
    "GroupIdentity",
    "GroupRandomness",
    "GroupedBranchAssembler",
    "NativeCollectionConfig",
    "NativeCollectionResult",
    "PRIVILEGED_FEATURE_NAMES",
    "group_randomness",
    "collect_native_groups",
    "privileged_features",
]
