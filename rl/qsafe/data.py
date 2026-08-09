"""Group-preserving PyTorch views over evidence-safe branch datasets."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from runtime.inference.actions import action_to_qpos, qpos_to_action
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_VIEW,
    build_recovery_program_features,
    validate_recovery_program_binding,
)


FeatureView = Literal["deployable", "privileged"]
ViewRole = Literal["training", "calibration", "test"]
ActionView = Literal[
    "application_concat",
    "requested",
    "recovery_program_v1",
]

_RECOVERY_ACTION_VECTOR_FIELDS = (
    "init_qpos",
    "action_offset",
    "joint_min",
    "joint_max",
)
_RECOVERY_ACTION_PROJECTION = (
    "clip_normalized_then_joint_bounds_then_slew_then_filter")
_RECOVERY_ACTION_CONTRACT_FIELDS = frozenset({
    "q_target_semantic",
    *_RECOVERY_ACTION_VECTOR_FIELDS,
    "projection",
    "max_joint_delta",
    "use_action_filter",
})


def _validate_recovery_action_application_contract(
    action_contract: object,
    recovery_binding: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Bind the collected action application to the recovery library exactly."""
    if not isinstance(action_contract, Mapping):
        raise ValueError(
            "recovery_program_v1 requires an action-application contract")
    if set(action_contract) != _RECOVERY_ACTION_CONTRACT_FIELDS:
        missing = sorted(_RECOVERY_ACTION_CONTRACT_FIELDS - set(action_contract))
        extra = sorted(set(action_contract) - _RECOVERY_ACTION_CONTRACT_FIELDS)
        raise ValueError(
            "recovery_program_v1 action-application contract must have the "
            f"exact locked keyset; missing={missing}, extra={extra}")
    recovery_manifest = recovery_binding.get("manifest")
    if not isinstance(recovery_manifest, Mapping):
        raise ValueError(
            "recovery_program_v1 recovery-program manifest is incomplete")
    recovery_projection = recovery_manifest.get("action_projection")
    if not isinstance(recovery_projection, Mapping):
        raise ValueError(
            "recovery_program_v1 recovery action projection is incomplete")

    if action_contract.get(
            "q_target_semantic") != "absolute_joint_position_sent":
        raise ValueError(
            "recovery_program_v1 action-application q_target semantics drifted")
    if action_contract.get("projection") != _RECOVERY_ACTION_PROJECTION:
        raise ValueError(
            "recovery_program_v1 action-application projection semantics "
            "drifted")

    projection_vectors: dict[str, np.ndarray] = {}
    for field in _RECOVERY_ACTION_VECTOR_FIELDS:
        try:
            collected = np.asarray(action_contract.get(field))
            recovery = np.asarray(recovery_projection.get(field))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "recovery_program_v1 action-application contract cannot be "
                "compared with the recovery action projection") from exc
        if collected.shape != (12,) or recovery.shape != (12,) or not (
                np.array_equal(collected, recovery, equal_nan=False)):
            raise ValueError(
                "recovery_program_v1 action-application contract field "
                f"{field!r} differs elementwise from the recovery action "
                "projection")
        projection_vectors[field] = np.asarray(
            collected, dtype=np.float32).copy()

    if recovery_projection.get("max_joint_delta") is not None or (
            recovery_projection.get("use_action_filter") is not False):
        raise ValueError(
            "recovery_program_v1 recovery action projection must disable "
            "slew limiting and action filtering")
    if action_contract.get("max_joint_delta") is not None:
        raise ValueError(
            "recovery_program_v1 action-application max_joint_delta "
            "semantics differ from the recovery action projection")
    if action_contract.get("use_action_filter") is not False:
        raise ValueError(
            "recovery_program_v1 action-application filter semantics differ "
            "from the recovery action projection")
    return projection_vectors


def _validate_recovery_candidate_projection(
    *,
    candidate_requested: np.ndarray,
    candidate_executed: np.ndarray,
    candidate_q_target: np.ndarray,
    candidate_mask: np.ndarray,
    projection_vectors: Mapping[str, np.ndarray],
) -> None:
    """Replay every valid K9 application tuple with the runtime projection."""
    requested = np.asarray(candidate_requested)
    executed = np.asarray(candidate_executed)
    q_target = np.asarray(candidate_q_target)
    mask = np.asarray(candidate_mask)
    for group_index, candidate_index in np.argwhere(mask):
        expected_q_target = action_to_qpos(
            requested[group_index, candidate_index],
            init_qpos=projection_vectors["init_qpos"],
            action_offset=projection_vectors["action_offset"],
            joint_min=projection_vectors["joint_min"],
            joint_max=projection_vectors["joint_max"],
        )
        if not np.array_equal(
                q_target[group_index, candidate_index],
                expected_q_target,
                equal_nan=False):
            raise ValueError(
                "recovery_program_v1 candidate_q_target is not the exact "
                "runtime action_to_qpos projection of candidate_requested "
                f"at group={group_index}, candidate={candidate_index}")
        expected_executed = qpos_to_action(
            q_target[group_index, candidate_index],
            init_qpos=projection_vectors["init_qpos"],
            action_offset=projection_vectors["action_offset"],
        )
        if not np.array_equal(
                executed[group_index, candidate_index],
                expected_executed,
                equal_nan=False):
            raise ValueError(
                "recovery_program_v1 candidate_executed is not the exact "
                "runtime qpos_to_action projection of candidate_q_target "
                f"at group={group_index}, candidate={candidate_index}")


def _validated_dataset_identity(
    dataset: GroupedBranchDataset,
    validation_report: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the validated immutable content/split identity of one fit set."""
    content_sha256 = validation_report.get("content_sha256")
    split = dataset.manifest.get("split")
    if not isinstance(content_sha256, str) or len(content_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in content_sha256):
        raise ValueError("dataset validation did not produce a content SHA-256")
    manifest_content_sha256 = dataset.manifest.get("content_sha256")
    if manifest_content_sha256 is not None:
        if not isinstance(manifest_content_sha256, str) or (
                manifest_content_sha256 != content_sha256):
            raise ValueError(
                "dataset manifest content SHA-256 differs from validated "
                "content")
        content_sha256 = manifest_content_sha256
    if not isinstance(split, str) or not split:
        raise ValueError("dataset manifest split must be nonempty text")
    return content_sha256, split


@dataclass(frozen=True)
class NormalizationStats:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    privileged_mean: np.ndarray | None = None
    privileged_std: np.ndarray | None = None
    fit_content_sha256: str | None = None
    fit_split: str | None = None

    def __post_init__(self) -> None:
        observation_mean = np.asarray(
            self.observation_mean, dtype=np.float32).reshape(-1).copy()
        observation_std = np.asarray(
            self.observation_std, dtype=np.float32).reshape(-1).copy()
        if observation_mean.shape != observation_std.shape or np.any(
                observation_std <= 0.0) or not np.all(
                    np.isfinite(observation_mean)) or not np.all(
                        np.isfinite(observation_std)):
            raise ValueError("invalid observation normalization statistics")
        object.__setattr__(self, "observation_mean", observation_mean)
        object.__setattr__(self, "observation_std", observation_std)
        observation_mean.setflags(write=False)
        observation_std.setflags(write=False)
        if (self.fit_content_sha256 is None) != (self.fit_split is None):
            raise ValueError(
                "normalization fit content SHA-256 and split must both be "
                "present or absent")
        if self.fit_content_sha256 is not None:
            if not isinstance(self.fit_content_sha256, str) or len(
                    self.fit_content_sha256) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in self.fit_content_sha256):
                raise ValueError(
                    "normalization fit_content_sha256 must be a lowercase "
                    "SHA-256 digest")
            if not isinstance(self.fit_split, str) or not self.fit_split:
                raise ValueError(
                    "normalization fit_split must be nonempty text")
        if self.privileged_mean is None or self.privileged_std is None:
            if self.privileged_mean is not None or self.privileged_std is not None:
                raise ValueError("privileged mean/std must both be present or absent")
            return
        privileged_mean = np.asarray(
            self.privileged_mean, dtype=np.float32).reshape(-1).copy()
        privileged_std = np.asarray(
            self.privileged_std, dtype=np.float32).reshape(-1).copy()
        if privileged_mean.shape != privileged_std.shape or np.any(
                privileged_std <= 0.0) or not np.all(
                    np.isfinite(privileged_mean)) or not np.all(
                        np.isfinite(privileged_std)):
            raise ValueError("invalid privileged normalization statistics")
        object.__setattr__(self, "privileged_mean", privileged_mean)
        object.__setattr__(self, "privileged_std", privileged_std)
        privileged_mean.setflags(write=False)
        privileged_std.setflags(write=False)

    def equivalent_to(self, other: object) -> bool:
        """Return exact equality for train-fitted preprocessing provenance."""
        if not isinstance(other, NormalizationStats):
            return False
        if self.fit_content_sha256 != other.fit_content_sha256 or (
                self.fit_split != other.fit_split):
            return False
        for left, right in (
            (self.observation_mean, other.observation_mean),
            (self.observation_std, other.observation_std),
            (self.privileged_mean, other.privileged_mean),
            (self.privileged_std, other.privileged_std),
        ):
            if (left is None) != (right is None):
                return False
            if left is not None and not np.array_equal(left, right):
                return False
        return True

    @classmethod
    def fit(
        cls,
        dataset: GroupedBranchDataset,
        privileged: PrivilegedBranchView | None = None,
    ) -> "NormalizationStats":
        validation_report = dataset.validate()
        fit_content_sha256, fit_split = _validated_dataset_identity(
            dataset, validation_report)
        observation = np.asarray(dataset["obs_history"], dtype=np.float64)
        observation_mean = observation.mean(axis=(0, 1))
        observation_std = np.maximum(observation.std(axis=(0, 1)), 1e-6)
        if privileged is None:
            return cls(
                observation_mean,
                observation_std,
                fit_content_sha256=fit_content_sha256,
                fit_split=fit_split,
            )
        privileged.validate(dataset)
        features = np.asarray(privileged.features, dtype=np.float64)
        return cls(
            observation_mean,
            observation_std,
            privileged_mean=features.mean(axis=0),
            privileged_std=np.maximum(features.std(axis=0), 1e-6),
            fit_content_sha256=fit_content_sha256,
            fit_split=fit_split,
        )


@dataclass(frozen=True)
class GroupBatch:
    observation_history: torch.Tensor
    nominal_action: torch.Tensor
    candidate_action: torch.Tensor
    candidate_mask: torch.Tensor
    fall: torch.Tensor
    first_failure_step: torch.Tensor
    max_tilt_rad: torch.Tensor
    min_height_m: torch.Tensor
    group_weight: torch.Tensor
    privileged_state: torch.Tensor | None


class TorchGroupedView:
    """A normalized dataset view whose item and batch unit remains a group."""

    def __init__(
        self,
        dataset: GroupedBranchDataset,
        normalization: NormalizationStats,
        privileged: PrivilegedBranchView | None = None,
        *,
        allow_mixed_command_speeds: bool = False,
        action_view: ActionView = "application_concat",
        view_role: ViewRole = "training",
    ):
        validation_report = dataset.validate()
        if "candidate_option_steps" in dataset.arrays:
            raise ValueError(
                "recovery-option datasets require a duration-aware v2 model "
                "and a passed independent-replica label gate; the v1 Q_safe "
                "view must not collapse distinct option durations")
        if privileged is not None:
            privileged.validate(dataset)
        command = np.asarray(dataset["command_vx"], dtype=np.float64)
        if not allow_mixed_command_speeds and float(command.max() - command.min()) > 1e-6:
            raise ValueError(
                "current 46D observation has no velocity command; mixed command "
                "speeds require separate Q_safe models")
        if normalization.observation_mean.shape != (46,):
            raise ValueError("normalization does not match 46D observation")
        if privileged is None and normalization.privileged_mean is not None:
            raise ValueError("deployable view must not receive privileged statistics")
        if privileged is not None:
            if normalization.privileged_mean is None or (
                    normalization.privileged_mean.shape != (
                        privileged.features.shape[1],)):
                raise ValueError("normalization does not match privileged feature width")
        if action_view not in (
                "application_concat", "requested", RECOVERY_PROGRAM_VIEW):
            raise ValueError(f"unknown action_view={action_view!r}")
        if view_role not in ("training", "calibration", "test"):
            raise ValueError(f"unknown view_role={view_role!r}")
        self.dataset = dataset
        self.normalization = normalization
        self.privileged = privileged
        self.feature_view: FeatureView = (
            "privileged" if privileged is not None else "deployable")
        self.action_view: ActionView = action_view
        self.view_role: ViewRole = view_role
        self._recovery_program_binding: dict[str, object] | None = None
        self._recovery_program_feature_manifest: dict[str, object] | None = None
        self.recovery_program_feature_contract_sha256: str | None = None
        self.recovery_library_fingerprint_sha256: str | None = None
        self.observation = (
            np.asarray(dataset["obs_history"], dtype=np.float32)
            - normalization.observation_mean[None, None, :]
        ) / normalization.observation_std[None, None, :]
        if not np.all(np.isfinite(self.observation)):
            raise ValueError("normalized observations contain non-finite values")
        self.mask = np.asarray(dataset["candidate_mask"], dtype=bool)
        nominal_requested = np.asarray(
            dataset["nominal_action_requested"], dtype=np.float32)
        candidate_requested_on_disk = np.asarray(dataset["candidate_requested"])
        if action_view == RECOVERY_PROGRAM_VIEW:
            current_content_sha256, current_split = _validated_dataset_identity(
                dataset, validation_report)
            if normalization.fit_content_sha256 is None or (
                    normalization.fit_split is None):
                raise ValueError(
                    "recovery_program_v1 requires normalization fit "
                    "content/split provenance")
            if view_role == "training" and (
                    normalization.fit_content_sha256
                    != current_content_sha256 or
                    normalization.fit_split != current_split):
                raise ValueError(
                    "recovery_program_v1 training view requires fit "
                    "normalization provenance matching this dataset content "
                    "and split")
            if view_role == "training":
                exact_fit = NormalizationStats.fit(dataset, privileged)
                if not normalization.equivalent_to(exact_fit):
                    raise ValueError(
                        "recovery_program_v1 training normalization must "
                        "exactly equal NormalizationStats.fit on this dataset")
            recovery_binding = dataset.manifest.get("recovery_program")
            feature_manifest = dataset.manifest.get(
                "recovery_program_feature_contract")
            if not isinstance(recovery_binding, dict) or not isinstance(
                    feature_manifest, dict):
                raise ValueError(
                    "recovery_program_v1 requires recovery-program and feature "
                    "manifest bindings")
            library_fingerprint = validate_recovery_program_binding(
                recovery_binding)
            projection_vectors = _validate_recovery_action_application_contract(
                dataset.manifest.get("action_application_contract"),
                recovery_binding,
            )
            feature_fingerprint = feature_manifest.get(
                "feature_contract_sha256")
            features = build_recovery_program_features(
                # Do not cast here.  V4's training/runtime bit contract requires
                # the evidence file itself to carry native float32 application
                # tuples; otherwise a float64 producer bug would be hidden at
                # the training boundary while runtime inference fails closed.
                candidate_requested=candidate_requested_on_disk,
                candidate_executed=np.asarray(dataset["candidate_executed"]),
                candidate_q_target=np.asarray(dataset["candidate_q_target"]),
                candidate_names=np.asarray(dataset["candidate_kind"]),
                candidate_behavior_steps=np.asarray(
                    dataset["candidate_behavior_steps"]),
                candidate_mask=np.asarray(dataset["candidate_mask"]),
                nominal_index=0,
                feature_manifest=feature_manifest,
                feature_manifest_fingerprint_sha256=feature_fingerprint,
                recovery_library_fingerprint_sha256=library_fingerprint,
            )
            _validate_recovery_candidate_projection(
                candidate_requested=candidate_requested_on_disk,
                candidate_executed=np.asarray(dataset["candidate_executed"]),
                candidate_q_target=np.asarray(dataset["candidate_q_target"]),
                candidate_mask=np.asarray(dataset["candidate_mask"]),
                projection_vectors=projection_vectors,
            )
            raw_candidate = features.candidate_descriptor
            self.nominal = features.nominal_descriptor
            self._recovery_program_binding = copy.deepcopy(recovery_binding)
            self._recovery_program_feature_manifest = copy.deepcopy(
                feature_manifest)
            self.recovery_program_feature_contract_sha256 = (
                features.feature_contract_sha256)
            self.recovery_library_fingerprint_sha256 = (
                features.recovery_library_fingerprint_sha256)
        elif action_view == "application_concat":
            candidate_requested = np.asarray(
                candidate_requested_on_disk, dtype=np.float32)
            raw_candidate = np.concatenate([
                candidate_requested,
                np.asarray(dataset["candidate_executed"], dtype=np.float32),
                np.asarray(dataset["candidate_q_target"], dtype=np.float32),
            ], axis=2)
            # Candidate zero is the complete nominal application tuple.  Only
            # its requested component has a separate nominal field on disk.
            self.nominal = raw_candidate[:, 0].copy()
        else:
            candidate_requested = np.asarray(
                candidate_requested_on_disk, dtype=np.float32)
            raw_candidate = candidate_requested
            self.nominal = nominal_requested
        # Invalid candidates are allowed to carry arbitrary sentinels on disk.
        # The network still evaluates its dense [G,K,*] tensor before the loss
        # mask is applied, so replace those sentinels with the finite nominal
        # action to prevent 0 * NaN terms from poisoning parameter gradients.
        self.candidate = np.where(
            self.mask[..., None], raw_candidate, self.nominal[:, None, :])
        if not np.all(np.isfinite(self.candidate)) or not np.all(
                np.isfinite(self.nominal)):
            raise ValueError("action features contain non-finite values")
        if not np.array_equal(
                self.candidate[:, 0], self.nominal, equal_nan=False):
            raise ValueError("candidate zero is not aligned with nominal action features")
        outcome_mask = self.mask[..., None]
        self.fall = np.where(
            outcome_mask,
            np.asarray(dataset["fall"], dtype=np.float32),
            0.0,
        ).astype(np.float32, copy=False)
        self.first_failure_step = np.where(
            outcome_mask,
            np.asarray(dataset["first_failure_step"], dtype=np.int64),
            dataset.horizon_steps + 1,
        ).astype(np.int64, copy=False)
        self.max_tilt = np.where(
            outcome_mask,
            np.asarray(dataset["max_tilt_rad"], dtype=np.float32),
            0.0,
        ).astype(np.float32, copy=False)
        self.min_height = np.where(
            outcome_mask,
            np.asarray(dataset["min_height_m"], dtype=np.float32),
            0.0,
        ).astype(np.float32, copy=False)
        probability = np.asarray(
            dataset["acceptance_probability"], dtype=np.float64)
        if action_view == RECOVERY_PROGRAM_VIEW:
            if not np.array_equal(
                    probability,
                    np.ones(dataset.group_count, dtype=np.float64),
                    equal_nan=False):
                raise ValueError(
                    "recovery_program_v1 requires exact unit acceptance "
                    "probability for every group; IPW is forbidden")
            group_weight = np.ones(dataset.group_count, dtype=np.float32)
        else:
            group_weight = (
                float(probability.min()) / probability).astype(np.float32)
            if not np.all(np.isfinite(group_weight)) or np.any(
                    group_weight <= 0.0):
                raise ValueError(
                    "acceptance-probability range cannot be represented by "
                    "positive float32 IPW weights")
        self.group_weight = group_weight
        self.privileged_features = None
        if privileged is not None:
            assert normalization.privileged_mean is not None
            assert normalization.privileged_std is not None
            self.privileged_features = (
                np.asarray(privileged.features, dtype=np.float32)
                - normalization.privileged_mean[None, :]
            ) / normalization.privileged_std[None, :]
            if not np.all(np.isfinite(self.privileged_features)):
                raise ValueError(
                    "normalized privileged features contain non-finite values")

    @property
    def group_count(self) -> int:
        return self.dataset.group_count

    @property
    def privileged_dim(self) -> int:
        return 0 if self.privileged_features is None else int(
            self.privileged_features.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.candidate.shape[2])

    @property
    def recovery_program_binding(self) -> dict[str, object] | None:
        """Return an isolated copy of the exact V4 recovery-library binding."""
        return copy.deepcopy(self._recovery_program_binding)

    @property
    def recovery_program_feature_manifest(self) -> dict[str, object] | None:
        """Return an isolated copy of the exact V4 feature manifest."""
        return copy.deepcopy(self._recovery_program_feature_manifest)

    @property
    def trajectory_id(self) -> np.ndarray:
        return np.asarray(self.dataset["trajectory_id"]).astype(str)

    @property
    def command_vx(self) -> float:
        command = np.asarray(self.dataset["command_vx"], dtype=np.float64)
        if float(command.max() - command.min()) > 1e-6:
            raise ValueError("mixed command-speed view has no scalar command_vx")
        return float(command[0])

    def batch(
        self,
        indices: Sequence[int] | np.ndarray,
        device: torch.device | str,
    ) -> GroupBatch:
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if len(selected) == 0 or np.any(selected < 0) or np.any(
                selected >= self.group_count):
            raise IndexError("group batch indices are empty or out of range")

        def tensor(value: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(
                np.ascontiguousarray(value[selected]), dtype=dtype,
                device=device)

        privileged = (
            None if self.privileged_features is None
            else tensor(self.privileged_features, torch.float32))
        return GroupBatch(
            observation_history=tensor(self.observation, torch.float32),
            nominal_action=tensor(self.nominal, torch.float32),
            candidate_action=tensor(self.candidate, torch.float32),
            candidate_mask=tensor(self.mask, torch.bool),
            fall=tensor(self.fall, torch.float32),
            first_failure_step=tensor(self.first_failure_step, torch.int64),
            max_tilt_rad=tensor(self.max_tilt, torch.float32),
            min_height_m=tensor(self.min_height, torch.float32),
            group_weight=tensor(self.group_weight, torch.float32),
            privileged_state=privileged,
        )

    def all_indices(self) -> np.ndarray:
        return np.arange(self.group_count, dtype=np.int64)


def trajectory_bootstrap_indices(
    trajectory_id: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Sample complete trajectory clusters with replacement."""
    trajectory = np.asarray(trajectory_id).astype(str).reshape(-1)
    unique = np.unique(trajectory)
    if len(unique) == 0:
        raise ValueError("trajectory bootstrap requires at least one cluster")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    indices = np.concatenate([
        np.flatnonzero(trajectory == name) for name in sampled
    ]).astype(np.int64)
    return indices, sampled.tolist()
