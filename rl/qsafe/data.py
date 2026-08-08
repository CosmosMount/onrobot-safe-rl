"""Group-preserving PyTorch views over evidence-safe branch datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch

from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView


FeatureView = Literal["deployable", "privileged"]
ActionView = Literal["application_concat", "requested"]


@dataclass(frozen=True)
class NormalizationStats:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    privileged_mean: np.ndarray | None = None
    privileged_std: np.ndarray | None = None

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
        dataset.validate()
        observation = np.asarray(dataset["obs_history"], dtype=np.float64)
        observation_mean = observation.mean(axis=(0, 1))
        observation_std = np.maximum(observation.std(axis=(0, 1)), 1e-6)
        if privileged is None:
            return cls(observation_mean, observation_std)
        privileged.validate(dataset)
        features = np.asarray(privileged.features, dtype=np.float64)
        return cls(
            observation_mean,
            observation_std,
            features.mean(axis=0),
            np.maximum(features.std(axis=0), 1e-6),
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
    ):
        dataset.validate()
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
        if action_view not in ("application_concat", "requested"):
            raise ValueError(f"unknown action_view={action_view!r}")
        self.dataset = dataset
        self.normalization = normalization
        self.privileged = privileged
        self.feature_view: FeatureView = (
            "privileged" if privileged is not None else "deployable")
        self.action_view: ActionView = action_view
        self.observation = (
            np.asarray(dataset["obs_history"], dtype=np.float32)
            - normalization.observation_mean[None, None, :]
        ) / normalization.observation_std[None, None, :]
        if not np.all(np.isfinite(self.observation)):
            raise ValueError("normalized observations contain non-finite values")
        self.mask = np.asarray(dataset["candidate_mask"], dtype=bool)
        nominal_requested = np.asarray(
            dataset["nominal_action_requested"], dtype=np.float32)
        candidate_requested = np.asarray(
            dataset["candidate_requested"], dtype=np.float32)
        if action_view == "application_concat":
            raw_candidate = np.concatenate([
                candidate_requested,
                np.asarray(dataset["candidate_executed"], dtype=np.float32),
                np.asarray(dataset["candidate_q_target"], dtype=np.float32),
            ], axis=2)
            # Candidate zero is the complete nominal application tuple.  Only
            # its requested component has a separate nominal field on disk.
            self.nominal = raw_candidate[:, 0].copy()
        else:
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
        group_weight = (
            float(probability.min()) / probability).astype(np.float32)
        if not np.all(np.isfinite(group_weight)) or np.any(group_weight <= 0.0):
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
