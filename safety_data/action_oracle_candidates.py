"""Executable candidate space for the pre-training action-oracle gate."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import numpy as np

from runtime.inference.actions import ActionApplier, qpos_to_action
from runtime.inference.observations import quat_to_euler_xyz
from safety_data.candidates import (
    CANDIDATE_KINDS as LOCAL_CANDIDATE_KINDS,
    CandidateSet,
)


ACTION_ORACLE_CANDIDATE_VERSION = "qsafe.action_oracle_candidates.v1"
STATE_DEPENDENT_KINDS = (
    "tilt_support_008",
    "tilt_support_016",
    "tilt_opposite_008",
    "tilt_opposite_016",
    "gyro_support_008",
    "gyro_support_016",
    "capture_step_008",
    "capture_step_016",
)
ACTION_ORACLE_KINDS = LOCAL_CANDIDATE_KINDS + STATE_DEPENDENT_KINDS
ACTION_ORACLE_COUNT = len(ACTION_ORACLE_KINDS)


@dataclass(frozen=True)
class ActionOracleCandidateSet:
    requested: np.ndarray
    executed: np.ndarray
    q_target: np.ndarray
    kind: np.ndarray
    mask: np.ndarray
    candidate_seed: int
    manifest_protocol: dict[str, Any]

    def __post_init__(self) -> None:
        for name in ("requested", "executed", "q_target"):
            value = np.asarray(getattr(self, name), dtype=np.float32).copy()
            if value.shape != (ACTION_ORACLE_COUNT, 12) or not np.all(
                    np.isfinite(value)):
                raise ValueError(f"{name} must be finite [24,12]")
            if name != "q_target" and (np.any(value < -1.0 - 1e-6) or np.any(
                    value > 1.0 + 1e-6)):
                raise ValueError(f"{name} must lie in [-1,1]")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        kind = np.asarray(self.kind, dtype=str).copy()
        mask = np.asarray(self.mask, dtype=bool).copy()
        if kind.shape != (ACTION_ORACLE_COUNT,) or tuple(kind) != ACTION_ORACLE_KINDS:
            raise ValueError("action oracle candidate kinds differ from protocol")
        if mask.shape != (ACTION_ORACLE_COUNT,) or not mask[0] or np.count_nonzero(
                mask) < 12:
            raise ValueError("action oracle candidates require at least 12 unique targets")
        kind.setflags(write=False); mask.setflags(write=False)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "candidate_seed", int(self.candidate_seed))
        object.__setattr__(self, "manifest_protocol",
                           copy.deepcopy(dict(self.manifest_protocol)))

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.mask))


def _deduplicate(targets: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    mask = np.zeros(len(targets), dtype=bool)
    retained: list[np.ndarray] = []
    for index, target in enumerate(targets):
        if not any(np.allclose(target, prior, atol=atol, rtol=0.0)
                   for prior in retained):
            mask[index] = True
            retained.append(target)
    return mask


def _support_target(
    base: np.ndarray, *, roll: float, pitch: float,
    magnitude: float, direction: float,
) -> np.ndarray:
    side = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    fore = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    raw = -(side * float(roll) + fore * float(pitch))
    scale = max(float(np.max(np.abs(raw))), 1e-6)
    extension = direction * float(magnitude) * np.clip(raw / scale, -1.0, 1.0)
    target = np.asarray(base, dtype=np.float32).copy()
    target[1::3] -= 0.50 * extension
    target[2::3] += extension
    return target


def _capture_target(
    base: np.ndarray, *, roll: float, pitch: float, magnitude: float,
) -> np.ndarray:
    side = np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32)
    fore = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    target = np.asarray(base, dtype=np.float32).copy()
    roll_scale = float(np.clip(roll / 0.35, -1.0, 1.0))
    pitch_scale = float(np.clip(pitch / 0.35, -1.0, 1.0))
    # Hip ab/adduction shifts the lateral support polygon; thigh motion shifts
    # front/rear foot placement.  Bounds/projection remain owned by ActionApplier.
    target[0::3] += float(magnitude) * side * roll_scale
    target[1::3] -= float(magnitude) * fore * pitch_scale
    return target


def build_action_oracle_candidates(
    local: CandidateSet, *, observation_history: np.ndarray,
    action_applier: ActionApplier,
) -> ActionOracleCandidateSet:
    """Append eight deployable state-feedback actions to the locked local set."""
    history = np.asarray(observation_history, dtype=np.float32)
    if history.shape != (5, 46) or not np.all(np.isfinite(history)):
        raise ValueError("observation_history must be finite [5,46]")
    newest = history[-1]
    roll, pitch, _ = quat_to_euler_xyz(newest[30:34])
    gyro_roll, gyro_pitch = map(float, newest[24:26])
    predicted_roll = float(roll) + 0.08 * gyro_roll
    predicted_pitch = float(pitch) + 0.08 * gyro_pitch
    base = np.asarray(local.q_target[0], dtype=np.float32)
    targets = [
        _support_target(base, roll=predicted_roll, pitch=predicted_pitch,
                        magnitude=0.08, direction=1.0),
        _support_target(base, roll=predicted_roll, pitch=predicted_pitch,
                        magnitude=0.16, direction=1.0),
        _support_target(base, roll=predicted_roll, pitch=predicted_pitch,
                        magnitude=0.08, direction=-1.0),
        _support_target(base, roll=predicted_roll, pitch=predicted_pitch,
                        magnitude=0.16, direction=-1.0),
        _support_target(base, roll=0.08 * gyro_roll, pitch=0.08 * gyro_pitch,
                        magnitude=0.08, direction=1.0),
        _support_target(base, roll=0.08 * gyro_roll, pitch=0.08 * gyro_pitch,
                        magnitude=0.16, direction=1.0),
        _capture_target(base, roll=predicted_roll, pitch=predicted_pitch,
                        magnitude=0.08),
        _capture_target(base, roll=predicted_roll, pitch=predicted_pitch,
                        magnitude=0.16),
    ]
    joint_min = np.asarray(action_applier.joint_min, dtype=np.float32)
    joint_max = np.asarray(action_applier.joint_max, dtype=np.float32)
    requested_extra = np.stack([
        np.clip(qpos_to_action(
            np.clip(target, joint_min, joint_max),
            init_qpos=action_applier.init_qpos,
            action_offset=action_applier.action_offset), -1.0, 1.0)
        for target in targets
    ]).astype(np.float32)
    previews = action_applier.preview_many(
        requested_extra, np.asarray(newest[:12], dtype=np.float32))
    projected_requested = np.stack([
        item.action_requested for item in previews], axis=0)
    projected_executed = np.stack([
        item.action_executed for item in previews], axis=0)
    projected_q_target = np.stack([
        item.action_q_target for item in previews], axis=0)
    requested = np.concatenate([local.requested, projected_requested], axis=0)
    executed = np.concatenate([local.executed, projected_executed], axis=0)
    q_target = np.concatenate([local.q_target, projected_q_target], axis=0)
    mask = _deduplicate(q_target)
    protocol = {
        "version": ACTION_ORACLE_CANDIDATE_VERSION,
        "count": ACTION_ORACLE_COUNT,
        "nominal_index": 0,
        "ordered_kinds": list(ACTION_ORACLE_KINDS),
        "local_candidate_protocol": local.manifest_protocol,
        "state_feedback_inputs": [
            "corrected_quaternion", "corrected_gyro", "current_joint_position",
        ],
        "lookahead_seconds_for_attitude": 0.08,
        "support_target_magnitudes_rad": [0.08, 0.16],
        "capture_step_magnitudes_rad": [0.08, 0.16],
        "future_branch_outcome_used": False,
        "privileged_state_used": False,
        "execution": "one_policy_step_then_receding_reobservation",
        "projection": "runtime_ActionApplier_preview_many",
    }
    return ActionOracleCandidateSet(
        requested=requested, executed=executed, q_target=q_target,
        kind=np.asarray(ACTION_ORACLE_KINDS), mask=mask,
        candidate_seed=local.candidate_seed, manifest_protocol=protocol)
