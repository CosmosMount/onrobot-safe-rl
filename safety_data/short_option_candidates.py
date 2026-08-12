"""Outcome-blind physical residual directions for short PPO options."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_SCALE = np.asarray([0.2, 0.4, 0.4] * 4, np.float32)
PROPOSALS = 64
DIRECTIONS = 5
DURATIONS = (1, 4, 8)
BETA = {
    1: np.asarray([1.0], np.float32),
    4: np.asarray([1.0, 0.75, 0.50, 0.25], np.float32),
    8: np.asarray([1.0, 0.875, 0.75, 0.625, 0.50, 0.375, 0.25, 0.125],
                  np.float32),
}


def normalized_physical_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, np.float32)
    second = np.asarray(second, np.float32)
    if first.shape[-1] != 12 or second.shape[-1] != 12:
        raise ValueError("physical actions must end in 12 joints")
    return np.sqrt(np.mean(np.square((first - second) / ACTION_SCALE), axis=-1))


def project_physical_target(
    target: np.ndarray, joint_lower: np.ndarray, joint_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, np.float32)
    lower = np.asarray(joint_lower, np.float32)
    upper = np.asarray(joint_upper, np.float32)
    if target.shape[-1] != 12 or lower.shape != (12,) or upper.shape != (12,):
        raise ValueError("projection requires 12D physical joint targets")
    if np.any(lower >= upper) or not np.all(np.isfinite(target)):
        raise ValueError("invalid physical target or joint limits")
    projected = np.clip(target, lower, upper).astype(np.float32)
    saturated = projected != target
    return projected, saturated


@dataclass(frozen=True)
class ResidualSelection:
    nominal: np.ndarray
    proposal_indices: np.ndarray
    selected_targets: np.ndarray
    residuals: np.ndarray
    nominal_distance: np.ndarray


def select_farthest_residuals(
    nominal: np.ndarray, proposals: np.ndarray,
) -> ResidualSelection:
    """Greedy farthest-point selection; no outcome can enter this API."""
    nominal = np.asarray(nominal, np.float32)
    proposals = np.asarray(proposals, np.float32)
    if nominal.shape != (12,) or proposals.shape != (PROPOSALS, 12):
        raise ValueError("selection requires one nominal and exactly 64 proposals")
    if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(proposals)):
        raise ValueError("candidate actions must be finite")
    selected: list[int] = []
    reference = nominal[None]
    for _ in range(DIRECTIONS):
        minimum = np.min(
            normalized_physical_distance(proposals[:, None, :], reference[None, :, :]),
            axis=1,
        )
        if selected:
            minimum[np.asarray(selected)] = -np.inf
        index = int(np.argmax(minimum))  # first index is deterministic tie-break
        if not np.isfinite(minimum[index]) or minimum[index] <= 0.0:
            raise RuntimeError("fewer than five distinct physical residual directions")
        selected.append(index)
        reference = np.concatenate((reference, proposals[index:index + 1]), axis=0)
    indices = np.asarray(selected, np.int16)
    targets = proposals[indices].copy()
    residuals = targets - nominal
    return ResidualSelection(
        nominal=nominal.copy(), proposal_indices=indices,
        selected_targets=targets, residuals=residuals,
        nominal_distance=normalized_physical_distance(targets, nominal).astype(np.float32),
    )


def option_candidate_layout(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.asarray(residuals, np.float32)
    if residuals.shape != (DIRECTIONS, 12):
        raise ValueError("option layout requires five 12D residuals")
    durations = np.asarray([0] + [value for value in DURATIONS for _ in range(DIRECTIONS)],
                           np.int8)
    directions = np.asarray([-1] + list(range(DIRECTIONS)) * len(DURATIONS), np.int8)
    return durations, directions


def apply_closed_loop_residual(
    nominal_physical: np.ndarray,
    residual: np.ndarray,
    *,
    duration: int,
    option_step: int,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if duration not in BETA or not 0 <= option_step < duration:
        raise ValueError("invalid preregistered duration or active option step")
    combined = (np.asarray(nominal_physical, np.float32)
                + BETA[duration][option_step] * np.asarray(residual, np.float32))
    return project_physical_target(combined, joint_lower, joint_upper)
