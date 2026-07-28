"""Counterfactual action branches stored separately from online SAC replay.

The branch collector restores one exact simulator integration state for every
candidate action.  A candidate is executed for the first policy interval and a
frozen continuation policy is used afterwards.  One maximum-horizon rollout is
enough to derive labels for all shorter horizons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import pickle
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np


FORMAT_VERSION = 'counterfactual_branch_v1'


@dataclass(frozen=True)
class BranchSnapshot:
    """Exact simulator state plus policy-side temporal context."""

    simulator_state: np.ndarray
    observation: np.ndarray
    previous_action: np.ndarray
    previous_executed_action: np.ndarray
    command_speed: float
    episode_id: int = 0
    policy_step: int = 0


@dataclass(frozen=True)
class BranchMeasurement:
    failure: bool
    near_failure: bool
    base_tilt_rad: float
    base_height_m: float
    contact_count: int = 0
    undesired_contact_count: int = 0
    max_contact_force: float = 0.0


@dataclass(frozen=True)
class HorizonOutcome:
    horizon: int
    failure: bool
    near_failure: bool
    time_to_failure: int
    max_tilt_rad: float
    min_base_height_m: float
    max_contact_count: int
    max_undesired_contact_count: int
    max_contact_force: float

    @property
    def binary_risk(self) -> float:
        return float(self.failure)


@dataclass(frozen=True)
class CandidateBranch:
    snapshot_index: int
    candidate_index: int
    candidate_family: str
    observation: np.ndarray
    action: np.ndarray
    nominal_action: np.ndarray
    previous_action: np.ndarray
    command_speed: float
    action_distance: float
    outcomes: dict[int, HorizonOutcome]
    nominal_safety_improvement: dict[int, float] = field(default_factory=dict)


class BranchBackend(Protocol):
    """Minimal backend needed by the simulator-independent branch engine."""

    def restore_state(self, state: np.ndarray) -> None:
        ...

    def observation(self, previous_action: np.ndarray,
                    previous_executed_action: np.ndarray,
                    command_speed: float) -> np.ndarray:
        ...

    def step_action(self, action: np.ndarray) -> BranchMeasurement:
        ...


ContinuationPolicy = Callable[[np.ndarray], np.ndarray]


def summarize_measurements(
        measurements: Sequence[BranchMeasurement],
        horizon: int) -> HorizonOutcome:
    if horizon <= 0:
        raise ValueError('branch horizon must be positive')
    selected = list(measurements[:horizon])
    if not selected:
        raise ValueError('cannot summarize an empty branch')
    failure_steps = [
        index for index, item in enumerate(selected, start=1)
        if item.failure
    ]
    return HorizonOutcome(
        horizon=int(horizon),
        failure=bool(failure_steps),
        near_failure=any(item.near_failure for item in selected),
        time_to_failure=(failure_steps[0] if failure_steps else -1),
        max_tilt_rad=max(item.base_tilt_rad for item in selected),
        min_base_height_m=min(item.base_height_m for item in selected),
        max_contact_count=max(item.contact_count for item in selected),
        max_undesired_contact_count=max(
            item.undesired_contact_count for item in selected),
        max_contact_force=max(item.max_contact_force for item in selected),
    )


def rollout_candidate(
        backend: BranchBackend,
        snapshot: BranchSnapshot,
        candidate_action: np.ndarray,
        continuation_policy: ContinuationPolicy,
        *,
        horizons: Sequence[int] = (8, 16, 32),
) -> dict[int, HorizonOutcome]:
    """Restore ``snapshot`` and evaluate one first-step counterfactual action."""
    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or horizons[0] <= 0:
        raise ValueError('branch horizons must be positive')
    backend.restore_state(np.asarray(snapshot.simulator_state).copy())
    action = np.asarray(candidate_action, dtype=np.float32)
    previous_action = np.asarray(
        snapshot.previous_action, dtype=np.float32).copy()
    previous_executed = np.asarray(
        snapshot.previous_executed_action, dtype=np.float32).copy()
    measurements: list[BranchMeasurement] = []

    for step in range(horizons[-1]):
        if step:
            observation = backend.observation(
                previous_action, previous_executed, snapshot.command_speed)
            action = np.asarray(
                continuation_policy(observation), dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        measurement = backend.step_action(action)
        measurements.append(measurement)
        previous_action = action.copy()
        previous_executed = action.copy()
        if measurement.failure:
            break

    # A terminated branch remains failed at every later horizon.  Repeat its
    # terminal measurement so shorter physical execution does not lose labels.
    if len(measurements) < horizons[-1]:
        measurements.extend(
            [measurements[-1]] * (horizons[-1] - len(measurements)))
    return {
        horizon: summarize_measurements(measurements, horizon)
        for horizon in horizons
    }


def evaluate_snapshot_candidates(
        backend: BranchBackend,
        snapshot: BranchSnapshot,
        candidates: Sequence[tuple[str, np.ndarray]],
        continuation_policy: ContinuationPolicy,
        *,
        snapshot_index: int,
        horizons: Sequence[int] = (8, 16, 32),
) -> list[CandidateBranch]:
    """Evaluate candidates and attach improvement relative to ``nominal``."""
    if not candidates:
        raise ValueError('at least one branch candidate is required')
    nominal_indices = [
        index for index, (family, _) in enumerate(candidates)
        if family == 'nominal'
    ]
    if len(nominal_indices) != 1:
        raise ValueError('candidates must contain exactly one nominal action')
    nominal_index = nominal_indices[0]
    raw: list[tuple[str, np.ndarray, dict[int, HorizonOutcome]]] = []
    for family, action in candidates:
        clipped = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        raw.append((
            str(family), clipped,
            rollout_candidate(
                backend, snapshot, clipped, continuation_policy,
                horizons=horizons)))
    nominal_action = raw[nominal_index][1]
    nominal_outcomes = raw[nominal_index][2]

    records = []
    for candidate_index, (family, action, outcomes) in enumerate(raw):
        improvement = {
            horizon: (
                nominal_outcomes[horizon].binary_risk
                - outcomes[horizon].binary_risk)
            for horizon in outcomes
        }
        records.append(CandidateBranch(
            snapshot_index=int(snapshot_index),
            candidate_index=int(candidate_index),
            candidate_family=family,
            observation=np.asarray(
                snapshot.observation, dtype=np.float32).copy(),
            action=action.copy(),
            nominal_action=nominal_action.copy(),
            previous_action=np.asarray(
                snapshot.previous_action, dtype=np.float32).copy(),
            command_speed=float(snapshot.command_speed),
            action_distance=float(np.linalg.norm(action - nominal_action)),
            outcomes=outcomes,
            nominal_safety_improvement=improvement,
        ))
    return records


def make_candidate_actions(
        nominal_action: np.ndarray,
        previous_action: np.ndarray,
        *,
        rng: np.random.Generator,
        perturbation_count: int = 8,
        perturbation_std: float = 0.15,
        contraction: float = 0.90,
) -> list[tuple[str, np.ndarray]]:
    """Construct the four required candidate families."""
    if perturbation_count < 0:
        raise ValueError('perturbation_count must be non-negative')
    nominal = np.asarray(nominal_action, dtype=np.float32)
    previous = np.asarray(previous_action, dtype=np.float32)
    if nominal.shape != previous.shape:
        raise ValueError('nominal and previous actions must have equal shape')
    candidates: list[tuple[str, np.ndarray]] = [
        ('nominal', nominal.copy()),
    ]
    for _ in range(perturbation_count):
        delta = rng.normal(0.0, perturbation_std, size=nominal.shape)
        candidates.append((
            'nominal_delta',
            np.clip(nominal + delta, -1.0, 1.0).astype(np.float32)))
    candidates.extend([
        ('previous', previous.copy()),
        ('contracted_previous',
         np.clip(contraction * previous, -1.0, 1.0).astype(np.float32)),
    ])
    return candidates


def save_counterfactual_artifact(
        path: str | Path,
        *,
        snapshots: Sequence[BranchSnapshot],
        branches: Sequence[CandidateBranch],
        metadata: dict[str, object] | None = None) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': FORMAT_VERSION,
        'metadata': dict(metadata or {}),
        'snapshots': list(snapshots),
        'branches': list(branches),
    }
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    try:
        with temporary.open('wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_counterfactual_artifact(path: str | Path) -> dict[str, object]:
    with Path(path).open('rb') as stream:
        payload = pickle.load(stream)
    if payload.get('format') != FORMAT_VERSION:
        raise ValueError(
            f'Unsupported counterfactual artifact: {payload.get("format")}')
    if 'snapshots' not in payload or 'branches' not in payload:
        raise ValueError('Incomplete counterfactual branch artifact')
    return payload

