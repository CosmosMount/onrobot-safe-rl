"""Native compound-snapshot same-state branch evaluator."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from train.mujoco_snapshot_env import BranchSnapshot, MujocoSnapshotEnv


class ContinuationPolicy(Protocol):
    def __call__(
        self, observation_history: np.ndarray, step: int,
        rng: np.random.Generator,
    ) -> np.ndarray: ...


class DisturbanceProgram(Protocol):
    def __call__(
        self, env: MujocoSnapshotEnv, step: int,
        rng: np.random.Generator,
    ) -> None: ...


def _capture_branch_state(component: Any, name: str) -> Any:
    capture = getattr(component, "capture_branch_state", None)
    restore = getattr(component, "restore_branch_state", None)
    if callable(capture) != callable(restore):
        raise TypeError(
            f"{name} must implement both capture_branch_state and "
            "restore_branch_state, or neither")
    return copy.deepcopy(capture()) if callable(capture) else None


def _restore_branch_state(component: Any, state: Any) -> None:
    restore = getattr(component, "restore_branch_state", None)
    if callable(restore):
        restore(copy.deepcopy(state))


@dataclass(frozen=True)
class NativeGroupEvaluation:
    candidate_requested: np.ndarray
    candidate_executed: np.ndarray
    candidate_q_target: np.ndarray
    fall: np.ndarray
    first_failure_step: np.ndarray
    max_tilt_rad: np.ndarray
    min_height_m: np.ndarray


def _rng(seed: int, stream: int, step: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([
        int(seed), int(stream), int(step)]))


def evaluate_same_state_group(
    env: MujocoSnapshotEnv,
    snapshot: BranchSnapshot,
    candidates: np.ndarray,
    replica_seeds: np.ndarray,
    *,
    horizon_steps: int,
    continuation_policy: ContinuationPolicy,
    disturbance_program: DisturbanceProgram | None = None,
) -> NativeGroupEvaluation:
    """Evaluate K actions with R common-random-number continuations."""
    candidate_values = np.asarray(candidates, dtype=np.float32)
    raw_seeds = np.asarray(replica_seeds)
    if raw_seeds.dtype.kind not in "iu" or raw_seeds.ndim != 1:
        raise ValueError("replica_seeds must be a one-dimensional integer array")
    if np.any(raw_seeds < 0):
        raise ValueError("replica seeds must be nonnegative")
    seeds = raw_seeds.astype(np.uint64, copy=False)
    if candidate_values.ndim != 2 or candidate_values.shape[1] != env.cfg.num_joints:
        raise ValueError("candidates must have shape [K, action_dim]")
    if len(candidate_values) < 2 or len(seeds) < 1:
        raise ValueError("same-state groups require K>=2 and R>=1")
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("replica seeds must be unique")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    candidate_count = len(candidate_values)
    replica_count = len(seeds)
    requested = np.empty((candidate_count, env.cfg.num_joints), np.float32)
    executed = np.empty_like(requested)
    q_target = np.empty_like(requested)
    fall = np.zeros((candidate_count, replica_count), dtype=bool)
    first_failure = np.full(
        (candidate_count, replica_count), horizon_steps + 1, dtype=np.int16)
    max_tilt = np.zeros((candidate_count, replica_count), dtype=np.float32)
    min_height = np.full(
        (candidate_count, replica_count), np.inf, dtype=np.float32)
    continuation_state = _capture_branch_state(
        continuation_policy, "continuation_policy")
    disturbance_state = (
        None if disturbance_program is None
        else _capture_branch_state(disturbance_program, "disturbance_program"))

    try:
        for candidate_index, first_action in enumerate(candidate_values):
            for replica_index, seed in enumerate(seeds):
                env.restore(snapshot)
                _restore_branch_state(continuation_policy, continuation_state)
                if disturbance_program is not None:
                    _restore_branch_state(disturbance_program, disturbance_state)
                for step in range(horizon_steps):
                    if disturbance_program is not None:
                        disturbance_program(
                            env, step, _rng(int(seed), 1, step))
                    action = (
                        first_action
                        if step == 0
                        else continuation_policy(
                            env.record_observation(), step,
                            _rng(int(seed), 0, step)))
                    result = env.step(action)
                    if step == 0:
                        if replica_index == 0:
                            requested[candidate_index] = (
                                result.application.action_requested)
                            executed[candidate_index] = (
                                result.application.action_executed)
                            q_target[candidate_index] = (
                                result.application.action_q_target)
                        else:
                            if not np.array_equal(
                                    requested[candidate_index],
                                    result.application.action_requested):
                                raise RuntimeError(
                                    "first action projection changed across "
                                    "CRN replicas")
                            if not np.array_equal(
                                    executed[candidate_index],
                                    result.application.action_executed):
                                raise RuntimeError(
                                    "executed first action changed across "
                                    "CRN replicas")
                            if not np.array_equal(
                                    q_target[candidate_index],
                                    result.application.action_q_target):
                                raise RuntimeError(
                                    "first q_target changed across CRN replicas")
                    max_tilt[candidate_index, replica_index] = max(
                        max_tilt[candidate_index, replica_index],
                        result.tilt_rad)
                    min_height[candidate_index, replica_index] = min(
                        min_height[candidate_index, replica_index],
                        result.height_m)
                    if result.failure:
                        fall[candidate_index, replica_index] = True
                        first_failure[candidate_index, replica_index] = step + 1
                        break
    finally:
        env.restore(snapshot)
        _restore_branch_state(continuation_policy, continuation_state)
        if disturbance_program is not None:
            _restore_branch_state(disturbance_program, disturbance_state)
    return NativeGroupEvaluation(
        candidate_requested=requested,
        candidate_executed=executed,
        candidate_q_target=q_target,
        fall=fall,
        first_failure_step=first_failure,
        max_tilt_rad=max_tilt,
        min_height_m=min_height,
    )
