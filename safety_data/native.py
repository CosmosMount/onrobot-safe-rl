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
class ReplicaSeedBundle:
    """One group's paired replica identities and independent RNG seeds.

    ``crn_id`` is an identity used to align the same replica across candidates;
    it is deliberately not an RNG seed.  Continuation actions derive only from
    ``rollout_seed`` and disturbances derive only from ``perturbation_seed``.
    All three arrays therefore map directly to the grouped dataset schema.

    Passing a legacy one-dimensional integer array to
    :func:`evaluate_same_state_group` remains supported.  It is converted with
    :meth:`from_legacy`, where all three arrays are equal, and marked with the
    explicit ``legacy_equal_seeds_v1`` contract.
    """

    crn_id: np.ndarray
    rollout_seed: np.ndarray
    perturbation_seed: np.ndarray
    seed_contract: str = "explicit_three_stream_v1"

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        expected_shape: tuple[int, ...] | None = None
        for name in ("crn_id", "rollout_seed", "perturbation_seed"):
            raw = np.asarray(getattr(self, name))
            if raw.dtype.kind not in "iu" or raw.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional integer array")
            if raw.size == 0:
                raise ValueError(f"{name} must contain at least one replica")
            if np.any(raw < 0):
                raise ValueError(f"{name} must contain nonnegative integers")
            if len(np.unique(raw)) != len(raw):
                raise ValueError(
                    f"{name} must be unique across replicas within the group")
            if expected_shape is None:
                expected_shape = raw.shape
            elif raw.shape != expected_shape:
                raise ValueError(
                    "crn_id, rollout_seed, and perturbation_seed must have "
                    "identical one-dimensional shapes")
            value = np.array(raw, dtype=np.uint64, copy=True)
            value.setflags(write=False)
            arrays[name] = value
        if self.seed_contract not in (
                "explicit_three_stream_v1", "legacy_equal_seeds_v1"):
            raise ValueError("unknown replica seed contract")
        if self.seed_contract == "legacy_equal_seeds_v1" and not (
                np.array_equal(arrays["crn_id"], arrays["rollout_seed"])
                and np.array_equal(
                    arrays["crn_id"], arrays["perturbation_seed"])):
            raise ValueError(
                "legacy_equal_seeds_v1 requires equal CRN, rollout, and "
                "perturbation arrays")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_legacy(cls, replica_seeds: np.ndarray) -> ReplicaSeedBundle:
        """Convert the former single seed vector without hiding its coupling."""
        raw = np.asarray(replica_seeds)
        return cls(
            crn_id=raw,
            rollout_seed=raw,
            perturbation_seed=raw,
            seed_contract="legacy_equal_seeds_v1",
        )

    @property
    def replica_count(self) -> int:
        return int(self.crn_id.shape[0])


@dataclass(frozen=True)
class NativeGroupEvaluation:
    candidate_requested: np.ndarray
    candidate_executed: np.ndarray
    candidate_q_target: np.ndarray
    fall: np.ndarray
    first_failure_step: np.ndarray
    max_tilt_rad: np.ndarray
    min_height_m: np.ndarray
    crn_id: np.ndarray
    rollout_seed: np.ndarray
    perturbation_seed: np.ndarray
    seed_contract: str


def _rng(seed: int, stream: int, step: int) -> np.random.Generator:
    """Derive one step RNG from one seed plus a fixed component domain tag."""
    return np.random.default_rng(np.random.SeedSequence([
        int(seed), int(stream), int(step)]))


def _checked_horizon_steps(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError("horizon_steps must be a positive integer")
    horizon = int(value)
    if horizon <= 0:
        raise ValueError("horizon_steps must be a positive integer")
    # The grouped evidence schema stores the H+1 non-failure sentinel in
    # int16.  Reject rather than silently wrapping an out-of-contract horizon.
    if horizon + 1 > np.iinfo(np.int16).max:
        raise ValueError("horizon_steps is too large for int16 failure labels")
    return horizon


def _checked_option_steps(
    value: np.ndarray | None,
    *,
    candidate_count: int,
) -> np.ndarray:
    """Return locked per-candidate recovery-option durations.

    Candidate zero is the nominal action and therefore has no residual option;
    its duration is locked to one step.  Recovery candidates may hold a
    linearly decaying residual for one through four steps.  ``None`` preserves
    the former one-step evaluator exactly.
    """
    if value is None:
        return np.ones(candidate_count, dtype=np.int64)
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or raw.ndim != 1 or raw.shape != (
            candidate_count,):
        raise ValueError("option_steps must be a one-dimensional integer [K] array")
    if raw[0] != 1:
        raise ValueError("nominal candidate option_steps[0] must equal 1")
    if np.any(raw < 1) or np.any(raw > 4):
        raise ValueError("recovery candidate option_steps must lie in [1, 4]")
    return np.array(raw, dtype=np.int64, copy=True)


def evaluate_same_state_group(
    env: MujocoSnapshotEnv,
    snapshot: BranchSnapshot,
    candidates: np.ndarray,
    replica_seeds: ReplicaSeedBundle | np.ndarray,
    *,
    horizon_steps: int,
    continuation_policy: ContinuationPolicy,
    disturbance_program: DisturbanceProgram | None = None,
    option_steps: np.ndarray | None = None,
) -> NativeGroupEvaluation:
    """Evaluate K actions/options with paired common-random continuations.

    Prefer an explicit :class:`ReplicaSeedBundle`.  A one-dimensional integer
    vector is accepted only as a backward-compatible equal-seed bundle; the
    returned ``seed_contract`` makes that legacy coupling observable.

    ``option_steps`` defaults to one for every candidate, which is exactly the
    former one-action branch contract.  When recovery candidate ``k`` has
    length ``L > 1``, step zero remains its exact requested action.  At steps
    ``1 .. L-1`` the evaluator calls the same seeded nominal continuation as
    every other candidate and adds the locked residual
    ``((L-step)/L) * (candidate[k] - candidate[0])``, clipping the combined
    normalized action to ``[-1, 1]``.  Steps at or after ``L`` use the nominal
    continuation unchanged.  Candidate zero must always have length one.
    """
    candidate_values = np.asarray(candidates, dtype=np.float32)
    seeds = (
        replica_seeds
        if isinstance(replica_seeds, ReplicaSeedBundle)
        else ReplicaSeedBundle.from_legacy(replica_seeds)
    )
    if candidate_values.ndim != 2 or candidate_values.shape[1] != env.cfg.num_joints:
        raise ValueError("candidates must have shape [K, action_dim]")
    if len(candidate_values) < 2:
        raise ValueError("same-state groups require K>=2 and R>=1")
    if not np.all(np.isfinite(candidate_values)):
        raise ValueError("candidates must contain only finite actions")
    if np.any(candidate_values < -1.0 - 1e-6) or np.any(
            candidate_values > 1.0 + 1e-6):
        raise ValueError("candidates must lie in normalized [-1, 1]")
    candidate_values = np.clip(candidate_values, -1.0, 1.0)
    horizon_steps = _checked_horizon_steps(horizon_steps)

    candidate_count = len(candidate_values)
    option_lengths = _checked_option_steps(
        option_steps, candidate_count=candidate_count)
    replica_count = seeds.replica_count
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
        nominal_action = candidate_values[0]
        for candidate_index, first_action in enumerate(candidate_values):
            option_length = int(option_lengths[candidate_index])
            option_residual = first_action - nominal_action
            for replica_index, (
                    crn_id, rollout_seed, perturbation_seed) in enumerate(zip(
                        seeds.crn_id,
                        seeds.rollout_seed,
                        seeds.perturbation_seed,
                        strict=True)):
                env.restore(snapshot)
                _restore_branch_state(continuation_policy, continuation_state)
                if disturbance_program is not None:
                    _restore_branch_state(disturbance_program, disturbance_state)
                for step in range(horizon_steps):
                    if disturbance_program is not None:
                        disturbance_program(
                            env, step, _rng(int(perturbation_seed), 1, step))
                    if step == 0:
                        action = first_action
                    else:
                        action = continuation_policy(
                            env.record_observation(), step,
                            _rng(int(rollout_seed), 0, step))
                        if step < option_length:
                            decay = (option_length - step) / option_length
                            action = np.clip(
                                action + decay * option_residual, -1.0, 1.0)
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
                                    f"CRN replica {int(crn_id)}")
                            if not np.array_equal(
                                    executed[candidate_index],
                                    result.application.action_executed):
                                raise RuntimeError(
                                    "executed first action changed across "
                                    f"CRN replica {int(crn_id)}")
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
        crn_id=seeds.crn_id.copy(),
        rollout_seed=seeds.rollout_seed.copy(),
        perturbation_seed=seeds.perturbation_seed.copy(),
        seed_contract=seeds.seed_contract,
    )
