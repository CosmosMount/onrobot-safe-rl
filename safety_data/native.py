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


class RecoveryProgram(Protocol):
    """Closed-loop recovery behavior selected by candidate index.

    Candidate zero is the nominal continuation and has duration zero.  The
    evaluator calls the program only for an active nonnominal candidate.  At
    continuation steps, ``nominal_action`` has already been sampled so common
    rollout RNG streams advance identically before recovery overrides it.
    """

    @property
    def behavior_steps(self) -> np.ndarray: ...

    def __call__(
        self,
        candidate_index: int,
        observation_history: np.ndarray,
        step: int,
        nominal_action: np.ndarray,
    ) -> np.ndarray: ...


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


def _checked_behavior_steps(
    recovery_program: RecoveryProgram,
    *,
    candidate_count: int,
    horizon_steps: int,
) -> np.ndarray:
    """Validate the closed-loop durations declared by a recovery program."""
    if not callable(recovery_program):
        raise TypeError("recovery_program must be callable")
    raw = np.asarray(recovery_program.behavior_steps)
    if raw.dtype.kind not in "iu" or raw.ndim != 1 or raw.shape != (
            candidate_count,):
        raise ValueError(
            "recovery_program.behavior_steps must be a one-dimensional "
            "integer [K] array")
    if raw[0] != 0:
        raise ValueError(
            "nominal candidate recovery_program.behavior_steps[0] must equal 0")
    if np.any(raw[1:] < 1) or np.any(raw[1:] > horizon_steps):
        raise ValueError(
            "nonnominal recovery_program behavior steps must lie in [1, H]")
    return np.array(raw, dtype=np.int64, copy=True)


def _checked_recovery_action(
    value: np.ndarray,
    *,
    action_dim: int,
    candidate_index: int,
    step: int,
) -> np.ndarray:
    """Return one finite, normalized action from a recovery program."""
    raw = np.asarray(value)
    if raw.shape != (action_dim,):
        raise ValueError(
            "recovery_program output must have shape [action_dim]; "
            f"candidate={candidate_index}, step={step}")
    try:
        action = np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "recovery_program output must contain numeric actions; "
            f"candidate={candidate_index}, step={step}") from exc
    if not np.all(np.isfinite(action)):
        raise ValueError(
            "recovery_program output must contain only finite actions; "
            f"candidate={candidate_index}, step={step}")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError(
            "recovery_program output must lie in normalized [-1, 1]; "
            f"candidate={candidate_index}, step={step}")
    return np.array(action, dtype=np.float32, copy=True)


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
    recovery_program: RecoveryProgram | None = None,
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

    ``recovery_program`` enables an independent closed-loop behavior and is
    mutually exclusive with ``option_steps``.  Candidate zero then has
    behavior duration zero and remains nominal.  At step zero the program gets
    the non-recording environment history view; its action must exactly match
    the corresponding ``candidates`` preview before the environment advances.
    At subsequent active steps, the paired nominal continuation is evaluated
    first and then overridden by the recovery program.
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
    if recovery_program is not None and option_steps is not None:
        raise ValueError("recovery_program and option_steps are mutually exclusive")
    if recovery_program is None:
        option_lengths = _checked_option_steps(
            option_steps, candidate_count=candidate_count)
        behavior_lengths = None
    else:
        option_lengths = None
        behavior_lengths = _checked_behavior_steps(
            recovery_program,
            candidate_count=candidate_count,
            horizon_steps=horizon_steps,
        )
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
    recovery_state = (
        None if recovery_program is None
        else _capture_branch_state(recovery_program, "recovery_program"))

    try:
        nominal_action = candidate_values[0]
        for candidate_index, first_action in enumerate(candidate_values):
            if recovery_program is None:
                assert option_lengths is not None
                option_length = int(option_lengths[candidate_index])
                option_residual = first_action - nominal_action
                behavior_length = 0
            else:
                assert behavior_lengths is not None
                option_length = 0
                option_residual = None
                behavior_length = int(behavior_lengths[candidate_index])
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
                if recovery_program is not None:
                    _restore_branch_state(recovery_program, recovery_state)
                for step in range(horizon_steps):
                    if disturbance_program is not None:
                        disturbance_program(
                            env, step, _rng(int(perturbation_seed), 1, step))
                    if step == 0:
                        if recovery_program is not None and candidate_index > 0:
                            action = _checked_recovery_action(
                                recovery_program(
                                    candidate_index,
                                    env.observation_history(),
                                    step,
                                    nominal_action.copy(),
                                ),
                                action_dim=env.cfg.num_joints,
                                candidate_index=candidate_index,
                                step=step,
                            )
                            if not np.array_equal(action, first_action):
                                raise RuntimeError(
                                    "recovery_program step-zero action differs "
                                    "from candidates preview; "
                                    f"candidate={candidate_index}")
                        else:
                            action = first_action
                    else:
                        observation_history = env.record_observation()
                        nominal_continuation = continuation_policy(
                            observation_history, step,
                            _rng(int(rollout_seed), 0, step))
                        action = nominal_continuation
                        if (recovery_program is not None
                                and candidate_index > 0
                                and step < behavior_length):
                            action = _checked_recovery_action(
                                recovery_program(
                                    candidate_index,
                                    observation_history,
                                    step,
                                    np.asarray(
                                        nominal_continuation,
                                        dtype=np.float32,
                                    ).copy(),
                                ),
                                action_dim=env.cfg.num_joints,
                                candidate_index=candidate_index,
                                step=step,
                            )
                        elif recovery_program is None and step < option_length:
                            assert option_residual is not None
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
        if recovery_program is not None:
            _restore_branch_state(recovery_program, recovery_state)
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
