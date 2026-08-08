"""Exact-snapshot repeated Q_safe versus nominal closed-loop evaluation.

The paired unit is one unique simulator snapshot from one source trajectory.
Both arms restore that same compound snapshot and receive the same stochastic
actor-noise and disturbance seeds at each policy step.  Candidate-search RNG
is a separate shield-only namespace, so adding candidates cannot change the
nominal continuation stream.

This module intentionally separates the mechanism runner from state-source
sampling.  A claim-bearing caller must supply independently sampled snapshots;
the strong-impulse ``native_poc_v1`` collector is development data and cannot
stand in for a natural/online Phase-1 endpoint.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from rl.qsafe.artifact import LoadedQSafeArtifact
from rl.qsafe.runtime import QSafeRuntimeResult, run_qsafe_step
from rl.qsafe.selector import SelectorConfig
from safety_data.candidates import (
    ACTOR_SAMPLE_COUNT,
    CandidateSet,
    EvidenceCandidateConfig,
    build_evidence_candidates,
)
from train.mujoco_snapshot_env import BranchSnapshot, MujocoSnapshotEnv


PAIRED_SEED_CONTRACT = "qsafe.paired_closed_loop_four_stream_v1"
_STREAM_NOMINAL_ACTOR = 0
_STREAM_DISTURBANCE = 1
_STREAM_CANDIDATE_ACTOR = 2
_STREAM_CANDIDATE_GEOMETRY = 3


class FrozenActor(Protocol):
    @property
    def training_step(self) -> int: ...

    def manifest(self) -> Mapping[str, Any]: ...

    def sample_action(
        self, observation: np.ndarray, rng: np.random.Generator,
    ) -> np.ndarray: ...

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray: ...


class FrozenRewardQ(Protocol):
    @property
    def training_step(self) -> int: ...

    def manifest(self) -> Mapping[str, Any]: ...

    def conservative_values(
        self, observation: np.ndarray, requested_actions: np.ndarray,
    ) -> np.ndarray: ...


class DisturbanceProgram(Protocol):
    def __call__(
        self, env: MujocoSnapshotEnv, step: int,
        rng: np.random.Generator,
    ) -> None: ...


class ShieldController(Protocol):
    def decide(
        self,
        env: MujocoSnapshotEnv,
        observation_history: np.ndarray,
        nominal_action: np.ndarray,
        *,
        pair_seed: int,
        step: int,
    ) -> "ShieldStepDecision": ...


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _derived_seed(pair_seed: int, stream: int, step: int, substream: int = 0) -> int:
    """Derive a platform-stable 63-bit seed from explicit domain components."""
    values = (
        _nonnegative_int(pair_seed, "pair_seed"),
        _nonnegative_int(stream, "stream"),
        _nonnegative_int(step, "step"),
        _nonnegative_int(substream, "substream"),
    )
    digest = hashlib.sha256(b"qsafe_paired_closed_loop_seed_v1\0")
    for value in values:
        digest.update(value.to_bytes(16, "little", signed=False))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def _rng(pair_seed: int, stream: int, step: int, substream: int = 0) -> np.random.Generator:
    return np.random.default_rng(
        _derived_seed(pair_seed, stream, step, substream))


def _capture_component(component: Any, name: str) -> Any:
    capture = getattr(component, "capture_branch_state", None)
    restore = getattr(component, "restore_branch_state", None)
    if callable(capture) != callable(restore):
        raise TypeError(
            f"{name} must implement both capture_branch_state and "
            "restore_branch_state, or neither")
    return copy.deepcopy(capture()) if callable(capture) else None


def _restore_component(component: Any, state: Any) -> None:
    restore = getattr(component, "restore_branch_state", None)
    if callable(restore):
        restore(copy.deepcopy(state))


def _checked_action(value: Any, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    if action.shape != (12,) or not np.all(np.isfinite(action)):
        raise ValueError(f"{name} must be a finite 12-D action")
    if np.any(action < -1.0 - 1e-6) or np.any(action > 1.0 + 1e-6):
        raise ValueError(f"{name} must lie in normalized [-1,1]")
    return np.clip(action, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class ShieldStepDecision:
    """Compact, immutable audit record for one repeated shield decision."""

    selected_action: np.ndarray
    selected_index: int
    intervened: bool
    reason: str
    requested_delta_rms: float
    q_target_delta_rms: float
    nominal_risk_lcb: float
    selected_risk_ucb: float
    selected_benefit_lcb: float

    def __post_init__(self) -> None:
        action = _checked_action(self.selected_action, "selected_action")
        action.setflags(write=False)
        object.__setattr__(self, "selected_action", action)
        index = _nonnegative_int(self.selected_index, "selected_index")
        if index >= 16:
            raise ValueError("selected_index must be below 16")
        object.__setattr__(self, "selected_index", index)
        if not isinstance(self.intervened, (bool, np.bool_)):
            raise ValueError("intervened must be boolean")
        object.__setattr__(self, "intervened", bool(self.intervened))
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be nonempty text")
        for name in (
            "requested_delta_rms",
            "q_target_delta_rms",
            "nominal_risk_lcb",
            "selected_risk_ucb",
            "selected_benefit_lcb",
        ):
            value = _finite_float(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if not self.intervened and self.selected_index != 0:
            raise ValueError("a non-intervention decision must select nominal index zero")

    @classmethod
    def from_runtime(cls, result: QSafeRuntimeResult) -> "ShieldStepDecision":
        selection = result.selection
        index = int(selection.selected_index)
        return cls(
            selected_action=result.selected_requested_action,
            selected_index=index,
            intervened=selection.intervened,
            reason=selection.reason,
            requested_delta_rms=float(selection.requested_delta_rms[index]),
            q_target_delta_rms=float(selection.q_target_delta_rms[index]),
            nominal_risk_lcb=float(selection.nominal_risk_lcb),
            selected_risk_ucb=float(selection.risk_ucb[index]),
            selected_benefit_lcb=float(selection.benefit_lcb[index]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "selected_action": self.selected_action.tolist(),
        }


def _require_bound_actor_critic(
    actor: FrozenActor,
    reward_q: FrozenRewardQ,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actor_manifest = copy.deepcopy(dict(actor.manifest()))
    critic_manifest = copy.deepcopy(dict(reward_q.manifest()))
    if int(actor.training_step) != int(reward_q.training_step):
        raise ValueError("actor and reward critic training steps differ")
    if actor_manifest.get("training_step") != critic_manifest.get("training_step"):
        raise ValueError("actor and reward critic manifests disagree on training step")
    if actor_manifest.get("observation_dim") != 46 or (
            critic_manifest.get("observation_dim") != 46):
        raise ValueError("actor and reward critic must use the deployable 46D observation")
    if actor_manifest.get("action_dim") != 12 or (
            critic_manifest.get("action_dim") != 12):
        raise ValueError("actor and reward critic must use the normalized 12D action")
    if actor_manifest.get("config_sha256") != critic_manifest.get("config_sha256"):
        raise ValueError("actor and reward critic config hashes differ")
    actor_path = actor_manifest.get("actor_path")
    critic_path = critic_manifest.get("critic_path")
    if not isinstance(actor_path, str) or not isinstance(critic_path, str):
        raise ValueError("actor and reward critic manifests require checkpoint paths")
    if Path(actor_path).resolve().parent != Path(critic_path).resolve().parent:
        raise ValueError("actor and reward critic are not from one checkpoint directory")
    return actor_manifest, critic_manifest


class RepeatedQSafeShield:
    """Compose frozen actor/reward-Q/Q_safe into one deterministic step gate."""

    def __init__(
        self,
        *,
        actor: FrozenActor,
        reward_q: FrozenRewardQ,
        qsafe_artifact: LoadedQSafeArtifact,
        selector_config: SelectorConfig,
        expected_command_speed_mps: float,
        candidate_config: EvidenceCandidateConfig | None = None,
    ) -> None:
        if not isinstance(qsafe_artifact, LoadedQSafeArtifact):
            raise TypeError("qsafe_artifact must be a LoadedQSafeArtifact")
        if not isinstance(selector_config, SelectorConfig):
            raise TypeError("selector_config must be a SelectorConfig")
        self.actor_manifest, self.reward_q_manifest = _require_bound_actor_critic(
            actor, reward_q)
        provenance = qsafe_artifact.manifest.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance.get(
                "runtime_binding_verified", False):
            raise ValueError(
                "Q_safe artifact has no verified continuation-policy binding")
        continuation_contract = provenance.get("continuation_policy_contract")
        if not isinstance(continuation_contract, Mapping) or not (
                continuation_contract.get("verified", False)):
            raise ValueError(
                "Q_safe artifact continuation-policy contract is unverified")
        for key in (
            "policy_fingerprint_sha256", "actor_state_dict_sha256",
            "config_sha256", "training_step", "observation_dim",
            "actor_observation_dim", "action_dim",
        ):
            if continuation_contract.get(key) != self.actor_manifest.get(key):
                raise ValueError(
                    "Q_safe continuation-policy binding disagrees with runtime "
                    f"actor field {key!r}")
        self.actor = actor
        self.reward_q = reward_q
        self.qsafe_artifact = qsafe_artifact
        self.selector_config = selector_config
        self.expected_command_speed_mps = _finite_float(
            expected_command_speed_mps, "expected_command_speed_mps")
        self.candidate_config = (
            EvidenceCandidateConfig()
            if candidate_config is None else candidate_config)
        if not isinstance(self.candidate_config, EvidenceCandidateConfig):
            raise TypeError("candidate_config must be EvidenceCandidateConfig")

    def manifest(self) -> dict[str, Any]:
        return {
            "version": "qsafe.repeated_shield.v1",
            "seed_contract": PAIRED_SEED_CONTRACT,
            "actor": copy.deepcopy(self.actor_manifest),
            "reward_q": copy.deepcopy(self.reward_q_manifest),
            "qsafe_artifact": str(self.qsafe_artifact.path),
            "qsafe_schema_version": self.qsafe_artifact.manifest.get(
                "schema_version"),
            "qsafe_component_sha256": copy.deepcopy(
                self.qsafe_artifact.manifest.get("component_sha256")),
            "selector_config": asdict(self.selector_config),
            "candidate_protocol": self.candidate_config.manifest_protocol(),
            "expected_command_speed_mps": self.expected_command_speed_mps,
        }

    def decide(
        self,
        env: MujocoSnapshotEnv,
        observation_history: np.ndarray,
        nominal_action: np.ndarray,
        *,
        pair_seed: int,
        step: int,
    ) -> ShieldStepDecision:
        history = np.asarray(observation_history, dtype=np.float32)
        if history.shape != (5, 46) or not np.all(np.isfinite(history)):
            raise ValueError("observation_history must be finite [5,46]")
        nominal = _checked_action(nominal_action, "nominal_action")
        pair_seed = _nonnegative_int(pair_seed, "pair_seed")
        step = _nonnegative_int(step, "step")
        observation = history[-1]
        deterministic = self.actor.deterministic_action(observation)
        actor_samples = np.stack([
            self.actor.sample_action(
                observation,
                _rng(pair_seed, _STREAM_CANDIDATE_ACTOR, step, sample_index),
            )
            for sample_index in range(ACTOR_SAMPLE_COUNT)
        ])
        candidates: CandidateSet = build_evidence_candidates(
            nominal=nominal,
            deterministic_mean=deterministic,
            previous_requested=env.previous_action_requested,
            actor_samples=actor_samples,
            action_applier=env.action_applier,
            current_qpos=np.asarray(
                env.data.qpos[env.qpos_addresses], dtype=np.float32),
            candidate_seed=_derived_seed(
                pair_seed, _STREAM_CANDIDATE_GEOMETRY, step),
            config=self.candidate_config,
        )
        reward_values = self.reward_q.conservative_values(
            observation, candidates.requested)
        runtime = run_qsafe_step(
            self.qsafe_artifact,
            history,
            candidates,
            reward_values,
            self.selector_config,
            expected_command_speed_mps=self.expected_command_speed_mps,
        )
        return ShieldStepDecision.from_runtime(runtime)


@dataclass(frozen=True)
class ClosedLoopRollout:
    fall: bool
    first_failure_step: int
    steps_executed: int
    max_tilt_rad: float
    min_height_m: float
    interventions: int
    no_eligible_steps: int
    requested_delta_rms_sum: float
    selection_reasons: Mapping[str, int]
    decisions: tuple[ShieldStepDecision, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fall, (bool, np.bool_)):
            raise ValueError("fall must be boolean")
        object.__setattr__(self, "fall", bool(self.fall))
        first_failure = _positive_int(
            self.first_failure_step, "first_failure_step")
        steps = _nonnegative_int(self.steps_executed, "steps_executed")
        interventions = _nonnegative_int(self.interventions, "interventions")
        no_eligible = _nonnegative_int(
            self.no_eligible_steps, "no_eligible_steps")
        if interventions > steps or no_eligible > steps:
            raise ValueError("intervention counters cannot exceed executed steps")
        if self.fall and first_failure != steps:
            raise ValueError("a fall must occur on the final executed step")
        if not self.fall and first_failure <= steps:
            raise ValueError("non-fall failure sentinel must exceed executed steps")
        for name in (
            "max_tilt_rad", "min_height_m", "requested_delta_rms_sum",
        ):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        decisions = tuple(self.decisions)
        if any(not isinstance(item, ShieldStepDecision) for item in decisions):
            raise ValueError("decisions must contain ShieldStepDecision records")
        if decisions and len(decisions) != steps:
            raise ValueError("shield decisions must align with every executed step")
        if decisions and interventions != sum(
                int(item.intervened) for item in decisions):
            raise ValueError("interventions disagree with shield decisions")
        reasons: dict[str, int] = {}
        for name, raw_count in dict(self.selection_reasons).items():
            if not isinstance(name, str) or not name:
                raise ValueError("selection reason names must be nonempty text")
            count = _nonnegative_int(raw_count, f"selection_reasons[{name!r}]")
            reasons[name] = count
        if decisions and sum(reasons.values()) != len(decisions):
            raise ValueError("selection reason counts disagree with decisions")
        if no_eligible != reasons.get("no_eligible", 0):
            raise ValueError("no_eligible_steps disagrees with selection reasons")
        object.__setattr__(self, "first_failure_step", first_failure)
        object.__setattr__(self, "steps_executed", steps)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "no_eligible_steps", no_eligible)
        object.__setattr__(self, "selection_reasons", MappingProxyType(reasons))
        object.__setattr__(self, "decisions", decisions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fall": self.fall,
            "first_failure_step": self.first_failure_step,
            "steps_executed": self.steps_executed,
            "max_tilt_rad": self.max_tilt_rad,
            "min_height_m": self.min_height_m,
            "interventions": self.interventions,
            "no_eligible_steps": self.no_eligible_steps,
            "requested_delta_rms_sum": self.requested_delta_rms_sum,
            "selection_reasons": dict(self.selection_reasons),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


@dataclass(frozen=True)
class PairedClosedLoopOutcome:
    pair_id: str
    state_hash: str
    trajectory_id: str
    source_seed: int
    pair_seed: int
    horizon_steps: int
    nominal: ClosedLoopRollout
    shield: ClosedLoopRollout
    seed_contract: str = PAIRED_SEED_CONTRACT

    def __post_init__(self) -> None:
        for name in ("pair_id", "state_hash", "trajectory_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty text")
        object.__setattr__(
            self, "source_seed", _nonnegative_int(self.source_seed, "source_seed"))
        object.__setattr__(
            self, "pair_seed", _nonnegative_int(self.pair_seed, "pair_seed"))
        horizon = _positive_int(self.horizon_steps, "horizon_steps")
        if horizon > np.iinfo(np.int16).max - 1:
            raise ValueError("horizon_steps is too large for H+1 failure sentinel")
        if not isinstance(self.nominal, ClosedLoopRollout) or not isinstance(
                self.shield, ClosedLoopRollout):
            raise ValueError("nominal and shield must be ClosedLoopRollout records")
        for name, rollout in (("nominal", self.nominal), ("shield", self.shield)):
            if rollout.steps_executed > horizon:
                raise ValueError(f"{name} rollout exceeds paired horizon")
            expected_failure = (
                rollout.first_failure_step <= horizon)
            if rollout.fall != expected_failure:
                raise ValueError(f"{name} failure sentinel disagrees with horizon")
        if self.seed_contract != PAIRED_SEED_CONTRACT:
            raise ValueError("unknown paired seed contract")
        object.__setattr__(self, "horizon_steps", horizon)

    @property
    def fall_reduction(self) -> int:
        return int(self.nominal.fall) - int(self.shield.fall)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "state_hash": self.state_hash,
            "trajectory_id": self.trajectory_id,
            "source_seed": self.source_seed,
            "pair_seed": self.pair_seed,
            "horizon_steps": self.horizon_steps,
            "seed_contract": self.seed_contract,
            "fall_reduction": self.fall_reduction,
            "nominal": self.nominal.to_dict(),
            "shield": self.shield.to_dict(),
        }


def _rollout_arm(
    env: MujocoSnapshotEnv,
    actor: FrozenActor,
    shield: ShieldController | None,
    disturbance_program: DisturbanceProgram | None,
    *,
    pair_seed: int,
    horizon_steps: int,
) -> ClosedLoopRollout:
    initial = env.measurement()
    if initial.failure:
        raise ValueError("paired snapshot is already failed before either arm")
    max_tilt = float(initial.tilt_rad)
    min_height = float(initial.height_m)
    first_failure = horizon_steps + 1
    steps_executed = 0
    decisions: list[ShieldStepDecision] = []
    reasons: dict[str, int] = {}

    for step in range(horizon_steps):
        if disturbance_program is not None:
            disturbance_program(
                env,
                step,
                _rng(pair_seed, _STREAM_DISTURBANCE, step),
            )
        # Snapshot capture is required to include the current policy-time
        # frame.  Avoid silently inserting a duplicate frame at step zero.
        history = (
            env.observation_history()
            if step == 0 else env.record_observation())
        nominal = actor.sample_action(
            history[-1], _rng(pair_seed, _STREAM_NOMINAL_ACTOR, step))
        action = nominal
        if shield is not None:
            decision = shield.decide(
                env,
                history,
                nominal,
                pair_seed=pair_seed,
                step=step,
            )
            if not isinstance(decision, ShieldStepDecision):
                raise TypeError("shield controller returned an invalid decision")
            decisions.append(decision)
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
            action = decision.selected_action
        result = env.step(action)
        steps_executed = step + 1
        max_tilt = max(max_tilt, float(result.tilt_rad))
        min_height = min(min_height, float(result.height_m))
        if result.failure:
            first_failure = step + 1
            break
    interventions = sum(int(item.intervened) for item in decisions)
    return ClosedLoopRollout(
        fall=first_failure <= horizon_steps,
        first_failure_step=first_failure,
        steps_executed=steps_executed,
        max_tilt_rad=max_tilt,
        min_height_m=min_height,
        interventions=interventions,
        no_eligible_steps=int(reasons.get("no_eligible", 0)),
        requested_delta_rms_sum=float(sum(
            item.requested_delta_rms for item in decisions if item.intervened)),
        selection_reasons=copy.deepcopy(reasons),
        decisions=tuple(decisions),
    )


def evaluate_paired_snapshot(
    env: MujocoSnapshotEnv,
    snapshot: BranchSnapshot,
    *,
    actor: FrozenActor,
    shield: ShieldController,
    pair_id: str,
    trajectory_id: str,
    source_seed: int,
    pair_seed: int,
    horizon_steps: int = 32,
    disturbance_program: DisturbanceProgram | None = None,
    arm_order: tuple[str, str] = ("nominal", "shield"),
) -> PairedClosedLoopOutcome:
    """Run both arms from one snapshot with matched explicit RNG streams.

    ``arm_order`` exists for determinism auditing.  A valid component must
    produce the same outcome in either order because environment, actor,
    shield and disturbance branch state are restored before each arm.
    """
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("pair_id must be nonempty text")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory_id must be nonempty text")
    source_seed = _nonnegative_int(source_seed, "source_seed")
    pair_seed = _nonnegative_int(pair_seed, "pair_seed")
    horizon_steps = _positive_int(horizon_steps, "horizon_steps")
    if horizon_steps > np.iinfo(np.int16).max - 1:
        raise ValueError("horizon_steps is too large for H+1 failure sentinel")
    if tuple(arm_order) not in (("nominal", "shield"), ("shield", "nominal")):
        raise ValueError("arm_order must contain nominal and shield exactly once")
    if not isinstance(snapshot, BranchSnapshot):
        raise TypeError("snapshot must be a BranchSnapshot")
    if snapshot.application_state.observation_history.shape[0] == 0:
        raise ValueError("paired snapshot has no recorded observation history")
    if disturbance_program is not None:
        scheduled_steps = getattr(disturbance_program, "policy_steps", None)
        if scheduled_steps is None:
            raise TypeError(
                "paired disturbance_program must expose explicit policy_steps")
        if 0 in scheduled_steps:
            raise ValueError(
                "step-zero disturbance would invalidate the captured history; "
                "schedule it after step zero")

    actor_state = _capture_component(actor, "actor")
    shield_state = _capture_component(shield, "shield")
    disturbance_state = (
        None if disturbance_program is None
        else _capture_component(disturbance_program, "disturbance_program"))
    outcomes: dict[str, ClosedLoopRollout] = {}
    try:
        for arm in arm_order:
            env.restore(snapshot)
            _restore_component(actor, actor_state)
            _restore_component(shield, shield_state)
            if disturbance_program is not None:
                _restore_component(disturbance_program, disturbance_state)
            outcomes[arm] = _rollout_arm(
                env,
                actor,
                shield if arm == "shield" else None,
                disturbance_program,
                pair_seed=pair_seed,
                horizon_steps=horizon_steps,
            )
    finally:
        env.restore(snapshot)
        _restore_component(actor, actor_state)
        _restore_component(shield, shield_state)
        if disturbance_program is not None:
            _restore_component(disturbance_program, disturbance_state)
    return PairedClosedLoopOutcome(
        pair_id=pair_id,
        state_hash=snapshot.compound_sha256(),
        trajectory_id=trajectory_id,
        source_seed=source_seed,
        pair_seed=pair_seed,
        horizon_steps=horizon_steps,
        nominal=outcomes["nominal"],
        shield=outcomes["shield"],
    )


@dataclass(frozen=True)
class PairedClosedLoopGateThresholds:
    min_independent_pairs: int = 1000
    min_absolute_fall_reduction: float = 0.03
    min_reduction_ci_low: float = 0.0
    require_improved_gt_worsened: bool = True

    def validate(self) -> None:
        _positive_int(self.min_independent_pairs, "min_independent_pairs")
        for name in (
            "min_absolute_fall_reduction", "min_reduction_ci_low",
        ):
            value = _finite_float(getattr(self, name), name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [-1,1]")
        if not isinstance(self.require_improved_gt_worsened, (bool, np.bool_)):
            raise ValueError("require_improved_gt_worsened must be boolean")


@dataclass(frozen=True)
class PairedClosedLoopSummary:
    independent_pairs: int
    source_seeds: tuple[int, ...]
    nominal_falls: int
    shield_falls: int
    nominal_fall_rate: float
    shield_fall_rate: float
    absolute_fall_reduction: float
    absolute_fall_reduction_ci95: tuple[float, float]
    relative_fall_reduction: float
    improved_pairs: int
    worsened_pairs: int
    unchanged_pairs: int
    intervention_rate_per_shield_step: float
    mean_interventions_per_pair: float
    gate_checks: Mapping[str, bool]
    paired_closed_loop_gate: bool
    bootstrap_replicates: int
    bootstrap_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_paired_closed_loop(
    outcomes: Sequence[PairedClosedLoopOutcome],
    *,
    thresholds: PairedClosedLoopGateThresholds = PairedClosedLoopGateThresholds(),
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260809,
) -> PairedClosedLoopSummary:
    """Validate independent paired units and compute the preregistered gate."""
    thresholds.validate()
    replicates = _positive_int(bootstrap_replicates, "bootstrap_replicates")
    seed = _nonnegative_int(bootstrap_seed, "bootstrap_seed")
    records = list(outcomes)
    if not records or any(not isinstance(item, PairedClosedLoopOutcome)
                          for item in records):
        raise ValueError("outcomes must contain PairedClosedLoopOutcome records")
    horizon = {item.horizon_steps for item in records}
    if len(horizon) != 1:
        raise ValueError("all paired outcomes must use one horizon")
    if next(iter(horizon)) != 32:
        raise ValueError("Phase-1 paired gate requires the preregistered H32 horizon")
    if any(item.seed_contract != PAIRED_SEED_CONTRACT for item in records):
        raise ValueError("paired outcomes use an unknown seed contract")
    identity_fields = {
        "pair_id": [item.pair_id for item in records],
        "state_hash": [item.state_hash for item in records],
        "trajectory_id": [item.trajectory_id for item in records],
        "pair_seed": [item.pair_seed for item in records],
    }
    for name, values in identity_fields.items():
        if len(set(values)) != len(values):
            raise ValueError(f"paired outcomes contain duplicate {name}")

    nominal = np.asarray([item.nominal.fall for item in records], dtype=np.int8)
    shield = np.asarray([item.shield.fall for item in records], dtype=np.int8)
    difference = nominal - shield
    count = len(records)
    estimate = float(np.mean(difference))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, count, size=(replicates, count))
    draws = difference[sampled].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    nominal_rate = float(np.mean(nominal))
    shield_rate = float(np.mean(shield))
    relative = (
        float(estimate / nominal_rate)
        if nominal_rate > 0.0 else float("nan"))
    improved = int(np.count_nonzero(difference == 1))
    worsened = int(np.count_nonzero(difference == -1))
    shield_steps = sum(item.shield.steps_executed for item in records)
    interventions = sum(item.shield.interventions for item in records)
    checks = {
        "independent_pairs": count >= thresholds.min_independent_pairs,
        "absolute_fall_reduction": (
            estimate >= thresholds.min_absolute_fall_reduction),
        "reduction_ci_low": float(low) > thresholds.min_reduction_ci_low,
        "improved_gt_worsened": (
            improved > worsened
            if thresholds.require_improved_gt_worsened else True),
    }
    return PairedClosedLoopSummary(
        independent_pairs=count,
        source_seeds=tuple(sorted({int(item.source_seed) for item in records})),
        nominal_falls=int(np.sum(nominal)),
        shield_falls=int(np.sum(shield)),
        nominal_fall_rate=nominal_rate,
        shield_fall_rate=shield_rate,
        absolute_fall_reduction=estimate,
        absolute_fall_reduction_ci95=(float(low), float(high)),
        relative_fall_reduction=relative,
        improved_pairs=improved,
        worsened_pairs=worsened,
        unchanged_pairs=count - improved - worsened,
        intervention_rate_per_shield_step=(
            float(interventions / shield_steps) if shield_steps else 0.0),
        mean_interventions_per_pair=float(interventions / count),
        gate_checks=checks,
        paired_closed_loop_gate=all(checks.values()),
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )


__all__ = [
    "ClosedLoopRollout",
    "PAIRED_SEED_CONTRACT",
    "PairedClosedLoopGateThresholds",
    "PairedClosedLoopOutcome",
    "PairedClosedLoopSummary",
    "RepeatedQSafeShield",
    "ShieldStepDecision",
    "evaluate_paired_snapshot",
    "summarize_paired_closed_loop",
]
