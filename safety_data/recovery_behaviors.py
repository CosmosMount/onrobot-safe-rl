"""Preregistered deployable closed-loop recovery behaviors for v3 triage.

The library is intentionally small and deterministic.  Every feedback law
reads only the corrected five-frame deployable observation history; no MuJoCo
state, privileged measurement, reward, or branch outcome is accepted by this
module.  Candidate zero is the early actor's externally supplied nominal
action.  Candidates one through eight implement the fixed recovery behaviors
from ``config/qsafe_closed_loop_recovery_triage_v3.yaml``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol

import numpy as np

from runtime.inference.actions import ActionApplier, qpos_to_action


RECOVERY_BEHAVIOR_PROTOCOL_VERSION = (
    "qsafe.closed_loop_recovery_behaviors.v3")
RECOVERY_BEHAVIOR_COUNT = 9
RECOVERY_BEHAVIOR_KINDS = (
    "nominal",
    "mature_actor_L10",
    "mature_actor_L25",
    "mature_actor_L50",
    "joint_brake_L10",
    "halfway_neutral_L10",
    "halfway_neutral_L25",
    "ramp_neutral_L25",
    "ramp_crouch_L25",
)
RECOVERY_BEHAVIOR_STEPS = (0, 10, 25, 50, 10, 10, 25, 25, 25)
# Short aliases make the dataset-facing contract unambiguous to callers.
CANDIDATE_KINDS = RECOVERY_BEHAVIOR_KINDS
CANDIDATE_BEHAVIOR_STEPS = RECOVERY_BEHAVIOR_STEPS

OBSERVATION_HISTORY_SHAPE = (5, 46)
OBSERVATION_JOINT_Q_SLICE = (0, 12)
OBSERVATION_PREVIOUS_Q_TARGET_SLICE = (34, 46)
Q_NEUTRAL_PER_LEG_RAD = (0.05, 0.70, -1.40)
Q_CROUCH_PER_LEG_RAD = (0.05, 0.90, -1.60)
RAMP_MAX_DELTA_RAD = 0.04

MATURE_POLICY_TRAINING_STEP = 500_000
MATURE_POLICY_CONFIG_SHA256 = (
    "ebf312ef27f64326a6ef478e0f86273af9f9cbaf61dd09b632fb10868524f726")
MATURE_POLICY_ACTOR_SHA256 = (
    "958e2c4345aca511723e4b9299554c9f1594617322ad7d62c0ace84319cbbed0")
MATURE_POLICY_STATE_DICT_SHA256 = (
    "2074a3f00152df8e96cddf380623a9eb4bb63f84538dafca59cdf509f9559409")
MATURE_POLICY_FINGERPRINT_SHA256 = (
    "f01e7cc36b9020631171c3dfe502d7426877aee0c33debd0a5c197206efc0908")
MATURE_CHECKPOINT_FINGERPRINT_SHA256 = (
    "2597eeb238d586bdd895e436b0e814aae23677484878da1eba5d1258b06f97cc")

_EXPECTED_ACTION_OFFSET = np.asarray(
    [0.2, 0.4, 0.4] * 4, dtype=np.float32)
_Q_NEUTRAL = np.asarray(Q_NEUTRAL_PER_LEG_RAD * 4, dtype=np.float32)
_Q_CROUCH = np.asarray(Q_CROUCH_PER_LEG_RAD * 4, dtype=np.float32)
_MATURE_POLICY_IDENTITY = {
    "training_step": MATURE_POLICY_TRAINING_STEP,
    "config_sha256": MATURE_POLICY_CONFIG_SHA256,
    "actor_sha256": MATURE_POLICY_ACTOR_SHA256,
    "actor_state_dict_sha256": MATURE_POLICY_STATE_DICT_SHA256,
    "policy_fingerprint_sha256": MATURE_POLICY_FINGERPRINT_SHA256,
    "checkpoint_fingerprint_sha256": MATURE_CHECKPOINT_FINGERPRINT_SHA256,
    "observation_dim": 46,
    "actor_observation_dim": 46,
    "action_dim": 12,
}

if len(RECOVERY_BEHAVIOR_KINDS) != RECOVERY_BEHAVIOR_COUNT or len(
        RECOVERY_BEHAVIOR_STEPS) != RECOVERY_BEHAVIOR_COUNT:
    raise AssertionError("v3 K9 recovery behavior constants disagree")


class DeterministicRecoveryPolicy(Protocol):
    """Minimal interface required from the frozen mature DroQ actor."""

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray: ...

    def manifest(self) -> Mapping[str, Any]: ...


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checked_history(value: np.ndarray) -> np.ndarray:
    history = np.asarray(value, dtype=np.float32)
    if history.shape != OBSERVATION_HISTORY_SHAPE:
        raise ValueError("observation_history must have exact shape [5,46]")
    if not np.all(np.isfinite(history)):
        raise ValueError("observation_history must contain only finite values")
    return history


def _checked_action(value: np.ndarray, name: str) -> np.ndarray:
    action = np.asarray(value, dtype=np.float32).reshape(-1)
    if action.shape != (12,) or not np.all(np.isfinite(action)):
        raise ValueError(f"{name} must be a finite 12D action")
    if np.any(action < -1.0 - 1e-6) or np.any(action > 1.0 + 1e-6):
        raise ValueError(f"{name} must lie in normalized [-1, 1]")
    return np.clip(action, -1.0, 1.0).astype(np.float32, copy=True)


def _checked_index(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError("candidate_index must be an integer")
    index = int(value)
    if not 0 <= index < RECOVERY_BEHAVIOR_COUNT:
        raise ValueError(
            f"candidate_index must lie in [0, {RECOVERY_BEHAVIOR_COUNT - 1}]")
    return index


def _checked_step(value: int, *, candidate_index: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError("step must be a nonnegative integer")
    step = int(value)
    if step < 0:
        raise ValueError("step must be a nonnegative integer")
    duration = RECOVERY_BEHAVIOR_STEPS[candidate_index]
    if candidate_index == 0:
        if step != 0:
            raise ValueError("nominal candidate is previewed only at step zero")
    elif step >= duration:
        raise ValueError(
            f"candidate {candidate_index} is inactive at step {step}; "
            f"duration is {duration}")
    return step


@dataclass(frozen=True)
class RecoveryBehaviorConfig:
    """Non-tunable values locked by the v3 preregistration."""

    q_neutral_per_leg_rad: tuple[float, float, float] = Q_NEUTRAL_PER_LEG_RAD
    q_crouch_per_leg_rad: tuple[float, float, float] = Q_CROUCH_PER_LEG_RAD
    ramp_max_delta_rad: float = RAMP_MAX_DELTA_RAD
    observation_history_shape: tuple[int, int] = OBSERVATION_HISTORY_SHAPE
    observation_joint_q_slice: tuple[int, int] = OBSERVATION_JOINT_Q_SLICE
    observation_previous_q_target_slice: tuple[int, int] = (
        OBSERVATION_PREVIOUS_Q_TARGET_SLICE)

    def __post_init__(self) -> None:
        expected = {
            "q_neutral_per_leg_rad": Q_NEUTRAL_PER_LEG_RAD,
            "q_crouch_per_leg_rad": Q_CROUCH_PER_LEG_RAD,
            "ramp_max_delta_rad": RAMP_MAX_DELTA_RAD,
            "observation_history_shape": OBSERVATION_HISTORY_SHAPE,
            "observation_joint_q_slice": OBSERVATION_JOINT_Q_SLICE,
            "observation_previous_q_target_slice": (
                OBSERVATION_PREVIOUS_Q_TARGET_SLICE),
        }
        actual: dict[str, Any] = {}
        try:
            for name in ("q_neutral_per_leg_rad", "q_crouch_per_leg_rad"):
                raw = tuple(getattr(self, name))
                if len(raw) != 3 or any(
                        isinstance(item, (bool, np.bool_))
                        or not isinstance(item, (int, float, np.integer, np.floating))
                        for item in raw):
                    raise ValueError(f"{name} must contain three numbers")
                value = tuple(float(item) for item in raw)
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{name} must be finite")
                actual[name] = value
            if isinstance(self.ramp_max_delta_rad, (bool, np.bool_)) or not isinstance(
                    self.ramp_max_delta_rad,
                    (int, float, np.integer, np.floating)):
                raise ValueError("ramp_max_delta_rad must be numeric")
            actual["ramp_max_delta_rad"] = float(self.ramp_max_delta_rad)
            for name in (
                    "observation_history_shape", "observation_joint_q_slice",
                    "observation_previous_q_target_slice"):
                raw = tuple(getattr(self, name))
                if len(raw) != 2 or any(
                        isinstance(item, (bool, np.bool_))
                        or not isinstance(item, (int, np.integer))
                        for item in raw):
                    raise ValueError(f"{name} must contain two integers")
                actual[name] = tuple(int(item) for item in raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "recovery behavior parameters are malformed") from exc
        if actual != expected:
            raise ValueError(
                "recovery behavior parameters must exactly match the "
                "preregistered v3 protocol")
        for name, value in actual.items():
            object.__setattr__(self, name, value)

    def manifest_protocol(self) -> dict[str, Any]:
        """Return the exact JSON-safe K9 candidate protocol."""
        return {
            "protocol_version": RECOVERY_BEHAVIOR_PROTOCOL_VERSION,
            "count": RECOVERY_BEHAVIOR_COUNT,
            "nominal_index": 0,
            "ordered_names": list(RECOVERY_BEHAVIOR_KINDS),
            "behavior_steps_array": "candidate_behavior_steps",
            "behavior_override_steps": list(RECOVERY_BEHAVIOR_STEPS),
            "observation_history_shape": list(OBSERVATION_HISTORY_SHAPE),
            "observation_joint_q_slice": list(OBSERVATION_JOINT_Q_SLICE),
            "observation_previous_q_target_slice": list(
                OBSERVATION_PREVIOUS_Q_TARGET_SLICE),
            "t0_nominal_semantics": "deterministic_early_actor_mean",
            "mature_actor_semantics": "deterministic_newest_46d_frame",
            "post_option_continuation": "stochastic_same_early_actor",
            "reselection_during_option": "forbidden",
            "q_neutral_per_leg_rad": list(Q_NEUTRAL_PER_LEG_RAD),
            "q_crouch_per_leg_rad": list(Q_CROUCH_PER_LEG_RAD),
            "halfway_neutral_formula": (
                "0.5_times_latest_q_plus_0.5_times_q_neutral"),
            "ramp_max_delta_rad_per_joint_per_policy_step": (
                RAMP_MAX_DELTA_RAD),
            "target_to_action": "runtime_qpos_to_action",
            "projection": "unchanged_runtime_ActionApplier",
            "kp": 60.0,
            "kd": 5.0,
            "max_joint_delta": None,
            "use_action_filter": False,
            "tie_priority": list(RECOVERY_BEHAVIOR_KINDS),
        }

    def protocol_sha256(self) -> str:
        return _canonical_sha256(self.manifest_protocol())


@dataclass(frozen=True)
class RecoveryBehaviorPreview:
    """Immutable first-step requested/executed/target records for K9."""

    requested: np.ndarray
    executed: np.ndarray
    q_target: np.ndarray
    kind: np.ndarray
    mask: np.ndarray
    behavior_steps: np.ndarray
    manifest_protocol: dict[str, Any]
    library_fingerprint_sha256: str

    def __post_init__(self) -> None:
        for name in ("requested", "executed", "q_target"):
            value = np.asarray(getattr(self, name), dtype=np.float32).copy()
            if value.shape != (RECOVERY_BEHAVIOR_COUNT, 12):
                raise ValueError(
                    f"{name} must have shape {(RECOVERY_BEHAVIOR_COUNT, 12)}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            if name != "q_target" and (
                    np.any(value < -1.0 - 1e-6)
                    or np.any(value > 1.0 + 1e-6)):
                raise ValueError(f"{name} must lie in normalized [-1, 1]")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        kind = np.asarray(self.kind, dtype=str).copy()
        if kind.shape != (RECOVERY_BEHAVIOR_COUNT,) or tuple(
                kind.tolist()) != RECOVERY_BEHAVIOR_KINDS:
            raise ValueError("kind does not match the locked K9 order")
        mask = np.asarray(self.mask, dtype=bool).copy()
        if mask.shape != (RECOVERY_BEHAVIOR_COUNT,) or not np.all(mask):
            raise ValueError("all K9 recovery behaviors must remain valid")
        steps = np.asarray(self.behavior_steps)
        if steps.dtype.kind not in "iu" or steps.shape != (
                RECOVERY_BEHAVIOR_COUNT,) or tuple(
                    steps.tolist()) != RECOVERY_BEHAVIOR_STEPS:
            raise ValueError("behavior_steps does not match the locked K9 order")
        steps = steps.astype(np.int64, copy=True)

        manifest = copy.deepcopy(dict(self.manifest_protocol))
        if manifest != RecoveryBehaviorConfig().manifest_protocol():
            raise ValueError("manifest_protocol does not match the K9 arrays")
        fingerprint = str(self.library_fingerprint_sha256)
        if len(fingerprint) != 64 or any(
                char not in "0123456789abcdef" for char in fingerprint):
            raise ValueError("library_fingerprint_sha256 must be lowercase SHA-256")

        for value in (kind, mask, steps):
            value.setflags(write=False)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "behavior_steps", steps)
        object.__setattr__(self, "manifest_protocol", manifest)
        object.__setattr__(self, "library_fingerprint_sha256", fingerprint)

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.mask))


class RecoveryBehaviorLibrary:
    """Stateless K9 feedback controller bound to the locked mature actor."""

    def __init__(
        self,
        mature_policy: DeterministicRecoveryPolicy,
        action_applier: ActionApplier,
        config: RecoveryBehaviorConfig | None = None,
    ) -> None:
        self._config = RecoveryBehaviorConfig() if config is None else config
        if not isinstance(self._config, RecoveryBehaviorConfig):
            raise TypeError("config must be RecoveryBehaviorConfig")
        if not callable(getattr(mature_policy, "deterministic_action", None)):
            raise TypeError("mature_policy must implement deterministic_action")
        manifest_method = getattr(mature_policy, "manifest", None)
        if not callable(manifest_method):
            raise TypeError("mature_policy must provide an evidence manifest")
        raw_manifest = manifest_method()
        if not isinstance(raw_manifest, Mapping):
            raise TypeError("mature_policy manifest must be a mapping")
        policy_manifest = copy.deepcopy(dict(raw_manifest))
        for key, expected in _MATURE_POLICY_IDENTITY.items():
            if policy_manifest.get(key) != expected:
                raise ValueError(
                    "mature_policy does not match the preregistered identity: "
                    f"field {key!r}")

        if not isinstance(action_applier, ActionApplier):
            raise TypeError("action_applier must be ActionApplier")
        if action_applier.max_joint_delta is not None:
            raise ValueError("v3 requires ActionApplier.max_joint_delta=None")
        if action_applier.action_filter is not None:
            raise ValueError("v3 requires ActionApplier.action_filter=None")
        projection_vectors: dict[str, np.ndarray] = {}
        for name in ("init_qpos", "action_offset", "joint_min", "joint_max"):
            value = np.asarray(getattr(action_applier, name), dtype=np.float32)
            if value.shape != (12,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"action_applier.{name} must be a finite 12D vector")
            projection_vectors[name] = value.copy()
        if np.any(projection_vectors["action_offset"] <= 0.0):
            raise ValueError("action_applier.action_offset must be positive")
        if not np.array_equal(projection_vectors["init_qpos"], _Q_NEUTRAL):
            raise ValueError("action_applier.init_qpos must equal locked q_neutral")
        if not np.array_equal(
                projection_vectors["action_offset"], _EXPECTED_ACTION_OFFSET):
            raise ValueError(
                "action_applier.action_offset must match the locked policy config")
        if np.any(projection_vectors["joint_min"] >= projection_vectors["joint_max"]):
            raise ValueError("action_applier joint bounds are invalid")
        if np.any(_Q_NEUTRAL < projection_vectors["joint_min"]) or np.any(
                _Q_NEUTRAL > projection_vectors["joint_max"]):
            raise ValueError("locked q_neutral lies outside action_applier bounds")
        if np.any(_Q_CROUCH < projection_vectors["joint_min"]) or np.any(
                _Q_CROUCH > projection_vectors["joint_max"]):
            raise ValueError("locked q_crouch lies outside action_applier bounds")

        self._mature_policy = mature_policy
        self._action_applier = action_applier
        self._policy_identity = {
            key: copy.deepcopy(policy_manifest[key])
            for key in _MATURE_POLICY_IDENTITY
        }
        self._projection_vectors = projection_vectors
        self._behavior_steps = np.asarray(
            RECOVERY_BEHAVIOR_STEPS, dtype=np.int64)
        self._behavior_steps.setflags(write=False)
        self._manifest = {
            "candidate_protocol": self._config.manifest_protocol(),
            "candidate_protocol_sha256": self._config.protocol_sha256(),
            "mature_policy_identity": copy.deepcopy(self._policy_identity),
            "action_projection": {
                name: value.tolist()
                for name, value in self._projection_vectors.items()
            } | {
                "max_joint_delta": None,
                "use_action_filter": False,
            },
            "input_boundary": "corrected_deployable_5x46_only",
            "privileged_inputs": "forbidden",
        }
        self._fingerprint = _canonical_sha256(self._manifest)

    @property
    def candidate_count(self) -> int:
        return RECOVERY_BEHAVIOR_COUNT

    @property
    def behavior_steps(self) -> np.ndarray:
        # Return an isolated snapshot.  A NumPy array that owns its storage can
        # have ``WRITEABLE`` re-enabled even after ``setflags(write=False)``;
        # exposing the owning protocol array would therefore let a caller
        # mutate the evaluator's locked durations behind the manifest's back.
        result = self._behavior_steps.copy()
        result.setflags(write=False)
        return result

    @property
    def durations(self) -> np.ndarray:
        """Alias used by generic native recovery-program evaluators."""
        return self.behavior_steps

    def manifest_protocol(self) -> dict[str, Any]:
        return self._config.manifest_protocol()

    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    def fingerprint(self) -> str:
        return self._fingerprint

    def capture_branch_state(self) -> None:
        """Return the only valid state for this deliberately stateless law."""
        return None

    def restore_branch_state(self, state: None) -> None:
        if state is not None:
            raise ValueError("RecoveryBehaviorLibrary branch state must be None")

    def _mature_action(self, history: np.ndarray) -> np.ndarray:
        value = self._mature_policy.deterministic_action(history[-1].copy())
        return _checked_action(value, "mature_policy action")

    def _joint_target_action(self, q_target: np.ndarray) -> np.ndarray:
        return qpos_to_action(
            q_target,
            init_qpos=self._projection_vectors["init_qpos"],
            action_offset=self._projection_vectors["action_offset"],
        )

    def __call__(
        self,
        candidate_index: int,
        observation_history: np.ndarray,
        step: int,
        nominal_action: np.ndarray,
    ) -> np.ndarray:
        """Return one active behavior action without consuming any RNG."""
        index = _checked_index(candidate_index)
        _checked_step(step, candidate_index=index)
        history = _checked_history(observation_history)
        nominal = _checked_action(nominal_action, "nominal_action")
        if index == 0:
            return nominal
        if index in (1, 2, 3):
            return self._mature_action(history)

        newest = history[-1]
        q_measured = newest[slice(*OBSERVATION_JOINT_Q_SLICE)]
        if index == 4:
            q_target = q_measured
        elif index in (5, 6):
            q_target = 0.5 * q_measured + 0.5 * _Q_NEUTRAL
        else:
            previous_q_target = newest[
                slice(*OBSERVATION_PREVIOUS_Q_TARGET_SLICE)]
            goal = _Q_NEUTRAL if index == 7 else _Q_CROUCH
            q_target = previous_q_target + np.clip(
                goal - previous_q_target,
                -RAMP_MAX_DELTA_RAD,
                RAMP_MAX_DELTA_RAD,
            )
        return self._joint_target_action(q_target)

    def preview_candidates(
        self,
        observation_history: np.ndarray,
        nominal_action: np.ndarray,
    ) -> np.ndarray:
        """Build the immutable K9 step-zero requested-action matrix."""
        history = _checked_history(observation_history)
        nominal = _checked_action(nominal_action, "nominal_action")
        actions = np.empty((RECOVERY_BEHAVIOR_COUNT, 12), dtype=np.float32)
        actions[0] = nominal
        mature = self._mature_action(history)
        actions[1:4] = mature
        for index in range(4, RECOVERY_BEHAVIOR_COUNT):
            actions[index] = self(index, history, 0, nominal)
        actions.setflags(write=False)
        return actions

    def preview_projected(
        self,
        observation_history: np.ndarray,
        nominal_action: np.ndarray,
    ) -> RecoveryBehaviorPreview:
        """Project all step-zero actions independently from the same state."""
        history = _checked_history(observation_history)
        actions = self.preview_candidates(history, nominal_action)
        current_qpos = history[-1, slice(*OBSERVATION_JOINT_Q_SLICE)]
        projected = self._action_applier.preview_many(actions, current_qpos)
        return RecoveryBehaviorPreview(
            requested=np.stack([
                value.action_requested for value in projected]),
            executed=np.stack([
                value.action_executed for value in projected]),
            q_target=np.stack([
                value.action_q_target for value in projected]),
            kind=np.asarray(RECOVERY_BEHAVIOR_KINDS),
            mask=np.ones(RECOVERY_BEHAVIOR_COUNT, dtype=bool),
            behavior_steps=self._behavior_steps,
            manifest_protocol=self.manifest_protocol(),
            library_fingerprint_sha256=self.fingerprint(),
        )


def build_recovery_behavior_library(
    mature_policy: DeterministicRecoveryPolicy,
    action_applier: ActionApplier,
    *,
    config: RecoveryBehaviorConfig | None = None,
) -> RecoveryBehaviorLibrary:
    """Construct the locked v3 controller with explicit runtime bindings."""
    return RecoveryBehaviorLibrary(
        mature_policy=mature_policy,
        action_applier=action_applier,
        config=config,
    )


__all__ = [
    "CANDIDATE_BEHAVIOR_STEPS",
    "CANDIDATE_KINDS",
    "MATURE_CHECKPOINT_FINGERPRINT_SHA256",
    "MATURE_POLICY_ACTOR_SHA256",
    "MATURE_POLICY_CONFIG_SHA256",
    "MATURE_POLICY_FINGERPRINT_SHA256",
    "MATURE_POLICY_STATE_DICT_SHA256",
    "MATURE_POLICY_TRAINING_STEP",
    "OBSERVATION_HISTORY_SHAPE",
    "OBSERVATION_JOINT_Q_SLICE",
    "OBSERVATION_PREVIOUS_Q_TARGET_SLICE",
    "Q_CROUCH_PER_LEG_RAD",
    "Q_NEUTRAL_PER_LEG_RAD",
    "RAMP_MAX_DELTA_RAD",
    "RECOVERY_BEHAVIOR_COUNT",
    "RECOVERY_BEHAVIOR_KINDS",
    "RECOVERY_BEHAVIOR_PROTOCOL_VERSION",
    "RECOVERY_BEHAVIOR_STEPS",
    "DeterministicRecoveryPolicy",
    "RecoveryBehaviorConfig",
    "RecoveryBehaviorLibrary",
    "RecoveryBehaviorPreview",
    "build_recovery_behavior_library",
]
