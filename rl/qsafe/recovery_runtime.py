"""Persistent K9 recovery-option runtime with auditable action ownership.

The controller in this module owns the option state machine and verifies the
complete frozen-Q_safe inference proof at every idle decision.  A caller
cannot inject a bare K9 index.  A selected non-nominal behavior owns exactly
its preregistered number of policy steps (unless the episode terminates), after
which the controller is spent until an explicit episode reset.

Actor evaluation is kept outside the controller, but its randomness is not:
``CounterBasedActorShadow`` derives one explicit 12D standard-normal vector
from a Stage-C/Stage-D counter key and supplies it to a deterministic actor.
``step`` requires that proposal even when it will be rejected, preserving one
draw per absolute policy step and making the rejected nominal action audit-only
data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import re
from typing import Any, Mapping, Protocol

import numpy as np

from rl.qsafe.artifact import LoadedQSafeArtifact
from rl.qsafe.recovery_inference import RecoveryQSafeInference
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_BEHAVIOR_STEPS,
    RECOVERY_PROGRAM_NAMES,
    build_recovery_program_features,
    make_recovery_program_feature_manifest,
    validate_recovery_program_binding,
)
from rl.qsafe.recovery_selector import (
    RecoverySelectorBundle,
)
from runtime.inference.actions import ActionApplier, ActionProjection
from safety_data.recovery_behaviors import (
    OBSERVATION_HISTORY_SHAPE,
    RECOVERY_BEHAVIOR_COUNT,
    RECOVERY_BEHAVIOR_KINDS,
    RECOVERY_BEHAVIOR_STEPS,
    RecoveryBehaviorConfig,
    RecoveryBehaviorLibrary,
)


_ACTION_DIM = 12
_OBSERVATION_DIM = OBSERVATION_HISTORY_SHAPE[1]
_K9_NAMES = tuple(RECOVERY_BEHAVIOR_KINDS)
_K9_STEPS = tuple(int(value) for value in RECOVERY_BEHAVIOR_STEPS)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_STREAM_KIND = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ACTOR_COUNTER_DOMAIN = b"qsafe.recovery_actor_shadow.v1\0"
_STAGE_C_DOMAIN = _ACTOR_COUNTER_DOMAIN + b"stage_c\0"
_STAGE_D_DOMAIN = _ACTOR_COUNTER_DOMAIN + b"stage_d\0"
_ACTOR_EXTERNAL_NOISE_CONTRACT = {
    "schema_version": "qsafe.external_noise_actor.v1",
    "randomness_source": "caller_supplied_only",
    "distribution": "standard_normal",
    "noise_shape": [12],
    "noise_dtype": "float32",
    "deterministic_for_equal_inputs": True,
    "stateful_rng": "forbidden",
}
_ACTOR_SNAPSHOT_SCHEMA = "qsafe.actor_snapshot.v1"
_ACTOR_SNAPSHOT_FIELDS = {
    "schema_version",
    "actor_state_sha256",
    "actor_weight_version",
    "actor_update_hash_chain_sha256",
}
_ACTOR_PROPOSAL_TOKEN = object()
_RUNTIME_ISSUE_TOKEN = object()


class RecoveryRuntimeState(str, Enum):
    """The four states locked by the persistent-option protocol."""

    IDLE = "idle"
    OPTION = "option"
    SPENT_UNTIL_RESET = "spent_until_reset"
    TERMINAL = "terminal"


class RecoveryActionOwner(str, Enum):
    """Source that owns the action actually sent to the runtime."""

    NOMINAL_ACTOR = "nominal_actor"
    RECOVERY_BEHAVIOR = "recovery_behavior"


class RecoveryBehaviorActionProvider(Protocol):
    """Locked K9 behavior interface implemented by RecoveryBehaviorLibrary."""

    candidate_count: int

    @property
    def behavior_steps(self) -> np.ndarray: ...

    def manifest_protocol(self) -> Mapping[str, Any]: ...

    def manifest(self) -> Mapping[str, Any]: ...

    def fingerprint(self) -> str: ...

    def require_live_integrity(self) -> None: ...

    def preview_candidates(
        self,
        observation_history: np.ndarray,
        nominal_action: np.ndarray,
    ) -> np.ndarray: ...

    def __call__(
        self,
        candidate_index: int,
        observation_history: np.ndarray,
        step: int,
        nominal_action: np.ndarray,
    ) -> np.ndarray: ...


class DeterministicExternalNoiseActorProvider(Protocol):
    """Actor whose only stochastic input is an explicit caller-owned vector."""

    def external_noise_contract(self) -> Mapping[str, Any]: ...

    def manifest(self) -> Mapping[str, Any]: ...

    def fingerprint(self) -> str: ...

    def actor_snapshot_manifest(
        self, absolute_step: int,
    ) -> Mapping[str, Any]: ...

    def actor_snapshot_fingerprint(self, absolute_step: int) -> str: ...

    def action_from_external_noise(
        self,
        observation: np.ndarray,
        standard_normal_noise: np.ndarray,
    ) -> np.ndarray: ...


class ActionProjectionProvider(Protocol):
    """Projection boundary bound to the recovery library's full manifest."""

    def manifest(self) -> Mapping[str, Any]: ...

    def fingerprint(self) -> str: ...

    def require_live_integrity(self) -> None: ...

    def preview_many(
        self,
        requested_actions: np.ndarray,
        current_observation: np.ndarray,
    ) -> tuple[ActionProjection, ...]: ...

    def __call__(
        self,
        requested_action: np.ndarray,
        current_observation: np.ndarray,
    ) -> ActionProjection: ...


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


def _vector(
    value: Any,
    *,
    name: str,
    width: int,
    normalized: bool,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if result.shape != (width,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {width}D vector")
    if normalized and (
            np.any(result < -1.0 - 1e-6)
            or np.any(result > 1.0 + 1e-6)):
        raise ValueError(f"{name} must lie in normalized [-1, 1]")
    if normalized:
        result = np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


def _observation(value: Any, *, name: str = "current_observation") -> np.ndarray:
    return _vector(
        value,
        name=name,
        width=_OBSERVATION_DIM,
        normalized=False,
    )


def _history(value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float32).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_history must be numeric") from exc
    if result.shape != OBSERVATION_HISTORY_SHAPE:
        raise ValueError("observation_history must have exact shape [5,46]")
    if not np.all(np.isfinite(result)):
        raise ValueError("observation_history must contain only finite values")
    return result


def _observation_sha256(observation: np.ndarray) -> str:
    canonical = np.ascontiguousarray(observation, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _history_sha256(history: np.ndarray) -> str:
    canonical = np.ascontiguousarray(history, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: Mapping[str, Any], *, name: str) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _u64(value: Any, *, name: str) -> int:
    result = _integer(value, name=name)
    if result > (1 << 64) - 1:
        raise ValueError(f"{name} must fit unsigned 64-bit encoding")
    return result


def _u64le(value: int) -> bytes:
    return value.to_bytes(8, byteorder="little", signed=False)


def _stream_bytes(value: Any) -> tuple[str, bytes]:
    if not isinstance(value, str) or _STREAM_KIND.fullmatch(value) is None:
        raise ValueError(
            "stream_kind must match lowercase [a-z][a-z0-9_]{0,63}")
    encoded = value.encode("ascii")
    if value != "nominal_actor":
        raise ValueError("actor counter stream_kind must be exactly 'nominal_actor'")
    return value, len(encoded).to_bytes(2, "little") + encoded


def _validated_actor_snapshot(
    manifest: Any,
    fingerprint: Any,
) -> tuple[dict[str, Any], str]:
    """Validate one portable actor-weight snapshot and its canonical hash."""
    if not isinstance(manifest, Mapping) or set(manifest) != (
            _ACTOR_SNAPSHOT_FIELDS):
        raise ValueError(
            "actor snapshot manifest must have the exact v1 field set")
    checked = copy.deepcopy(dict(manifest))
    if checked.get("schema_version") != _ACTOR_SNAPSHOT_SCHEMA:
        raise ValueError("actor snapshot schema_version drifted")
    checked["actor_state_sha256"] = _sha256(
        checked.get("actor_state_sha256"), name="actor snapshot state SHA-256")
    checked["actor_update_hash_chain_sha256"] = _sha256(
        checked.get("actor_update_hash_chain_sha256"),
        name="actor snapshot update-chain SHA-256",
    )
    checked["actor_weight_version"] = _u64(
        checked.get("actor_weight_version"),
        name="actor snapshot weight version",
    )
    checked_fingerprint = _sha256(
        fingerprint, name="actor snapshot fingerprint")
    expected = _canonical_sha256(checked, name="actor snapshot manifest")
    if checked_fingerprint != expected:
        raise ValueError("actor snapshot manifest/fingerprint mismatch")
    return checked, checked_fingerprint


def _counter_key_fields(key: "ActorCounterKey") -> dict[str, Any]:
    if type(key) is StageCActorCounterKey:
        return {
            "stage": "C",
            "state_hash_sha256": key.state_hash_sha256,
            "replica": key.replica,
            "absolute_step": key.absolute_step,
            "stream_kind": key.stream_kind,
            "draw_index": key.draw_index,
        }
    if type(key) is StageDActorCounterKey:
        return {
            "stage": "D",
            "training_seed": key.training_seed,
            "absolute_exposure_step": key.absolute_exposure_step,
            "stream_kind": key.stream_kind,
            "draw_index": key.draw_index,
        }
    raise TypeError("counter key must be an exact Stage-C or Stage-D key")


class BoundActionProjectionProvider:
    """Bind an ``ActionApplier`` to its exact JSON projection manifest.

    The wrapper derives, rather than accepts, every manifest value from the
    wrapped applier.  The controller subsequently requires this full mapping
    and its canonical fingerprint to equal the recovery library binding.
    """

    def __init__(self, action_applier: ActionApplier) -> None:
        if type(action_applier) is not ActionApplier:
            raise TypeError("action_applier must be the exact ActionApplier")
        vectors: dict[str, np.ndarray] = {}
        for name in ("init_qpos", "action_offset", "joint_min", "joint_max"):
            value = np.asarray(getattr(action_applier, name), dtype=np.float32)
            if value.shape != (_ACTION_DIM,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"action_applier.{name} must be a finite 12D vector")
            vectors[name] = value.copy()
        max_delta = action_applier.max_joint_delta
        if max_delta is not None:
            raise ValueError(
                "claim-bearing recovery projection forbids max_joint_delta")
        if action_applier.action_filter is not None:
            raise ValueError(
                "claim-bearing recovery projection forbids action filters")
        self._action_applier = action_applier
        self._action_applier_callables = (
            ActionApplier.project,
            ActionApplier.preview_many,
            ActionApplier.executed_action,
        )
        self._manifest = {
            name: value.tolist() for name, value in vectors.items()
        } | {
            "max_joint_delta": None,
            "use_action_filter": False,
        }
        self._fingerprint = _canonical_sha256(
            self._manifest, name="action projection manifest")

    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._manifest)

    def fingerprint(self) -> str:
        return self._fingerprint

    def _current_manifest(self) -> dict[str, Any]:
        if type(self._action_applier) is not ActionApplier or (
                self._action_applier.max_joint_delta is not None) or (
                self._action_applier.action_filter is not None):
            raise ValueError("bound action projection mutated")
        if self._action_applier_callables != (
                ActionApplier.project,
                ActionApplier.preview_many,
                ActionApplier.executed_action) or any(
                    name in self._action_applier.__dict__
                    for name in ("project", "preview_many", "executed_action")):
            raise ValueError("bound action projection callable surface mutated")
        vectors: dict[str, list[float]] = {}
        for name in ("init_qpos", "action_offset", "joint_min", "joint_max"):
            value = np.asarray(getattr(self._action_applier, name))
            if value.dtype != np.dtype(np.float32) or value.shape != (
                    _ACTION_DIM,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"bound action projection {name} mutated")
            vectors[name] = value.tolist()
        return vectors | {
            "max_joint_delta": None,
            "use_action_filter": False,
        }

    def require_live_integrity(self) -> None:
        current = self._current_manifest()
        if current != self._manifest or self._fingerprint != _canonical_sha256(
                current, name="action projection manifest"):
            raise ValueError("bound action projection manifest mutated")

    @staticmethod
    def _qpos(current_observation: np.ndarray) -> np.ndarray:
        observation = _observation(current_observation)
        return observation[:_ACTION_DIM].copy()

    def preview_many(
        self,
        requested_actions: np.ndarray,
        current_observation: np.ndarray,
    ) -> tuple[ActionProjection, ...]:
        self.require_live_integrity()
        actions = np.asarray(requested_actions)
        if actions.dtype != np.dtype(np.float32) or actions.shape != (
                RECOVERY_BEHAVIOR_COUNT, _ACTION_DIM) or not np.all(
                np.isfinite(actions)):
            raise ValueError(
                "projection preview requires finite float32 K9 actions")
        result = self._action_applier.preview_many(
            actions.copy(), self._qpos(current_observation))
        self.require_live_integrity()
        return result

    def __call__(
        self,
        requested_action: np.ndarray,
        current_observation: np.ndarray,
    ) -> ActionProjection:
        self.require_live_integrity()
        action = _vector(
            requested_action,
            name="projection requested action",
            width=_ACTION_DIM,
            normalized=True,
        )
        result = self._action_applier.project(
            action, self._qpos(current_observation))
        self.require_live_integrity()
        return result


@dataclass(frozen=True)
class StageCActorCounterKey:
    """Exact Stage-C actor RNG key for one state replica and policy step."""

    state_hash_sha256: str
    replica: int
    absolute_step: int
    stream_kind: str
    draw_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_hash_sha256", _sha256(
            self.state_hash_sha256, name="state_hash_sha256"))
        object.__setattr__(self, "replica", _u64(
            self.replica, name="replica"))
        object.__setattr__(self, "absolute_step", _u64(
            self.absolute_step, name="absolute_step"))
        stream, _ = _stream_bytes(self.stream_kind)
        object.__setattr__(self, "stream_kind", stream)
        draw = _u64(self.draw_index, name="draw_index")
        if draw != 0:
            raise ValueError("one-draw actor shadow requires draw_index zero")
        object.__setattr__(self, "draw_index", draw)

    def seed_payload(self) -> bytes:
        _, stream = _stream_bytes(self.stream_kind)
        return b"".join((
            _STAGE_C_DOMAIN,
            bytes.fromhex(self.state_hash_sha256),
            _u64le(self.replica),
            _u64le(self.absolute_step),
            stream,
            _u64le(self.draw_index),
        ))


@dataclass(frozen=True)
class StageDActorCounterKey:
    """Exact Stage-D actor RNG key for one seed and exposure step."""

    training_seed: int
    absolute_exposure_step: int
    stream_kind: str
    draw_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_seed", _u64(
            self.training_seed, name="training_seed"))
        object.__setattr__(self, "absolute_exposure_step", _u64(
            self.absolute_exposure_step,
            name="absolute_exposure_step",
        ))
        stream, _ = _stream_bytes(self.stream_kind)
        object.__setattr__(self, "stream_kind", stream)
        draw = _u64(self.draw_index, name="draw_index")
        if draw != 0:
            raise ValueError("one-draw actor shadow requires draw_index zero")
        object.__setattr__(self, "draw_index", draw)

    def seed_payload(self) -> bytes:
        _, stream = _stream_bytes(self.stream_kind)
        return b"".join((
            _STAGE_D_DOMAIN,
            _u64le(self.training_seed),
            _u64le(self.absolute_exposure_step),
            stream,
            _u64le(self.draw_index),
        ))


@dataclass(frozen=True)
class StageCActorCounterDomain:
    """Stage-C key fields shared by all consecutive steps in one replica."""

    state_hash_sha256: str
    replica: int
    stream_kind: str = "nominal_actor"

    def __post_init__(self) -> None:
        key = StageCActorCounterKey(
            self.state_hash_sha256,
            self.replica,
            0,
            self.stream_kind,
        )
        object.__setattr__(self, "state_hash_sha256", key.state_hash_sha256)
        object.__setattr__(self, "replica", key.replica)
        object.__setattr__(self, "stream_kind", key.stream_kind)

    def key(self, absolute_step: int) -> StageCActorCounterKey:
        return StageCActorCounterKey(
            state_hash_sha256=self.state_hash_sha256,
            replica=self.replica,
            absolute_step=absolute_step,
            stream_kind=self.stream_kind,
        )


@dataclass(frozen=True)
class StageDActorCounterDomain:
    """Stage-D key fields shared by all exposure steps for one train seed."""

    training_seed: int
    stream_kind: str = "nominal_actor"

    def __post_init__(self) -> None:
        key = StageDActorCounterKey(
            self.training_seed,
            0,
            self.stream_kind,
        )
        object.__setattr__(self, "training_seed", key.training_seed)
        object.__setattr__(self, "stream_kind", key.stream_kind)

    def key(self, absolute_step: int) -> StageDActorCounterKey:
        return StageDActorCounterKey(
            training_seed=self.training_seed,
            absolute_exposure_step=absolute_step,
            stream_kind=self.stream_kind,
        )


ActorCounterKey = StageCActorCounterKey | StageDActorCounterKey
ActorCounterDomain = StageCActorCounterDomain | StageDActorCounterDomain


def _counter_seed_and_noise(
    counter_key: ActorCounterKey,
) -> tuple[bytes, np.ndarray]:
    digest = hashlib.sha256(counter_key.seed_payload()).digest()
    seed = int.from_bytes(digest, byteorder="little", signed=False)
    generator = np.random.Generator(np.random.PCG64(seed))
    return digest, generator.standard_normal(
        _ACTION_DIM, dtype=np.float32)


def _actor_proposal_live_sha256(proposal: "NominalActorProposal") -> str:
    digest = hashlib.sha256(b"qsafe.nominal_actor_proposal.live.v1\0")
    digest.update(_u64le(proposal.absolute_step))
    digest.update(proposal.observation_sha256.encode("ascii"))
    digest.update(proposal.counter_seed_sha256.encode("ascii"))
    digest.update(proposal.actor_provider_fingerprint_sha256.encode("ascii"))
    digest.update(proposal.actor_snapshot_fingerprint_sha256.encode("ascii"))
    digest.update(json.dumps(
        proposal.actor_snapshot_manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8"))
    digest.update(proposal.counter_key.seed_payload())
    for name, value in (
        ("action", proposal.action),
        ("external_noise", proposal.external_noise),
    ):
        array = np.ascontiguousarray(value, dtype="<f4")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class NominalActorProposal:
    """Auditable proof of one counter-keyed external actor-noise draw."""

    absolute_step: int
    action: np.ndarray
    observation_sha256: str
    counter_key: ActorCounterKey
    counter_seed_sha256: str
    external_noise: np.ndarray
    actor_provider_fingerprint_sha256: str
    actor_snapshot_manifest: dict[str, Any]
    actor_snapshot_fingerprint_sha256: str
    _proposal_token: object | None = field(
        default=None, repr=False, compare=False)
    _proposal_live_sha256: str | None = field(
        default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._proposal_token is not _ACTOR_PROPOSAL_TOKEN:
            raise ValueError(
                "nominal actor proposal must come from CounterBasedActorShadow")
        step = _integer(self.absolute_step, name="absolute_step")
        action = _vector(
            self.action,
            name="nominal actor proposal",
            width=_ACTION_DIM,
            normalized=True,
        )
        digest = _sha256(
            self.observation_sha256, name="observation_sha256")
        if not isinstance(
                self.counter_key,
                (StageCActorCounterKey, StageDActorCounterKey)):
            raise TypeError("counter_key must be an exact Stage-C or Stage-D key")
        key_step = (
            self.counter_key.absolute_step
            if isinstance(self.counter_key, StageCActorCounterKey)
            else self.counter_key.absolute_exposure_step
        )
        if key_step != step:
            raise ValueError("counter key step differs from proposal step")
        seed_hash = _sha256(
            self.counter_seed_sha256, name="counter_seed_sha256")
        expected_seed_hash = hashlib.sha256(
            self.counter_key.seed_payload()).hexdigest()
        if seed_hash != expected_seed_hash:
            raise ValueError("counter_seed_sha256 does not bind counter key")
        noise = np.asarray(self.external_noise)
        if noise.dtype != np.dtype(np.float32) or noise.shape != (
                _ACTION_DIM,) or not np.all(np.isfinite(noise)):
            raise ValueError("external_noise must be finite float32 shape [12]")
        _, expected_noise = _counter_seed_and_noise(self.counter_key)
        if not np.array_equal(noise, expected_noise):
            raise ValueError("external_noise does not match counter-keyed PCG64")
        actor_fingerprint = _sha256(
            self.actor_provider_fingerprint_sha256,
            name="actor_provider_fingerprint_sha256",
        )
        snapshot, snapshot_fingerprint = _validated_actor_snapshot(
            self.actor_snapshot_manifest,
            self.actor_snapshot_fingerprint_sha256,
        )
        object.__setattr__(self, "absolute_step", step)
        object.__setattr__(self, "action", _readonly(action))
        object.__setattr__(self, "observation_sha256", digest)
        object.__setattr__(self, "counter_seed_sha256", seed_hash)
        object.__setattr__(self, "external_noise", _readonly(noise))
        object.__setattr__(
            self, "actor_provider_fingerprint_sha256", actor_fingerprint)
        object.__setattr__(self, "actor_snapshot_manifest", snapshot)
        object.__setattr__(
            self,
            "actor_snapshot_fingerprint_sha256",
            snapshot_fingerprint,
        )
        if self._proposal_live_sha256 is not None and (
                self._proposal_live_sha256
                != _actor_proposal_live_sha256(self)):
            raise ValueError("nominal actor proposal live digest mismatch")

    def require_live_integrity(self) -> None:
        snapshot, snapshot_fingerprint = _validated_actor_snapshot(
            self.actor_snapshot_manifest,
            self.actor_snapshot_fingerprint_sha256,
        )
        if self._proposal_token is not _ACTOR_PROPOSAL_TOKEN or (
                self._proposal_live_sha256 is None) or (
                self._proposal_live_sha256
                != _actor_proposal_live_sha256(self)) or (
                snapshot != self.actor_snapshot_manifest) or (
                snapshot_fingerprint
                != self.actor_snapshot_fingerprint_sha256):
            raise ValueError("nominal actor proposal is forged or mutated")

    @property
    def semantic(self) -> str:
        return "one_counter_keyed_external_noise_draw_per_absolute_step"

    @property
    def proposal_sha256(self) -> str:
        self.require_live_integrity()
        assert self._proposal_live_sha256 is not None
        return self._proposal_live_sha256

    def audit_fields(self) -> dict[str, Any]:
        """Return canonical JSON-safe actor RNG and weight identity fields."""
        self.require_live_integrity()
        noise_bytes = np.ascontiguousarray(
            self.external_noise, dtype="<f4").tobytes(order="C")
        action_bytes = np.ascontiguousarray(
            self.action, dtype="<f4").tobytes(order="C")
        return {
            "nominal_actor_proposal_sha256": self.proposal_sha256,
            "nominal_actor_observation_sha256": self.observation_sha256,
            "nominal_actor_counter_key": _counter_key_fields(self.counter_key),
            "nominal_actor_counter_payload_sha256": hashlib.sha256(
                self.counter_key.seed_payload()).hexdigest(),
            "nominal_actor_counter_seed_sha256": self.counter_seed_sha256,
            "nominal_actor_external_noise_f4_sha256": hashlib.sha256(
                noise_bytes).hexdigest(),
            "nominal_actor_action_f4_sha256": hashlib.sha256(
                action_bytes).hexdigest(),
            "nominal_actor_provider_fingerprint_sha256": (
                self.actor_provider_fingerprint_sha256),
            "nominal_actor_snapshot_manifest": copy.deepcopy(
                self.actor_snapshot_manifest),
            "nominal_actor_snapshot_fingerprint_sha256": (
                self.actor_snapshot_fingerprint_sha256),
        }


class CounterBasedActorShadow:
    """Generate one explicit PCG64 actor-noise draw per consecutive step.

    Legacy callables and actors that own their RNG are not accepted.  The
    provider must declare the exact external-noise contract and implement only
    deterministic evaluation from observation plus the supplied 12D noise.
    Each returned action is evaluated twice with equal inputs and must be
    bit-identical, a runtime challenge that fails closed on hidden state.
    """

    def __init__(
        self,
        actor_provider: DeterministicExternalNoiseActorProvider,
        counter_domain: ActorCounterDomain,
        *,
        first_absolute_step: int = 0,
    ) -> None:
        contract_method = getattr(actor_provider, "external_noise_contract", None)
        action_method = getattr(actor_provider, "action_from_external_noise", None)
        manifest_method = getattr(actor_provider, "manifest", None)
        fingerprint_method = getattr(actor_provider, "fingerprint", None)
        snapshot_manifest_method = getattr(
            actor_provider, "actor_snapshot_manifest", None)
        snapshot_fingerprint_method = getattr(
            actor_provider, "actor_snapshot_fingerprint", None)
        if not all(callable(method) for method in (
                contract_method, action_method, manifest_method,
                fingerprint_method, snapshot_manifest_method,
                snapshot_fingerprint_method)):
            raise TypeError(
                "actor_provider must implement deterministic external-noise "
                "API plus static and snapshot manifest/fingerprint")
        contract = contract_method()
        if not isinstance(contract, Mapping) or dict(contract) != (
                _ACTOR_EXTERNAL_NOISE_CONTRACT):
            raise ValueError("actor provider external-noise contract has drifted")
        actor_manifest = manifest_method()
        if not isinstance(actor_manifest, Mapping):
            raise TypeError("actor provider manifest must be a mapping")
        actor_manifest = copy.deepcopy(dict(actor_manifest))
        if actor_manifest.get("external_noise_contract") != (
                _ACTOR_EXTERNAL_NOISE_CONTRACT):
            raise ValueError(
                "actor provider manifest does not bind external-noise contract")
        actor_fingerprint = _sha256(
            fingerprint_method(), name="actor provider fingerprint")
        if actor_fingerprint != _canonical_sha256(
                actor_manifest, name="actor provider manifest"):
            raise ValueError("actor provider manifest/fingerprint mismatch")
        if not isinstance(
                counter_domain,
                (StageCActorCounterDomain, StageDActorCounterDomain)):
            raise TypeError("counter_domain must be an exact Stage-C/Stage-D domain")
        self._actor_provider = actor_provider
        self._actor_provider_manifest = actor_manifest
        self._actor_provider_fingerprint_sha256 = actor_fingerprint
        self._counter_domain = counter_domain
        self._next_absolute_step = _integer(
            first_absolute_step, name="first_absolute_step")
        self._consumed_count = 0
        self._pending_proposals: dict[int, str] = {}
        self._frozen_stage_c_snapshot: tuple[dict[str, Any], str] | None = None
        self._last_stage_d_snapshot: tuple[dict[str, Any], str] | None = None

    def _require_static_provider_integrity(self) -> None:
        contract = self._actor_provider.external_noise_contract()
        if not isinstance(contract, Mapping) or dict(contract) != (
                _ACTOR_EXTERNAL_NOISE_CONTRACT):
            raise ValueError("actor provider external-noise contract mutated")
        manifest = self._actor_provider.manifest()
        if not isinstance(manifest, Mapping) or dict(manifest) != (
                self._actor_provider_manifest):
            raise ValueError("actor provider static manifest mutated")
        fingerprint = _sha256(
            self._actor_provider.fingerprint(),
            name="actor provider fingerprint",
        )
        if fingerprint != self._actor_provider_fingerprint_sha256 or (
                fingerprint != _canonical_sha256(
                    self._actor_provider_manifest,
                    name="actor provider manifest",
                )):
            raise ValueError("actor provider static fingerprint mutated")

    def _snapshot(
        self,
        step: int,
        *,
        advance: bool,
    ) -> tuple[dict[str, Any], str]:
        self._require_static_provider_integrity()
        manifest = self._actor_provider.actor_snapshot_manifest(step)
        fingerprint = self._actor_provider.actor_snapshot_fingerprint(step)
        checked = _validated_actor_snapshot(manifest, fingerprint)
        self._require_static_provider_integrity()
        if isinstance(self._counter_domain, StageCActorCounterDomain):
            frozen = self._frozen_stage_c_snapshot
            if frozen is not None and checked != frozen:
                raise ValueError(
                    "Stage-C actor snapshot changed after its first proposal")
            if advance and frozen is None:
                self._frozen_stage_c_snapshot = copy.deepcopy(checked)
            return checked

        previous = self._last_stage_d_snapshot
        if previous is not None:
            previous_manifest, previous_fingerprint = previous
            previous_version = int(previous_manifest["actor_weight_version"])
            current_version = int(checked[0]["actor_weight_version"])
            if current_version < previous_version:
                raise ValueError("Stage-D actor weight version moved backwards")
            if current_version == previous_version and checked != previous:
                raise ValueError(
                    "Stage-D actor snapshot changed without a version advance")
            if current_version > previous_version and (
                    checked[0]["actor_state_sha256"]
                    == previous_manifest["actor_state_sha256"] or
                    checked[0]["actor_update_hash_chain_sha256"]
                    == previous_manifest["actor_update_hash_chain_sha256"] or
                    checked[1] == previous_fingerprint):
                raise ValueError(
                    "Stage-D actor version advanced without new weights and "
                    "update-chain identity")
        if advance:
            self._last_stage_d_snapshot = copy.deepcopy(checked)
        return checked

    @property
    def next_absolute_step(self) -> int:
        return self._next_absolute_step

    @property
    def consumed_count(self) -> int:
        return self._consumed_count

    @property
    def actor_provider_fingerprint_sha256(self) -> str:
        return self._actor_provider_fingerprint_sha256

    @staticmethod
    def external_noise_contract() -> dict[str, Any]:
        return copy.deepcopy(_ACTOR_EXTERNAL_NOISE_CONTRACT)

    def consume(
        self,
        *,
        absolute_step: int,
        current_observation: np.ndarray,
    ) -> NominalActorProposal:
        step = _integer(absolute_step, name="absolute_step")
        if step != self._next_absolute_step:
            raise RuntimeError(
                "actor shadow consumption must be consecutive: expected "
                f"absolute step {self._next_absolute_step}, got {step}"
            )
        observation = _observation(current_observation)
        counter_key = self._counter_domain.key(step)
        digest, noise = _counter_seed_and_noise(counter_key)
        snapshot_before = self._snapshot(step, advance=False)
        action_method = self._actor_provider.action_from_external_noise
        first = _vector(
            action_method(observation.copy(), noise.copy()),
            name="external-noise actor action",
            width=_ACTION_DIM,
            normalized=True,
        )
        snapshot_between = self._snapshot(step, advance=False)
        second = _vector(
            action_method(observation.copy(), noise.copy()),
            name="external-noise actor repeat action",
            width=_ACTION_DIM,
            normalized=True,
        )
        snapshot_after = self._snapshot(step, advance=False)
        if snapshot_before != snapshot_between or (
                snapshot_before != snapshot_after):
            raise RuntimeError(
                "actor snapshot changed during equal-input evaluation")
        if not np.array_equal(first, second):
            raise RuntimeError(
                "actor provider is stateful or nondeterministic for equal "
                "external-noise inputs")
        proposal = NominalActorProposal(
            absolute_step=step,
            action=first,
            observation_sha256=_observation_sha256(observation),
            counter_key=counter_key,
            counter_seed_sha256=digest.hex(),
            external_noise=noise,
            actor_provider_fingerprint_sha256=(
                self._actor_provider_fingerprint_sha256),
            actor_snapshot_manifest=snapshot_before[0],
            actor_snapshot_fingerprint_sha256=snapshot_before[1],
            _proposal_token=_ACTOR_PROPOSAL_TOKEN,
        )
        proposal = replace(
            proposal,
            _proposal_live_sha256=_actor_proposal_live_sha256(proposal),
        )
        # Advance only after one explicit draw and deterministic actor proof.
        committed_snapshot = self._snapshot(step, advance=True)
        if committed_snapshot != snapshot_before:
            raise RuntimeError("actor snapshot changed before proposal issuance")
        self._next_absolute_step += 1
        self._consumed_count += 1
        self._pending_proposals[step] = str(proposal._proposal_live_sha256)
        return proposal

    def validate_issued(self, proposal: NominalActorProposal) -> None:
        """Validate a still-pending proposal from this exact shadow."""
        if type(proposal) is not NominalActorProposal:
            raise TypeError("proposal must be NominalActorProposal")
        proposal.require_live_integrity()
        current_snapshot = self._snapshot(
            proposal.absolute_step, advance=False)
        if current_snapshot != (
                proposal.actor_snapshot_manifest,
                proposal.actor_snapshot_fingerprint_sha256):
            raise ValueError("nominal proposal actor snapshot is no longer current")
        expected_key = self._counter_domain.key(proposal.absolute_step)
        if proposal.counter_key != expected_key or (
                proposal.actor_provider_fingerprint_sha256
                != self._actor_provider_fingerprint_sha256):
            raise ValueError(
                "nominal proposal belongs to another actor shadow/domain")
        if self._pending_proposals.get(proposal.absolute_step) != (
                proposal._proposal_live_sha256):
            raise ValueError(
                "nominal proposal was not issued or was already committed")

    def commit_issued(self, proposal: NominalActorProposal) -> None:
        """Consume one previously validated proposal exactly once."""
        self.validate_issued(proposal)
        del self._pending_proposals[proposal.absolute_step]


def _digest_f4(digest: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype="<f4")
    digest.update(name.encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))


def _replay_live_sha256(replay: "RecoveryReplayAction") -> str:
    digest = hashlib.sha256(b"qsafe.recovery_replay_action.live.v1\0")
    scalar = {
        "absolute_step": replay.absolute_step,
        "owner": replay.owner.value,
        "behavior_index": replay.behavior_index,
        "behavior_name": replay.behavior_name,
        "behavior_step": replay.behavior_step,
        "nominal_is_log_only": replay.nominal_is_log_only,
        "nominal_proposal_sha256": replay.nominal_proposal_sha256,
        "training_action_semantic": replay.training_action_semantic,
    }
    digest.update(json.dumps(
        scalar, allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True).encode("utf-8"))
    for name, value in (
        ("action", replay.action),
        ("action_nominal", replay.action_nominal),
        ("action_executed", replay.action_executed),
        ("action_q_target", replay.action_q_target),
    ):
        _digest_f4(digest, name, value)
    return digest.hexdigest()


@dataclass(frozen=True)
class RecoveryReplayAction:
    """Single-step replay semantics for an applied runtime action."""

    absolute_step: int
    action: np.ndarray
    action_nominal: np.ndarray
    action_executed: np.ndarray
    action_q_target: np.ndarray
    owner: RecoveryActionOwner
    behavior_index: int
    behavior_name: str
    behavior_step: int
    nominal_is_log_only: bool
    nominal_proposal_sha256: str
    _issue_token: object | None = field(
        default=None, repr=False, compare=False)
    _live_sha256: str | None = field(
        default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issue_token is not _RUNTIME_ISSUE_TOKEN:
            raise ValueError(
                "replay action must be issued by PersistentRecoveryController")
        absolute_step = _integer(self.absolute_step, name="absolute_step")
        action = _vector(
            self.action, name="replay training action", width=_ACTION_DIM,
            normalized=True)
        nominal = _vector(
            self.action_nominal, name="replay nominal action", width=_ACTION_DIM,
            normalized=True)
        executed = _vector(
            self.action_executed, name="replay executed action",
            width=_ACTION_DIM, normalized=True)
        q_target = _vector(
            self.action_q_target, name="replay q-target", width=_ACTION_DIM,
            normalized=False)
        try:
            owner = RecoveryActionOwner(self.owner)
        except ValueError as exc:
            raise ValueError("replay owner is invalid") from exc
        index = _integer(self.behavior_index, name="behavior_index")
        if index >= RECOVERY_BEHAVIOR_COUNT:
            raise ValueError("behavior_index is outside locked K9")
        step = _integer(self.behavior_step, name="behavior_step")
        if self.behavior_name != _K9_NAMES[index]:
            raise ValueError("behavior_name does not match locked K9 index")
        log_only = _boolean(
            self.nominal_is_log_only, name="nominal_is_log_only")
        proposal_sha256 = _sha256(
            self.nominal_proposal_sha256,
            name="nominal_proposal_sha256",
        )
        expected_recovery = owner is RecoveryActionOwner.RECOVERY_BEHAVIOR
        if expected_recovery != log_only:
            raise ValueError(
                "rejected nominal is log-only exactly when recovery owns action")
        if expected_recovery:
            if index == 0 or not 0 <= step < _K9_STEPS[index]:
                raise ValueError("recovery replay behavior step is inactive")
        elif index != 0 or step != 0:
            raise ValueError("nominal replay record must use K9 index/step zero")

        for name, value in (
            ("action", action),
            ("action_nominal", nominal),
            ("action_executed", executed),
            ("action_q_target", q_target),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "absolute_step", absolute_step)
        object.__setattr__(self, "behavior_index", index)
        object.__setattr__(self, "behavior_step", step)
        object.__setattr__(self, "nominal_is_log_only", log_only)
        object.__setattr__(
            self, "nominal_proposal_sha256", proposal_sha256)
        if self._live_sha256 is not None and self._live_sha256 != (
                _replay_live_sha256(self)):
            raise ValueError("replay action live digest mismatch")

    @property
    def training_action_semantic(self) -> str:
        return "actual_requested_action"

    def require_live_integrity(self) -> None:
        for name in (
                "action", "action_nominal", "action_executed",
                "action_q_target"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(
                    np.float32) or value.shape != (_ACTION_DIM,) or (
                    value.flags.writeable) or not np.all(np.isfinite(value)):
                raise ValueError("replay action arrays are mutable or malformed")
        if self._issue_token is not _RUNTIME_ISSUE_TOKEN or (
                self._live_sha256 is None) or self._live_sha256 != (
                    _replay_live_sha256(self)):
            raise ValueError("replay action is forged or mutated")

    def transition_fields(self) -> dict[str, Any]:
        """Return replay fields with the actual request as training action."""
        self.require_live_integrity()
        return {
            "action": self.action.copy(),
            "action_nominal": self.action_nominal.copy(),
            "action_executed": self.action_executed.copy(),
            "action_q_target": self.action_q_target.copy(),
            "action_owner": self.owner.value,
            "recovery_behavior_index": int(self.behavior_index),
            "recovery_behavior_name": self.behavior_name,
            "recovery_behavior_step": int(self.behavior_step),
            "action_nominal_log_only": bool(self.nominal_is_log_only),
            "training_action_semantic": self.training_action_semantic,
            "recovery_runtime_absolute_step": int(self.absolute_step),
            "nominal_actor_proposal_sha256": self.nominal_proposal_sha256,
            "recovery_replay_proof_sha256": self._live_sha256,
        }

    def validate_transition_fields(self, fields: Mapping[str, Any]) -> None:
        """Fail closed if a transition trains on the rejected nominal action."""
        self.require_live_integrity()
        if not isinstance(fields, Mapping):
            raise TypeError("replay transition fields must be a mapping")
        expected_arrays = {
            "action": self.action,
            "action_nominal": self.action_nominal,
            "action_executed": self.action_executed,
            "action_q_target": self.action_q_target,
        }
        missing = [name for name in expected_arrays if name not in fields]
        if missing:
            raise ValueError(f"replay transition is missing action fields: {missing}")
        for name, expected in expected_arrays.items():
            actual = _vector(
                fields[name],
                name=f"replay transition {name}",
                width=_ACTION_DIM,
                normalized=name != "action_q_target",
            )
            if not np.array_equal(actual, expected):
                if name == "action":
                    raise ValueError(
                        "replay training action must be the actual requested "
                        "action, never the rejected nominal proposal"
                    )
                raise ValueError(f"replay transition {name} changed after execution")

        expected_scalars = {
            "action_owner": self.owner.value,
            "recovery_behavior_index": self.behavior_index,
            "recovery_behavior_name": self.behavior_name,
            "recovery_behavior_step": self.behavior_step,
            "action_nominal_log_only": self.nominal_is_log_only,
            "training_action_semantic": self.training_action_semantic,
            "recovery_runtime_absolute_step": self.absolute_step,
            "nominal_actor_proposal_sha256": self.nominal_proposal_sha256,
            "recovery_replay_proof_sha256": self._live_sha256,
        }
        for name, expected in expected_scalars.items():
            if fields.get(name) != expected:
                raise ValueError(f"replay transition {name} violates runtime semantics")


def _runtime_step_live_sha256(step: "RecoveryRuntimeStep") -> str:
    digest = hashlib.sha256(b"qsafe.recovery_runtime_step.live.v1\0")
    scalar = {
        "absolute_step": step.absolute_step,
        "episode_index": step.episode_index,
        "state_before": step.state_before.value,
        "state_after_action": step.state_after_action.value,
        "owner": step.owner.value,
        "behavior_index": step.behavior_index,
        "behavior_name": step.behavior_name,
        "behavior_step": step.behavior_step,
        "behavior_duration": step.behavior_duration,
        "nominal_rejected": step.nominal_rejected,
        "replay_sha256": step.replay._live_sha256,
        "nominal_proposal_sha256": step.nominal_proposal.proposal_sha256,
    }
    digest.update(json.dumps(
        scalar, allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True).encode("utf-8"))
    for name, value in (
        ("actual_requested", step.actual_requested),
        ("actual_executed", step.actual_executed),
        ("actual_q_target", step.actual_q_target),
        ("rejected_nominal", step.rejected_nominal),
    ):
        _digest_f4(digest, name, value)
    return digest.hexdigest()


@dataclass(frozen=True)
class RecoveryRuntimeStep:
    """Immutable, auditable action decision for one absolute policy step."""

    absolute_step: int
    episode_index: int
    state_before: RecoveryRuntimeState
    state_after_action: RecoveryRuntimeState
    owner: RecoveryActionOwner
    behavior_index: int
    behavior_name: str
    behavior_step: int
    behavior_duration: int
    actual_requested: np.ndarray
    actual_executed: np.ndarray
    actual_q_target: np.ndarray
    rejected_nominal: np.ndarray
    nominal_rejected: bool
    replay: RecoveryReplayAction
    nominal_proposal: NominalActorProposal
    _issue_token: object | None = field(
        default=None, repr=False, compare=False)
    _live_sha256: str | None = field(
        default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issue_token is not _RUNTIME_ISSUE_TOKEN:
            raise ValueError(
                "runtime step must be issued by PersistentRecoveryController")
        absolute_step = _integer(self.absolute_step, name="absolute_step")
        episode_index = _integer(self.episode_index, name="episode_index")
        try:
            state_before = RecoveryRuntimeState(self.state_before)
            state_after = RecoveryRuntimeState(self.state_after_action)
            owner = RecoveryActionOwner(self.owner)
        except ValueError as exc:
            raise ValueError("runtime step state/owner is invalid") from exc
        behavior_index = _integer(
            self.behavior_index, name="behavior_index")
        behavior_step = _integer(self.behavior_step, name="behavior_step")
        behavior_duration = _integer(
            self.behavior_duration, name="behavior_duration")
        if behavior_index >= RECOVERY_BEHAVIOR_COUNT or (
                self.behavior_name != _K9_NAMES[behavior_index]) or (
                behavior_duration != _K9_STEPS[behavior_index]):
            raise ValueError("runtime step behavior does not match locked K9")
        nominal_rejected = _boolean(
            self.nominal_rejected, name="nominal_rejected")
        for name in (
                "actual_requested", "actual_executed", "actual_q_target",
                "rejected_nominal"):
            value = _vector(
                getattr(self, name),
                name=f"runtime step {name}",
                width=_ACTION_DIM,
                normalized=name != "actual_q_target",
            )
            object.__setattr__(self, name, _readonly(value))
        if type(self.nominal_proposal) is not NominalActorProposal:
            raise TypeError("runtime step nominal proposal is invalid")
        self.nominal_proposal.require_live_integrity()
        if self.nominal_proposal.absolute_step != absolute_step or not (
                np.array_equal(
                    self.nominal_proposal.action, self.rejected_nominal)):
            raise ValueError("runtime step nominal proposal does not match action")
        if type(self.replay) is not RecoveryReplayAction:
            raise TypeError("runtime step replay proof is invalid")
        self.replay.require_live_integrity()
        if self.replay.absolute_step != absolute_step or (
                self.replay.nominal_proposal_sha256
                != self.nominal_proposal.proposal_sha256) or not all((
                    np.array_equal(self.replay.action, self.actual_requested),
                    np.array_equal(
                        self.replay.action_executed, self.actual_executed),
                    np.array_equal(
                        self.replay.action_q_target, self.actual_q_target),
                    np.array_equal(
                        self.replay.action_nominal, self.rejected_nominal),
                )) or self.replay.owner is not owner or (
                    self.replay.behavior_index != behavior_index) or (
                    self.replay.behavior_name != self.behavior_name) or (
                    self.replay.behavior_step != behavior_step) or (
                    self.replay.nominal_is_log_only != nominal_rejected):
            raise ValueError("runtime step and replay proof disagree")
        object.__setattr__(self, "absolute_step", absolute_step)
        object.__setattr__(self, "episode_index", episode_index)
        object.__setattr__(self, "state_before", state_before)
        object.__setattr__(self, "state_after_action", state_after)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "behavior_index", behavior_index)
        object.__setattr__(self, "behavior_step", behavior_step)
        object.__setattr__(self, "behavior_duration", behavior_duration)
        object.__setattr__(self, "nominal_rejected", nominal_rejected)
        if self._live_sha256 is not None and self._live_sha256 != (
                _runtime_step_live_sha256(self)):
            raise ValueError("runtime step live digest mismatch")

    def require_live_integrity(self) -> None:
        self.nominal_proposal.require_live_integrity()
        self.replay.require_live_integrity()
        for name in (
                "actual_requested", "actual_executed", "actual_q_target",
                "rejected_nominal"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.dtype != np.dtype(
                    np.float32) or value.shape != (_ACTION_DIM,) or (
                    value.flags.writeable) or not np.all(np.isfinite(value)):
                raise ValueError("runtime step arrays are mutable or malformed")
        if self._issue_token is not _RUNTIME_ISSUE_TOKEN or (
                self._live_sha256 is None) or self._live_sha256 != (
                    _runtime_step_live_sha256(self)):
            raise ValueError("runtime step is forged or mutated")

    def transition_fields(self) -> dict[str, Any]:
        """Return replay fields plus a complete actor/runtime audit record."""
        self.require_live_integrity()
        fields = self.replay.transition_fields()
        fields.update(self.nominal_proposal.audit_fields())
        fields.update({
            "recovery_runtime_step_sha256": self._live_sha256,
            "recovery_runtime_episode_index": self.episode_index,
            "recovery_runtime_state_before": self.state_before.value,
            "recovery_runtime_state_after_action": (
                self.state_after_action.value),
            "recovery_behavior_duration": self.behavior_duration,
        })
        return fields


@dataclass(frozen=True)
class RecoveryOutcome:
    """Outcome acknowledgment paired with exactly one emitted action."""

    absolute_step: int
    fell: bool
    terminated: bool
    state_after_outcome: RecoveryRuntimeState


def _validate_k9_provider(
    provider: RecoveryBehaviorActionProvider,
) -> tuple[str, dict[str, Any]]:
    if type(provider) is not RecoveryBehaviorLibrary:
        raise TypeError(
            "recovery_behavior_provider must be the exact locked "
            "RecoveryBehaviorLibrary")
    provider.require_live_integrity()
    if getattr(provider, "candidate_count", None) != RECOVERY_BEHAVIOR_COUNT:
        raise ValueError("recovery behavior provider must expose exact K9 count")
    try:
        raw_steps = np.asarray(provider.behavior_steps)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "recovery behavior provider must expose exact K9 steps") from exc
    if raw_steps.dtype.kind not in "iu" or raw_steps.shape != (
            RECOVERY_BEHAVIOR_COUNT,) or tuple(
                int(value) for value in raw_steps) != _K9_STEPS:
        raise ValueError("recovery behavior provider steps do not match locked K9")
    manifest_method = getattr(provider, "manifest_protocol", None)
    if not callable(manifest_method):
        raise TypeError("recovery behavior provider must expose manifest_protocol")
    manifest = manifest_method()
    if not isinstance(manifest, Mapping):
        raise TypeError("recovery behavior manifest must be a mapping")
    expected = RecoveryBehaviorConfig().manifest_protocol()
    if dict(manifest) != expected:
        changed = sorted(
            name for name in set(manifest) | set(expected)
            if manifest.get(name) != expected.get(name))
        raise ValueError(
            "recovery behavior manifest does not exactly match locked K9; "
            f"changed fields={changed}")
    full_manifest_method = getattr(provider, "manifest", None)
    fingerprint_method = getattr(provider, "fingerprint", None)
    preview_method = getattr(provider, "preview_candidates", None)
    if not callable(full_manifest_method) or not callable(
            fingerprint_method) or not callable(preview_method):
        raise TypeError(
            "recovery behavior provider must expose preview and full "
            "manifest/fingerprint")
    full_manifest = full_manifest_method()
    binding = {
        "manifest": full_manifest,
        "fingerprint_sha256": fingerprint_method(),
    }
    fingerprint = validate_recovery_program_binding(binding)
    return fingerprint, copy.deepcopy(dict(full_manifest))


def _validate_projection_provider(
    provider: ActionProjectionProvider,
    *,
    expected_manifest: Mapping[str, Any],
) -> None:
    if type(provider) is not BoundActionProjectionProvider:
        raise TypeError(
            "action_projection_provider must be BoundActionProjectionProvider")
    provider.require_live_integrity()
    manifest_method = getattr(provider, "manifest", None)
    fingerprint_method = getattr(provider, "fingerprint", None)
    preview_method = getattr(provider, "preview_many", None)
    if not callable(provider) or not callable(manifest_method) or not callable(
            fingerprint_method) or not callable(preview_method):
        raise TypeError(
            "action projection provider must expose call/preview and full "
            "manifest/fingerprint")
    manifest = manifest_method()
    if not isinstance(manifest, Mapping):
        raise TypeError("action projection manifest must be a mapping")
    checked_manifest = copy.deepcopy(dict(manifest))
    if checked_manifest != dict(expected_manifest):
        raise ValueError(
            "action projection manifest differs from recovery library")
    fingerprint = _sha256(
        fingerprint_method(), name="action projection fingerprint")
    if fingerprint != _canonical_sha256(
            checked_manifest, name="action projection manifest"):
        raise ValueError("action projection manifest fingerprint mismatch")


def _proof_action_matrix(
    value: Any,
    *,
    name: str,
    normalized: bool,
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(
            np.float32) or value.shape != (
                RECOVERY_BEHAVIOR_COUNT, _ACTION_DIM):
        raise ValueError(f"decision proof {name} must be float32 shape [9,12]")
    if value.flags.writeable:
        raise ValueError(f"decision proof {name} must be immutable")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"decision proof {name} must be finite")
    if normalized and (
            np.any(value < -1.0 - 1e-6)
            or np.any(value > 1.0 + 1e-6)):
        raise ValueError(f"decision proof {name} must lie in [-1,1]")
    return value.copy()


def _projection_tuple(
    value: Any,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        requested = _vector(
            value.action_requested,
            name=f"{name} requested",
            width=_ACTION_DIM,
            normalized=True,
        )
        executed = _vector(
            value.action_executed,
            name=f"{name} executed",
            width=_ACTION_DIM,
            normalized=True,
        )
        q_target = _vector(
            value.action_q_target,
            name=f"{name} q-target",
            width=_ACTION_DIM,
            normalized=False,
        )
    except AttributeError as exc:
        raise TypeError(
            f"{name} must expose requested/executed/q-target") from exc
    return requested, executed, q_target


def _validate_idle_decision_proof(
    proof: Any,
    *,
    history: np.ndarray,
    expected_library_fingerprint: str,
    selector_bundle: RecoverySelectorBundle,
    expected_artifact_manifest_sha256: str,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if type(proof) is not RecoveryQSafeInference:
        raise TypeError(
            "idle step requires an exact RecoveryQSafeInference decision proof")
    proof.require_live_integrity(selector_bundle)
    if proof.history_sha256 != _history_sha256(history):
        raise ValueError("decision proof belongs to another observation history")
    if proof.recovery_library_fingerprint_sha256 != (
            expected_library_fingerprint):
        raise ValueError("decision proof recovery library fingerprint mismatch")
    if proof.selector_bundle_sha256 != selector_bundle.bundle_sha256:
        raise ValueError("decision proof selector bundle fingerprint mismatch")
    if proof.artifact_manifest_sha256 != expected_artifact_manifest_sha256:
        raise ValueError("decision proof Q_safe artifact identity mismatch")
    if proof.feature_contract_sha256 != make_recovery_program_feature_manifest(
            expected_library_fingerprint)["feature_contract_sha256"]:
        raise ValueError("decision proof feature contract fingerprint mismatch")

    requested = _proof_action_matrix(
        proof.raw_candidate_requested,
        name="raw_candidate_requested",
        normalized=True,
    )
    executed = _proof_action_matrix(
        proof.raw_candidate_executed,
        name="raw_candidate_executed",
        normalized=True,
    )
    q_target = _proof_action_matrix(
        proof.raw_candidate_q_target,
        name="raw_candidate_q_target",
        normalized=False,
    )
    mask = proof.candidate_mask
    if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(
            np.bool_) or mask.shape != (RECOVERY_BEHAVIOR_COUNT,) or not np.all(
                mask) or mask.flags.writeable:
        raise ValueError(
            "decision proof candidate_mask must be immutable all-true K9")
    feature_manifest = make_recovery_program_feature_manifest(
        expected_library_fingerprint)
    features = build_recovery_program_features(
        candidate_requested=requested,
        candidate_executed=executed,
        candidate_q_target=q_target,
        candidate_names=np.asarray(RECOVERY_PROGRAM_NAMES, dtype=str),
        candidate_behavior_steps=np.asarray(
            RECOVERY_PROGRAM_BEHAVIOR_STEPS, dtype=np.int64),
        candidate_mask=mask.copy(),
        nominal_index=0,
        feature_manifest=feature_manifest,
        feature_manifest_fingerprint_sha256=feature_manifest[
            "feature_contract_sha256"],
        recovery_library_fingerprint_sha256=expected_library_fingerprint,
    )
    for actual, expected, name in (
        (proof.nominal_action_features, features.nominal_descriptor,
         "nominal_action_features"),
        (proof.candidate_action_features, features.candidate_descriptor,
         "candidate_action_features"),
    ):
        if not isinstance(actual, np.ndarray) or actual.dtype != np.dtype(
                np.float32) or actual.flags.writeable or not np.array_equal(
                actual, expected):
            raise ValueError(
                f"decision proof {name} does not encode the raw K9 preview")

    selected = _integer(proof.selected_index, name="proof selected_index")
    if selected >= RECOVERY_BEHAVIOR_COUNT:
        raise ValueError("decision proof selected index is outside locked K9")
    if not isinstance(proof.intervened, (bool, np.bool_)) or bool(
            proof.intervened) != (selected != 0):
        raise ValueError("decision proof intervention flag is inconsistent")
    if not bool(mask[selected]):
        raise ValueError("decision proof selected an unavailable K9 candidate")
    return selected, requested, executed, q_target


class PersistentRecoveryController:
    """One-shot-per-episode persistent K9 option controller."""

    def __init__(
        self,
        recovery_behavior_provider: RecoveryBehaviorActionProvider,
        action_projection_provider: ActionProjectionProvider,
        nominal_actor_shadow: CounterBasedActorShadow,
        *,
        qsafe_artifact: LoadedQSafeArtifact,
        selector_bundle: RecoverySelectorBundle,
    ) -> None:
        fingerprint, library_manifest = _validate_k9_provider(
            recovery_behavior_provider)
        if not isinstance(qsafe_artifact, LoadedQSafeArtifact):
            raise TypeError("qsafe_artifact must be LoadedQSafeArtifact")
        qsafe_artifact.require_live_integrity()
        if not isinstance(selector_bundle, RecoverySelectorBundle):
            raise TypeError("selector_bundle must be RecoverySelectorBundle")
        checked_selector = selector_bundle.validated()
        provenance = qsafe_artifact.manifest.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("Q_safe artifact recovery provenance is missing")
        expected_library = validate_recovery_program_binding(
            provenance.get("recovery_program"))
        serialized_selector = provenance.get("recovery_selector_bundle")
        if not isinstance(serialized_selector, Mapping) or dict(
                serialized_selector) != checked_selector.to_dict() or (
                provenance.get("recovery_selector_bundle_sha256")
                != checked_selector.bundle_sha256):
            raise ValueError(
                "controller selector bundle differs from Q_safe artifact")
        if fingerprint != expected_library:
            raise ValueError(
                "runtime recovery library fingerprint differs from artifact")
        projection_manifest = library_manifest.get("action_projection")
        if not isinstance(projection_manifest, Mapping):
            raise ValueError(
                "recovery library lacks a full action projection manifest")
        _validate_projection_provider(
            action_projection_provider,
            expected_manifest=projection_manifest,
        )
        if type(nominal_actor_shadow) is not CounterBasedActorShadow:
            raise TypeError(
                "nominal_actor_shadow must be CounterBasedActorShadow")
        self._behavior_provider = recovery_behavior_provider
        self._recovery_library_manifest = library_manifest
        self._recovery_library_fingerprint_sha256 = fingerprint
        self._selector_bundle = checked_selector
        self._selector_bundle_sha256 = checked_selector.bundle_sha256
        self._qsafe_artifact = qsafe_artifact
        self._artifact_manifest_sha256 = (
            qsafe_artifact.claim_identity_sha256)
        self._projection_provider = action_projection_provider
        self._nominal_actor_shadow = nominal_actor_shadow
        self._state = RecoveryRuntimeState.IDLE
        self._active_behavior_index: int | None = None
        self._active_behavior_step = 0
        self._intervention_used = False
        self._terminal = False
        self._awaiting_outcome_step: int | None = None
        self._last_absolute_step: int | None = None
        self._episode_index = 0
        self._has_emitted_action = False

    def _require_live_runtime_bindings(self) -> None:
        fingerprint, manifest = _validate_k9_provider(
            self._behavior_provider)
        if fingerprint != self._recovery_library_fingerprint_sha256 or (
                manifest != self._recovery_library_manifest):
            raise RuntimeError("runtime recovery behavior binding changed")
        projection_manifest = self._recovery_library_manifest.get(
            "action_projection")
        if not isinstance(projection_manifest, Mapping):
            raise RuntimeError("runtime recovery projection binding disappeared")
        _validate_projection_provider(
            self._projection_provider,
            expected_manifest=projection_manifest,
        )

    @property
    def state(self) -> RecoveryRuntimeState:
        return self._state

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def intervention_used(self) -> bool:
        return self._intervention_used

    @property
    def episode_index(self) -> int:
        return self._episode_index

    @property
    def active_behavior_index(self) -> int | None:
        return self._active_behavior_index

    def reset(self) -> None:
        """Re-arm only after an acknowledged episode terminal.

        The initial, unused idle controller accepts an idempotent reset for
        integration convenience.  Once an action has been emitted, reset is
        forbidden until ``observe_outcome`` acknowledges a terminal step.
        """
        if self._awaiting_outcome_step is not None:
            raise RuntimeError("cannot reset before acknowledging the last outcome")
        if self._has_emitted_action and not self._terminal:
            raise RuntimeError("reset requires an acknowledged episode terminal")
        if self._has_emitted_action:
            self._episode_index += 1
        self._state = RecoveryRuntimeState.IDLE
        self._active_behavior_index = None
        self._active_behavior_step = 0
        self._intervention_used = False
        self._terminal = False
        self._has_emitted_action = False

    def _planned_owner(
        self,
        decision_index: int | None,
    ) -> tuple[
        RecoveryActionOwner,
        int,
        int,
        RecoveryRuntimeState,
        int | None,
        int,
        bool,
    ]:
        """Return action ownership and the post-success state without mutation."""
        if self._state is RecoveryRuntimeState.IDLE:
            if decision_index in (None, 0):
                return (
                    RecoveryActionOwner.NOMINAL_ACTOR,
                    0,
                    0,
                    RecoveryRuntimeState.IDLE,
                    None,
                    0,
                    self._intervention_used,
                )
            index = int(decision_index)
            duration = _K9_STEPS[index]
            if duration <= 0:
                raise ValueError("selected recovery behavior has no active duration")
            next_step = 1
            next_state = (
                RecoveryRuntimeState.SPENT_UNTIL_RESET
                if next_step == duration
                else RecoveryRuntimeState.OPTION
            )
            return (
                RecoveryActionOwner.RECOVERY_BEHAVIOR,
                index,
                0,
                next_state,
                None if next_state is RecoveryRuntimeState.SPENT_UNTIL_RESET
                else index,
                0 if next_state is RecoveryRuntimeState.SPENT_UNTIL_RESET
                else next_step,
                True,
            )

        if decision_index is not None:
            raise RuntimeError(
                "reselection is forbidden outside idle; omit selection until reset")

        if self._state is RecoveryRuntimeState.OPTION:
            if self._active_behavior_index is None:
                raise AssertionError("option state has no active behavior")
            index = self._active_behavior_index
            step = self._active_behavior_step
            duration = _K9_STEPS[index]
            if not 0 <= step < duration:
                raise AssertionError("active option step is outside locked duration")
            next_step = step + 1
            next_state = (
                RecoveryRuntimeState.SPENT_UNTIL_RESET
                if next_step == duration
                else RecoveryRuntimeState.OPTION
            )
            return (
                RecoveryActionOwner.RECOVERY_BEHAVIOR,
                index,
                step,
                next_state,
                None if next_state is RecoveryRuntimeState.SPENT_UNTIL_RESET
                else index,
                0 if next_state is RecoveryRuntimeState.SPENT_UNTIL_RESET
                else next_step,
                True,
            )

        return (
            RecoveryActionOwner.NOMINAL_ACTOR,
            0,
            0,
            RecoveryRuntimeState.SPENT_UNTIL_RESET,
            None,
            0,
            True,
        )

    def step(
        self,
        *,
        absolute_step: int,
        current_observation: np.ndarray,
        observation_history: np.ndarray,
        nominal_proposal: NominalActorProposal,
        decision_proof: RecoveryQSafeInference | None = None,
    ) -> RecoveryRuntimeStep:
        """Emit one applied action and require a subsequent outcome acknowledgment.

        Every idle action requires the complete inference proof that selected
        it.  Proofs are forbidden while an option owns the actuator and after
        the one permitted intervention has been spent.  A bare behavior index
        is deliberately not part of this API.
        """
        if self._terminal:
            raise RuntimeError("terminal controller must be reset before stepping")
        self._require_live_runtime_bindings()
        self._qsafe_artifact.require_live_integrity()
        if self._qsafe_artifact.claim_identity_sha256 != (
                self._artifact_manifest_sha256):
            raise RuntimeError("controller Q_safe artifact identity changed")
        if self._awaiting_outcome_step is not None:
            raise RuntimeError("last emitted action has no acknowledged outcome")
        step = _integer(absolute_step, name="absolute_step")
        if self._last_absolute_step is not None and step != (
                self._last_absolute_step + 1):
            raise RuntimeError(
                "controller steps must use consecutive absolute counters: "
                f"expected {self._last_absolute_step + 1}, got {step}"
            )
        if type(nominal_proposal) is not NominalActorProposal:
            raise TypeError(
                "nominal_proposal must come from CounterBasedActorShadow")
        self._nominal_actor_shadow.validate_issued(nominal_proposal)
        observation = _observation(current_observation)
        history = _history(observation_history)
        if not np.array_equal(history[-1], observation):
            raise ValueError(
                "current_observation must exactly equal newest history frame")
        if nominal_proposal.absolute_step != step:
            raise ValueError("nominal actor proposal belongs to another absolute step")
        if nominal_proposal.observation_sha256 != _observation_sha256(observation):
            raise ValueError("nominal actor proposal belongs to another observation")
        proof_requested: np.ndarray | None = None
        proof_executed: np.ndarray | None = None
        proof_q_target: np.ndarray | None = None
        if self._state is RecoveryRuntimeState.IDLE:
            (
                selection,
                proof_requested,
                proof_executed,
                proof_q_target,
            ) = _validate_idle_decision_proof(
                decision_proof,
                history=history,
                expected_library_fingerprint=(
                    self._recovery_library_fingerprint_sha256),
                selector_bundle=self._selector_bundle,
                expected_artifact_manifest_sha256=(
                    self._artifact_manifest_sha256),
            )
            raw_preview = self._behavior_provider.preview_candidates(
                history.copy(), nominal_proposal.action.copy())
            if not isinstance(raw_preview, np.ndarray) or raw_preview.dtype != (
                    np.dtype(np.float32)) or raw_preview.shape != (
                    RECOVERY_BEHAVIOR_COUNT, _ACTION_DIM) or not np.all(
                    np.isfinite(raw_preview)):
                raise ValueError(
                    "recovery behavior provider returned malformed K9 preview")
            if not np.array_equal(raw_preview, proof_requested):
                raise ValueError(
                    "decision proof requested K9 preview differs from library")
            projected_preview = self._projection_provider.preview_many(
                raw_preview.copy(), observation.copy())
            if not isinstance(projected_preview, tuple) or len(
                    projected_preview) != RECOVERY_BEHAVIOR_COUNT:
                raise ValueError(
                    "projection provider must return an exact K9 preview tuple")
            preview_requested: list[np.ndarray] = []
            preview_executed: list[np.ndarray] = []
            preview_q_target: list[np.ndarray] = []
            for index, projected in enumerate(projected_preview):
                values = _projection_tuple(
                    projected, name=f"K9 projection preview[{index}]")
                preview_requested.append(values[0])
                preview_executed.append(values[1])
                preview_q_target.append(values[2])
            for actual, expected, name in (
                (np.stack(preview_requested), proof_requested, "requested"),
                (np.stack(preview_executed), proof_executed, "executed"),
                (np.stack(preview_q_target), proof_q_target, "q-target"),
            ):
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        f"decision proof {name} K9 preview differs from "
                        "projection provider")
        else:
            if decision_proof is not None:
                raise RuntimeError(
                    "new decision proofs are forbidden outside idle")
            selection = None
        state_before = self._state
        (
            owner,
            behavior_index,
            behavior_step,
            next_state,
            next_active_index,
            next_active_step,
            next_intervention_used,
        ) = self._planned_owner(selection)

        nominal_action = nominal_proposal.action.copy()
        if owner is RecoveryActionOwner.RECOVERY_BEHAVIOR:
            requested = _vector(
                self._behavior_provider(
                    behavior_index,
                    history.copy(),
                    behavior_step,
                    nominal_action.copy(),
                ),
                name="recovery behavior requested action",
                width=_ACTION_DIM,
                normalized=True,
            )
        else:
            requested = nominal_action

        projection = self._projection_provider(
            requested.copy(), observation.copy())
        (
            projected_requested,
            projected_executed,
            projected_q_target,
        ) = _projection_tuple(projection, name="applied action projection")
        if not np.array_equal(projected_requested, requested):
            raise ValueError(
                "projection changed action_requested instead of only applying it")
        if proof_requested is not None:
            for actual, expected, name in (
                (projected_requested, proof_requested[behavior_index],
                 "requested"),
                (projected_executed, proof_executed[behavior_index],
                 "executed"),
                (projected_q_target, proof_q_target[behavior_index],
                 "q-target"),
            ):
                if not np.array_equal(actual, expected):
                    raise ValueError(
                        "applied candidate first-step " + name
                        + " differs from decision proof")

        nominal_rejected = owner is RecoveryActionOwner.RECOVERY_BEHAVIOR
        replay = RecoveryReplayAction(
            absolute_step=step,
            action=projected_requested,
            action_nominal=nominal_action,
            action_executed=projected_executed,
            action_q_target=projected_q_target,
            owner=owner,
            behavior_index=behavior_index,
            behavior_name=_K9_NAMES[behavior_index],
            behavior_step=behavior_step,
            nominal_is_log_only=nominal_rejected,
            nominal_proposal_sha256=nominal_proposal.proposal_sha256,
            _issue_token=_RUNTIME_ISSUE_TOKEN,
        )
        replay = replace(
            replay, _live_sha256=_replay_live_sha256(replay))
        result = RecoveryRuntimeStep(
            absolute_step=step,
            episode_index=self._episode_index,
            state_before=state_before,
            state_after_action=next_state,
            owner=owner,
            behavior_index=behavior_index,
            behavior_name=_K9_NAMES[behavior_index],
            behavior_step=behavior_step,
            behavior_duration=_K9_STEPS[behavior_index],
            actual_requested=projected_requested,
            actual_executed=projected_executed,
            actual_q_target=projected_q_target,
            rejected_nominal=nominal_action,
            nominal_rejected=nominal_rejected,
            replay=replay,
            nominal_proposal=nominal_proposal,
            _issue_token=_RUNTIME_ISSUE_TOKEN,
        )
        result = replace(
            result, _live_sha256=_runtime_step_live_sha256(result))
        result.require_live_integrity()

        # Commit the state only after provider, projection, and replay
        # validation all succeed.
        self._require_live_runtime_bindings()
        self._nominal_actor_shadow.commit_issued(nominal_proposal)
        self._state = next_state
        self._active_behavior_index = next_active_index
        self._active_behavior_step = next_active_step
        self._intervention_used = next_intervention_used
        self._awaiting_outcome_step = step
        self._last_absolute_step = step
        self._has_emitted_action = True
        result.require_live_integrity()
        return result

    def observe_outcome(
        self,
        *,
        absolute_step: int,
        fell: bool,
        terminated: bool,
    ) -> RecoveryOutcome:
        """Acknowledge the environment result for the last emitted action."""
        step = _integer(absolute_step, name="absolute_step")
        fell_value = _boolean(fell, name="fell")
        terminated_value = _boolean(terminated, name="terminated")
        if self._awaiting_outcome_step is None:
            raise RuntimeError("no emitted action is awaiting an outcome")
        if step != self._awaiting_outcome_step:
            raise ValueError("outcome absolute step does not match emitted action")
        if fell_value and not terminated_value:
            raise ValueError("fall must be terminal")

        self._awaiting_outcome_step = None
        if terminated_value:
            self._terminal = True
            self._state = RecoveryRuntimeState.TERMINAL
            self._active_behavior_index = None
            self._active_behavior_step = 0
        return RecoveryOutcome(
            absolute_step=step,
            fell=fell_value,
            terminated=terminated_value,
            state_after_outcome=self._state,
        )


__all__ = [
    "ActionProjectionProvider",
    "ActorCounterDomain",
    "ActorCounterKey",
    "BoundActionProjectionProvider",
    "CounterBasedActorShadow",
    "DeterministicExternalNoiseActorProvider",
    "NominalActorProposal",
    "PersistentRecoveryController",
    "RecoveryActionOwner",
    "RecoveryBehaviorActionProvider",
    "RecoveryOutcome",
    "RecoveryReplayAction",
    "RecoveryRuntimeState",
    "RecoveryRuntimeStep",
    "StageCActorCounterDomain",
    "StageCActorCounterKey",
    "StageDActorCounterDomain",
    "StageDActorCounterKey",
]
