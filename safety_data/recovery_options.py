"""Locked recovery-option expansion for Objective-1 mechanism triage.

The v2 protocol deliberately reuses the audited K=16 evidence candidate
geometry.  It selects seven fixed non-nominal first actions and expands each
into four recovery-option durations.  Duration changes the continuation
semantics, not the first action, so all four options for one residual template
have identical previewed requested/executed/q-target arrays.

Only the eight distinct first actions (nominal plus seven templates) are
subject to physical q-target deduplication.  Once those eight actions pass,
all K=29 options are valid because option duration is part of the intervention.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from safety_data.candidates import (
    CANDIDATE_PROTOCOL_VERSION as BASE_CANDIDATE_PROTOCOL_VERSION,
    CandidateSet,
    EvidenceCandidateConfig,
    InsufficientCandidateSupportError,
    build_evidence_candidates,
)


RECOVERY_OPTION_PROTOCOL_VERSION = "qsafe.recovery_option_candidates.v2"
RECOVERY_OPTION_COUNT = 29
RECOVERY_OPTION_BASE_INDICES = (1, 2, 4, 6, 7, 10, 11)
RECOVERY_OPTION_TEMPLATE_IDS = (
    "deterministic_mean",
    "previous_requested",
    "contraction_0.50",
    "local_actor_sample_0",
    "local_actor_sample_1",
    "symmetric_direction_0_plus",
    "symmetric_direction_0_minus",
)
RECOVERY_OPTION_DURATIONS = (1, 2, 3, 4)
RECOVERY_OPTION_KINDS = (
    "nominal",
    *(
        f"{template_id}_L{duration}"
        for template_id in RECOVERY_OPTION_TEMPLATE_IDS
        for duration in RECOVERY_OPTION_DURATIONS
    ),
)
RECOVERY_OPTION_STEPS = (
    1,
    *(
        duration
        for _ in RECOVERY_OPTION_TEMPLATE_IDS
        for duration in RECOVERY_OPTION_DURATIONS
    ),
)

_LOCKED_ACTOR_SAMPLE_MAX_DELTA_RMS = 0.50
_LOCKED_PERTURBATION_RADIUS_RMS = 0.25
_LOCKED_Q_TARGET_DEDUP_ATOL = 1e-6
_LOCKED_MIN_UNIQUE_BASE_CANDIDATES = 8
_SELECTED_BASE_INDICES = (0, *RECOVERY_OPTION_BASE_INDICES)

if len(RECOVERY_OPTION_BASE_INDICES) != len(RECOVERY_OPTION_TEMPLATE_IDS):
    raise AssertionError("recovery-option base indices and templates disagree")
if len(RECOVERY_OPTION_KINDS) != RECOVERY_OPTION_COUNT or len(
        RECOVERY_OPTION_STEPS) != RECOVERY_OPTION_COUNT:
    raise AssertionError("recovery-option K=29 protocol constants disagree")


@dataclass(frozen=True)
class RecoveryOptionCandidateConfig:
    """Fixed candidate configuration preregistered for v2 triage.

    The radius attributes intentionally mirror :class:`EvidenceCandidateConfig`
    so the native collector can record its existing profile fields unchanged.
    Unlike the reusable K=16 configuration, these values are not tunable: a
    deviation would no longer match the preregistered triage protocol.
    """

    actor_sample_max_delta_rms: float = _LOCKED_ACTOR_SAMPLE_MAX_DELTA_RMS
    perturbation_radius_rms: float = _LOCKED_PERTURBATION_RADIUS_RMS
    q_target_dedup_atol: float = _LOCKED_Q_TARGET_DEDUP_ATOL
    min_unique_base_candidates: int = _LOCKED_MIN_UNIQUE_BASE_CANDIDATES

    def __post_init__(self) -> None:
        # Reuse the base protocol's type, finiteness, and range validation.
        try:
            base = EvidenceCandidateConfig(
                actor_sample_max_delta_rms=self.actor_sample_max_delta_rms,
                perturbation_radius_rms=self.perturbation_radius_rms,
                q_target_dedup_atol=self.q_target_dedup_atol,
                min_unique_candidates=self.min_unique_base_candidates,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "recovery-option base candidate parameters are invalid"
            ) from exc

        locked = {
            "actor_sample_max_delta_rms": _LOCKED_ACTOR_SAMPLE_MAX_DELTA_RMS,
            "perturbation_radius_rms": _LOCKED_PERTURBATION_RADIUS_RMS,
            "q_target_dedup_atol": _LOCKED_Q_TARGET_DEDUP_ATOL,
            "min_unique_base_candidates": _LOCKED_MIN_UNIQUE_BASE_CANDIDATES,
        }
        actual = {
            "actor_sample_max_delta_rms": base.actor_sample_max_delta_rms,
            "perturbation_radius_rms": base.perturbation_radius_rms,
            "q_target_dedup_atol": base.q_target_dedup_atol,
            "min_unique_base_candidates": base.min_unique_candidates,
        }
        if actual != locked:
            raise ValueError(
                "recovery-option candidate parameters must exactly match the "
                "preregistered v2 triage protocol"
            )
        for name, value in actual.items():
            object.__setattr__(self, name, value)

    def base_config(self) -> EvidenceCandidateConfig:
        """Return the exact K=16 configuration used before expansion."""
        return EvidenceCandidateConfig(
            actor_sample_max_delta_rms=self.actor_sample_max_delta_rms,
            perturbation_radius_rms=self.perturbation_radius_rms,
            q_target_dedup_atol=self.q_target_dedup_atol,
            min_unique_candidates=self.min_unique_base_candidates,
        )

    def manifest_protocol(self) -> dict[str, Any]:
        """Return the exact candidate section of the v2 triage protocol."""
        return {
            "protocol_version": RECOVERY_OPTION_PROTOCOL_VERSION,
            "count": RECOVERY_OPTION_COUNT,
            "nominal_index": 0,
            "ordered_kinds": list(RECOVERY_OPTION_KINDS),
            "base_generator": BASE_CANDIDATE_PROTOCOL_VERSION,
            "base_generator_parameters": {
                "actor_sample_max_delta_rms": float(
                    self.actor_sample_max_delta_rms),
                "perturbation_radius_rms": float(
                    self.perturbation_radius_rms),
                "q_target_dedup_atol": float(self.q_target_dedup_atol),
                "min_unique_base_candidates": int(
                    self.min_unique_base_candidates),
            },
            "residual_templates": [
                {
                    "template_id": template_id,
                    "base_candidate_index": base_index,
                }
                for template_id, base_index in zip(
                    RECOVERY_OPTION_TEMPLATE_IDS,
                    RECOVERY_OPTION_BASE_INDICES,
                    strict=True,
                )
            ],
            "option_steps": list(RECOVERY_OPTION_DURATIONS),
            "ordering": "nominal_then_template_major_duration_minor",
            "option_count_formula": "1_plus_7_templates_times_4_durations",
            "duplicate_rule": (
                "duration_distinguishes_options_with_the_same_first_q_target"
            ),
            "option_steps_array": "candidate_option_steps",
            "option_semantics": "linear_decay_actor_residual_v1",
        }

    def build_candidates(self, **raw_args: Any) -> RecoveryOptionCandidateSet:
        """Build K=29 options from the raw arguments accepted by K=16."""
        if "config" in raw_args:
            raise TypeError(
                "RecoveryOptionCandidateConfig.build_candidates owns config")
        return build_recovery_option_candidates(config=self, **raw_args)


@dataclass(frozen=True)
class RecoveryOptionCandidateSet:
    """One immutable, fixed-width K=29 recovery-option candidate set."""

    requested: np.ndarray
    executed: np.ndarray
    q_target: np.ndarray
    kind: np.ndarray
    mask: np.ndarray
    option_steps: np.ndarray
    candidate_seed: int
    manifest_protocol: dict[str, Any]

    def __post_init__(self) -> None:
        arrays = {
            "requested": np.asarray(self.requested, dtype=np.float32).copy(),
            "executed": np.asarray(self.executed, dtype=np.float32).copy(),
            "q_target": np.asarray(self.q_target, dtype=np.float32).copy(),
        }
        for name, value in arrays.items():
            if value.shape != (RECOVERY_OPTION_COUNT, 12):
                raise ValueError(
                    f"{name} must have shape {(RECOVERY_OPTION_COUNT, 12)}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            if name in ("requested", "executed") and (
                    np.any(value < -1.0 - 1e-6)
                    or np.any(value > 1.0 + 1e-6)):
                raise ValueError(f"{name} must lie in normalized [-1, 1]")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        kind = np.asarray(self.kind, dtype=str).copy()
        if kind.shape != (RECOVERY_OPTION_COUNT,) or tuple(
                kind.tolist()) != RECOVERY_OPTION_KINDS:
            raise ValueError("kind does not match the locked recovery-option order")
        if len(set(kind.tolist())) != RECOVERY_OPTION_COUNT:
            raise ValueError("recovery-option kinds must be unique")

        mask = np.asarray(self.mask, dtype=bool).copy()
        if mask.shape != (RECOVERY_OPTION_COUNT,):
            raise ValueError(f"mask must have shape {(RECOVERY_OPTION_COUNT,)}")
        if not np.all(mask):
            raise ValueError("all K=29 recovery options must remain valid")

        option_steps = np.asarray(self.option_steps)
        if option_steps.dtype.kind not in "iu" or option_steps.shape != (
                RECOVERY_OPTION_COUNT,):
            raise ValueError("option_steps must be an integer [29] array")
        option_steps = option_steps.astype(np.int64, copy=True)
        if tuple(option_steps.tolist()) != RECOVERY_OPTION_STEPS:
            raise ValueError(
                "option_steps does not match template-major duration-minor order")

        seed = _candidate_seed(self.candidate_seed)
        manifest = copy.deepcopy(dict(self.manifest_protocol))
        if manifest != RecoveryOptionCandidateConfig().manifest_protocol():
            raise ValueError(
                "manifest_protocol does not match the locked recovery-option "
                "candidate protocol")

        kind.setflags(write=False)
        mask.setflags(write=False)
        option_steps.setflags(write=False)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "option_steps", option_steps)
        object.__setattr__(self, "candidate_seed", seed)
        object.__setattr__(self, "manifest_protocol", manifest)

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.mask))


def _candidate_seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise ValueError("candidate_seed must be an explicit integer")
    seed = int(value)
    if not 0 <= seed < 2**64:
        raise ValueError("candidate_seed must be in uint64 range")
    return seed


def _selected_base_valid_count(
    candidates: CandidateSet,
    *,
    q_target_dedup_atol: float,
) -> int:
    """Count valid, unique q targets among the eight selected base actions."""
    retained: list[np.ndarray] = []
    valid_count = 0
    for index in _SELECTED_BASE_INDICES:
        if not bool(candidates.mask[index]):
            continue
        target = np.asarray(candidates.q_target[index], dtype=np.float32)
        if any(
            np.allclose(target, previous, rtol=0.0, atol=q_target_dedup_atol)
            for previous in retained
        ):
            continue
        retained.append(target)
        valid_count += 1
    return valid_count


def build_recovery_option_candidates(
    *,
    nominal: np.ndarray,
    deterministic_mean: np.ndarray,
    previous_requested: np.ndarray,
    actor_samples: np.ndarray,
    action_applier: Any,
    current_qpos: np.ndarray,
    candidate_seed: int,
    config: RecoveryOptionCandidateConfig | None = None,
) -> RecoveryOptionCandidateSet:
    """Build the preregistered K=29 recovery-option candidate set.

    Physical support is checked entirely from action previews before any
    branch outcome can be evaluated.  The selected eight K=16 first actions
    must all be valid and mutually unique.  Duration replicas then deliberately
    reuse those previews because only their continuation semantics differ.
    """
    protocol_config = config or RecoveryOptionCandidateConfig()
    if not isinstance(protocol_config, RecoveryOptionCandidateConfig):
        raise TypeError("config must be RecoveryOptionCandidateConfig")

    base = build_evidence_candidates(
        nominal=nominal,
        deterministic_mean=deterministic_mean,
        previous_requested=previous_requested,
        actor_samples=actor_samples,
        action_applier=action_applier,
        current_qpos=current_qpos,
        candidate_seed=candidate_seed,
        config=protocol_config.base_config(),
    )
    selected_valid_count = _selected_base_valid_count(
        base, q_target_dedup_atol=protocol_config.q_target_dedup_atol)
    if selected_valid_count != _LOCKED_MIN_UNIQUE_BASE_CANDIDATES:
        raise InsufficientCandidateSupportError(
            selected_valid_count, _LOCKED_MIN_UNIQUE_BASE_CANDIDATES)

    expansion_indices = np.asarray([
        0,
        *(
            base_index
            for base_index in RECOVERY_OPTION_BASE_INDICES
            for _ in RECOVERY_OPTION_DURATIONS
        ),
    ], dtype=np.int64)
    if expansion_indices.shape != (RECOVERY_OPTION_COUNT,):
        raise AssertionError("internal recovery-option expansion is not K=29")

    return RecoveryOptionCandidateSet(
        requested=base.requested[expansion_indices],
        executed=base.executed[expansion_indices],
        q_target=base.q_target[expansion_indices],
        kind=np.asarray(RECOVERY_OPTION_KINDS),
        mask=np.ones(RECOVERY_OPTION_COUNT, dtype=bool),
        option_steps=np.asarray(RECOVERY_OPTION_STEPS, dtype=np.int64),
        candidate_seed=base.candidate_seed,
        manifest_protocol=protocol_config.manifest_protocol(),
    )


__all__ = [
    "RECOVERY_OPTION_BASE_INDICES",
    "RECOVERY_OPTION_COUNT",
    "RECOVERY_OPTION_DURATIONS",
    "RECOVERY_OPTION_KINDS",
    "RECOVERY_OPTION_PROTOCOL_VERSION",
    "RECOVERY_OPTION_STEPS",
    "RECOVERY_OPTION_TEMPLATE_IDS",
    "RecoveryOptionCandidateConfig",
    "RecoveryOptionCandidateSet",
    "build_recovery_option_candidates",
]
