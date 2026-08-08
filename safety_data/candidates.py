"""Locked candidate construction for Q-safe evidence branches.

The protocol intentionally has a fixed, auditable order.  Candidate zero is
always the action that the unshielded policy would have requested.  All
candidate actions are projected from one action-filter baseline through
``ActionApplier.preview_many`` before duplicates are identified in the
physically meaningful absolute joint-target space.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from robots.go2.joint_layout import validate_joint_vector
from runtime.inference.actions import ActionApplier


CANDIDATE_PROTOCOL_VERSION = "qsafe.evidence_candidates.v1"
CANDIDATE_COUNT = 16
ACTOR_SAMPLE_COUNT = 4
CONTRACTION_FACTORS = (0.25, 0.50, 0.75)
PERTURBATION_DIRECTION_COUNT = 3
CANDIDATE_KINDS = (
    "nominal",
    "deterministic_mean",
    "previous_requested",
    "contraction_0.25",
    "contraction_0.50",
    "contraction_0.75",
    "local_actor_sample_0",
    "local_actor_sample_1",
    "local_actor_sample_2",
    "local_actor_sample_3",
    "symmetric_direction_0_plus",
    "symmetric_direction_0_minus",
    "symmetric_direction_1_plus",
    "symmetric_direction_1_minus",
    "symmetric_direction_2_plus",
    "symmetric_direction_2_minus",
)


class CandidateProtocolError(ValueError):
    """The fixed evidence-candidate contract cannot be satisfied."""


@dataclass(frozen=True)
class EvidenceCandidateConfig:
    """Tunable radii within the otherwise locked K=16 protocol.

    Radii are root-mean-square distances in normalized policy-action space.
    Actor samples farther than ``actor_sample_max_delta_rms`` from the
    deterministic actor mean are radially contracted.  Symmetric perturbation
    pairs use ``perturbation_radius_rms`` unless action bounds require an equal
    reduction of both sides of the pair.
    """

    # Development boundary-PoC defaults locked after the native_poc_v1 scan.
    # They are not Phase-2 range-expansion parameters.
    actor_sample_max_delta_rms: float = 0.50
    perturbation_radius_rms: float = 0.25
    q_target_dedup_atol: float = 1e-6
    min_unique_candidates: int = 8

    def __post_init__(self) -> None:
        for name in (
            "actor_sample_max_delta_rms",
            "perturbation_radius_rms",
            "q_target_dedup_atol",
        ):
            raw_value = getattr(self, name)
            if isinstance(raw_value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a finite number")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, value)
        if self.actor_sample_max_delta_rms <= 0.0:
            raise ValueError("actor_sample_max_delta_rms must be positive")
        if self.perturbation_radius_rms <= 0.0:
            raise ValueError("perturbation_radius_rms must be positive")
        if self.q_target_dedup_atol < 0.0:
            raise ValueError("q_target_dedup_atol must be non-negative")
        if isinstance(self.min_unique_candidates, (bool, np.bool_)) or not isinstance(
            self.min_unique_candidates, (int, np.integer)
        ):
            raise ValueError("min_unique_candidates must be an integer")
        object.__setattr__(
            self, "min_unique_candidates", int(self.min_unique_candidates))
        if not 8 <= self.min_unique_candidates <= CANDIDATE_COUNT:
            raise ValueError(
                f"min_unique_candidates must be in [8, {CANDIDATE_COUNT}]")

    def manifest_protocol(self) -> dict[str, Any]:
        """Return the JSON-safe dataset-manifest candidate contract."""
        return {
            "version": CANDIDATE_PROTOCOL_VERSION,
            "count": CANDIDATE_COUNT,
            "nominal_index": 0,
            "ordered_kinds": list(CANDIDATE_KINDS),
            "deterministic_mean_index": 1,
            "previous_requested_index": 2,
            "contractions": {
                "indices": [3, 4, 5],
                "factors": list(CONTRACTION_FACTORS),
                "formula": (
                    "previous_requested + factor * "
                    "(nominal - previous_requested)"
                ),
            },
            "local_actor_samples": {
                "indices": [6, 7, 8, 9],
                "count": ACTOR_SAMPLE_COUNT,
                "center": "deterministic_mean",
                "max_delta_rms": float(self.actor_sample_max_delta_rms),
                "localization": "radial_contraction_then_action_bounds",
            },
            "symmetric_perturbations": {
                "indices": [[10, 11], [12, 13], [14, 15]],
                "direction_count": PERTURBATION_DIRECTION_COUNT,
                "center": "nominal",
                "radius_rms": float(self.perturbation_radius_rms),
                "pair_order": ["plus", "minus"],
                "direction_sampler": (
                    "numpy_pcg64_normal_modified_gram_schmidt_v1"
                ),
                "seed_argument": "candidate_seed",
            },
            "projection": {
                "method": "ActionApplier.preview_many",
                "filter_baseline": "same_for_every_candidate",
                "deduplicate_on": "absolute_action_q_target",
                "dedup_atol": float(self.q_target_dedup_atol),
                "duplicate_handling": "retain_slot_and_clear_mask",
            },
            "minimum_unique_candidates": int(self.min_unique_candidates),
        }


@dataclass(frozen=True)
class CandidateSet:
    """One projected, fixed-width evidence candidate set.

    Duplicate slots remain present so indices and kinds are stable, but their
    mask is false.  ``candidate_seed`` is explicit per group and must be
    persisted alongside group metadata; ``manifest_protocol`` is invariant
    across groups using the same configuration.
    """

    requested: np.ndarray
    executed: np.ndarray
    q_target: np.ndarray
    kind: np.ndarray
    mask: np.ndarray
    candidate_seed: int
    manifest_protocol: dict[str, Any]

    def __post_init__(self) -> None:
        arrays = {
            "requested": np.asarray(self.requested, dtype=np.float32).copy(),
            "executed": np.asarray(self.executed, dtype=np.float32).copy(),
            "q_target": np.asarray(self.q_target, dtype=np.float32).copy(),
        }
        for name, value in arrays.items():
            if value.shape != (CANDIDATE_COUNT, 12):
                raise ValueError(
                    f"{name} must have shape {(CANDIDATE_COUNT, 12)}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            if name in ("requested", "executed") and (
                    np.any(value < -1.0 - 1e-6)
                    or np.any(value > 1.0 + 1e-6)):
                raise ValueError(f"{name} must lie in normalized [-1, 1]")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

        kind = np.asarray(self.kind, dtype=str).copy()
        mask = np.asarray(self.mask, dtype=bool).copy()
        if kind.shape != (CANDIDATE_COUNT,):
            raise ValueError(f"kind must have shape {(CANDIDATE_COUNT,)}")
        if tuple(kind.tolist()) != CANDIDATE_KINDS:
            raise ValueError("kind does not match the locked candidate order")
        if mask.shape != (CANDIDATE_COUNT,):
            raise ValueError(f"mask must have shape {(CANDIDATE_COUNT,)}")
        if not mask[0]:
            raise ValueError("candidate zero must remain valid")
        candidate_seed = _candidate_seed(self.candidate_seed)
        manifest_protocol = copy.deepcopy(dict(self.manifest_protocol))
        if manifest_protocol.get("version") != CANDIDATE_PROTOCOL_VERSION or (
                manifest_protocol.get("count") != CANDIDATE_COUNT) or (
                manifest_protocol.get("nominal_index") != 0) or tuple(
                    manifest_protocol.get("ordered_kinds", ())) != CANDIDATE_KINDS:
            raise ValueError("manifest_protocol does not match candidate arrays")
        kind.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "candidate_seed", candidate_seed)
        object.__setattr__(self, "manifest_protocol", manifest_protocol)

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.mask))


def _normalized_action(name: str, value: np.ndarray) -> np.ndarray:
    action = validate_joint_vector(name, value).copy()
    if np.any(action < -1.0 - 1e-6) or np.any(action > 1.0 + 1e-6):
        raise CandidateProtocolError(f"{name} must lie in normalized [-1, 1]")
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _candidate_seed(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise CandidateProtocolError("candidate_seed must be an explicit integer")
    seed = int(value)
    if not 0 <= seed < 2**64:
        raise CandidateProtocolError("candidate_seed must be in uint64 range")
    return seed


def _local_actor_samples(
    actor_samples: np.ndarray,
    deterministic_mean: np.ndarray,
    radius_rms: float,
) -> np.ndarray:
    samples = np.asarray(actor_samples, dtype=np.float32)
    if samples.shape != (ACTOR_SAMPLE_COUNT, 12):
        raise CandidateProtocolError(
            "actor_samples must contain exactly four normalized 12-D actions")
    if not np.all(np.isfinite(samples)):
        raise CandidateProtocolError("actor_samples must contain only finite values")
    if np.any(samples < -1.0 - 1e-6) or np.any(samples > 1.0 + 1e-6):
        raise CandidateProtocolError("actor_samples must lie in normalized [-1, 1]")

    samples = np.clip(samples, -1.0, 1.0)
    delta = samples - deterministic_mean[None, :]
    rms = np.sqrt(np.mean(np.square(delta), axis=1))
    scale = np.minimum(1.0, radius_rms / np.maximum(rms, 1e-12))
    localized = deterministic_mean[None, :] + delta * scale[:, None]
    return np.clip(localized, -1.0, 1.0).astype(np.float32)


def _seeded_directions(candidate_seed: int) -> np.ndarray:
    """Generate three deterministic, mutually orthogonal unit-RMS directions."""
    generator = np.random.Generator(np.random.PCG64(candidate_seed))
    raw = generator.standard_normal((PERTURBATION_DIRECTION_COUNT, 12))
    # Avoid a LAPACK-dependent QR sign/basis convention in an evidence seed
    # contract.  With only three 12-D vectors, explicit modified Gram-Schmidt
    # is cheap and its ordered arithmetic is transparent.
    basis: list[np.ndarray] = []
    for raw_direction in raw:
        direction = raw_direction.copy()
        for previous in basis:
            direction -= np.dot(direction, previous) * previous
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-12:  # practically impossible
            raise CandidateProtocolError(
                "candidate_seed generated a degenerate perturbation basis")
        direction /= norm
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction *= -1.0
        basis.append(direction)
    return (np.stack(basis) * np.sqrt(12.0)).astype(np.float32)


def _symmetric_pair(
    center: np.ndarray,
    direction: np.ndarray,
    requested_radius_rms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a bound-safe pair with equal opposite displacement from center."""
    center64 = np.asarray(center, dtype=np.float64)
    direction64 = np.asarray(direction, dtype=np.float64)
    nonzero = np.abs(direction64) > 1e-12
    if not np.any(nonzero):
        raise CandidateProtocolError("perturbation direction must be nonzero")
    room = np.maximum(0.0, 1.0 - np.abs(center64))
    bound_radius = float(
        np.min(room[nonzero] / np.abs(direction64[nonzero])))
    radius = min(float(requested_radius_rms), bound_radius)
    # If an action-bound component is active, take the immediately smaller
    # representable radius.  This prevents a final float32 roundoff clip from
    # breaking the requested plus/minus symmetry.
    if radius > 0.0 and radius == bound_radius:
        radius = float(np.nextafter(radius, 0.0))
    delta = radius * direction64
    plus = center64 + delta
    minus = center64 - delta
    return plus.astype(np.float32), minus.astype(np.float32)


def _unique_q_target_mask(q_target: np.ndarray, atol: float) -> np.ndarray:
    targets = np.asarray(q_target, dtype=np.float32)
    if targets.ndim != 2 or not np.all(np.isfinite(targets)):
        raise CandidateProtocolError(
            "projected q_target must be a finite [candidate, joint] array")
    mask = np.zeros(len(targets), dtype=bool)
    retained: list[np.ndarray] = []
    for index, target in enumerate(targets):
        duplicate = any(
            np.allclose(target, previous, rtol=0.0, atol=atol)
            for previous in retained
        )
        if not duplicate:
            mask[index] = True
            retained.append(target)
    return mask


def build_evidence_candidates(
    *,
    nominal: np.ndarray,
    deterministic_mean: np.ndarray,
    previous_requested: np.ndarray,
    actor_samples: np.ndarray,
    action_applier: ActionApplier,
    current_qpos: np.ndarray,
    candidate_seed: int,
    config: EvidenceCandidateConfig | None = None,
) -> CandidateSet:
    """Build and project the locked K=16 Q-safe evidence candidates.

    ``candidate_seed`` is deliberately required instead of accepting ambient
    global RNG state.  The caller must generate the four actor samples in their
    recorded order using the same experiment's explicit actor RNG namespace.
    This function uses the seed for the three local perturbation directions.
    """
    protocol_config = config or EvidenceCandidateConfig()
    seed = _candidate_seed(candidate_seed)
    nominal_action = _normalized_action("nominal", nominal)
    mean_action = _normalized_action("deterministic_mean", deterministic_mean)
    previous_action = _normalized_action("previous_requested", previous_requested)
    current_position = validate_joint_vector("current_qpos", current_qpos)
    local_samples = _local_actor_samples(
        actor_samples,
        mean_action,
        protocol_config.actor_sample_max_delta_rms,
    )

    requested: list[np.ndarray] = [
        nominal_action,
        mean_action,
        previous_action,
    ]
    requested.extend(
        previous_action + factor * (nominal_action - previous_action)
        for factor in CONTRACTION_FACTORS
    )
    requested.extend(local_samples)
    for direction in _seeded_directions(seed):
        plus, minus = _symmetric_pair(
            nominal_action,
            direction,
            protocol_config.perturbation_radius_rms,
        )
        requested.extend((plus, minus))

    requested_array = np.asarray(requested, dtype=np.float32)
    if requested_array.shape != (CANDIDATE_COUNT, 12):
        raise AssertionError("internal candidate count violated the fixed protocol")

    projections = action_applier.preview_many(requested_array, current_position)
    if len(projections) != CANDIDATE_COUNT:
        raise CandidateProtocolError(
            "ActionApplier.preview_many returned the wrong candidate count")
    projected_requested = np.stack(
        [projection.action_requested for projection in projections], axis=0)
    projected_executed = np.stack(
        [projection.action_executed for projection in projections], axis=0)
    projected_q_target = np.stack(
        [projection.action_q_target for projection in projections], axis=0)
    mask = _unique_q_target_mask(
        projected_q_target, protocol_config.q_target_dedup_atol)
    valid_count = int(np.count_nonzero(mask))
    if valid_count < protocol_config.min_unique_candidates:
        raise CandidateProtocolError(
            "candidate projection produced only "
            f"{valid_count} unique q_target values; at least "
            f"{protocol_config.min_unique_candidates} are required")

    return CandidateSet(
        requested=projected_requested,
        executed=projected_executed,
        q_target=projected_q_target,
        kind=np.asarray(CANDIDATE_KINDS),
        mask=mask,
        candidate_seed=seed,
        manifest_protocol=protocol_config.manifest_protocol(),
    )


__all__ = [
    "ACTOR_SAMPLE_COUNT",
    "CANDIDATE_COUNT",
    "CANDIDATE_KINDS",
    "CANDIDATE_PROTOCOL_VERSION",
    "CandidateProtocolError",
    "CandidateSet",
    "EvidenceCandidateConfig",
    "build_evidence_candidates",
]
