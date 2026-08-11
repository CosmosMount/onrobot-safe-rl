"""Outcome-free matched-random placebo fitting for persistent K9 recovery.

The placebo matches frozen Q_safe *decisions*, never fall labels.  It bins the
current state's nominal-risk LCB, samples abstention/intervention, then samples
a duration-by-first-action-distance cell.  Candidate choice is uniform inside
that cell and an empty cell deterministically abstains without fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping

import numpy as np

from rl.qsafe.recovery_calibration import (
    canonical_sha256,
    execution_lock_sha256,
)
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_BEHAVIOR_STEPS,
    RECOVERY_PROGRAM_CANDIDATE_COUNT,
    RECOVERY_PROGRAM_NOMINAL_INDEX,
)
from rl.qsafe.recovery_selector import RecoverySelectorConfig


MATCHED_RANDOM_PLACEBO_SCHEMA_VERSION = "qsafe.matched_random_placebo.v1"
MATCHED_RANDOM_PLACEBO_RNG_DOMAIN = (
    "qsafe_state_dependent_recovery_v4_placebo\0")
MATCHED_RANDOM_PLACEBO_DURATIONS = (10, 25, 50)
MATCHED_RANDOM_PLACEBO_RISK_BINS = 10
MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES = 4
MAX_INTERVENTION_RATE_MISMATCH = 0.02
MAX_DURATION_TOTAL_VARIATION = 0.05
MAX_ACTION_DISTANCE_KS = 0.10

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_UINT64_MAX = (1 << 64) - 1
_UINT256_MAX = (1 << 256) - 1
_SELECTOR_TRIGGER_GRID = (0.10, 0.20, 0.30, 0.40, 0.50)
_SELECTOR_BENEFIT_GRID = (0.00, 0.02, 0.05, 0.08, 0.12)
_SELECTOR_RISK_GRID = (0.25, 0.40, 0.55, 0.70)


def _readonly(value: Any, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _selector_payload(config: RecoverySelectorConfig) -> dict[str, float]:
    if not isinstance(config, RecoverySelectorConfig):
        raise TypeError("selector_config must be RecoverySelectorConfig")
    checked = _validate_frozen_selector_config(config)
    return {
        name: float(getattr(checked, name))
        for name in checked.__dataclass_fields__
    }


def _selector_from_payload(value: Any) -> RecoverySelectorConfig:
    fields = set(RecoverySelectorConfig.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("placebo selector support fields are not exact")
    return _validate_frozen_selector_config(RecoverySelectorConfig(**{
        name: _finite(value.get(name), name) for name in fields
    }))


def _validate_frozen_selector_config(
    config: RecoverySelectorConfig,
) -> RecoverySelectorConfig:
    if not isinstance(config, RecoverySelectorConfig):
        raise TypeError("selector_config must be RecoverySelectorConfig")
    checked = config.validated()
    if checked.nominal_risk_lcb_trigger not in _SELECTOR_TRIGGER_GRID or (
            checked.min_benefit_lcb not in _SELECTOR_BENEFIT_GRID) or (
                checked.max_risk_ucb not in _SELECTOR_RISK_GRID):
        raise ValueError("placebo selector config is outside the frozen grid")
    if checked.max_epistemic_std != 0.20 or (
            checked.max_action_delta_rms != 0.50) or (
                checked.max_q_target_delta_rms != 0.25):
        raise ValueError("placebo selector support differs from the frozen gates")
    return checked


@dataclass(frozen=True)
class PlaceboFitMetrics:
    target_intervention_rate: float
    realized_intervention_rate: float
    absolute_intervention_rate_error: float
    duration_histogram_total_variation: float
    first_action_distance_ecdf_distance: float
    eligible: bool

    def __post_init__(self) -> None:
        for name in (
            "target_intervention_rate",
            "realized_intervention_rate",
            "absolute_intervention_rate_error",
            "duration_histogram_total_variation",
            "first_action_distance_ecdf_distance",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        expected = bool(
            self.absolute_intervention_rate_error
            <= MAX_INTERVENTION_RATE_MISMATCH
            and self.duration_histogram_total_variation
            <= MAX_DURATION_TOTAL_VARIATION
            and self.first_action_distance_ecdf_distance
            <= MAX_ACTION_DISTANCE_KS
        )
        if type(self.eligible) is not bool or self.eligible != expected:
            raise ValueError("placebo fit eligibility disagrees with frozen limits")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_intervention_rate": float(self.target_intervention_rate),
            "realized_intervention_rate": float(self.realized_intervention_rate),
            "absolute_intervention_rate_error": float(
                self.absolute_intervention_rate_error),
            "duration_histogram_total_variation": float(
                self.duration_histogram_total_variation),
            "first_action_distance_ecdf_distance": float(
                self.first_action_distance_ecdf_distance),
            "eligible": bool(self.eligible),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "PlaceboFitMetrics":
        expected = {
            "target_intervention_rate",
            "realized_intervention_rate",
            "absolute_intervention_rate_error",
            "duration_histogram_total_variation",
            "first_action_distance_ecdf_distance",
            "eligible",
        }
        if not isinstance(value, Mapping) or set(value) != expected or type(
                value.get("eligible")) is not bool:
            raise ValueError("placebo fit metric fields are not exact")
        return cls(
            target_intervention_rate=_finite(
                value.get("target_intervention_rate"),
                "target_intervention_rate"),
            realized_intervention_rate=_finite(
                value.get("realized_intervention_rate"),
                "realized_intervention_rate"),
            absolute_intervention_rate_error=_finite(
                value.get("absolute_intervention_rate_error"),
                "absolute_intervention_rate_error"),
            duration_histogram_total_variation=_finite(
                value.get("duration_histogram_total_variation"),
                "duration_histogram_total_variation"),
            first_action_distance_ecdf_distance=_finite(
                value.get("first_action_distance_ecdf_distance"),
                "first_action_distance_ecdf_distance"),
            eligible=value["eligible"],
        )


@dataclass(frozen=True)
class MatchedRandomPlaceboBundle:
    selector_bundle_sha256: str
    execution_lock_sha256: str
    fit_rng_assignment_count: int
    fit_rng_assignment_sha256: str
    selector_config: RecoverySelectorConfig
    nominal_risk_bin_edges: np.ndarray
    first_action_distance_edges: np.ndarray
    intervention_probability: np.ndarray
    conditional_cell_probability: np.ndarray
    fit_metrics: PlaceboFitMetrics

    def __post_init__(self) -> None:
        selector_hash = _sha256(
            self.selector_bundle_sha256, "selector_bundle_sha256")
        lock_hash = execution_lock_sha256(self.execution_lock_sha256)
        if isinstance(self.fit_rng_assignment_count, (bool, np.bool_)) or (
                not isinstance(self.fit_rng_assignment_count, (int, np.integer))) or (
                    int(self.fit_rng_assignment_count) <= 0):
            raise ValueError("fit_rng_assignment_count must be a positive integer")
        assignment_hash = _sha256(
            self.fit_rng_assignment_sha256, "fit_rng_assignment_sha256")
        config = _validate_frozen_selector_config(self.selector_config)
        risk_edges = np.asarray(self.nominal_risk_bin_edges, dtype=np.float64)
        distance_edges = np.asarray(
            self.first_action_distance_edges, dtype=np.float64)
        intervention = np.asarray(
            self.intervention_probability, dtype=np.float64)
        cells = np.asarray(
            self.conditional_cell_probability, dtype=np.float64)
        if risk_edges.shape != (MATCHED_RANDOM_PLACEBO_RISK_BINS + 1,) or (
                not np.all(np.isfinite(risk_edges))) or np.any(
                    np.diff(risk_edges) < 0.0):
            raise ValueError("nominal-risk decile edges are invalid")
        if distance_edges.shape != (
                MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES + 1,) or not np.all(
                    np.isfinite(distance_edges)) or np.any(
                        np.diff(distance_edges) < 0.0):
            raise ValueError("action-distance quartile edges are invalid")
        if intervention.shape != (MATCHED_RANDOM_PLACEBO_RISK_BINS,) or (
                not np.all(np.isfinite(intervention))) or np.any(
                    (intervention < 0.0) | (intervention > 1.0)):
            raise ValueError("placebo intervention probabilities are invalid")
        expected_cells = (
            MATCHED_RANDOM_PLACEBO_RISK_BINS,
            len(MATCHED_RANDOM_PLACEBO_DURATIONS),
            MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES,
        )
        if cells.shape != expected_cells or not np.all(np.isfinite(cells)) or (
                np.any(cells < 0.0)):
            raise ValueError("placebo conditional cell table is invalid")
        sums = cells.sum(axis=(1, 2))
        for row in range(MATCHED_RANDOM_PLACEBO_RISK_BINS):
            expected = 0.0 if intervention[row] == 0.0 else 1.0
            if not math.isclose(
                    float(sums[row]), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    "placebo conditional cell rows must sum to zero or one")
        if not isinstance(self.fit_metrics, PlaceboFitMetrics):
            raise TypeError("fit_metrics must be PlaceboFitMetrics")
        object.__setattr__(self, "selector_bundle_sha256", selector_hash)
        object.__setattr__(self, "execution_lock_sha256", lock_hash)
        object.__setattr__(
            self, "fit_rng_assignment_count", int(self.fit_rng_assignment_count))
        object.__setattr__(
            self, "fit_rng_assignment_sha256", assignment_hash)
        object.__setattr__(self, "selector_config", config)
        object.__setattr__(
            self, "nominal_risk_bin_edges", _readonly(risk_edges))
        object.__setattr__(
            self, "first_action_distance_edges", _readonly(distance_edges))
        object.__setattr__(
            self, "intervention_probability", _readonly(intervention))
        object.__setattr__(
            self, "conditional_cell_probability", _readonly(cells))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": MATCHED_RANDOM_PLACEBO_SCHEMA_VERSION,
            "fitted_role": (
                "selector_calibration_only_after_qsafe_selector_frozen"),
            "selector_bundle_sha256": self.selector_bundle_sha256,
            "execution_lock_sha256": self.execution_lock_sha256,
            "fit_rng_assignment": {
                "group_count": self.fit_rng_assignment_count,
                "assignment_sha256": self.fit_rng_assignment_sha256,
                "assignment_hash_payload": (
                    "ordered_source_u64le_group_fingerprint_raw32_"
                    "draw_index_u64le_derived_seed_uint256le32"),
            },
            "outcome_based_reweighting": "forbidden",
            "reads_qsafe_option_ranking": False,
            "own_current_state_only": True,
            "selector_trigger_and_support": _selector_payload(
                self.selector_config),
            "nominal_risk_bins": {
                "count": MATCHED_RANDOM_PLACEBO_RISK_BINS,
                "quantile_edges_0_to_1": self.nominal_risk_bin_edges.tolist(),
                "edge_method": "numpy_linear_quantile",
                "assignment": "interior_edges_searchsorted_right",
            },
            "durations": list(MATCHED_RANDOM_PLACEBO_DURATIONS),
            "first_action_distance_quantiles": {
                "count": MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES,
                "quantile_edges_0_to_1": (
                    self.first_action_distance_edges.tolist()),
                "edge_method": (
                    "numpy_linear_quantile_over_supported_nonnominal_options"),
                "assignment": "interior_edges_searchsorted_right",
            },
            "table": {
                "intervention_probability": (
                    self.intervention_probability.tolist()),
                "conditional_duration_by_distance_probability": (
                    self.conditional_cell_probability.tolist()),
                "cell_order": "duration_outer_then_distance_quartile",
            },
            "sampling": {
                "bit_generator": "numpy_PCG64",
                "rng_domain_literal": MATCHED_RANDOM_PLACEBO_RNG_DOMAIN,
                "preassigned_seed_required": True,
                "seed_payload_order": [
                    "domain_literal_ascii_with_terminal_nul",
                    "source_seed_u64le",
                    "group_fingerprint_sha256_raw32",
                    "draw_index_u64le",
                ],
                "seed_integer": "sha256_payload_as_uint256_little_endian",
                "draw_order": [
                    "intervene_uniform",
                    "cell_uniform",
                    "eligible_option_uniform",
                ],
                "option_sampling": "uniform_within_eligible_cell",
                "empty_cell": "deterministic_abstain_no_fallback",
                "candidate_order": "locked_K9_index",
            },
            "fit": {
                "method": (
                    "preassigned_uniform_threshold_matching_outcome_free"),
                "objective_order": [
                    "absolute_intervention_rate_error",
                    "duration_histogram_total_variation",
                    "first_action_distance_ecdf_distance",
                ],
                "tie_break": "lexicographic_table_order",
                "limits": {
                    "max_intervention_rate_mismatch": (
                        MAX_INTERVENTION_RATE_MISMATCH),
                    "max_duration_total_variation": (
                        MAX_DURATION_TOTAL_VARIATION),
                    "max_action_distance_ks": MAX_ACTION_DISTANCE_KS,
                },
                "metrics": self.fit_metrics.to_dict(),
            },
        }

    @property
    def bundle_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def to_dict(self) -> dict[str, Any]:
        result = self.payload()
        result["bundle_sha256"] = self.bundle_sha256
        return result

    @classmethod
    def from_dict(cls, value: Any) -> "MatchedRandomPlaceboBundle":
        expected = {
            "schema_version",
            "fitted_role",
            "selector_bundle_sha256",
            "execution_lock_sha256",
            "fit_rng_assignment",
            "outcome_based_reweighting",
            "reads_qsafe_option_ranking",
            "own_current_state_only",
            "selector_trigger_and_support",
            "nominal_risk_bins",
            "durations",
            "first_action_distance_quantiles",
            "table",
            "sampling",
            "fit",
            "bundle_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("placebo bundle fields are not exact")
        if value.get("schema_version") != MATCHED_RANDOM_PLACEBO_SCHEMA_VERSION or (
                value.get("fitted_role") != (
                    "selector_calibration_only_after_qsafe_selector_frozen")) or (
                value.get("outcome_based_reweighting") != "forbidden") or (
                value.get("reads_qsafe_option_ranking") is not False) or (
                value.get("own_current_state_only") is not True):
            raise ValueError("placebo bundle frozen semantics drifted")
        risk = value.get("nominal_risk_bins")
        distance = value.get("first_action_distance_quantiles")
        table = value.get("table")
        sampling = value.get("sampling")
        fit = value.get("fit")
        assignment = value.get("fit_rng_assignment")
        if not all(isinstance(item, Mapping) for item in (
                risk, distance, table, sampling, fit, assignment)):
            raise ValueError("placebo bundle nested sections are required")
        expected_risk = {
            "count", "quantile_edges_0_to_1", "edge_method", "assignment",
        }
        expected_distance = {
            "count", "quantile_edges_0_to_1", "edge_method", "assignment",
        }
        expected_table = {
            "intervention_probability",
            "conditional_duration_by_distance_probability",
            "cell_order",
        }
        expected_sampling = {
            "bit_generator", "rng_domain_literal", "preassigned_seed_required",
            "seed_payload_order", "seed_integer", "draw_order",
            "option_sampling", "empty_cell", "candidate_order",
        }
        expected_fit = {
            "method", "objective_order", "tie_break", "limits", "metrics",
        }
        if set(risk) != expected_risk or set(distance) != expected_distance or (
                set(table) != expected_table) or set(sampling) != expected_sampling or (
                    set(fit) != expected_fit):
            raise ValueError("placebo bundle nested fields are not exact")
        if set(assignment) != {
            "group_count", "assignment_sha256", "assignment_hash_payload",
        } or assignment.get("assignment_hash_payload") != (
            "ordered_source_u64le_group_fingerprint_raw32_"
            "draw_index_u64le_derived_seed_uint256le32"
        ):
            raise ValueError("placebo fit RNG assignment fields drifted")
        limits = fit.get("limits")
        if not isinstance(limits, Mapping) or dict(limits) != {
            "max_intervention_rate_mismatch": MAX_INTERVENTION_RATE_MISMATCH,
            "max_duration_total_variation": MAX_DURATION_TOTAL_VARIATION,
            "max_action_distance_ks": MAX_ACTION_DISTANCE_KS,
        }:
            raise ValueError("placebo fit limits drifted")
        frozen_checks = (
            risk.get("count") == MATCHED_RANDOM_PLACEBO_RISK_BINS,
            risk.get("edge_method") == "numpy_linear_quantile",
            risk.get("assignment") == "interior_edges_searchsorted_right",
            distance.get("count") == MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES,
            distance.get("edge_method") == (
                "numpy_linear_quantile_over_supported_nonnominal_options"),
            distance.get("assignment") == "interior_edges_searchsorted_right",
            value.get("durations") == list(MATCHED_RANDOM_PLACEBO_DURATIONS),
            table.get("cell_order") == (
                "duration_outer_then_distance_quartile"),
            sampling.get("bit_generator") == "numpy_PCG64",
            sampling.get("rng_domain_literal") == MATCHED_RANDOM_PLACEBO_RNG_DOMAIN,
            sampling.get("preassigned_seed_required") is True,
            sampling.get("seed_payload_order") == [
                "domain_literal_ascii_with_terminal_nul",
                "source_seed_u64le",
                "group_fingerprint_sha256_raw32",
                "draw_index_u64le",
            ],
            sampling.get("seed_integer") == (
                "sha256_payload_as_uint256_little_endian"),
            sampling.get("draw_order") == [
                "intervene_uniform", "cell_uniform", "eligible_option_uniform"],
            sampling.get("option_sampling") == "uniform_within_eligible_cell",
            sampling.get("empty_cell") == (
                "deterministic_abstain_no_fallback"),
            sampling.get("candidate_order") == "locked_K9_index",
            fit.get("method") == (
                "preassigned_uniform_threshold_matching_outcome_free"),
            fit.get("objective_order") == [
                "absolute_intervention_rate_error",
                "duration_histogram_total_variation",
                "first_action_distance_ecdf_distance",
            ],
            fit.get("tie_break") == "lexicographic_table_order",
        )
        if not all(frozen_checks):
            raise ValueError("placebo bundle frozen table semantics drifted")
        bundle = cls(
            selector_bundle_sha256=value.get("selector_bundle_sha256"),
            execution_lock_sha256=value.get("execution_lock_sha256"),
            fit_rng_assignment_count=assignment.get("group_count"),
            fit_rng_assignment_sha256=assignment.get("assignment_sha256"),
            selector_config=_selector_from_payload(
                value.get("selector_trigger_and_support")),
            nominal_risk_bin_edges=np.asarray(
                risk.get("quantile_edges_0_to_1")),
            first_action_distance_edges=np.asarray(
                distance.get("quantile_edges_0_to_1")),
            intervention_probability=np.asarray(
                table.get("intervention_probability")),
            conditional_cell_probability=np.asarray(
                table.get("conditional_duration_by_distance_probability")),
            fit_metrics=PlaceboFitMetrics.from_dict(fit.get("metrics")),
        )
        if value.get("bundle_sha256") != bundle.bundle_sha256:
            raise ValueError("placebo bundle hash mismatch")
        return bundle


@dataclass(frozen=True)
class MatchedRandomPlaceboDecision:
    selected_index: int
    intervened: bool
    reason: str
    risk_bin: int
    duration_steps: int
    action_distance_quartile: int


def _uint64(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)) or int(value) < 0 or int(value) > (
                _UINT64_MAX):
        raise ValueError(f"{name} must be an unsigned 64-bit integer")
    return int(value)


def _uint256_seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)) or int(value) < 0 or int(value) > (
                _UINT256_MAX):
        raise ValueError("placebo RNG seed must be an unsigned 256-bit integer")
    return int(value)


def derive_matched_random_placebo_seed(
    *,
    source_seed: Any,
    group_fingerprint_sha256: Any,
    draw_index: Any = 0,
) -> int:
    """Derive the execution-locked PCG64 seed as a little-endian uint256."""
    source = _uint64(source_seed, "source_seed")
    index = _uint64(draw_index, "draw_index")
    fingerprint = _sha256(
        group_fingerprint_sha256, "group_fingerprint_sha256")
    payload = b"".join((
        MATCHED_RANDOM_PLACEBO_RNG_DOMAIN.encode("ascii"),
        source.to_bytes(8, "little", signed=False),
        bytes.fromhex(fingerprint),
        index.to_bytes(8, "little", signed=False),
    ))
    return int.from_bytes(hashlib.sha256(payload).digest(), "little")


def _pcg64_draws(seed: Any) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(_uint256_seed(seed)))
    return rng.random(3)


def _binary_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in "biu" or not np.all(
            np.isin(raw, (0, 1, False, True))):
        raise ValueError(f"{name} must be binary shape {shape}")
    return raw.astype(bool, copy=True)


def _duration_array(value: Any, groups: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    shape = ((RECOVERY_PROGRAM_CANDIDATE_COUNT,) if groups is None else
             (groups, RECOVERY_PROGRAM_CANDIDATE_COUNT))
    if raw.shape != shape or raw.dtype.kind not in "iu":
        raise ValueError(f"candidate_duration_steps must be integer shape {shape}")
    allowed = np.asarray((0, *MATCHED_RANDOM_PLACEBO_DURATIONS))
    if not np.all(np.isin(raw, allowed)) or np.any(raw[..., 0] != 0):
        raise ValueError("candidate durations must be nominal zero or 10/25/50")
    expected = np.asarray(RECOVERY_PROGRAM_BEHAVIOR_STEPS, dtype=raw.dtype)
    if groups is None:
        exact = np.array_equal(raw, expected)
    else:
        exact = np.array_equal(raw, np.broadcast_to(expected, raw.shape))
    if not exact:
        raise ValueError("candidate durations differ from the locked K9 order")
    return raw.astype(np.int64, copy=True)


def _distance_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    distance = np.asarray(value, dtype=np.float64)
    if distance.shape != shape or not np.all(np.isfinite(distance)) or np.any(
            distance < 0.0):
        raise ValueError(f"first_action_distance must be finite nonnegative {shape}")
    return distance.copy()


def _bin(value: float, edges: np.ndarray) -> int:
    return int(np.searchsorted(edges, value, side="right"))


def _select_from_draws(
    bundle: MatchedRandomPlaceboBundle,
    *,
    nominal_risk_lcb: float,
    candidate_support_mask: np.ndarray,
    candidate_duration_steps: np.ndarray,
    first_action_distance: np.ndarray,
    draws: np.ndarray,
) -> MatchedRandomPlaceboDecision:
    risk = _finite(nominal_risk_lcb, "nominal_risk_lcb")
    if risk < 0.0 or risk > 1.0:
        raise ValueError("nominal_risk_lcb must lie in [0,1]")
    support = _binary_array(
        candidate_support_mask,
        (RECOVERY_PROGRAM_CANDIDATE_COUNT,),
        "candidate_support_mask",
    )
    duration = _duration_array(candidate_duration_steps)
    distance = _distance_array(
        first_action_distance, (RECOVERY_PROGRAM_CANDIDATE_COUNT,))
    if not support[RECOVERY_PROGRAM_NOMINAL_INDEX]:
        raise ValueError("placebo support must retain the nominal candidate")
    if distance[RECOVERY_PROGRAM_NOMINAL_INDEX] != 0.0:
        raise ValueError("nominal first-action distance must equal zero")
    draws = np.asarray(draws, dtype=np.float64)
    if draws.shape != (3,) or not np.all(np.isfinite(draws)) or np.any(
            (draws < 0.0) | (draws >= 1.0)):
        raise ValueError("placebo draws must be three uniforms in [0,1)")
    risk_bin = _bin(risk, bundle.nominal_risk_bin_edges[1:-1])
    if risk < bundle.selector_config.nominal_risk_lcb_trigger:
        return MatchedRandomPlaceboDecision(
            selected_index=0,
            intervened=False,
            reason="state_below_trigger",
            risk_bin=risk_bin,
            duration_steps=0,
            action_distance_quartile=-1,
        )
    if draws[0] >= bundle.intervention_probability[risk_bin]:
        return MatchedRandomPlaceboDecision(
            selected_index=0,
            intervened=False,
            reason="table_abstain",
            risk_bin=risk_bin,
            duration_steps=0,
            action_distance_quartile=-1,
        )
    probability = bundle.conditional_cell_probability[risk_bin].reshape(-1)
    if probability.sum() == 0.0:
        return MatchedRandomPlaceboDecision(
            selected_index=0,
            intervened=False,
            reason="empty_table_row",
            risk_bin=risk_bin,
            duration_steps=0,
            action_distance_quartile=-1,
        )
    cell = int(np.searchsorted(
        np.cumsum(probability), draws[1], side="right"))
    cell = min(cell, len(probability) - 1)
    duration_index, distance_quartile = divmod(
        cell, MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES)
    chosen_duration = MATCHED_RANDOM_PLACEBO_DURATIONS[duration_index]
    option_quartile = np.searchsorted(
        bundle.first_action_distance_edges[1:-1], distance, side="right")
    eligible = (
        support
        & (duration == chosen_duration)
        & (option_quartile == distance_quartile)
    )
    eligible[RECOVERY_PROGRAM_NOMINAL_INDEX] = False
    indices = np.flatnonzero(eligible)
    if len(indices) == 0:
        return MatchedRandomPlaceboDecision(
            selected_index=0,
            intervened=False,
            reason="empty_cell",
            risk_bin=risk_bin,
            duration_steps=0,
            action_distance_quartile=distance_quartile,
        )
    offset = min(int(math.floor(draws[2] * len(indices))), len(indices) - 1)
    selected = int(indices[offset])
    return MatchedRandomPlaceboDecision(
        selected_index=selected,
        intervened=True,
        reason="selected_uniform_within_cell",
        risk_bin=risk_bin,
        duration_steps=chosen_duration,
        action_distance_quartile=distance_quartile,
    )


def select_matched_random_placebo(
    bundle: MatchedRandomPlaceboBundle,
    *,
    nominal_risk_lcb: Any,
    candidate_support_mask: Any,
    candidate_duration_steps: Any,
    first_action_distance: Any,
    source_seed: Any,
    group_fingerprint_sha256: Any,
    draw_index: Any = 0,
) -> MatchedRandomPlaceboDecision:
    """Derive the preassigned PCG64 stream and sample one placebo decision."""
    if not isinstance(bundle, MatchedRandomPlaceboBundle):
        raise TypeError("bundle must be MatchedRandomPlaceboBundle")
    seed = derive_matched_random_placebo_seed(
        source_seed=source_seed,
        group_fingerprint_sha256=group_fingerprint_sha256,
        draw_index=draw_index,
    )
    return _select_from_draws(
        bundle,
        nominal_risk_lcb=nominal_risk_lcb,
        candidate_support_mask=np.asarray(candidate_support_mask),
        candidate_duration_steps=np.asarray(candidate_duration_steps),
        first_action_distance=np.asarray(first_action_distance),
        draws=_pcg64_draws(seed),
    )


def _threshold_for_count(draws: np.ndarray, target_count: int) -> float:
    values = np.sort(np.asarray(draws, dtype=np.float64), kind="stable")
    if len(values) == 0:
        return 0.0
    target = int(np.clip(target_count, 0, len(values)))
    if target == 0:
        return 0.0
    if target == len(values):
        return float(np.nextafter(values[-1], 1.0))
    lower = values[target - 1]
    upper = values[target]
    if lower < upper:
        return float(np.nextafter(lower, 1.0))
    # A tied boundary cannot realize the requested count.  Select the closer
    # attainable side; an exact distance tie chooses the smaller threshold.
    below = int(np.sum(values < lower))
    through = int(np.sum(values <= lower))
    if abs(target - below) <= abs(through - target):
        return float(lower)
    return float(np.nextafter(lower, 1.0))


def _scaled_cell_counts(target: np.ndarray, total: int) -> np.ndarray:
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if total == 0:
        return np.zeros_like(target)
    if target.sum() <= 0:
        raise ValueError("positive placebo intervention count has no target cell")
    raw = target.astype(np.float64) * (float(total) / float(target.sum()))
    result = np.floor(raw).astype(np.int64)
    remaining = int(total - result.sum())
    remainder = raw - result
    order = np.lexsort((np.arange(len(target)), -remainder))
    result[order[:remaining]] += 1
    return result


def _categorical_probabilities_for_counts(
    draws: np.ndarray,
    target_counts: np.ndarray,
) -> np.ndarray:
    target = np.asarray(target_counts, dtype=np.int64).reshape(-1)
    if len(draws) != int(target.sum()):
        raise ValueError("categorical target counts must sum to draw count")
    if len(draws) == 0:
        return np.zeros(len(target), dtype=np.float64)
    cumulative = np.cumsum(target)[:-1]
    boundaries = np.asarray([
        _threshold_for_count(draws, int(count)) for count in cumulative
    ])
    cdf = np.concatenate([[0.0], boundaries, [1.0]])
    probability = np.diff(cdf)
    if np.any(probability < -1e-15):
        raise AssertionError("categorical thresholds are not monotone")
    probability = np.maximum(probability, 0.0)
    probability /= probability.sum()
    return probability


def _histogram(values: np.ndarray, categories: tuple[int, ...]) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(len(categories), dtype=np.float64)
    return np.asarray([
        np.mean(values == category) for category in categories
    ], dtype=np.float64)


def _ecdf_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=np.float64), kind="stable")
    right = np.sort(np.asarray(right, dtype=np.float64), kind="stable")
    if len(left) == 0 and len(right) == 0:
        return 0.0
    if len(left) == 0 or len(right) == 0:
        return 1.0
    support = np.unique(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, support, side="right") / len(left)
    right_cdf = np.searchsorted(right, support, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _rng_assignments(
    source_seed: Any,
    group_fingerprint_sha256: Any,
    draw_index: Any,
    groups: int,
) -> tuple[np.ndarray, list[int], str]:
    raw_source = np.asarray(source_seed)
    raw_fingerprint = np.asarray(group_fingerprint_sha256)
    raw_draw_index = np.asarray(draw_index)
    if raw_source.shape != (groups,) or raw_source.dtype.kind not in "iu":
        raise ValueError("placebo_source_seed must be integer shape [G]")
    if raw_fingerprint.shape != (groups,):
        raise ValueError("group_fingerprint_sha256 must have shape [G]")
    if raw_draw_index.ndim == 0:
        raw_draw_index = np.full(groups, raw_draw_index.item())
    if raw_draw_index.shape != (groups,) or raw_draw_index.dtype.kind not in "iu":
        raise ValueError("placebo_draw_index must be integer scalar or [G]")
    source = np.asarray([
        _uint64(value, "placebo_source_seed") for value in raw_source.tolist()
    ], dtype=np.uint64)
    fingerprints = np.asarray([
        _sha256(value, "group_fingerprint_sha256")
        for value in raw_fingerprint.tolist()
    ])
    if len(np.unique(fingerprints)) != groups:
        raise ValueError("placebo group fingerprints must be unique")
    indices = np.asarray([
        _uint64(value, "placebo_draw_index")
        for value in raw_draw_index.tolist()
    ], dtype=np.uint64)
    seeds = [
        derive_matched_random_placebo_seed(
            source_seed=int(source[group]),
            group_fingerprint_sha256=str(fingerprints[group]),
            draw_index=int(indices[group]),
        )
        for group in range(groups)
    ]
    if len(set(seeds)) != groups:
        raise ValueError("derived placebo fit seeds must be unique")
    digest = hashlib.sha256(b"qsafe.placebo.fit_rng_assignments.v1\0")
    for group in range(groups):
        digest.update(int(source[group]).to_bytes(8, "little"))
        digest.update(bytes.fromhex(str(fingerprints[group])))
        digest.update(int(indices[group]).to_bytes(8, "little"))
        digest.update(int(seeds[group]).to_bytes(32, "little"))
    return source, seeds, digest.hexdigest()


def fit_matched_random_placebo(
    *,
    nominal_risk_lcb: Any,
    qsafe_selected_index: Any,
    candidate_support_mask: Any,
    candidate_duration_steps: Any,
    first_action_distance: Any,
    placebo_source_seed: Any,
    group_fingerprint_sha256: Any,
    placebo_draw_index: Any = 0,
    selector_config: RecoverySelectorConfig,
    selector_bundle_sha256: str,
    execution_lock: str | Mapping[str, Any],
) -> MatchedRandomPlaceboBundle:
    """Fit a canonical outcome-free placebo table to frozen Q_safe decisions."""
    risk = np.asarray(nominal_risk_lcb, dtype=np.float64).reshape(-1)
    groups = len(risk)
    if groups == 0 or not np.all(np.isfinite(risk)) or np.any(
            (risk < 0.0) | (risk > 1.0)):
        raise ValueError("nominal_risk_lcb must be nonempty probabilities [G]")
    selected = np.asarray(qsafe_selected_index)
    if selected.shape != (groups,) or selected.dtype.kind not in "iu" or np.any(
            (selected < 0) | (selected >= RECOVERY_PROGRAM_CANDIDATE_COUNT)):
        raise ValueError("qsafe_selected_index must be integer K9 indices [G]")
    selected = selected.astype(np.int64, copy=True)
    support = _binary_array(
        candidate_support_mask,
        (groups, RECOVERY_PROGRAM_CANDIDATE_COUNT),
        "candidate_support_mask",
    )
    duration_raw = np.asarray(candidate_duration_steps)
    if duration_raw.shape == (RECOVERY_PROGRAM_CANDIDATE_COUNT,):
        duration_raw = np.broadcast_to(duration_raw, (
            groups, RECOVERY_PROGRAM_CANDIDATE_COUNT))
    duration = _duration_array(duration_raw, groups)
    distance = _distance_array(
        first_action_distance,
        (groups, RECOVERY_PROGRAM_CANDIDATE_COUNT),
    )
    if not np.all(support[:, RECOVERY_PROGRAM_NOMINAL_INDEX]):
        raise ValueError("placebo support must retain every nominal candidate")
    if not np.all(distance[:, RECOVERY_PROGRAM_NOMINAL_INDEX] == 0.0):
        raise ValueError("nominal first-action distance must equal zero")
    _, seeds, assignment_hash = _rng_assignments(
        placebo_source_seed,
        group_fingerprint_sha256,
        placebo_draw_index,
        groups,
    )
    config = _validate_frozen_selector_config(selector_config)
    selector_hash = _sha256(selector_bundle_sha256, "selector_bundle_sha256")
    lock_hash = execution_lock_sha256(execution_lock)
    intervention = selected != RECOVERY_PROGRAM_NOMINAL_INDEX
    row = np.arange(groups)
    if np.any(intervention & ~support[row, selected]):
        raise ValueError("Q_safe selected an option outside placebo support")
    if np.any((risk < config.nominal_risk_lcb_trigger) & intervention):
        raise ValueError("Q_safe intervention violates the frozen nominal trigger")
    if np.any(intervention & ~np.isin(
            duration[row, selected], MATCHED_RANDOM_PLACEBO_DURATIONS)):
        raise ValueError("Q_safe selected an unregistered option duration")

    risk_edges = np.quantile(
        risk,
        np.arange(0, MATCHED_RANDOM_PLACEBO_RISK_BINS + 1)
        / MATCHED_RANDOM_PLACEBO_RISK_BINS,
        method="linear",
    )
    nonnominal = np.ones_like(support)
    nonnominal[:, RECOVERY_PROGRAM_NOMINAL_INDEX] = False
    distance_pool = distance[support & nonnominal]
    if len(distance_pool) == 0:
        raise ValueError("placebo fitting requires a supported nonnominal option")
    distance_edges = np.quantile(
        distance_pool,
        np.arange(0, MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES + 1)
        / MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES,
        method="linear",
    )
    risk_bin = np.searchsorted(risk_edges[1:-1], risk, side="right")
    distance_bin = np.searchsorted(
        distance_edges[1:-1], distance, side="right")
    duration_index = np.full_like(duration, -1, dtype=np.int64)
    for index, value in enumerate(MATCHED_RANDOM_PLACEBO_DURATIONS):
        duration_index[duration == value] = index
    selected_cell = np.full(groups, -1, dtype=np.int64)
    selected_cell[intervention] = (
        duration_index[row[intervention], selected[intervention]]
        * MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES
        + distance_bin[row[intervention], selected[intervention]]
    )
    draws = np.stack([_pcg64_draws(value) for value in seeds])

    table_intervention = np.zeros(
        MATCHED_RANDOM_PLACEBO_RISK_BINS, dtype=np.float64)
    table_cells = np.zeros((
        MATCHED_RANDOM_PLACEBO_RISK_BINS,
        len(MATCHED_RANDOM_PLACEBO_DURATIONS),
        MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES,
    ), dtype=np.float64)
    trigger_active = risk >= config.nominal_risk_lcb_trigger
    for bin_index in range(MATCHED_RANDOM_PLACEBO_RISK_BINS):
        in_bin_active = (risk_bin == bin_index) & trigger_active
        target_count = int(np.sum(intervention & (risk_bin == bin_index)))
        table_intervention[bin_index] = _threshold_for_count(
            draws[in_bin_active, 0], target_count)
        proposed = in_bin_active & (
            draws[:, 0] < table_intervention[bin_index])
        proposed_count = int(np.sum(proposed))
        if proposed_count == 0:
            table_intervention[bin_index] = 0.0
            continue
        target_cells = np.bincount(
            selected_cell[intervention & (risk_bin == bin_index)],
            minlength=(len(MATCHED_RANDOM_PLACEBO_DURATIONS)
                       * MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES),
        )
        scaled = _scaled_cell_counts(target_cells, proposed_count)
        probability = _categorical_probabilities_for_counts(
            draws[proposed, 1], scaled)
        table_cells[bin_index] = probability.reshape(
            len(MATCHED_RANDOM_PLACEBO_DURATIONS),
            MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES,
        )

    placeholder_metrics = PlaceboFitMetrics(
        target_intervention_rate=float(np.mean(intervention)),
        realized_intervention_rate=0.0,
        absolute_intervention_rate_error=float(np.mean(intervention)),
        duration_histogram_total_variation=1.0,
        first_action_distance_ecdf_distance=1.0,
        eligible=False,
    )
    provisional = MatchedRandomPlaceboBundle(
        selector_bundle_sha256=selector_hash,
        execution_lock_sha256=lock_hash,
        fit_rng_assignment_count=groups,
        fit_rng_assignment_sha256=assignment_hash,
        selector_config=config,
        nominal_risk_bin_edges=risk_edges,
        first_action_distance_edges=distance_edges,
        intervention_probability=table_intervention,
        conditional_cell_probability=table_cells,
        fit_metrics=placeholder_metrics,
    )
    placebo_selected = np.zeros(groups, dtype=np.int64)
    for group in range(groups):
        placebo_selected[group] = _select_from_draws(
            provisional,
            nominal_risk_lcb=risk[group],
            candidate_support_mask=support[group],
            candidate_duration_steps=duration[group],
            first_action_distance=distance[group],
            draws=draws[group],
        ).selected_index
    placebo_intervention = placebo_selected != RECOVERY_PROGRAM_NOMINAL_INDEX
    target_rate = float(np.mean(intervention))
    realized_rate = float(np.mean(placebo_intervention))
    rate_error = abs(target_rate - realized_rate)
    target_duration = duration[row[intervention], selected[intervention]]
    realized_duration = duration[
        row[placebo_intervention], placebo_selected[placebo_intervention]]
    duration_tv = float(0.5 * np.sum(np.abs(
        _histogram(target_duration, MATCHED_RANDOM_PLACEBO_DURATIONS)
        - _histogram(realized_duration, MATCHED_RANDOM_PLACEBO_DURATIONS)
    )))
    target_distance = distance[row[intervention], selected[intervention]]
    realized_distance = distance[
        row[placebo_intervention], placebo_selected[placebo_intervention]]
    distance_ks = _ecdf_distance(target_distance, realized_distance)
    metrics = PlaceboFitMetrics(
        target_intervention_rate=target_rate,
        realized_intervention_rate=realized_rate,
        absolute_intervention_rate_error=rate_error,
        duration_histogram_total_variation=duration_tv,
        first_action_distance_ecdf_distance=distance_ks,
        eligible=bool(
            rate_error <= MAX_INTERVENTION_RATE_MISMATCH
            and duration_tv <= MAX_DURATION_TOTAL_VARIATION
            and distance_ks <= MAX_ACTION_DISTANCE_KS
        ),
    )
    return MatchedRandomPlaceboBundle(
        selector_bundle_sha256=selector_hash,
        execution_lock_sha256=lock_hash,
        fit_rng_assignment_count=groups,
        fit_rng_assignment_sha256=assignment_hash,
        selector_config=config,
        nominal_risk_bin_edges=risk_edges,
        first_action_distance_edges=distance_edges,
        intervention_probability=table_intervention,
        conditional_cell_probability=table_cells,
        fit_metrics=metrics,
    )


__all__ = [
    "MATCHED_RANDOM_PLACEBO_DISTANCE_QUARTILES",
    "MATCHED_RANDOM_PLACEBO_DURATIONS",
    "MATCHED_RANDOM_PLACEBO_RISK_BINS",
    "MATCHED_RANDOM_PLACEBO_RNG_DOMAIN",
    "MATCHED_RANDOM_PLACEBO_SCHEMA_VERSION",
    "MatchedRandomPlaceboBundle",
    "MatchedRandomPlaceboDecision",
    "PlaceboFitMetrics",
    "derive_matched_random_placebo_seed",
    "fit_matched_random_placebo",
    "select_matched_random_placebo",
]
