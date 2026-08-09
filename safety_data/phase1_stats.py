"""Preregistered Phase 1 online three-arm evidence statistics.

The independent statistical unit in this module is a complete training seed.
Each seed must contain exactly one baseline, treatment, and matched-placebo run
at the same fixed exposure.  The helpers intentionally know nothing about
Phase 2: their only terminal decision is ``phase1_pass``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Any, Iterable, Mapping

import numpy as np


ARMS = ("baseline", "treatment", "placebo")
PHASE1_ROUTES = ("fresh_030", "shift_027", "shift_033")
CONFIRMATION_SEEDS = tuple(range(201, 211))
_ROUTE_SPEED_MPS = {
    "fresh_030": 0.30,
    "shift_027": 0.27,
    "shift_033": 0.33,
}


class Phase1EvidenceError(ValueError):
    """Raised when a Phase 1 evidence table is incomplete or ambiguous."""


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_))


def _finite_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase1EvidenceError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise Phase1EvidenceError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class OnlineRun:
    """One completed arm for one independently trained policy seed."""

    route: str
    training_seed: int
    arm: str
    target_speed_mps: float
    exposure_policy_steps: int
    falls: int
    mean_task_return: float
    forward_velocity_error_mps: float
    deadline_misses: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "OnlineRun":
        expected = set(cls.__dataclass_fields__)
        unknown = set(value) - expected
        if unknown:
            raise Phase1EvidenceError(
                f"online run has unknown fields: {sorted(unknown)}")
        try:
            return cls(**{field: value[field] for field in cls.__dataclass_fields__})
        except KeyError as exc:
            raise Phase1EvidenceError(
                f"online run is missing required field {exc.args[0]!r}") from exc

    def validate(self) -> None:
        if self.route not in PHASE1_ROUTES:
            raise Phase1EvidenceError(f"unknown Phase 1 route {self.route!r}")
        if self.arm not in ARMS:
            raise Phase1EvidenceError(f"unknown online arm {self.arm!r}")
        if not _is_integer(self.training_seed):
            raise Phase1EvidenceError("training_seed must be an integer")
        if not _is_integer(self.exposure_policy_steps) or self.exposure_policy_steps <= 0:
            raise Phase1EvidenceError("exposure_policy_steps must be a positive integer")
        if not _is_integer(self.falls) or not 0 <= self.falls <= self.exposure_policy_steps:
            raise Phase1EvidenceError(
                "falls must be a nonnegative integer no greater than exposure")
        if not _is_integer(self.deadline_misses) or not (
            0 <= self.deadline_misses <= self.exposure_policy_steps
        ):
            raise Phase1EvidenceError(
                "deadline_misses must be a nonnegative integer no greater than exposure")
        speed = _finite_float(self.target_speed_mps, "target_speed_mps")
        if not math.isclose(
            speed, _ROUTE_SPEED_MPS[self.route], rel_tol=0.0, abs_tol=1e-9
        ):
            raise Phase1EvidenceError(
                f"route {self.route!r} requires target speed "
                f"{_ROUTE_SPEED_MPS[self.route]:.2f} m/s, got {speed}")
        _finite_float(self.mean_task_return, "mean_task_return")
        velocity_error = _finite_float(
            self.forward_velocity_error_mps, "forward_velocity_error_mps")
        if velocity_error < 0.0:
            raise Phase1EvidenceError(
                "forward_velocity_error_mps must be nonnegative")


@dataclass(frozen=True)
class RouteSpec:
    """Fixed, preregistered provenance and exposure for one Phase 1 route."""

    route: str
    expected_seeds: tuple[int, ...] = CONFIRMATION_SEEDS
    expected_exposure_policy_steps: int = 500_000
    starts_from_zero: bool = False
    independently_finetuned_target_actor: bool = False
    placebo_matching_verified: bool = False

    def validate(self) -> None:
        if self.route not in PHASE1_ROUTES:
            raise Phase1EvidenceError(f"unknown Phase 1 route {self.route!r}")
        if not self.expected_seeds:
            raise Phase1EvidenceError("expected_seeds must not be empty")
        if any(not _is_integer(seed) for seed in self.expected_seeds):
            raise Phase1EvidenceError("expected_seeds must contain only integers")
        if len(set(self.expected_seeds)) != len(self.expected_seeds):
            raise Phase1EvidenceError("expected_seeds contains duplicates")
        if not _is_integer(self.expected_exposure_policy_steps) or (
            self.expected_exposure_policy_steps <= 0
        ):
            raise Phase1EvidenceError(
                "expected_exposure_policy_steps must be a positive integer")
        for name in (
            "starts_from_zero",
            "independently_finetuned_target_actor",
            "placebo_matching_verified",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise Phase1EvidenceError(f"{name} must be boolean")

    @property
    def actor_provenance_verified(self) -> bool:
        if self.route == "fresh_030":
            return bool(self.starts_from_zero)
        return bool(self.independently_finetuned_target_actor)


@dataclass(frozen=True)
class OnlineGateThresholds:
    min_relative_fall_reduction: float = 0.20
    min_absolute_falls_per_1000_reduction: float = 0.40
    min_reduction_ci_low: float = 0.0
    min_treatment_vs_placebo_reduction_ci_low: float = 0.0
    min_return_ratio: float = 0.95
    max_forward_velocity_error_increase_mps: float = 0.03
    max_runtime_deadline_miss_rate: float = 0.001
    max_exact_label_swap_p_value: float = 0.05

    def validate(self) -> None:
        values = {
            name: _finite_float(value, name)
            for name, value in asdict(self).items()
        }
        if not 0.0 <= values["min_relative_fall_reduction"] <= 1.0:
            raise Phase1EvidenceError(
                "min_relative_fall_reduction must lie in [0, 1]")
        for name in (
            "min_absolute_falls_per_1000_reduction",
            "min_return_ratio",
            "max_forward_velocity_error_increase_mps",
            "max_runtime_deadline_miss_rate",
        ):
            if values[name] < 0.0:
                raise Phase1EvidenceError(f"{name} must be nonnegative")
        if not 0.0 <= values["max_exact_label_swap_p_value"] <= 1.0:
            raise Phase1EvidenceError(
                "max_exact_label_swap_p_value must lie in [0, 1]")
        if values["max_runtime_deadline_miss_rate"] > 1.0:
            raise Phase1EvidenceError(
                "max_runtime_deadline_miss_rate must lie in [0, 1]")


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    confidence: float


@dataclass(frozen=True)
class RouteEvidence:
    route: str
    seeds: tuple[int, ...]
    exposure_policy_steps_per_seed_arm: int
    fall_count: Mapping[str, int]
    total_exposure_policy_steps: Mapping[str, int]
    fall_rate_per_1000: Mapping[str, float]
    mean_task_return: Mapping[str, float]
    forward_velocity_error_mps: Mapping[str, float]
    deadline_miss_rate: Mapping[str, float]
    absolute_fall_reduction_per_1000: ConfidenceInterval
    relative_fall_reduction: ConfidenceInterval
    treatment_vs_placebo_reduction_per_1000: ConfidenceInterval
    exact_paired_label_swap_p_value: float
    exact_label_swap_permutations: int
    return_ratio: float
    forward_velocity_error_increase_mps: float
    treatment_deadline_miss_rate: float
    gate_checks: Mapping[str, bool]
    route_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CommonGateStatus:
    """Common evidence gates that are shared by every online claim route."""

    data_gate: bool
    label_reliability_gate: bool
    mechanics_gate: bool
    model_gate: bool
    paired_closed_loop_gate: bool

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (bool, np.bool_)):
                raise Phase1EvidenceError(f"common gate {name} must be boolean")


@dataclass(frozen=True)
class Phase1EvidenceDecision:
    common_gate_checks: Mapping[str, bool]
    common_mechanism_gates: bool
    fresh_030_online: bool
    shift_027_online: bool
    shift_033_online: bool
    small_shift_online: bool
    online_route_expression: bool
    phase1_pass: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _coerce_runs(records: Iterable[OnlineRun | Mapping[str, Any]]) -> list[OnlineRun]:
    runs: list[OnlineRun] = []
    for record in records:
        if isinstance(record, OnlineRun):
            run = record
        elif isinstance(record, Mapping):
            run = OnlineRun.from_mapping(record)
        else:
            raise Phase1EvidenceError(
                "online records must be OnlineRun instances or mappings")
        run.validate()
        runs.append(run)
    if not runs:
        raise Phase1EvidenceError("online evidence table must not be empty")
    return runs


def _validated_table(
    records: Iterable[OnlineRun | Mapping[str, Any]], spec: RouteSpec
) -> dict[str, np.ndarray]:
    spec.validate()
    runs = _coerce_runs(records)
    expected_seed_set = set(spec.expected_seeds)
    observed_seed_set = {int(run.training_seed) for run in runs}
    if observed_seed_set != expected_seed_set:
        missing = sorted(expected_seed_set - observed_seed_set)
        extra = sorted(observed_seed_set - expected_seed_set)
        raise Phase1EvidenceError(
            f"training seed set mismatch: missing={missing}, extra={extra}")

    by_key: dict[tuple[int, str], OnlineRun] = {}
    for run in runs:
        if run.route != spec.route:
            raise Phase1EvidenceError(
                f"route mismatch: spec={spec.route!r}, record={run.route!r}")
        key = (int(run.training_seed), run.arm)
        if key in by_key:
            raise Phase1EvidenceError(
                f"duplicate online arm for seed={key[0]}, arm={key[1]!r}")
        if run.exposure_policy_steps != spec.expected_exposure_policy_steps:
            raise Phase1EvidenceError(
                f"seed={run.training_seed}, arm={run.arm!r} has exposure "
                f"{run.exposure_policy_steps}, expected "
                f"{spec.expected_exposure_policy_steps}")
        by_key[key] = run

    required_keys = set(itertools.product(spec.expected_seeds, ARMS))
    observed_keys = set(by_key)
    if observed_keys != required_keys:
        missing = sorted(required_keys - observed_keys)
        extra = sorted(observed_keys - required_keys)
        raise Phase1EvidenceError(
            f"three-arm table is incomplete: missing={missing}, extra={extra}")

    shape = (len(spec.expected_seeds), len(ARMS))
    falls = np.empty(shape, dtype=np.float64)
    exposure = np.empty(shape, dtype=np.float64)
    task_return = np.empty(shape, dtype=np.float64)
    velocity_error = np.empty(shape, dtype=np.float64)
    deadline_misses = np.empty(shape, dtype=np.float64)
    for seed_index, seed in enumerate(spec.expected_seeds):
        for arm_index, arm in enumerate(ARMS):
            run = by_key[(seed, arm)]
            falls[seed_index, arm_index] = run.falls
            exposure[seed_index, arm_index] = run.exposure_policy_steps
            task_return[seed_index, arm_index] = run.mean_task_return
            velocity_error[seed_index, arm_index] = run.forward_velocity_error_mps
            deadline_misses[seed_index, arm_index] = run.deadline_misses
    return {
        "falls": falls,
        "exposure": exposure,
        "task_return": task_return,
        "velocity_error": velocity_error,
        "deadline_misses": deadline_misses,
    }


def _pooled_rate_per_1000(falls: np.ndarray, exposure: np.ndarray) -> float:
    return float(1000.0 * np.sum(falls) / np.sum(exposure))


def _weighted_average(values: np.ndarray, exposure: np.ndarray) -> float:
    return float(np.sum(values * exposure) / np.sum(exposure))


def _bootstrap_intervals(
    falls: np.ndarray,
    exposure: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[ConfidenceInterval, ConfidenceInterval, ConfidenceInterval]:
    if not _is_integer(replicates) or replicates <= 0:
        raise Phase1EvidenceError("bootstrap_replicates must be a positive integer")
    if not _is_integer(seed) or seed < 0:
        raise Phase1EvidenceError("bootstrap_seed must be a nonnegative integer")
    confidence = _finite_float(confidence, "bootstrap_confidence")
    if not 0.0 < confidence < 1.0:
        raise Phase1EvidenceError("bootstrap_confidence must lie in (0, 1)")

    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    baseline = arm_index["baseline"]
    treatment = arm_index["treatment"]
    placebo = arm_index["placebo"]
    point_rates = np.asarray([
        _pooled_rate_per_1000(falls[:, index], exposure[:, index])
        for index in range(len(ARMS))
    ])
    point_absolute = float(point_rates[baseline] - point_rates[treatment])
    point_relative = (
        float(point_absolute / point_rates[baseline])
        if point_rates[baseline] > 0.0 else float("nan")
    )
    point_placebo = float(point_rates[placebo] - point_rates[treatment])

    # One sampled index selects the complete baseline/treatment/placebo cluster
    # for a training seed.  Sampling arm rows independently would be invalid.
    rng = np.random.default_rng(int(seed))
    sampled_seed = rng.integers(
        0, falls.shape[0], size=(int(replicates), falls.shape[0]))
    sampled_falls = falls[sampled_seed, :].sum(axis=1)
    sampled_exposure = exposure[sampled_seed, :].sum(axis=1)
    rates = 1000.0 * sampled_falls / sampled_exposure
    absolute = rates[:, baseline] - rates[:, treatment]
    relative = np.divide(
        absolute,
        rates[:, baseline],
        out=np.full_like(absolute, np.nan),
        where=rates[:, baseline] > 0.0,
    )
    versus_placebo = rates[:, placebo] - rates[:, treatment]
    alpha = 1.0 - confidence

    def interval(estimate: float, draws: np.ndarray) -> ConfidenceInterval:
        finite = np.asarray(draws, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return ConfidenceInterval(estimate, float("nan"), float("nan"), confidence)
        low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
        return ConfidenceInterval(estimate, float(low), float(high), confidence)

    return (
        interval(point_absolute, absolute),
        interval(point_relative, relative),
        interval(point_placebo, versus_placebo),
    )


def exact_paired_label_swap_p_value(
    baseline_falls: np.ndarray,
    treatment_falls: np.ndarray,
    exposure: np.ndarray,
) -> tuple[float, int]:
    """Exact one-sided paired randomization p-value for fewer treatment falls."""

    baseline = np.asarray(baseline_falls, dtype=np.float64).reshape(-1)
    treatment = np.asarray(treatment_falls, dtype=np.float64).reshape(-1)
    paired_exposure = np.asarray(exposure, dtype=np.float64).reshape(-1)
    if not (len(baseline) == len(treatment) == len(paired_exposure) > 0):
        raise Phase1EvidenceError("paired label-swap arrays must be nonempty and aligned")
    if len(baseline) > 20:
        raise Phase1EvidenceError(
            "exact label-swap enumeration is limited to at most 20 paired seeds")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(treatment)):
        raise Phase1EvidenceError("fall counts must be finite")
    if not np.all(np.isfinite(paired_exposure)) or np.any(paired_exposure <= 0.0):
        raise Phase1EvidenceError("paired exposure must be finite and positive")
    if (
        np.any(baseline < 0.0)
        or np.any(treatment < 0.0)
        or np.any(baseline > paired_exposure)
        or np.any(treatment > paired_exposure)
        or np.any(baseline != np.floor(baseline))
        or np.any(treatment != np.floor(treatment))
    ):
        raise Phase1EvidenceError(
            "paired fall counts must be nonnegative integers no greater than exposure")

    difference = baseline - treatment
    observed = float(1000.0 * np.sum(difference) / np.sum(paired_exposure))
    permutations = 1 << len(difference)
    indices = np.arange(permutations, dtype=np.uint64)[:, None]
    bit = np.arange(len(difference), dtype=np.uint64)[None, :]
    sign = np.where(((indices >> bit) & 1) == 0, 1.0, -1.0)
    permuted = 1000.0 * (sign @ difference) / np.sum(paired_exposure)
    tolerance = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(observed))
    p_value = float(np.count_nonzero(permuted >= observed - tolerance) / permutations)
    return p_value, permutations


def evaluate_online_route(
    records: Iterable[OnlineRun | Mapping[str, Any]],
    spec: RouteSpec,
    *,
    thresholds: OnlineGateThresholds = OnlineGateThresholds(),
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence: float = 0.95,
) -> RouteEvidence:
    """Validate and evaluate one fixed-exposure Phase 1 online claim route."""

    thresholds.validate()
    table = _validated_table(records, spec)
    falls = table["falls"]
    exposure = table["exposure"]
    arm_index = {arm: index for index, arm in enumerate(ARMS)}

    fall_rate = {
        arm: _pooled_rate_per_1000(
            falls[:, arm_index[arm]], exposure[:, arm_index[arm]])
        for arm in ARMS
    }
    fall_count = {
        arm: int(np.sum(falls[:, arm_index[arm]])) for arm in ARMS
    }
    total_exposure = {
        arm: int(np.sum(exposure[:, arm_index[arm]])) for arm in ARMS
    }
    mean_task_return = {
        arm: _weighted_average(
            table["task_return"][:, arm_index[arm]],
            exposure[:, arm_index[arm]],
        )
        for arm in ARMS
    }
    velocity_error = {
        arm: _weighted_average(
            table["velocity_error"][:, arm_index[arm]],
            exposure[:, arm_index[arm]],
        )
        for arm in ARMS
    }
    deadline_miss_rate = {
        arm: float(
            np.sum(table["deadline_misses"][:, arm_index[arm]])
            / np.sum(exposure[:, arm_index[arm]]))
        for arm in ARMS
    }
    absolute_ci, relative_ci, placebo_ci = _bootstrap_intervals(
        falls,
        exposure,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence=bootstrap_confidence,
    )
    baseline = arm_index["baseline"]
    treatment = arm_index["treatment"]
    exact_p, exact_permutations = exact_paired_label_swap_p_value(
        falls[:, baseline], falls[:, treatment], exposure[:, baseline])

    baseline_return = mean_task_return["baseline"]
    treatment_return = mean_task_return["treatment"]
    return_ratio = (
        float(treatment_return / baseline_return)
        if baseline_return > 0.0 else float("nan")
    )
    baseline_velocity_error = velocity_error["baseline"]
    treatment_velocity_error = velocity_error["treatment"]
    velocity_error_increase = float(
        treatment_velocity_error - baseline_velocity_error)
    treatment_deadline_miss_rate = deadline_miss_rate["treatment"]

    checks = {
        "fixed_seed_arm_exposure_complete": True,
        "confirmation_seed_set": bool(
            len(spec.expected_seeds) == len(CONFIRMATION_SEEDS)
            and set(spec.expected_seeds) == set(CONFIRMATION_SEEDS)),
        "confirmation_fixed_exposure": bool(
            spec.expected_exposure_policy_steps == 500_000),
        "actor_provenance_verified": spec.actor_provenance_verified,
        "placebo_matching_verified": bool(spec.placebo_matching_verified),
        "relative_fall_reduction": bool(
            math.isfinite(relative_ci.estimate)
            and relative_ci.estimate >= thresholds.min_relative_fall_reduction),
        "absolute_fall_reduction_per_1000": bool(
            absolute_ci.estimate
            >= thresholds.min_absolute_falls_per_1000_reduction),
        "absolute_fall_reduction_ci_low": bool(
            absolute_ci.low > thresholds.min_reduction_ci_low),
        "treatment_vs_placebo_reduction_ci_low": bool(
            placebo_ci.low
            > thresholds.min_treatment_vs_placebo_reduction_ci_low),
        "exact_paired_label_swap": bool(
            exact_p <= thresholds.max_exact_label_swap_p_value),
        "task_return_ratio": bool(
            math.isfinite(return_ratio)
            and return_ratio >= thresholds.min_return_ratio),
        "forward_velocity_error_increase": bool(
            velocity_error_increase
            <= thresholds.max_forward_velocity_error_increase_mps),
        "runtime_deadline_miss_rate": bool(
            treatment_deadline_miss_rate
            <= thresholds.max_runtime_deadline_miss_rate),
    }
    return RouteEvidence(
        route=spec.route,
        seeds=tuple(int(seed) for seed in spec.expected_seeds),
        exposure_policy_steps_per_seed_arm=int(
            spec.expected_exposure_policy_steps),
        fall_count=fall_count,
        total_exposure_policy_steps=total_exposure,
        fall_rate_per_1000=fall_rate,
        mean_task_return=mean_task_return,
        forward_velocity_error_mps=velocity_error,
        deadline_miss_rate=deadline_miss_rate,
        absolute_fall_reduction_per_1000=absolute_ci,
        relative_fall_reduction=relative_ci,
        treatment_vs_placebo_reduction_per_1000=placebo_ci,
        exact_paired_label_swap_p_value=exact_p,
        exact_label_swap_permutations=exact_permutations,
        return_ratio=return_ratio,
        forward_velocity_error_increase_mps=velocity_error_increase,
        treatment_deadline_miss_rate=treatment_deadline_miss_rate,
        gate_checks=checks,
        route_pass=all(checks.values()),
    )


def compile_phase1_evidence(
    common_gates: CommonGateStatus,
    route_evidence: Mapping[str, RouteEvidence],
) -> Phase1EvidenceDecision:
    """Compile the preregistered Phase 1 route expression, and nothing else."""

    common_gates.validate()
    unknown = set(route_evidence) - set(PHASE1_ROUTES)
    if unknown:
        raise Phase1EvidenceError(
            f"unknown route evidence supplied: {sorted(unknown)}")
    for route, evidence in route_evidence.items():
        if not isinstance(evidence, RouteEvidence):
            raise Phase1EvidenceError(
                f"route evidence for {route!r} must be a RouteEvidence")
        if route != evidence.route:
            raise Phase1EvidenceError(
                f"route evidence key {route!r} does not match {evidence.route!r}")

    common_checks = {
        name: bool(value) for name, value in asdict(common_gates).items()
    }
    common_pass = all(common_checks.values())
    route_pass = {
        route: bool(route_evidence.get(route).route_pass)
        if route in route_evidence else False
        for route in PHASE1_ROUTES
    }
    small_shift = route_pass["shift_027"] and route_pass["shift_033"]
    online_expression = route_pass["fresh_030"] or small_shift
    return Phase1EvidenceDecision(
        common_gate_checks=common_checks,
        common_mechanism_gates=common_pass,
        fresh_030_online=route_pass["fresh_030"],
        shift_027_online=route_pass["shift_027"],
        shift_033_online=route_pass["shift_033"],
        small_shift_online=small_shift,
        online_route_expression=online_expression,
        phase1_pass=bool(common_pass and online_expression),
    )


__all__ = [
    "ARMS",
    "CONFIRMATION_SEEDS",
    "PHASE1_ROUTES",
    "CommonGateStatus",
    "ConfidenceInterval",
    "OnlineGateThresholds",
    "OnlineRun",
    "Phase1EvidenceDecision",
    "Phase1EvidenceError",
    "RouteEvidence",
    "RouteSpec",
    "compile_phase1_evidence",
    "evaluate_online_route",
    "exact_paired_label_swap_p_value",
]
