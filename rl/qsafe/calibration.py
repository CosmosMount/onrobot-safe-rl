"""Calibration-only grid search for a locked repeated Q_safe selector.

Safety labels are used only from the declared development calibration split.
The output is a single :class:`SelectorConfig` (or an explicit abstain/fail
result) that can be evaluated once on held-out model and paired-closed-loop
data.  Candidate rows remain grouped and trajectory clusters are resampled as
whole units for the calibration confidence interval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

from rl.qsafe.selector import CandidateBatch, SelectorConfig, select_candidate


@dataclass(frozen=True)
class SelectorCalibrationInputs:
    member_risk: np.ndarray
    empirical_risk: np.ndarray
    requested: np.ndarray
    executed: np.ndarray
    q_target: np.ndarray
    reward_q: np.ndarray
    candidate_mask: np.ndarray
    acceptance_probability: np.ndarray
    trajectory_id: np.ndarray
    source_seed: np.ndarray

    def validated(self) -> "SelectorCalibrationInputs":
        risk = np.asarray(self.member_risk, dtype=np.float64)
        empirical = np.asarray(self.empirical_risk, dtype=np.float64)
        if risk.ndim != 3 or risk.shape[0] < 2:
            raise ValueError("member_risk must have shape [M>=2,G,K]")
        members, groups, candidates = risk.shape
        if groups == 0 or candidates != 16:
            raise ValueError("selector calibration requires nonempty K=16 groups")
        if empirical.shape != (groups, candidates):
            raise ValueError("empirical_risk must have shape [G,K]")
        arrays = {}
        for name in ("requested", "executed", "q_target"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (groups, candidates, 12):
                raise ValueError(f"{name} must have shape [G,16,12]")
            arrays[name] = value.copy()
        reward = np.asarray(self.reward_q, dtype=np.float64)
        if reward.shape != (groups, candidates):
            raise ValueError("reward_q must have shape [G,K]")
        raw_mask = np.asarray(self.candidate_mask)
        if raw_mask.shape != (groups, candidates) or raw_mask.dtype.kind not in "biu" or (
                not np.all(np.isin(raw_mask, (0, 1, False, True)))):
            raise ValueError("candidate_mask must be binary [G,K]")
        mask = raw_mask.astype(bool)
        if not np.all(mask[:, 0]) or np.any(mask.sum(axis=1) < 8):
            raise ValueError("every calibration group requires nominal plus K>=8 support")
        if not np.all(np.isfinite(risk)) or np.any((risk < 0.0) | (risk > 1.0)):
            raise ValueError("member_risk must be finite probabilities")
        if not np.all(np.isfinite(empirical[mask])) or np.any(
                (empirical[mask] < 0.0) | (empirical[mask] > 1.0)):
            raise ValueError("valid empirical risks must be finite probabilities")
        if not all(np.all(np.isfinite(value)) for value in (
                *arrays.values(), reward)):
            raise ValueError("selector action/reward inputs must be finite")
        for name in ("requested", "executed"):
            if np.any(arrays[name] < -1.0 - 1e-6) or np.any(
                    arrays[name] > 1.0 + 1e-6):
                raise ValueError(f"{name} must lie in normalized [-1,1]")
        probability = np.asarray(
            self.acceptance_probability, dtype=np.float64).reshape(-1)
        trajectory = np.asarray(self.trajectory_id).astype(str).reshape(-1)
        source_seed = np.asarray(self.source_seed).reshape(-1)
        if probability.shape != (groups,) or not np.all(np.isfinite(probability)) or (
                np.any(probability <= 0.0) or np.any(probability > 1.0)):
            raise ValueError("acceptance_probability must be finite [G] in (0,1]")
        if trajectory.shape != (groups,) or np.any(trajectory == ""):
            raise ValueError("trajectory_id must be nonempty [G]")
        if source_seed.shape != (groups,) or source_seed.dtype.kind not in "iu" or (
                np.any(source_seed < 0)):
            raise ValueError("source_seed must be nonnegative integer [G]")
        if np.any(source_seed.astype(np.uint64) > np.iinfo(np.int64).max):
            raise ValueError("source_seed exceeds the supported int64 range")
        return SelectorCalibrationInputs(
            member_risk=risk.copy(),
            empirical_risk=empirical.copy(),
            requested=arrays["requested"],
            executed=arrays["executed"],
            q_target=arrays["q_target"],
            reward_q=reward.copy(),
            candidate_mask=mask.copy(),
            acceptance_probability=probability.copy(),
            trajectory_id=trajectory.copy(),
            source_seed=source_seed.astype(np.int64, copy=True),
        )


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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _grid_values(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"selector grid {name} must be a nonempty sequence")
    result = tuple(_finite(item, f"selector grid {name}") for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"selector grid {name} contains duplicates")
    return result


@dataclass(frozen=True)
class SelectorCalibrationSpec:
    min_independent_groups: int
    min_trajectory_clusters: int
    min_source_seeds: int
    require_calibration_absolute_reduction: float
    require_calibration_reduction_ci_low: float
    max_intervention_rate: float
    uncertainty_beta: float
    max_epistemic_std: float
    max_action_delta_rms: float
    max_q_target_delta_rms: float
    nominal_risk_lcb_trigger: tuple[float, ...]
    min_benefit_lcb: tuple[float, ...]
    max_risk_ucb: tuple[float, ...]
    reward_q_margin: tuple[float, ...]

    @classmethod
    def from_protocol(cls, value: Mapping[str, Any]) -> "SelectorCalibrationSpec":
        if not isinstance(value, Mapping):
            raise ValueError("selector_calibration protocol must be a mapping")
        if value.get("split_role") != "development_calibration_only" or (
                value.get("test_policy") != "evaluate_locked_configuration_once") or (
                value.get("no_feasible_configuration")
                != "abstain_and_fail_paired_gate"):
            raise ValueError("selector calibration split/failure policy is not locked")
        if value.get("empirical_weighting") != (
                "inverse_acceptance_probability_group_macro") or value.get(
                    "objective") != "maximize_empirical_absolute_fall_reduction":
            raise ValueError("selector calibration estimand/objective is not locked")
        if value.get("tie_break_order") != [
            "larger_calibration_ci_low",
            "larger_calibration_reduction",
            "lower_intervention_rate",
            "smaller_reward_q_margin",
            "larger_min_benefit_lcb",
            "smaller_nominal_risk_trigger",
            "smaller_max_risk_ucb",
        ]:
            raise ValueError("selector calibration tie-break order is not locked")
        fixed = value.get("fixed_gates")
        grid = value.get("grid")
        if not isinstance(fixed, Mapping) or not isinstance(grid, Mapping):
            raise ValueError("selector calibration fixed_gates/grid are required")
        return cls(
            min_independent_groups=_positive_int(
                value.get("min_independent_groups"), "min_independent_groups"),
            min_trajectory_clusters=_positive_int(
                value.get("min_trajectory_clusters"), "min_trajectory_clusters"),
            min_source_seeds=_positive_int(
                value.get("min_source_seeds"), "min_source_seeds"),
            require_calibration_absolute_reduction=_finite(
                value.get("require_calibration_absolute_reduction"),
                "require_calibration_absolute_reduction"),
            require_calibration_reduction_ci_low=_finite(
                value.get("require_calibration_reduction_ci_low"),
                "require_calibration_reduction_ci_low"),
            max_intervention_rate=_finite(
                value.get("max_intervention_rate"), "max_intervention_rate"),
            uncertainty_beta=_finite(
                value.get("uncertainty_beta"), "uncertainty_beta"),
            max_epistemic_std=_finite(
                fixed.get("max_epistemic_std"), "max_epistemic_std"),
            max_action_delta_rms=_finite(
                fixed.get("max_action_delta_rms"), "max_action_delta_rms"),
            max_q_target_delta_rms=_finite(
                fixed.get("max_q_target_delta_rms"),
                "max_q_target_delta_rms"),
            nominal_risk_lcb_trigger=_grid_values(
                grid.get("nominal_risk_lcb_trigger"),
                "nominal_risk_lcb_trigger"),
            min_benefit_lcb=_grid_values(
                grid.get("min_benefit_lcb"), "min_benefit_lcb"),
            max_risk_ucb=_grid_values(
                grid.get("max_risk_ucb"), "max_risk_ucb"),
            reward_q_margin=_grid_values(
                grid.get("reward_q_margin"), "reward_q_margin"),
        ).validated()

    def validated(self) -> "SelectorCalibrationSpec":
        for name in (
            "require_calibration_absolute_reduction",
            "require_calibration_reduction_ci_low",
            "max_intervention_rate",
            "uncertainty_beta",
            "max_epistemic_std",
            "max_action_delta_rms",
            "max_q_target_delta_rms",
        ):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.max_intervention_rate > 1.0:
            raise ValueError("max_intervention_rate must not exceed one")
        if self.require_calibration_absolute_reduction > 1.0 or (
                self.require_calibration_reduction_ci_low > 1.0):
            raise ValueError("calibration reduction thresholds must not exceed one")
        return self

    def configurations(self) -> list[SelectorConfig]:
        return [
            SelectorConfig(
                nominal_risk_lcb_trigger=trigger,
                min_benefit_lcb=benefit,
                max_risk_ucb=max_risk,
                max_epistemic_std=self.max_epistemic_std,
                max_action_delta_rms=self.max_action_delta_rms,
                max_q_target_delta_rms=self.max_q_target_delta_rms,
                reward_q_margin=reward_margin,
                uncertainty_beta=self.uncertainty_beta,
            )
            for trigger, benefit, max_risk, reward_margin in itertools.product(
                self.nominal_risk_lcb_trigger,
                self.min_benefit_lcb,
                self.max_risk_ucb,
                self.reward_q_margin,
            )
        ]


@dataclass(frozen=True)
class SelectorCalibrationRow:
    config: SelectorConfig
    absolute_fall_reduction: float
    reduction_ci95_low: float
    reduction_ci95_high: float
    intervention_rate: float
    selected_empirical_fall_risk: float
    nominal_empirical_fall_risk: float
    selection_reason_counts: Mapping[str, int]
    feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "absolute_fall_reduction": self.absolute_fall_reduction,
            "reduction_ci95": [self.reduction_ci95_low, self.reduction_ci95_high],
            "intervention_rate": self.intervention_rate,
            "selected_empirical_fall_risk": self.selected_empirical_fall_risk,
            "nominal_empirical_fall_risk": self.nominal_empirical_fall_risk,
            "selection_reason_counts": dict(self.selection_reason_counts),
            "feasible": self.feasible,
        }


@dataclass(frozen=True)
class SelectorCalibrationResult:
    selector_config: SelectorConfig | None
    feasible: bool
    selected_row_index: int | None
    group_count: int
    trajectory_clusters: int
    source_seeds: tuple[int, ...]
    grid_configurations: int
    bootstrap_replicates: int
    bootstrap_seed: int
    rows: tuple[SelectorCalibrationRow, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_config": (
                None if self.selector_config is None
                else asdict(self.selector_config)),
            "feasible": self.feasible,
            "selected_row_index": self.selected_row_index,
            "group_count": self.group_count,
            "trajectory_clusters": self.trajectory_clusters,
            "source_seeds": list(self.source_seeds),
            "grid_configurations": self.grid_configurations,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "rows": [row.to_dict() for row in self.rows],
        }


def _bootstrap_group_counts(
    trajectory_id: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    unique, cluster_index = np.unique(trajectory_id, return_inverse=True)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(unique), size=(replicates, len(unique)), dtype=np.int64)
    counts = np.zeros((replicates, len(unique)), dtype=np.float64)
    rows = np.repeat(np.arange(replicates), len(unique))
    np.add.at(counts, (rows, sampled.reshape(-1)), 1.0)
    return counts[:, cluster_index]


def calibrate_selector(
    inputs: SelectorCalibrationInputs,
    spec: SelectorCalibrationSpec,
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260809,
) -> SelectorCalibrationResult:
    data = inputs.validated()
    spec = spec.validated()
    replicates = _positive_int(bootstrap_replicates, "bootstrap_replicates")
    seed = _nonnegative_int(bootstrap_seed, "bootstrap_seed")
    groups = data.empirical_risk.shape[0]
    trajectory_count = len(np.unique(data.trajectory_id))
    sources = tuple(sorted(set(map(int, data.source_seed))))
    if groups < spec.min_independent_groups:
        raise ValueError("selector calibration has too few independent groups")
    if trajectory_count < spec.min_trajectory_clusters:
        raise ValueError("selector calibration has too few trajectory clusters")
    if len(sources) < spec.min_source_seeds:
        raise ValueError("selector calibration has too few source seeds")

    configurations = spec.configurations()
    weights = data.acceptance_probability.min() / data.acceptance_probability
    weight_sum = float(weights.sum())
    nominal_risk = data.empirical_risk[:, 0]
    effects = np.empty((len(configurations), groups), dtype=np.float64)
    interventions = np.empty_like(effects)
    selected_risk = np.empty_like(effects)
    reason_tables: list[dict[str, int]] = []
    for config_index, config in enumerate(configurations):
        reasons: dict[str, int] = {}
        for group in range(groups):
            selection = select_candidate(
                data.member_risk[:, group, :],
                CandidateBatch(
                    requested=data.requested[group],
                    executed=data.executed[group],
                    q_target=data.q_target[group],
                    reward_q=data.reward_q[group],
                    mask=data.candidate_mask[group],
                ),
                config,
            )
            index = selection.selected_index
            selected_risk[config_index, group] = data.empirical_risk[group, index]
            effects[config_index, group] = (
                nominal_risk[group] - selected_risk[config_index, group])
            interventions[config_index, group] = float(selection.intervened)
            reasons[selection.reason] = reasons.get(selection.reason, 0) + 1
        reason_tables.append(reasons)

    point_effect = (effects * weights[None, :]).sum(axis=1) / weight_sum
    point_intervention = (
        interventions * weights[None, :]).sum(axis=1) / weight_sum
    point_selected_risk = (
        selected_risk * weights[None, :]).sum(axis=1) / weight_sum
    point_nominal_risk = float(np.sum(nominal_risk * weights) / weight_sum)
    bootstrap_counts = _bootstrap_group_counts(
        data.trajectory_id, replicates=replicates, seed=seed)
    bootstrap_weights = bootstrap_counts * weights[None, :]
    denominator = bootstrap_weights.sum(axis=1)
    if np.any(denominator <= 0.0):
        raise RuntimeError("trajectory bootstrap generated an empty replicate")
    draws = (
        bootstrap_weights @ effects.T) / denominator[:, None]
    ci_low, ci_high = np.quantile(draws, [0.025, 0.975], axis=0)

    rows = []
    for index, config in enumerate(configurations):
        feasible = bool(
            point_effect[index]
            >= spec.require_calibration_absolute_reduction
            and ci_low[index] > spec.require_calibration_reduction_ci_low
            and point_intervention[index] <= spec.max_intervention_rate)
        rows.append(SelectorCalibrationRow(
            config=config,
            absolute_fall_reduction=float(point_effect[index]),
            reduction_ci95_low=float(ci_low[index]),
            reduction_ci95_high=float(ci_high[index]),
            intervention_rate=float(point_intervention[index]),
            selected_empirical_fall_risk=float(point_selected_risk[index]),
            nominal_empirical_fall_risk=point_nominal_risk,
            selection_reason_counts=reason_tables[index],
            feasible=feasible,
        ))
    feasible_indices = [index for index, row in enumerate(rows) if row.feasible]
    selected_index = None
    selected_config = None
    if feasible_indices:
        selected_index = min(feasible_indices, key=lambda index: (
            -rows[index].reduction_ci95_low,
            -rows[index].absolute_fall_reduction,
            rows[index].intervention_rate,
            rows[index].config.reward_q_margin,
            -rows[index].config.min_benefit_lcb,
            rows[index].config.nominal_risk_lcb_trigger,
            rows[index].config.max_risk_ucb,
        ))
        selected_config = rows[selected_index].config
    return SelectorCalibrationResult(
        selector_config=selected_config,
        feasible=selected_config is not None,
        selected_row_index=selected_index,
        group_count=groups,
        trajectory_clusters=trajectory_count,
        source_seeds=sources,
        grid_configurations=len(configurations),
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        rows=tuple(rows),
    )


__all__ = [
    "SelectorCalibrationInputs",
    "SelectorCalibrationResult",
    "SelectorCalibrationRow",
    "SelectorCalibrationSpec",
    "calibrate_selector",
]
