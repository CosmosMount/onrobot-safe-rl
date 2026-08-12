"""Independent-oracle analysis for preregistered PPO short options."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FAMILIES = {"L1": np.arange(1, 6), "L4": np.arange(6, 11),
            "L8": np.arange(11, 16)}


@dataclass(frozen=True)
class Bootstrap:
    replicates: int = 10_000
    seed: int = 20260813


def _summary(values: np.ndarray, bootstrap: Bootstrap) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap requires a finite state-level vector")
    rng = np.random.default_rng(bootstrap.seed)
    estimates = np.empty(bootstrap.replicates, np.float64)
    for start in range(0, bootstrap.replicates, 512):
        stop = min(start + 512, bootstrap.replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        estimates[start:stop] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "one_sided_95_lcb": float(np.quantile(estimates, 0.05)),
        "two_sided_95_ci_low": float(np.quantile(estimates, 0.025)),
        "two_sided_95_ci_high": float(np.quantile(estimates, 0.975)),
    }


def _state_ordering(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states, candidates = first.shape
    agreement = np.zeros(states, np.float64)
    reproducible = np.zeros(states, bool)
    for state in range(states):
        scores: list[float] = []
        both_nontie = 0
        for left in range(candidates):
            for right in range(left + 1, candidates):
                first_sign = np.sign(first[state, left] - first[state, right])
                second_sign = np.sign(second[state, left] - second[state, right])
                if first_sign == 0 or second_sign == 0:
                    scores.append(0.5)
                else:
                    both_nontie += 1
                    scores.append(float(first_sign == second_sign))
        agreement[state] = np.mean(scores)
        reproducible[state] = both_nontie > 0 and agreement[state] > 0.5
    return agreement, reproducible


def _diagnostics(
    array: np.ndarray, selected: np.ndarray, candidate_indices: np.ndarray,
) -> dict[str, float]:
    states = np.arange(len(selected))
    selected_values = array[states, selected, 4:8]
    family_values = array[:, candidate_indices][:, :, 4:8]
    return {
        "selected_evaluation_mean": float(np.mean(selected_values)),
        "selected_evaluation_q95": float(np.quantile(selected_values, 0.95)),
        "family_evaluation_mean": float(np.mean(family_values)),
        "family_evaluation_q95": float(np.quantile(family_values, 0.95)),
    }


def analyze_short_option_oracle(
    *, h96_fall: np.ndarray, candidate_duration: np.ndarray,
    replacement_sum: np.ndarray, replacement_max: np.ndarray,
    projection_saturation_count: np.ndarray,
    joint_limit_saturation_count: np.ndarray,
    active_steps: np.ndarray,
    max_abs_roll: np.ndarray, max_abs_pitch: np.ndarray,
    max_angular_velocity: np.ndarray, min_base_height: np.ndarray,
    bootstrap: Bootstrap = Bootstrap(),
) -> dict[str, object]:
    fall = np.asarray(h96_fall, bool)
    if fall.shape != (600, 16, 8):
        raise ValueError("oracle dataset must be exactly [600,16,8]")
    duration = np.asarray(candidate_duration)
    expected = np.asarray([0] + [1] * 5 + [4] * 5 + [8] * 5)
    if duration.shape != (600, 16) or not np.array_equal(
            duration, np.broadcast_to(expected, duration.shape)):
        raise ValueError("candidate duration layout differs from preregistration")
    diagnostic_arrays = [replacement_sum, replacement_max,
                         projection_saturation_count, joint_limit_saturation_count,
                         active_steps, max_abs_roll, max_abs_pitch,
                         max_angular_velocity, min_base_height]
    if any(np.asarray(value).shape != fall.shape for value in diagnostic_arrays):
        raise ValueError("option diagnostic arrays must match branch outcome shape")
    if np.any(np.asarray(active_steps) <= 0):
        raise ValueError("every branch must execute at least its first action")

    nominal_eval = fall[:, 0, 4:8].mean(axis=1)
    report: dict[str, object] = {}
    reductions: dict[str, np.ndarray] = {}
    for name, nonnominal in FAMILIES.items():
        indices = np.concatenate(([0], nonnominal))
        discovery = fall[:, indices, :4].mean(axis=2)
        evaluation = fall[:, indices, 4:8].mean(axis=2)
        local_selected = np.argmin(discovery, axis=1)
        selected = indices[local_selected]
        selected_eval = evaluation[np.arange(600), local_selected]
        reduction = nominal_eval - selected_eval
        reductions[name] = reduction
        first_half = fall[:, indices, :4].mean(axis=2)
        second_half = fall[:, indices, 4:8].mean(axis=2)
        ordering, reproducible = _state_ordering(first_half, second_half)
        full_risk = fall[:, indices].mean(axis=2)
        spread = np.max(full_risk, axis=1) - np.min(full_risk, axis=1)
        active = np.asarray(active_steps)[np.arange(600), selected, 4:8]
        projection = np.asarray(projection_saturation_count)[
            np.arange(600), selected, 4:8]
        joint_limit = np.asarray(joint_limit_saturation_count)[
            np.arange(600), selected, 4:8]
        report[name] = {
            "nominal_evaluation_fall_rate": float(nominal_eval.mean()),
            "independent_oracle_evaluation_fall_rate": float(selected_eval.mean()),
            "independent_oracle_reduction": _summary(reduction, bootstrap),
            "rescue_states": int(np.sum(reduction > 0)),
            "harm_states": int(np.sum(reduction < 0)),
            "split_half_ordering_agreement": _summary(ordering, bootstrap),
            "reproducible_ordering_state_fraction": _summary(
                reproducible.astype(np.float64), bootstrap),
            "within_state_empirical_risk_range": _summary(spread, bootstrap),
            "replacement_magnitude": {
                "mean_per_active_step": float(np.mean(
                    np.asarray(replacement_sum)[np.arange(600), selected, 4:8]
                    / active)),
                "mean_branch_max": float(np.mean(
                    np.asarray(replacement_max)[np.arange(600), selected, 4:8])),
            },
            "saturation": {
                "projection_joint_fraction": float(np.sum(projection) / np.sum(active * 12)),
                "joint_limit_joint_fraction": float(np.sum(joint_limit) / np.sum(active * 12)),
            },
            "option_stability": {
                "max_abs_roll": _diagnostics(
                    np.asarray(max_abs_roll), selected, indices),
                "max_abs_pitch": _diagnostics(
                    np.asarray(max_abs_pitch), selected, indices),
                "max_angular_velocity": _diagnostics(
                    np.asarray(max_angular_velocity), selected, indices),
                "min_base_height": {
                    **_diagnostics(-np.asarray(min_base_height), selected, indices),
                    "reported_as_negative_for_q95_only": True,
                    "selected_evaluation_actual_mean": float(np.mean(
                        np.asarray(min_base_height)[np.arange(600), selected, 4:8])),
                },
            },
            "selected_candidate_index": selected.tolist(),
        }

    comparisons = {}
    for name in ("L4", "L8"):
        comparisons[f"{name}_minus_L1_oracle_reduction"] = _summary(
            reductions[name] - reductions["L1"], bootstrap)
    passing = []
    for name in ("L4", "L8"):
        item = report[name]
        reduction = item["independent_oracle_reduction"]
        ordering = item["split_half_ordering_agreement"]
        if (reduction["mean"] >= 0.03
                and reduction["one_sided_95_lcb"] > 0
                and ordering["two_sided_95_ci_low"] > 0.5
                and item["rescue_states"] > item["harm_states"]):
            passing.append(name)
    clearly_better = [name for name in passing if comparisons[
        f"{name}_minus_L1_oracle_reduction"]["one_sided_95_lcb"] > 0]
    return {
        "schema_version": "qsafe.short_option_oracle_analysis.v1",
        "states": 600, "candidates": 16, "replicas": 8,
        "duration_families": report,
        "paired_duration_comparisons": comparisons,
        "passing_long_families": passing,
        "long_families_clearly_better_than_L1": clearly_better,
        "short_option_candidate_space_supported": bool(passing),
        "one_step_action_timescale_insufficient": bool(clearly_better),
        "persistent_residual_option_route_stopped": not any(
            report[name]["independent_oracle_reduction"]["mean"] >= 0.03
            for name in ("L4", "L8")),
        "critic_training_authorized": False,
        "protected_outcomes_read_or_generated": False,
        "sac_transfer_run": False,
    }
