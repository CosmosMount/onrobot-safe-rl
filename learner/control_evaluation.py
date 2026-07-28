"""Control-facing evaluation of Q_safe on exact-state action branches."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from jaxrl.agents.safety_critic import binary_prediction_metrics
from learner.counterfactual_dataset import CandidateBranch


def _severity(branch: CandidateBranch, horizon: int) -> float:
    outcome = branch.outcomes[horizon]
    if outcome.failure:
        # Earlier failures are strictly worse while retaining failure > near.
        return 1.0 + 0.25 * (
            horizon - max(outcome.time_to_failure, 1)) / max(horizon, 1)
    return 0.5 if outcome.near_failure else 0.0


def _select_group(indices, branches, risks, support, horizon, epsilon):
    nominal = next(
        index for index in indices
        if branches[index].candidate_family == 'nominal')
    eligible = [
        index for index in indices
        if support[index] and risks[index] <= epsilon
    ]
    # Match the non-invasive selector: an accepted nominal is never replaced.
    if nominal in eligible:
        return nominal, True
    if eligible:
        return min(eligible, key=lambda index: risks[index]), True
    return nominal, False


def _selection_metrics(
        branches: Sequence[CandidateBranch],
        risks: np.ndarray,
        support: np.ndarray,
        groups: dict[int, list[int]],
        *,
        horizon: int,
        epsilon: float,
) -> dict[str, float]:
    selected = []
    nominal = []
    covered = []
    for indices in groups.values():
        selected_index, is_covered = _select_group(
            indices, branches, risks, support, horizon, epsilon)
        selected.append(selected_index)
        nominal.append(next(
            index for index in indices
            if branches[index].candidate_family == 'nominal'))
        covered.append(is_covered)
    selected = np.asarray(selected, dtype=np.int64)
    nominal = np.asarray(nominal, dtype=np.int64)
    covered = np.asarray(covered, dtype=bool)
    labels = np.asarray([
        branches[index].outcomes[horizon].failure for index in selected
    ], dtype=np.float32)
    selected_risks = risks[selected]
    selected_severity = np.asarray([
        _severity(branches[index], horizon) for index in selected
    ])
    nominal_failures = np.asarray([
        branches[index].outcomes[horizon].failure for index in nominal
    ], dtype=np.float32)
    selected_failures = labels
    regrets = []
    for selected_index, indices in zip(selected, groups.values()):
        best = min(_severity(branches[index], horizon) for index in indices)
        regrets.append(_severity(branches[selected_index], horizon) - best)
    calibration = binary_prediction_metrics(labels, selected_risks)
    covered_count = max(int(np.sum(covered)), 1)
    false_safe = float(np.sum(labels[covered]) / covered_count)
    return {
        'control_coverage': float(np.mean(covered)),
        'control_abstention_rate': float(1.0 - np.mean(covered)),
        'control_selected_false_safe_rate': false_safe,
        'control_selected_action_ece': float(
            calibration['Q_safe_calibration_ece']),
        'control_selected_action_brier': float(
            calibration['Q_safe_brier']),
        'control_top1_safety_regret': float(np.mean(regrets)),
        'control_nominal_failure_rate': float(np.mean(nominal_failures)),
        'control_selected_failure_rate': float(np.mean(selected_failures)),
        'control_nominal_relative_failure_reduction': float(np.mean(
            nominal_failures - selected_failures)),
        'control_selected_severity': float(np.mean(selected_severity)),
    }


def evaluate_control_facing(
        branches: Sequence[CandidateBranch],
        predicted_risks,
        *,
        horizon: int = 32,
        epsilon: float = 0.2,
        support=None,
        k_values: Sequence[int] = (4, 8, 16, 32),
        seed: int = 0,
) -> dict[str, object]:
    """Evaluate action ranking/selection against branch rollout outcomes."""
    if not branches:
        raise ValueError('control-facing evaluation needs branch records')
    risks = np.asarray(predicted_risks, dtype=np.float64).reshape(-1)
    if risks.shape[0] != len(branches):
        raise ValueError('predicted_risks must align with branches')
    if support is None:
        support_array = np.ones(len(branches), dtype=bool)
    else:
        support_array = np.asarray(support, dtype=bool).reshape(-1)
        if support_array.shape[0] != len(branches):
            raise ValueError('support must align with branches')
    if any(horizon not in branch.outcomes for branch in branches):
        raise ValueError(f'horizon {horizon} absent from branch artifact')

    groups: dict[int, list[int]] = defaultdict(list)
    for index, branch in enumerate(branches):
        groups[int(branch.snapshot_index)].append(index)
    pair_correct = 0.0
    pair_count = 0
    for indices in groups.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1:]:
                truth_delta = (
                    _severity(branches[left], horizon)
                    - _severity(branches[right], horizon))
                if truth_delta == 0.0:
                    continue
                prediction_delta = risks[left] - risks[right]
                pair_correct += float(
                    prediction_delta * truth_delta > 0.0)
                pair_count += 1
    result: dict[str, object] = _selection_metrics(
        branches, risks, support_array, groups,
        horizon=horizon, epsilon=epsilon)
    result.update({
        'control_pairwise_risk_ranking_accuracy': (
            float(pair_correct / pair_count) if pair_count else float('nan')),
        'control_pairwise_comparable_pairs': float(pair_count),
        'control_num_snapshots': float(len(groups)),
        'control_support_coverage': float(np.mean(support_array)),
        'control_horizon': float(horizon),
        'control_epsilon': float(epsilon),
    })

    coverage_curve = []
    for threshold in np.linspace(0.05, 0.95, 19):
        point = _selection_metrics(
            branches, risks, support_array, groups,
            horizon=horizon, epsilon=float(threshold))
        coverage_curve.append({
            'risk_threshold': float(threshold),
            'coverage': point['control_coverage'],
            'false_safe_rate': point[
                'control_selected_false_safe_rate'],
            'failure_reduction': point[
                'control_nominal_relative_failure_reduction'],
        })
    result['control_risk_coverage_curve'] = coverage_curve

    rng = np.random.default_rng(seed)
    alternative_order: dict[int, list[int]] = {}
    for snapshot_index, indices in groups.items():
        nominal = next(
            index for index in indices
            if branches[index].candidate_family == 'nominal')
        alternatives = [index for index in indices if index != nominal]
        alternative_order[snapshot_index] = rng.permutation(
            alternatives).tolist()
    k_curve = []
    for k in sorted({int(value) for value in k_values if int(value) >= 1}):
        subset_groups: dict[int, list[int]] = {}
        for snapshot_index, indices in groups.items():
            nominal = next(
                index for index in indices
                if branches[index].candidate_family == 'nominal')
            subset_groups[snapshot_index] = [
                nominal,
                *alternative_order[snapshot_index][:max(k - 1, 0)]]
        point = _selection_metrics(
            branches, risks, support_array, subset_groups,
            horizon=horizon, epsilon=epsilon)
        k_curve.append({
            'K': k,
            'coverage': point['control_coverage'],
            'false_safe_rate': point[
                'control_selected_false_safe_rate'],
            'failure_reduction': point[
                'control_nominal_relative_failure_reduction'],
            'top1_safety_regret': point[
                'control_top1_safety_regret'],
        })
    result['control_k_curve'] = k_curve
    if len(k_curve) >= 2:
        result['control_false_safe_increase_with_k'] = float(
            k_curve[-1]['false_safe_rate']
            - k_curve[0]['false_safe_rate'])
    else:
        result['control_false_safe_increase_with_k'] = float('nan')
    return result
