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


def _select_group(
        indices, branches, risks, support, horizon, epsilon,
        structured_fallback=False):
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
    if structured_fallback:
        contracted = next((
            index for index in indices
            if branches[index].candidate_family == 'contracted_previous'
        ), nominal)
        return contracted, False
    return nominal, False


def _selection_metrics(
        branches: Sequence[CandidateBranch],
        risks: np.ndarray,
        support: np.ndarray,
        groups: dict[int, list[int]],
        *,
        horizon: int,
        epsilon: float,
        structured_fallback: bool = False,
) -> dict[str, float]:
    selected = []
    nominal = []
    covered = []
    for indices in groups.values():
        selected_index, is_covered = _select_group(
            indices, branches, risks, support, horizon, epsilon,
            structured_fallback)
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
    replacements = covered & (selected != nominal)
    fallbacks = ~covered & (selected != nominal)
    failure_delta = nominal_failures - selected_failures
    replacement_contribution = float(np.mean(
        np.where(replacements, failure_delta, 0.0)))
    fallback_contribution = float(np.mean(
        np.where(fallbacks, failure_delta, 0.0)))
    total_reduction = float(np.mean(failure_delta))
    positive_total = max(total_reduction, 0.0)
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
        'control_nominal_relative_failure_reduction': total_reduction,
        'control_replacement_rate': float(np.mean(replacements)),
        'control_fallback_rate': float(np.mean(fallbacks)),
        'control_replacement_failure_contribution':
            replacement_contribution,
        'control_fallback_failure_contribution': fallback_contribution,
        'control_fallback_reduction_fraction': (
            max(fallback_contribution, 0.0) / positive_total
            if positive_total > 0.0 else 1.0),
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
        structured_fallback: bool = False,
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
                if not (support_array[left] and support_array[right]):
                    continue
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
        horizon=horizon, epsilon=epsilon,
        structured_fallback=structured_fallback)
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
            horizon=horizon, epsilon=float(threshold),
            structured_fallback=structured_fallback)
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
            horizon=horizon, epsilon=epsilon,
            structured_fallback=structured_fallback)
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


def evaluate_double_critic_control(
        branches: Sequence[CandidateBranch],
        selector_risks,
        validator_risks,
        *,
        horizon: int = 16,
        epsilon: float = 0.2,
        improvement_margin: float = 0.02,
        support=None,
) -> dict[str, float]:
    """Evaluate A-select/B-validate with no search performed by critic B."""
    if not branches:
        raise ValueError('double-critic evaluation needs branch records')
    selector = np.asarray(selector_risks, dtype=np.float64).reshape(-1)
    validator = np.asarray(validator_risks, dtype=np.float64).reshape(-1)
    if len(selector) != len(branches) or len(validator) != len(branches):
        raise ValueError('critic risks must align with branches')
    supported = (
        np.ones(len(branches), dtype=bool)
        if support is None else np.asarray(support, dtype=bool).reshape(-1))
    if len(supported) != len(branches):
        raise ValueError('support must align with branches')

    groups: dict[int, list[int]] = defaultdict(list)
    for index, branch in enumerate(branches):
        groups[int(branch.snapshot_index)].append(index)
    final_indices = []
    nominal_indices = []
    covered = []
    replacements = []
    validation_rejects = []
    abstentions = []
    selector_improvements = []
    validator_improvements = []
    for indices in groups.values():
        nominal = next(
            index for index in indices
            if branches[index].candidate_family == 'nominal')
        contracted = next((
            index for index in indices
            if branches[index].candidate_family == 'contracted_previous'
        ), nominal)
        nominal_safe = (
            supported[nominal]
            and selector[nominal] <= epsilon
            and validator[nominal] <= epsilon)
        if nominal_safe:
            final = nominal
            is_covered = True
            replacement = False
            validation_reject = False
            abstention = False
            selector_improvement = 0.0
            validator_improvement = 0.0
        else:
            eligible = [
                index for index in indices
                if index != nominal
                and supported[index]
                and selector[index] <= epsilon
            ]
            selected = (
                min(eligible, key=lambda index: selector[index])
                if eligible else None)
            if selected is None:
                valid = False
                selector_improvement = 0.0
                validator_improvement = 0.0
            else:
                selector_improvement = (
                    selector[nominal] - selector[selected])
                validator_improvement = (
                    validator[nominal] - validator[selected])
                valid = (
                    validator[selected] <= epsilon
                    and selector_improvement >= improvement_margin
                    and validator_improvement >= improvement_margin)
            if valid:
                final = int(selected)
                is_covered = True
                replacement = True
                validation_reject = False
                abstention = False
            else:
                final = contracted
                is_covered = False
                replacement = False
                validation_reject = selected is not None
                abstention = True
        final_indices.append(final)
        nominal_indices.append(nominal)
        covered.append(is_covered)
        replacements.append(replacement)
        validation_rejects.append(validation_reject)
        abstentions.append(abstention)
        selector_improvements.append(selector_improvement)
        validator_improvements.append(validator_improvement)

    final_indices = np.asarray(final_indices, dtype=np.int64)
    nominal_indices = np.asarray(nominal_indices, dtype=np.int64)
    covered = np.asarray(covered, dtype=bool)
    replacements = np.asarray(replacements, dtype=bool)
    validation_rejects = np.asarray(validation_rejects, dtype=bool)
    abstentions = np.asarray(abstentions, dtype=bool)
    final_failures = np.asarray([
        branches[index].outcomes[horizon].failure
        for index in final_indices], dtype=np.float32)
    nominal_failures = np.asarray([
        branches[index].outcomes[horizon].failure
        for index in nominal_indices], dtype=np.float32)
    count_covered = max(int(np.sum(covered)), 1)
    regrets = []
    for final, indices in zip(final_indices, groups.values()):
        best = min(_severity(branches[index], horizon) for index in indices)
        regrets.append(_severity(branches[final], horizon) - best)
    replacement_contribution = np.where(
        replacements, nominal_failures - final_failures, 0.0)
    abstention_contribution = np.where(
        abstentions, nominal_failures - final_failures, 0.0)
    return {
        'double_coverage': float(np.mean(covered)),
        'double_replacement_rate': float(np.mean(replacements)),
        'double_validation_reject_rate': float(
            np.mean(validation_rejects)),
        'double_abstention_rate': float(np.mean(abstentions)),
        'double_false_safe_rate': float(
            np.sum(final_failures[covered]) / count_covered),
        'double_nominal_failure_rate': float(np.mean(nominal_failures)),
        'double_selected_failure_rate': float(np.mean(final_failures)),
        'double_failure_reduction': float(np.mean(
            nominal_failures - final_failures)),
        'double_replacement_failure_reduction': float(np.mean(
            replacement_contribution)),
        'double_abstention_failure_reduction': float(np.mean(
            abstention_contribution)),
        'double_top1_safety_regret': float(np.mean(regrets)),
        'double_selector_improvement_mean': float(np.mean(
            selector_improvements)),
        'double_validator_improvement_mean': float(np.mean(
            validator_improvements)),
        'double_selected_A_B_disagreement': float(np.mean(np.abs(
            selector[final_indices] - validator[final_indices]))),
        'double_epsilon': float(epsilon),
        'double_improvement_margin': float(improvement_margin),
        'double_num_snapshots': float(len(groups)),
    }
