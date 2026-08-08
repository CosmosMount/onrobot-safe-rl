"""Group-macro diagnostics for same-state safety-action predictions."""

from __future__ import annotations

from typing import Any

import numpy as np

from safety_data.schema import DatasetValidationError, GroupedBranchDataset


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != values.shape:
        raise ValueError("metric values and weights must have equal shapes")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("metric weights must be finite and positive")
    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _group_weights(dataset: GroupedBranchDataset) -> np.ndarray:
    # Inverse probability weighting restores the natural state distribution
    # when boundary mining records its acceptance probability. Natural samples
    # have probability one and are therefore unchanged.
    probability = np.asarray(
        dataset["acceptance_probability"], dtype=np.float64)
    # p_min / p is proportional to 1 / p, but remains in (0, 1] even when
    # mining probabilities are extremely small.
    weights = float(np.min(probability)) / probability
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise DatasetValidationError("invalid inverse-acceptance weights")
    return weights


def _pair_scores(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    group_scores = np.full(target.shape[0], np.nan, dtype=np.float64)
    pair_counts = np.zeros(target.shape[0], dtype=np.int64)
    for group in range(target.shape[0]):
        indices = np.flatnonzero(mask[group])
        scores: list[float] = []
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1:]:
                target_delta = float(target[group, left] - target[group, right])
                if abs(target_delta) < minimum_gap or target_delta == 0.0:
                    continue
                prediction_delta = float(
                    prediction[group, left] - prediction[group, right])
                if prediction_delta == 0.0:
                    scores.append(0.5)
                else:
                    scores.append(float(
                        np.sign(target_delta) == np.sign(prediction_delta)))
        if scores:
            group_scores[group] = float(np.mean(scores))
            pair_counts[group] = len(scores)
    return group_scores, pair_counts


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[labels].sum())
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives))


def _equal_mass_ece(
    prediction: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    bins: int,
) -> tuple[float, np.ndarray]:
    if bins < 1:
        raise ValueError("ECE bins must be positive")
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    sample_weight = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    order = np.argsort(prediction, kind="mergesort")
    prediction = prediction[order]
    target = target[order]
    sample_weight = sample_weight[order]
    if not (
        len(prediction) == len(target) == len(sample_weight)
        and len(prediction) > 0
    ):
        raise ValueError("ECE inputs must be nonempty and have equal lengths")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("ECE predictions and targets must be finite")
    if not np.all(np.isfinite(sample_weight)) or np.any(sample_weight <= 0.0):
        raise ValueError("ECE sample weights must be finite and positive")
    # Equal predictions are indistinguishable to a calibrator. Aggregate each
    # tie before splitting weighted mass across bins so row order cannot alter
    # ECE when a tied block straddles a bin boundary.
    unique_prediction, inverse = np.unique(prediction, return_inverse=True)
    tied_weight = np.bincount(
        inverse, weights=sample_weight, minlength=len(unique_prediction))
    tied_target = np.bincount(
        inverse, weights=sample_weight * target,
        minlength=len(unique_prediction)) / tied_weight
    prediction = unique_prediction
    target = tied_target
    sample_weight = tied_weight
    total = float(sample_weight.sum())
    if total <= 0.0:
        return float("nan"), np.full(bins, np.nan)
    target_mass = total / bins
    bin_weight = np.zeros(bins, dtype=np.float64)
    bin_prediction = np.zeros(bins, dtype=np.float64)
    bin_target = np.zeros(bins, dtype=np.float64)
    sample = 0
    remaining = float(sample_weight[0])
    for bin_number in range(bins):
        desired = total - bin_weight.sum() if bin_number == bins - 1 else target_mass
        while desired > 1e-15 * total and sample < len(prediction):
            portion = min(remaining, desired)
            bin_weight[bin_number] += portion
            bin_prediction[bin_number] += portion * prediction[sample]
            bin_target[bin_number] += portion * target[sample]
            desired -= portion
            remaining -= portion
            if remaining <= 1e-15 * total:
                sample += 1
                if sample < len(prediction):
                    remaining = float(sample_weight[sample])
    nonempty = bin_weight > 0.0
    confidence = np.zeros(bins, dtype=np.float64)
    frequency = np.zeros(bins, dtype=np.float64)
    confidence[nonempty] = bin_prediction[nonempty] / bin_weight[nonempty]
    frequency[nonempty] = bin_target[nonempty] / bin_weight[nonempty]
    mass_fraction = bin_weight / total
    ece = float(np.sum(mass_fraction * np.abs(confidence - frequency)))
    return ece, mass_fraction


def _cluster_bootstrap_ci(
    values: np.ndarray,
    group_weights: np.ndarray,
    clusters: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    group_weights = np.asarray(group_weights, dtype=np.float64)
    clusters = np.asarray(clusters).astype(str)
    unique = np.unique(clusters)
    if replicates <= 0 or len(unique) < 2:
        return float("nan"), float("nan")
    members = {name: np.flatnonzero(clusters == name) for name in unique}
    rng = np.random.default_rng(seed)
    results = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[name] for name in sampled])
        results[replicate] = _weighted_mean(
            values[indices], group_weights[indices])
    finite = results[np.isfinite(results)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def evaluate_predictions(
    dataset: GroupedBranchDataset,
    predicted_risk: np.ndarray,
    *,
    strong_pair_gap: float = 0.25,
    ece_bins: int = 10,
    bootstrap_replicates: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate candidate risks without treating rows/replicas as groups.

    ``predicted_risk`` is one probability per group/candidate. Targets are the
    binomial branch frequencies over CRN replicas. Pair metrics are first
    averaged within each group; uncertainty resamples complete trajectories.
    """
    dataset.validate()
    prediction = np.asarray(predicted_risk, dtype=np.float64)
    mask = np.asarray(dataset["candidate_mask"], dtype=bool)
    if prediction.shape != mask.shape:
        raise DatasetValidationError(
            f"predicted_risk shape {prediction.shape}, expected {mask.shape}")
    if not np.all(np.isfinite(prediction[mask])):
        raise DatasetValidationError("predicted_risk contains non-finite values")
    if np.any(prediction[mask] < 0.0) or np.any(prediction[mask] > 1.0):
        raise DatasetValidationError("predicted_risk must lie in [0, 1]")

    fall = np.asarray(dataset["fall"], dtype=bool)
    target = np.mean(fall, axis=2, dtype=np.float64)
    weights = _group_weights(dataset)
    group_count = dataset.group_count
    nominal = target[:, 0]
    selected_index = np.empty(group_count, dtype=np.int64)
    oracle_index = np.empty(group_count, dtype=np.int64)
    brier_by_group = np.empty(group_count, dtype=np.float64)
    empirical_risk_mse_by_group = np.empty(group_count, dtype=np.float64)
    for group in range(group_count):
        valid = np.flatnonzero(mask[group])
        selected_index[group] = valid[np.argmin(prediction[group, valid])]
        oracle_index[group] = valid[np.argmin(target[group, valid])]
        brier_by_group[group] = float(np.mean(np.square(
            prediction[group, valid, None]
            - fall[group, valid].astype(np.float64))))
        empirical_risk_mse_by_group[group] = float(np.mean(np.square(
            prediction[group, valid] - target[group, valid])))
    selected_target = target[np.arange(group_count), selected_index]
    oracle_target = target[np.arange(group_count), oracle_index]
    top1_reduction = nominal - selected_target
    oracle_reduction = nominal - oracle_target
    achieved = _weighted_mean(top1_reduction, weights)
    available = _weighted_mean(oracle_reduction, weights)

    pair_by_group, pair_counts = _pair_scores(
        target, prediction, mask, minimum_gap=np.nextafter(0.0, 1.0))
    strong_by_group, strong_counts = _pair_scores(
        target, prediction, mask, minimum_gap=float(strong_pair_gap))
    pair_accuracy = _weighted_mean(pair_by_group, weights)
    strong_pair_accuracy = _weighted_mean(strong_by_group, weights)

    flat_prediction: list[float] = []
    flat_target: list[float] = []
    flat_calibration_weight: list[float] = []
    auc_labels: list[bool] = []
    auc_scores: list[float] = []
    for group in range(group_count):
        valid = np.flatnonzero(mask[group])
        candidate_weight = weights[group] / len(valid)
        for candidate in valid:
            flat_prediction.append(float(prediction[group, candidate]))
            flat_target.append(float(target[group, candidate]))
            flat_calibration_weight.append(candidate_weight)
            for outcome in fall[group, candidate]:
                auc_labels.append(bool(outcome))
                auc_scores.append(float(prediction[group, candidate]))

    clusters = _as_cluster_text(dataset["trajectory_id"])
    top1_ci = _cluster_bootstrap_ci(
        top1_reduction, weights, clusters,
        replicates=bootstrap_replicates, seed=bootstrap_seed)
    pair_ci = _cluster_bootstrap_ci(
        pair_by_group, weights, clusters,
        replicates=bootstrap_replicates, seed=bootstrap_seed + 1)
    ece, ece_mass = _equal_mass_ece(
        np.asarray(flat_prediction), np.asarray(flat_target),
        np.asarray(flat_calibration_weight), ece_bins)
    return {
        "groups": group_count,
        "valid_candidates": int(mask.sum()),
        "replicas": dataset.replica_count,
        "strong_pair_gap": float(strong_pair_gap),
        "pair_accuracy_group_macro": pair_accuracy,
        "pair_accuracy_ci95": list(pair_ci),
        "pair_groups": int(np.isfinite(pair_by_group).sum()),
        "pair_comparisons": int(pair_counts.sum()),
        "strong_pair_accuracy_group_macro": strong_pair_accuracy,
        "strong_pair_groups": int(np.isfinite(strong_by_group).sum()),
        "strong_pair_comparisons": int(strong_counts.sum()),
        "auroc_replica_diagnostic": _binary_auc(
            np.asarray(auc_labels), np.asarray(auc_scores)),
        "brier_group_macro": _weighted_mean(brier_by_group, weights),
        "empirical_risk_mse_group_macro": _weighted_mean(
            empirical_risk_mse_by_group, weights),
        "ece_equal_mass": ece,
        "ece_bins": int(ece_bins),
        "ece_bin_mass_fraction": ece_mass.tolist(),
        "ece_max_bin_mass_error": float(np.max(np.abs(
            ece_mass - 1.0 / ece_bins))),
        "nominal_fall_risk": _weighted_mean(nominal, weights),
        "selected_fall_risk": _weighted_mean(selected_target, weights),
        "top1_absolute_reduction": achieved,
        "top1_reduction_ci95": list(top1_ci),
        "oracle_fall_risk": _weighted_mean(oracle_target, weights),
        "oracle_absolute_reduction": available,
        "oracle_gap_capture": (
            float(achieved / available) if available > 0.0 else float("nan")),
    }


def _as_cluster_text(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise DatasetValidationError("trajectory_id must be a vector")
    return result.astype(str)
