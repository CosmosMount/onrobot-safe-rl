"""Pure development-only supervision diagnostics for counterfactual Q_safe."""

from __future__ import annotations

from itertools import combinations

import numpy as np


JOINT_NAMES = tuple(
    f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR")
    for joint in ("hip", "thigh", "calf"))
PAIR_I, PAIR_J = (np.asarray(value, np.int16) for value in zip(
    *combinations(range(16), 2), strict=True))


def _bootstrap_indices(count: int, seed: int, replicates: int = 10_000) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, count, size=(replicates, count), dtype=np.int32)


def _summary(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    samples = np.nanmean(values[indices], axis=1)
    return {
        "mean": float(np.nanmean(values)),
        "one_sided_95_lcb": float(np.nanquantile(samples, 0.05)),
        "two_sided_95_ci_low": float(np.nanquantile(samples, 0.025)),
        "two_sided_95_ci_high": float(np.nanquantile(samples, 0.975)),
    }


def _paired_difference(
    later: np.ndarray, earlier: np.ndarray, indices: np.ndarray,
) -> dict[str, float]:
    return _summary(
        np.asarray(later, np.float64) - np.asarray(earlier, np.float64), indices)


def _risk_pair_arrays(risk: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = risk[:, PAIR_I] - risk[:, PAIR_J]
    non_tie = np.count_nonzero(delta != 0, axis=1)
    strong = np.count_nonzero(np.abs(delta) >= 0.50, axis=1)
    return delta, non_tie, strong


def _ordering_agreement(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_delta = first[:, PAIR_I] - first[:, PAIR_J]
    second_delta = second[:, PAIR_I] - second[:, PAIR_J]
    valid = (first_delta != 0) & (second_delta != 0)
    correct = np.sign(first_delta) == np.sign(second_delta)
    result = np.full(len(first), np.nan, np.float64)
    counts = valid.sum(axis=1)
    keep = counts > 0
    result[keep] = (correct & valid).sum(axis=1)[keep] / counts[keep]
    return result


def _prefix_metrics(labels: np.ndarray, replicas: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    prefix = labels[:, :, :replicas]
    risk = prefix.mean(axis=2)
    _, non_tie, strong = _risk_pair_arrays(risk)
    spread = risk.max(axis=1) - risk.min(axis=1)
    half = replicas // 2
    first = prefix[:, :, :half].mean(axis=2)
    second = prefix[:, :, half:].mean(axis=2)
    agreement = _ordering_agreement(first, second)
    oracle = np.argmin(first, axis=1)
    row = np.arange(len(labels))
    reduction = second[:, 0] - second[row, oracle]
    arrays = {
        "spread": spread, "strong_state": strong > 0,
        "non_tie_pairs": non_tie, "strong_pairs": strong,
        "ordering_agreement": agreement, "oracle_reduction": reduction,
    }
    report = {
        "replicas": replicas,
        "empirical_candidate_fall_risk_mean": float(risk.mean()),
        "empirical_candidate_fall_risk_by_index": risk.mean(axis=0).tolist(),
        "within_state_risk_range_mean": float(spread.mean()),
        "within_state_risk_range_median": float(np.median(spread)),
        "non_tie_pairs_per_state_mean": float(non_tie.mean()),
        "strong_pairs_per_state_mean": float(strong.mean()),
        "strong_pair_state_coverage": float(np.mean(strong > 0)),
        "pair_ordering_agreement": float(np.nanmean(agreement)),
        "independent_oracle_reduction": float(reduction.mean()),
    }
    return report, arrays


def replica_scaling_analysis(first_fall_step: np.ndarray) -> dict[str, object]:
    labels = np.asarray(first_fall_step) <= 96
    if labels.shape[1:] != (16, 16):
        raise ValueError("replica scaling requires [states,16 candidates,16 replicas]")
    reports: dict[str, object] = {}
    arrays = {}
    indices = _bootstrap_indices(len(labels), 91001)
    for replicas in (4, 8, 16):
        report, values = _prefix_metrics(labels, replicas)
        report["state_bootstrap"] = {
            "within_state_risk_range_mean": _summary(values["spread"], indices),
            "strong_pair_state_coverage": _summary(
                values["strong_state"], indices),
            "pair_ordering_agreement": _summary(
                values["ordering_agreement"], indices),
            "independent_oracle_reduction": _summary(
                values["oracle_reduction"], indices),
        }
        reports[f"R{replicas}"] = report
        arrays[replicas] = values
    strong_delta = _paired_difference(
        arrays[16]["strong_state"], arrays[4]["strong_state"], indices)
    agreement_delta = _paired_difference(
        arrays[16]["ordering_agreement"], arrays[4]["ordering_agreement"], indices)
    reports["paired_differences"] = {
        "R16_minus_R4_strong_pair_state_coverage": strong_delta,
        "R16_minus_R4_pair_ordering_agreement": agreement_delta,
        "R8_minus_R4_strong_pair_state_coverage": _paired_difference(
            arrays[8]["strong_state"], arrays[4]["strong_state"], indices),
    }
    reports["r4_label_noise_likely"] = bool(
        strong_delta["one_sided_95_lcb"] > 0
        and agreement_delta["one_sided_95_lcb"] > 0)
    return reports


def _horizon_subset(
    first_fall: np.ndarray, mask: np.ndarray, horizon: int, seed: int,
) -> dict[str, object]:
    labels = first_fall[mask] <= horizon
    risk = labels.mean(axis=2)
    _, non_tie, strong = _risk_pair_arrays(risk)
    spread = risk.max(axis=1) - risk.min(axis=1)
    discovery = labels[:, :, :8].mean(axis=2)
    evaluation = labels[:, :, 8:].mean(axis=2)
    choice = np.argmin(discovery, axis=1)
    reduction = evaluation[:, 0] - evaluation[np.arange(len(labels)), choice]
    indices = _bootstrap_indices(len(labels), seed)
    return {
        "states": int(mask.sum()),
        "candidate_fall_risk_mean": float(risk.mean()),
        "within_state_risk_range_mean": float(spread.mean()),
        "within_state_risk_range_median": float(np.median(spread)),
        "non_tie_pairs_per_state_mean": float(non_tie.mean()),
        "strong_pairs_per_state_mean": float(strong.mean()),
        "strong_pair_state_coverage": float(np.mean(strong > 0)),
        "independent_oracle_headroom": _summary(reduction, indices),
        "_strong_state": strong > 0,
    }


def horizon_analysis(
    first_fall_step: np.ndarray,
    risk_stratum: np.ndarray,
    collector_seed: np.ndarray,
) -> dict[str, object]:
    first_fall = np.asarray(first_fall_step)
    stratum = np.asarray(risk_stratum).astype("U")
    collectors = np.asarray(collector_seed)
    n = len(first_fall)
    result: dict[str, object] = {}
    overall_arrays = {}
    for horizon_index, horizon in enumerate((16, 32, 64, 96)):
        subsets = {"overall": np.ones(n, bool)}
        subsets.update({name: stratum == name for name in ("boundary", "medium", "normal")})
        subsets.update({f"seed{seed}": collectors == seed for seed in (137, 138)})
        report = {}
        for subset_index, (name, mask) in enumerate(subsets.items()):
            metrics = _horizon_subset(
                first_fall, mask, horizon,
                92000 + 100 * horizon_index + subset_index)
            if name == "overall":
                overall_arrays[horizon] = metrics.pop("_strong_state")
            else:
                metrics.pop("_strong_state")
            report[name] = metrics
        result[f"H{horizon}"] = report
    indices = _bootstrap_indices(n, 92999)
    comparisons = {}
    for short in (16, 32, 64):
        comparisons[f"H{short}_minus_H96_strong_pair_coverage"] = _paired_difference(
            overall_arrays[short], overall_arrays[96], indices)
    result["paired_horizon_comparisons"] = comparisons
    qualifying = []
    for short in (16, 32):
        contrast = comparisons[f"H{short}_minus_H96_strong_pair_coverage"]
        oracle = result[f"H{short}"]["overall"]["independent_oracle_headroom"]
        if contrast["one_sided_95_lcb"] > 0 and oracle["one_sided_95_lcb"] > 0:
            qualifying.append(short)
    result["h96_credit_dilution_likely"] = bool(qualifying)
    result["qualifying_short_horizons"] = qualifying
    return result


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0
    normalized = (x - mean) / std
    design = np.column_stack((np.ones(len(x)), normalized))
    penalty = np.eye(design.shape[1]); penalty[0, 0] = 0
    coefficient = np.linalg.solve(design.T @ design + alpha * penalty,
                                  design.T @ y)
    return mean, std, coefficient


def _ridge_predict(model, x: np.ndarray) -> np.ndarray:
    mean, std, coefficient = model
    return np.column_stack((np.ones(len(x)), (x - mean) / std)) @ coefficient


def _grouped_cv_r2(x: np.ndarray, y: np.ndarray, state_id: np.ndarray) -> float:
    fold = np.asarray([int.from_bytes(value[:4], "little") % 5
                       for value in state_id], np.int8)
    prediction = np.empty(len(y), np.float64)
    for held_out in range(5):
        train = fold != held_out
        prediction[~train] = _ridge_predict(_ridge_fit(x[train], y[train]), x[~train])
    denominator = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - prediction) ** 2) / denominator) if denominator else 0.0


def _state_action_features(delta: np.ndarray, distance: np.ndarray) -> np.ndarray:
    candidates = delta[:, 1:]
    return np.concatenate((
        distance[:, 1:].mean(axis=1, keepdims=True),
        distance[:, 1:].max(axis=1, keepdims=True),
        np.maximum(candidates, 0).max(axis=1),
        np.maximum(-candidates, 0).max(axis=1),
        candidates.std(axis=1),
    ), axis=1)


def candidate_direction_analysis(
    first_fall_step: np.ndarray,
    critic_action: np.ndarray,
    candidate_distance: np.ndarray,
    candidate_distance_bin: np.ndarray,
    observation_history: np.ndarray,
    risk_stratum: np.ndarray,
    collector_seed: np.ndarray,
    state_id: np.ndarray,
) -> dict[str, object]:
    action = np.asarray(critic_action, np.float64)
    delta = action - action[:, :1]
    distance = np.asarray(candidate_distance, np.float64)
    bins = np.asarray(candidate_distance_bin).astype("U")
    risk = (np.asarray(first_fall_step) <= 96).mean(axis=2)
    risk_delta = risk - risk[:, :1]
    spread = risk.max(axis=1) - risk.min(axis=1)
    seed = np.asarray(collector_seed)
    state_ids = np.asarray(state_id, "S64")
    bootstrap = _bootstrap_indices(len(action), 93001)

    direction_report = []
    structured = []
    for joint, name in enumerate(JOINT_NAMES):
        for sign_name, positive in (("positive", True), ("negative", False)):
            state_effect = np.full(len(action), np.nan)
            for state in range(len(action)):
                mask = delta[state, 1:, joint] > 0 if positive else (
                    delta[state, 1:, joint] < 0)
                if np.any(mask):
                    state_effect[state] = np.mean(risk_delta[state, 1:][mask])
            summary = _summary(state_effect, bootstrap)
            qualifies = bool(
                abs(summary["mean"]) >= 0.05
                and (summary["two_sided_95_ci_low"] > 0
                     or summary["two_sided_95_ci_high"] < 0))
            direction_report.append({
                "joint": name, "direction": sign_name,
                "risk_delta_vs_nominal": summary, "structured_effect": qualifies,
            })
            if qualifies:
                structured.append(f"{name}:{sign_name}")

    features = _state_action_features(delta, distance)
    cv_r2 = _grouped_cv_r2(features, spread, state_ids)
    action_model = _ridge_fit(features, spread)
    predicted = _ridge_predict(action_model, features)
    observed_gap = float(spread[seed == 137].mean() - spread[seed == 138].mean())
    predicted_gap = float(predicted[seed == 137].mean() - predicted[seed == 138].mean())
    same_sign = bool(observed_gap != 0 and np.sign(observed_gap) == np.sign(predicted_gap))
    explained_fraction = (
        abs(predicted_gap / observed_gap) if observed_gap != 0 else 0.0)

    latest = np.asarray(observation_history, np.float64).reshape(len(action), -1)
    stratum = np.asarray(risk_stratum).astype("U")
    state_features = np.concatenate((latest,
        np.column_stack([stratum == value for value in ("boundary", "medium", "normal")]),
        risk[:, :1]), axis=1)
    state_cv_r2 = _grouped_cv_r2(state_features, spread, state_ids)

    seed_report = {}
    for collector in (137, 138):
        mask = seed == collector
        joint_change = np.abs(delta[mask, 1:]).mean(axis=(0, 1))
        positive_reach = np.maximum(delta[mask, 1:], 0).max(axis=1).mean(axis=0)
        negative_reach = np.maximum(-delta[mask, 1:], 0).max(axis=1).mean(axis=0)
        seed_report[str(collector)] = {
            "states": int(mask.sum()),
            "nominal_H96_risk": float(risk[mask, 0].mean()),
            "within_state_risk_range_mean": float(spread[mask].mean()),
            "strong_pair_state_coverage": float(np.mean(
                np.max(np.abs(risk[mask, :, None] - risk[mask, None, :]), axis=(1, 2)) >= 0.5)),
            "candidate_distance_mean": float(distance[mask, 1:].mean()),
            "candidate_distance_q50": float(np.median(distance[mask, 1:])),
            "candidate_distance_q90": float(np.quantile(distance[mask, 1:], 0.9)),
            "joint_mean_absolute_change": dict(zip(JOINT_NAMES, joint_change.tolist())),
            "joint_positive_reach": dict(zip(JOINT_NAMES, positive_reach.tolist())),
            "joint_negative_reach": dict(zip(JOINT_NAMES, negative_reach.tolist())),
        }
    bin_report = {}
    for name in ("near", "medium", "far"):
        mask = bins[:, 1:] == name
        bin_report[name] = {
            "mean_distance": float(distance[:, 1:][mask].mean()),
            "mean_signed_risk_delta": float(risk_delta[:, 1:][mask].mean()),
            "mean_absolute_risk_delta": float(np.abs(risk_delta[:, 1:][mask]).mean()),
        }
    flag = bool(
        structured and cv_r2 >= 0.05 and same_sign and explained_fraction >= 0.50)
    return {
        "joint_direction_effects": direction_report,
        "structured_joint_directions": structured,
        "distance_bin_relationship": bin_report,
        "seed_comparison": seed_report,
        "action_coverage_model": {
            "grouped_cv_r2": cv_r2,
            "observed_seed137_minus_seed138_risk_range_gap": observed_gap,
            "action_distribution_predicted_gap": predicted_gap,
            "same_sign": same_sign,
            "explained_fraction_absolute": float(explained_fraction),
        },
        "state_distribution_model_grouped_cv_r2": state_cv_r2,
        "candidate_direction_coverage_likely": flag,
    }
