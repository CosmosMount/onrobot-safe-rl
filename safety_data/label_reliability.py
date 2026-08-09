"""Independent-replica reliability gate for grouped Q_safe labels.

The primary estimand deliberately separates candidate discovery from outcome
audit.  Candidate minima are discovered on the manifest-locked discovery
replicas and evaluated only on the disjoint audit replicas.  This prevents the
same-sample minimum (the usual empirical ``oracle``) from being mistaken for
reproducible candidate-level signal.

Groups are macro averaged with inverse acceptance-probability weights and
uncertainty resamples complete source trajectories.  Metrics that reuse the
same outcomes for selection and evaluation are returned only under the
explicitly biased diagnostics section and never participate in the gate.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np

from safety_data.schema import DatasetValidationError, GroupedBranchDataset


PARTITION_SCHEMA_VERSION = "qsafe.independent_replica_partition.v2"
REPORT_SCHEMA_VERSION = "qsafe.independent_replica_label_gate.report.v1"

_PARTITION_KEYS = frozenset({
    "schema_version",
    "assignment_timing",
    "axis",
    "ordering",
    "discovery_indices",
    "audit_indices",
    "discovery_replicas",
    "audit_replicas",
    "exhaustive",
})

_THRESHOLD_KEYS = frozenset({
    "min_discovery_to_audit_absolute_reduction",
    "min_reduction_ci_low",
    "min_pair_order_agreement",
    "min_pair_order_agreement_ci_low",
    "bootstrap_replicates",
    "bootstrap_seed",
    "confidence_level",
})


class LabelReliabilityError(ValueError):
    """The independent-replica label gate cannot be evaluated safely."""


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise LabelReliabilityError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LabelReliabilityError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise LabelReliabilityError(f"{name} must be a finite number")
    return result


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise LabelReliabilityError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise LabelReliabilityError(f"{name} must be {qualifier}")
    return result


def _exact_mapping_keys(
    value: object,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LabelReliabilityError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise LabelReliabilityError(f"{name} keys must be strings")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise LabelReliabilityError(
            f"{name} keys do not match the locked contract: "
            f"missing={missing}, unknown={unknown}")
    return value


def _validated_thresholds(thresholds: Mapping[str, Any]) -> dict[str, Any]:
    raw = _exact_mapping_keys(thresholds, _THRESHOLD_KEYS, "thresholds")
    result: dict[str, Any] = {
        "min_discovery_to_audit_absolute_reduction": _finite_float(
            raw["min_discovery_to_audit_absolute_reduction"],
            "min_discovery_to_audit_absolute_reduction"),
        "min_reduction_ci_low": _finite_float(
            raw["min_reduction_ci_low"], "min_reduction_ci_low"),
        "min_pair_order_agreement": _finite_float(
            raw["min_pair_order_agreement"], "min_pair_order_agreement"),
        "min_pair_order_agreement_ci_low": _finite_float(
            raw["min_pair_order_agreement_ci_low"],
            "min_pair_order_agreement_ci_low"),
        "bootstrap_replicates": _integer(
            raw["bootstrap_replicates"], "bootstrap_replicates", minimum=1),
        "bootstrap_seed": _integer(
            raw["bootstrap_seed"], "bootstrap_seed", minimum=0),
        "confidence_level": _finite_float(
            raw["confidence_level"], "confidence_level"),
    }
    for name in (
        "min_discovery_to_audit_absolute_reduction",
        "min_reduction_ci_low",
    ):
        if not -1.0 <= result[name] <= 1.0:
            raise LabelReliabilityError(f"{name} must lie in [-1, 1]")
    for name in (
        "min_pair_order_agreement",
        "min_pair_order_agreement_ci_low",
    ):
        if not 0.0 <= result[name] <= 1.0:
            raise LabelReliabilityError(f"{name} must lie in [0, 1]")
    if not 0.0 < result["confidence_level"] < 1.0:
        raise LabelReliabilityError("confidence_level must lie in (0, 1)")
    return result


def _index_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise LabelReliabilityError(f"{name} must be a nonempty list of integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
                item, (int, np.integer)):
            raise LabelReliabilityError(
                f"{name} must be a nonempty list of integers")
        result.append(int(item))
    if len(set(result)) != len(result):
        raise LabelReliabilityError(f"{name} contains duplicate indices")
    if any(item < 0 for item in result):
        raise LabelReliabilityError(f"{name} contains a negative index")
    return result


def _validated_partition(
    manifest: Mapping[str, Any], replica_count: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    collection_protocol = manifest.get("collection_protocol")
    if not isinstance(collection_protocol, Mapping) or (
            "replica_partition" not in collection_protocol):
        raise LabelReliabilityError(
            "manifest.collection_protocol is missing the pre-outcome "
            "replica_partition")
    raw = _exact_mapping_keys(
        collection_protocol["replica_partition"], _PARTITION_KEYS,
        "manifest.collection_protocol.replica_partition")
    literals = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "assignment_timing": "before_candidate_outcomes",
        "axis": "replica",
        "ordering": "discovery_then_audit",
    }
    for name, expected in literals.items():
        if raw[name] != expected:
            raise LabelReliabilityError(
                f"replica_partition.{name}={raw[name]!r}, expected {expected!r}")
    if raw["exhaustive"] is not True:
        raise LabelReliabilityError("replica_partition.exhaustive must be true")

    discovery = _index_list(raw["discovery_indices"], "discovery_indices")
    audit = _index_list(raw["audit_indices"], "audit_indices")
    discovery_count = _integer(
        raw["discovery_replicas"], "discovery_replicas", minimum=1)
    audit_count = _integer(raw["audit_replicas"], "audit_replicas", minimum=1)
    if discovery_count != len(discovery):
        raise LabelReliabilityError(
            "discovery_replicas does not match discovery_indices")
    if audit_count != len(audit):
        raise LabelReliabilityError("audit_replicas does not match audit_indices")
    if set(discovery) & set(audit):
        raise LabelReliabilityError("discovery_indices and audit_indices overlap")

    # ``discovery_then_audit`` is stronger than mere disjointness: it makes the
    # one-way primary direction independently auditable from the manifest.
    expected_discovery = list(range(discovery_count))
    expected_audit = list(range(discovery_count, discovery_count + audit_count))
    if discovery != expected_discovery or audit != expected_audit:
        raise LabelReliabilityError(
            "discovery_then_audit requires contiguous discovery indices followed "
            "by contiguous audit indices")
    if discovery_count + audit_count != replica_count:
        raise LabelReliabilityError(
            "exhaustive replica partition does not cover the dataset replica axis")

    canonical = {
        name: (
            list(map(int, raw[name]))
            if name in ("discovery_indices", "audit_indices")
            else int(raw[name])
            if name in ("discovery_replicas", "audit_replicas")
            else bool(raw[name])
            if name == "exhaustive"
            else str(raw[name])
        )
        for name in sorted(_PARTITION_KEYS)
    }
    return (
        canonical,
        np.asarray(discovery, dtype=np.int64),
        np.asarray(audit, dtype=np.int64),
    )


def _validated_gate_arrays(dataset: GroupedBranchDataset) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    # Preserve the dataset validator's useful exception type/message.  No
    # metric is computed after validation fails.
    dataset.validate()

    mask = np.asarray(dataset["candidate_mask"])
    fall = np.asarray(dataset["fall"])
    acceptance = np.asarray(dataset["acceptance_probability"], dtype=np.float64)
    trajectory = np.asarray(dataset["trajectory_id"])
    if mask.ndim != 2 or mask.dtype.kind not in "biu" or not np.all(
            np.isin(mask, (0, 1, False, True))):
        raise DatasetValidationError("candidate_mask must be a binary [G, K] array")
    mask = mask.astype(bool, copy=False)
    if mask.shape[0] == 0 or mask.shape[1] < 2:
        raise DatasetValidationError("label gate requires nonempty groups and K >= 2")
    if np.any(~mask[:, 0]) or np.any(mask.sum(axis=1) < 2):
        raise DatasetValidationError(
            "candidate 0 and at least one alternative must be valid in every group")
    if fall.ndim != 3 or fall.shape[:2] != mask.shape or fall.shape[2] < 2:
        raise DatasetValidationError(
            "fall must have shape [G, K, R] with at least two replicas")
    valid = np.broadcast_to(mask[..., None], fall.shape)
    if not np.all(np.isin(fall[valid], (0, 1, False, True))):
        raise DatasetValidationError("fall labels must be binary")
    fall = fall.astype(np.float64, copy=False)
    groups = mask.shape[0]
    if acceptance.shape != (groups,) or not np.all(np.isfinite(acceptance)):
        raise DatasetValidationError(
            "acceptance_probability must be a finite [G] vector")
    if np.any(acceptance <= 0.0) or np.any(acceptance > 1.0):
        raise DatasetValidationError("acceptance_probability must lie in (0, 1]")
    if trajectory.shape != (groups,) or trajectory.dtype.kind not in "US":
        raise DatasetValidationError("trajectory_id must be a nonempty text [G] vector")
    trajectory = trajectory.astype(str, copy=False)
    if np.any(trajectory == ""):
        raise DatasetValidationError("trajectory_id contains an empty identifier")
    if len(np.unique(trajectory)) < 2:
        raise DatasetValidationError(
            "trajectory-cluster bootstrap requires at least two trajectories")
    # Scaling by min(p) is algebraically identical to 1/p after normalization,
    # and avoids overflow for extremely small but valid mining probabilities.
    weights = float(np.min(acceptance)) / acceptance
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise DatasetValidationError("invalid inverse-acceptance weights")
    return mask, fall, weights, trajectory


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or not np.all(np.isfinite(values)):
        raise LabelReliabilityError("metric values must be finite group vectors")
    denominator = float(np.sum(weights))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise LabelReliabilityError("metric weights have no positive finite mass")
    return float(np.sum(values * weights) / denominator)


def _selection_evaluation(
    selection_risk: np.ndarray,
    evaluation_risk: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = mask.shape[0]
    expected_selected = np.empty(groups, dtype=np.float64)
    reduction = np.empty(groups, dtype=np.float64)
    tie_count = np.empty(groups, dtype=np.int64)
    for group in range(groups):
        valid = np.flatnonzero(mask[group])
        discovery = selection_risk[group, valid]
        winners = valid[discovery == np.min(discovery)]
        # The selection rule has no outcome-based ordering within a discovery
        # tie.  Its estimand is therefore the expected audit risk under uniform
        # tie breaking, not whichever tied row happens to occur first.
        expected_selected[group] = float(np.mean(evaluation_risk[group, winners]))
        reduction[group] = float(
            evaluation_risk[group, 0] - expected_selected[group])
        tie_count[group] = len(winners)
    return expected_selected, reduction, tie_count


def _pair_agreement(
    discovery_risk: np.ndarray,
    audit_risk: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = mask.shape[0]
    scores = np.empty(groups, dtype=np.float64)
    comparisons = np.zeros(groups, dtype=np.int64)
    ties = np.zeros(groups, dtype=np.int64)
    for group in range(groups):
        valid = np.flatnonzero(mask[group])
        group_scores: list[float] = []
        for offset, left in enumerate(valid):
            for right in valid[offset + 1:]:
                discovery_delta = float(
                    discovery_risk[group, left] - discovery_risk[group, right])
                audit_delta = float(
                    audit_risk[group, left] - audit_risk[group, right])
                if discovery_delta == 0.0 or audit_delta == 0.0:
                    # A tie carries no directional evidence.  Scoring it 0.5
                    # avoids both optimistic tie agreement and row-order bias.
                    group_scores.append(0.5)
                    ties[group] += 1
                else:
                    group_scores.append(float(
                        np.sign(discovery_delta) == np.sign(audit_delta)))
        comparisons[group] = len(group_scores)
        scores[group] = float(np.mean(group_scores))
    return scores, comparisons, ties


def _cluster_bootstrap(
    values: tuple[np.ndarray, ...],
    weights: np.ndarray,
    trajectories: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> tuple[tuple[float, float], ...]:
    unique, cluster_index = np.unique(trajectories, return_inverse=True)
    cluster_count = len(unique)
    cluster_denominator = np.bincount(
        cluster_index, weights=weights, minlength=cluster_count)
    cluster_numerators = [
        np.bincount(
            cluster_index, weights=weights * value, minlength=cluster_count)
        for value in values
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty((len(values), replicates), dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        denominator = float(np.sum(cluster_denominator[sampled]))
        for metric, numerator in enumerate(cluster_numerators):
            draws[metric, replicate] = float(
                np.sum(numerator[sampled]) / denominator)
    alpha = (1.0 - confidence) / 2.0
    return tuple(
        tuple(map(float, np.quantile(metric_draws, [alpha, 1.0 - alpha])))
        for metric_draws in draws
    )


def _centered_candidate_correlation(
    discovery_risk: np.ndarray,
    audit_risk: np.ndarray,
    mask: np.ndarray,
    group_weights: np.ndarray,
) -> float | None:
    left: list[float] = []
    right: list[float] = []
    weights: list[float] = []
    for group in range(mask.shape[0]):
        valid = np.flatnonzero(mask[group])
        discovery = discovery_risk[group, valid]
        audit = audit_risk[group, valid]
        left.extend((discovery - np.mean(discovery)).tolist())
        right.extend((audit - np.mean(audit)).tolist())
        # Each group retains its IPW macro mass regardless of its K.
        weights.extend([float(group_weights[group] / len(valid))] * len(valid))
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weight))
    mean_x = float(np.sum(weight * x) / total)
    mean_y = float(np.sum(weight * y) / total)
    centered_x = x - mean_x
    centered_y = y - mean_y
    variance_x = float(np.sum(weight * np.square(centered_x)) / total)
    variance_y = float(np.sum(weight * np.square(centered_y)) / total)
    if variance_x <= 0.0 or variance_y <= 0.0:
        return None
    covariance = float(np.sum(weight * centered_x * centered_y) / total)
    result = covariance / math.sqrt(variance_x * variance_y)
    return float(np.clip(result, -1.0, 1.0))


def _ci_dict(low: float, high: float, confidence: float) -> dict[str, float]:
    return {
        "low": float(low),
        "high": float(high),
        "confidence_level": float(confidence),
    }


def evaluate_independent_replica_label_gate(
    dataset: GroupedBranchDataset,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the locked discovery-to-audit label-reliability gate.

    ``replica_partition`` must have been recorded in the dataset manifest
    before candidate outcomes.  Unknown or missing partition and threshold
    fields are rejected so protocol changes cannot silently alter the gate.
    The returned mapping contains only JSON-safe Python scalars and containers.
    """
    locked_thresholds = _validated_thresholds(thresholds)
    mask, fall, weights, trajectories = _validated_gate_arrays(dataset)
    manifest = getattr(dataset, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise LabelReliabilityError("dataset manifest must be a mapping")
    partition, discovery_indices, audit_indices = _validated_partition(
        manifest, fall.shape[2])

    discovery_risk = np.mean(
        fall[:, :, discovery_indices], axis=2, dtype=np.float64)
    audit_risk = np.mean(fall[:, :, audit_indices], axis=2, dtype=np.float64)
    full_risk = np.mean(fall, axis=2, dtype=np.float64)

    selected_audit, primary_group, discovery_ties = _selection_evaluation(
        discovery_risk, audit_risk, mask)
    selected_discovery, reverse_group, audit_ties = _selection_evaluation(
        audit_risk, discovery_risk, mask)
    symmetric_group = 0.5 * (primary_group + reverse_group)
    pair_group, pair_counts, pair_ties = _pair_agreement(
        discovery_risk, audit_risk, mask)

    full_selected, biased_oracle_group, full_ties = _selection_evaluation(
        full_risk, full_risk, mask)
    primary_estimate = _weighted_mean(primary_group, weights)
    pair_estimate = _weighted_mean(pair_group, weights)
    primary_ci, pair_ci = _cluster_bootstrap(
        (primary_group, pair_group), weights, trajectories,
        replicates=locked_thresholds["bootstrap_replicates"],
        seed=locked_thresholds["bootstrap_seed"],
        confidence=locked_thresholds["confidence_level"],
    )

    checks = {
        "discovery_to_audit_absolute_reduction": bool(
            primary_estimate
            >= locked_thresholds[
                "min_discovery_to_audit_absolute_reduction"]),
        "reduction_ci_low": bool(
            primary_ci[0] > locked_thresholds["min_reduction_ci_low"]),
        "pair_order_agreement": bool(
            pair_estimate >= locked_thresholds["min_pair_order_agreement"]),
        "pair_order_agreement_ci_low": bool(
            pair_ci[0]
            > locked_thresholds["min_pair_order_agreement_ci_low"]),
    }

    inverse_weight_sum = float(np.sum(weights))
    effective_groups = float(
        inverse_weight_sum ** 2 / np.sum(np.square(weights)))
    result: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "estimand": (
            "inverse_acceptance_weighted_group_macro; "
            "discover_min_on_discovery_replicas; uniform_tie_break; "
            "evaluate_only_on_audit_replicas"
        ),
        "groups": int(mask.shape[0]),
        "valid_candidates": int(np.sum(mask)),
        "trajectory_clusters": int(len(np.unique(trajectories))),
        "ipw_effective_groups": effective_groups,
        "replica_partition": partition,
        "thresholds": dict(locked_thresholds),
        "primary": {
            "discovery_to_audit_absolute_reduction": primary_estimate,
            "confidence_interval": _ci_dict(
                primary_ci[0], primary_ci[1],
                locked_thresholds["confidence_level"]),
            "nominal_audit_fall_risk": _weighted_mean(
                audit_risk[:, 0], weights),
            "uniform_tie_expected_selected_audit_fall_risk": _weighted_mean(
                selected_audit, weights),
            "groups_with_discovery_min_tie": int(
                np.count_nonzero(discovery_ties > 1)),
            "mean_discovery_min_tie_count": _weighted_mean(
                discovery_ties.astype(np.float64), weights),
        },
        "pair_order_agreement": {
            "estimate": pair_estimate,
            "confidence_interval": _ci_dict(
                pair_ci[0], pair_ci[1],
                locked_thresholds["confidence_level"]),
            "comparisons": int(np.sum(pair_counts)),
            "tie_comparisons": int(np.sum(pair_ties)),
            "tie_score": 0.5,
        },
        "diagnostics_not_gate_eligible": {
            "audit_to_discovery_absolute_reduction": _weighted_mean(
                reverse_group, weights),
            "symmetric_crossfit_absolute_reduction": _weighted_mean(
                symmetric_group, weights),
            "centered_candidate_half_correlation": (
                _centered_candidate_correlation(
                    discovery_risk, audit_risk, mask, weights)),
            "biased_same_replica_full_oracle": {
                "absolute_reduction": _weighted_mean(
                    biased_oracle_group, weights),
                "nominal_full_fall_risk": _weighted_mean(
                    full_risk[:, 0], weights),
                "selected_full_fall_risk": _weighted_mean(
                    full_selected, weights),
                "groups_with_min_tie": int(np.count_nonzero(full_ties > 1)),
                "gate_eligible": False,
                "bias_warning": (
                    "Selection and evaluation reuse the same replicas; this is "
                    "an optimistically biased diagnostic, not mechanism evidence."
                ),
            },
            "mean_audit_min_tie_count": _weighted_mean(
                audit_ties.astype(np.float64), weights),
            "uniform_tie_expected_selected_discovery_fall_risk": _weighted_mean(
                selected_discovery, weights),
        },
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    # Enforce the public contract here, rather than relying on callers to
    # discover NumPy scalars or NaN only when persisting an evidence report.
    json.dumps(result, allow_nan=False, sort_keys=True)
    return result


__all__ = [
    "LabelReliabilityError",
    "PARTITION_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "evaluate_independent_replica_label_gate",
]
