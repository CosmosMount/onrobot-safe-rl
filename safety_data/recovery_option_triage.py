"""Locked discovery/audit analysis for recovery-option mechanism triage."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from safety_data.label_reliability import (
    LabelReliabilityError,
    _cluster_bootstrap,
    _pair_agreement,
    _selection_evaluation,
    _validated_gate_arrays,
    _validated_partition,
    _weighted_mean,
)
from safety_data.recovery_options import (
    RECOVERY_OPTION_DURATIONS,
    RECOVERY_OPTION_KINDS,
    RECOVERY_OPTION_STEPS,
    RecoveryOptionCandidateConfig,
)
from safety_data.schema import GroupedBranchDataset


REPORT_SCHEMA_VERSION = "qsafe.recovery_option_triage.report.v1"


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LabelReliabilityError(f"{name} must be a mapping")
    return value


def _duration_indices(option_steps: np.ndarray, duration: int) -> np.ndarray:
    indices = np.flatnonzero(option_steps == duration)
    if duration != 1:
        indices = np.concatenate([np.asarray([0], dtype=np.int64), indices])
    if len(indices) != 8 or indices[0] != 0:
        raise LabelReliabilityError(
            f"duration L{duration} must contain nominal plus seven templates")
    return indices


def _source_effects(
    values: np.ndarray,
    weights: np.ndarray,
    source_seed: np.ndarray,
    expected_seeds: list[int],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for seed in expected_seeds:
        selected = np.flatnonzero(source_seed == seed)
        if not len(selected):
            raise LabelReliabilityError(f"source seed {seed} has no groups")
        result[str(seed)] = _weighted_mean(
            values[selected], weights[selected])
    return result


def _winner_digest(
    selection_risk: np.ndarray,
    mask: np.ndarray,
    original_indices: np.ndarray,
) -> str:
    winners: list[dict[str, Any]] = []
    for group in range(mask.shape[0]):
        valid = np.flatnonzero(mask[group])
        values = selection_risk[group, valid]
        selected = valid[values == np.min(values)]
        winners.append({
            "group": group,
            "candidate_indices": original_indices[selected].astype(int).tolist(),
        })
    return _canonical_sha256(winners)


def _ci(low_high: tuple[float, float], confidence: float) -> dict[str, float]:
    return {
        "low": float(low_high[0]),
        "high": float(low_high[1]),
        "confidence_level": float(confidence),
    }


def evaluate_recovery_option_triage(
    dataset: GroupedBranchDataset,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered duration/template discovery/audit decision.

    For each duration, discovery replicas choose a template independently per
    state and audit replicas evaluate the uniform expectation across any tied
    templates.  L1 is fixed as the one-step comparator.  Exactly one of L2--L4
    is selected globally by its discovery-only aggregate and then evaluated on
    audit replicas; an audit failure cannot fall through to a runner-up.
    """
    protocol = _require_mapping(protocol, "protocol")
    if protocol.get("protocol_name") != "objective1_recovery_option_triage_v2":
        raise LabelReliabilityError("unexpected recovery-option triage protocol")
    collection = _require_mapping(protocol.get("collection"), "collection")
    gates = _require_mapping(protocol.get("triage_gates"), "triage_gates")
    data_gate = _require_mapping(gates.get("data"), "triage_gates.data")
    if collection.get("candidates") != (
            RecoveryOptionCandidateConfig().manifest_protocol()):
        raise LabelReliabilityError("protocol candidate contract has drifted")

    mask, fall, weights, trajectories = _validated_gate_arrays(dataset)
    manifest = _require_mapping(dataset.manifest, "dataset.manifest")
    partition, discovery_indices, audit_indices = _validated_partition(
        manifest, fall.shape[2])
    if partition != collection.get("replica_partition"):
        raise LabelReliabilityError(
            "dataset replica partition differs from triage protocol")
    if manifest.get("candidate_protocol") != collection.get("candidates"):
        raise LabelReliabilityError(
            "dataset candidate protocol differs from triage protocol")

    groups, candidates = mask.shape
    source_seed = np.asarray(dataset["source_seed"])
    option_steps_matrix = np.asarray(dataset["candidate_option_steps"])
    candidate_kind = np.asarray(dataset["candidate_kind"]).astype(str)
    expected_steps = np.asarray(RECOVERY_OPTION_STEPS, dtype=np.int64)
    expected_kinds = np.asarray(RECOVERY_OPTION_KINDS, dtype=str)
    if option_steps_matrix.shape != (groups, candidates) or not np.all(
            option_steps_matrix == expected_steps[None, :]):
        raise LabelReliabilityError(
            "candidate_option_steps differs from the locked K29 order")
    if candidate_kind.shape != (groups, candidates) or not np.all(
            candidate_kind == expected_kinds[None, :]):
        raise LabelReliabilityError(
            "candidate_kind differs from the locked K29 order")
    if not np.all(mask):
        raise LabelReliabilityError(
            "triage requires all 29 options after pre-outcome support screening")

    expected_seeds = [int(value) for value in data_gate["required_source_seeds"]]
    observed_seeds = sorted(set(map(int, source_seed)))
    source_seed_group_counts = {
        str(seed): int(np.count_nonzero(source_seed == seed))
        for seed in expected_seeds
    }
    data_checks = {
        "independent_groups": groups == int(collection["total_groups"]),
        "trajectory_clusters": len(np.unique(trajectories))
        >= int(data_gate["min_trajectory_clusters"]),
        "source_seeds": observed_seeds == expected_seeds,
        "groups_per_source_seed": all(
            count == int(collection["groups_per_source_seed"])
            for count in source_seed_group_counts.values()),
        "candidates_per_group": candidates
        == int(data_gate["candidates_per_group"]),
        "discovery_replicas": len(discovery_indices)
        == int(data_gate["discovery_replicas"]),
        "audit_replicas": len(audit_indices)
        == int(data_gate["audit_replicas"]),
    }
    if not all(data_checks.values()):
        raise LabelReliabilityError(
            f"triage data gate failed before outcome analysis: {data_checks}")

    discovery_risk = np.mean(
        fall[:, :, discovery_indices], axis=2, dtype=np.float64)
    audit_risk = np.mean(
        fall[:, :, audit_indices], axis=2, dtype=np.float64)
    duration_results: dict[int, dict[str, Any]] = {}
    duration_group_effects: dict[int, np.ndarray] = {}
    duration_pair_groups: dict[int, np.ndarray] = {}
    duration_discovery_scores: dict[int, float] = {}
    duration_selected_audit: dict[int, np.ndarray] = {}

    for duration in RECOVERY_OPTION_DURATIONS:
        original_indices = _duration_indices(expected_steps, duration)
        duration_mask = mask[:, original_indices]
        duration_discovery = discovery_risk[:, original_indices]
        duration_audit = audit_risk[:, original_indices]
        selected_audit, audit_effect, tie_count = _selection_evaluation(
            duration_discovery, duration_audit, duration_mask)
        _, discovery_effect, _ = _selection_evaluation(
            duration_discovery, duration_discovery, duration_mask)
        pair_group, pair_counts, pair_ties = _pair_agreement(
            duration_discovery, duration_audit, duration_mask)
        duration_group_effects[duration] = audit_effect
        duration_pair_groups[duration] = pair_group
        duration_selected_audit[duration] = selected_audit
        duration_discovery_scores[duration] = _weighted_mean(
            discovery_effect, weights)
        duration_results[duration] = {
            "duration_steps": duration,
            "discovery_same_replica_selection_score_biased": (
                duration_discovery_scores[duration]),
            "audit_absolute_reduction": _weighted_mean(audit_effect, weights),
            "nominal_audit_fall_risk": _weighted_mean(
                duration_audit[:, 0], weights),
            "selected_audit_fall_risk": _weighted_mean(
                selected_audit, weights),
            "pair_order_agreement": _weighted_mean(pair_group, weights),
            "pair_comparisons": int(np.sum(pair_counts)),
            "pair_tie_comparisons": int(np.sum(pair_ties)),
            "groups_with_discovery_min_tie": int(
                np.count_nonzero(tie_count > 1)),
            "mean_discovery_min_tie_count": _weighted_mean(
                tie_count.astype(np.float64), weights),
            "discovery_winner_set_sha256": _winner_digest(
                duration_discovery, duration_mask, original_indices),
        }

    # L1 is fixed. L2--L4 are selected exactly once by discovery outcomes;
    # shortest duration is the preregistered exact-tie break.
    multistep_duration = min(
        (2, 3, 4), key=lambda duration: (
            -duration_discovery_scores[duration], duration))
    one_step_group = duration_group_effects[1]
    multistep_group = duration_group_effects[multistep_duration]
    improvement_group = multistep_group - one_step_group
    confidence = float(gates["confidence_level"])
    bootstrap = _require_mapping(
        _require_mapping(protocol.get("statistics"), "statistics").get(
            "bootstrap"), "statistics.bootstrap")
    intervals = _cluster_bootstrap(
        (
            one_step_group,
            duration_pair_groups[1],
            multistep_group,
            improvement_group,
        ),
        weights,
        trajectories,
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
        confidence=confidence,
    )
    one_step_effect = _weighted_mean(one_step_group, weights)
    one_step_pair = _weighted_mean(duration_pair_groups[1], weights)
    multistep_effect = _weighted_mean(multistep_group, weights)
    improvement = _weighted_mean(improvement_group, weights)
    one_step_sources = _source_effects(
        one_step_group, weights, source_seed, expected_seeds)
    multistep_sources = _source_effects(
        multistep_group, weights, source_seed, expected_seeds)

    gate_a = _require_mapping(gates.get("one_step_A"), "one_step_A")
    gate_b = _require_mapping(gates.get("multistep_B"), "multistep_B")
    checks_a = {
        "audit_absolute_reduction": one_step_effect
        >= float(gate_a["min_audit_absolute_reduction"]),
        "reduction_ci_low": intervals[0][0]
        > float(gate_a["min_reduction_lcb"]),
        "pair_order_agreement": one_step_pair
        >= float(gate_a["min_discovery_to_audit_pair_order_agreement"]),
        "each_source_seed_positive": all(
            value > 0.0 for value in one_step_sources.values()),
    }
    checks_b = {
        "audit_absolute_reduction": multistep_effect
        >= float(gate_b["min_audit_absolute_reduction"]),
        "reduction_ci_low": intervals[2][0]
        > float(gate_b["min_reduction_lcb"]),
        "improvement_over_L1": improvement
        >= float(gate_b["min_improvement_over_locked_L1"]),
        "improvement_over_L1_ci_low": intervals[3][0]
        > float(gate_b["min_improvement_over_L1_lcb"]),
        "each_source_seed_positive": all(
            value > 0.0 for value in multistep_sources.values()),
    }
    one_step_pass = bool(all(checks_a.values()))
    multistep_pass = bool(all(checks_b.values()))
    headroom_threshold = float(_require_mapping(
        gates.get("no_headroom_stop"), "no_headroom_stop")[
            "max_effect_for_all_options"])
    no_headroom_stop = bool(max(
        result["audit_absolute_reduction"]
        for result in duration_results.values()) < headroom_threshold)
    if multistep_pass:
        decision = "authorize_fresh_multistep_protocol_preregistration_only"
    elif one_step_pass:
        decision = "authorize_fresh_high_replica_one_step_preregistration_only"
    elif no_headroom_stop:
        decision = "stop_model_scaling_and_redesign_recovery_action_library"
    else:
        decision = "triage_failed_no_model_authorized"

    for index, duration in enumerate(RECOVERY_OPTION_DURATIONS):
        duration_results[duration]["audit_reduction_confidence_interval"] = _ci(
            _cluster_bootstrap(
                (duration_group_effects[duration],),
                weights,
                trajectories,
                replicates=int(bootstrap["replicates"]),
                seed=int(bootstrap["seed"]) + 100 + index,
                confidence=confidence,
            )[0],
            confidence,
        )
        duration_results[duration]["source_seed_effects"] = _source_effects(
            duration_group_effects[duration],
            weights,
            source_seed,
            expected_seeds,
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_name": protocol["protocol_name"],
        "protocol_contract_sha256": _canonical_sha256(protocol),
        "dataset_content_sha256": manifest.get("content_sha256"),
        "data_gate": {
            "pass": True,
            "checks": data_checks,
            "observed": {
                "groups": groups,
                "trajectory_clusters": int(len(np.unique(trajectories))),
                "source_seeds": observed_seeds,
                "source_seed_group_counts": source_seed_group_counts,
                "candidates": candidates,
                "discovery_replicas": len(discovery_indices),
                "audit_replicas": len(audit_indices),
            },
        },
        "replica_partition": partition,
        "selection": {
            "one_step_duration": 1,
            "selected_multistep_duration": multistep_duration,
            "multistep_duration_candidates": [2, 3, 4],
            "duration_tie_break": "shortest_duration",
            "audit_failure_policy": "do_not_try_runner_up",
            "uniform_candidate_tie_expectation": True,
        },
        "durations": {
            f"L{duration}": duration_results[duration]
            for duration in RECOVERY_OPTION_DURATIONS
        },
        "one_step_A": {
            "audit_absolute_reduction": one_step_effect,
            "confidence_interval": _ci(intervals[0], confidence),
            "pair_order_agreement": one_step_pair,
            "pair_confidence_interval": _ci(intervals[1], confidence),
            "source_seed_effects": one_step_sources,
            "checks": checks_a,
            "pass": one_step_pass,
        },
        "multistep_B": {
            "selected_duration": multistep_duration,
            "audit_absolute_reduction": multistep_effect,
            "confidence_interval": _ci(intervals[2], confidence),
            "improvement_over_locked_L1": improvement,
            "improvement_confidence_interval": _ci(intervals[3], confidence),
            "source_seed_effects": multistep_sources,
            "checks": checks_b,
            "pass": multistep_pass,
        },
        "no_headroom_stop": no_headroom_stop,
        "decision": decision,
        "model_training_authorized": False,
        "authorization_note": (
            "A/B pass authorizes only a fresh preregistered model/data protocol; "
            "this development triage cannot itself support an Objective-1 claim."
        ),
        "selector_calibration_authorized": False,
        "paired_closed_loop_authorized": False,
        "online_training_authorized": False,
        "phase2_authorized": False,
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "evaluate_recovery_option_triage",
]
