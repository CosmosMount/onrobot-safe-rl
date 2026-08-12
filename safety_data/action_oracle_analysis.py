"""Discovery/audit oracle analysis for action-conditioned candidate spaces."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from safety_data.action_qsafe_protocol import action_qsafe_protocol_sha256
from safety_data.schema import GroupedBranchDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_candidate(
    fall: np.ndarray, candidate_mask: np.ndarray,
    candidate_requested: np.ndarray,
) -> np.ndarray:
    discovery_risk = fall[:, :, :fall.shape[2] // 2].mean(axis=2)
    discovery_risk = np.where(candidate_mask, discovery_risk, np.inf)
    deviation = np.sqrt(np.mean(np.square(
        candidate_requested - candidate_requested[:, :1]), axis=2))
    selected = np.empty(len(fall), dtype=np.int64)
    for group in range(len(fall)):
        best = np.min(discovery_risk[group])
        tied = np.flatnonzero(discovery_risk[group] == best)
        selected[group] = int(tied[np.argmin(deviation[group, tied])])
    return selected


def _bootstrap_lcb(
    reduction: np.ndarray, actor_seed: np.ndarray, source_seed: np.ndarray,
    *, replicates: int, seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    actors = np.unique(actor_seed)
    draws = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        values: list[float] = []
        sampled_actors = rng.choice(actors, size=len(actors), replace=True)
        for actor in sampled_actors:
            sources = np.unique(source_seed[actor_seed == actor])
            sampled_sources = rng.choice(sources, size=len(sources), replace=True)
            actor_values: list[float] = []
            for source in sampled_sources:
                rows = np.flatnonzero(
                    (actor_seed == actor) & (source_seed == source))
                sampled_rows = rng.choice(rows, size=len(rows), replace=True)
                actor_values.append(float(np.mean(reduction[sampled_rows])))
            values.append(float(np.mean(actor_values)))
        draws[draw] = float(np.mean(values))
    return float(np.quantile(draws, 0.05, method="linear"))


def analyze_action_oracle(
    paths: Sequence[str | Path], *, role: str = "development",
    bootstrap_replicates: int = 10_000, bootstrap_seed: int = 20260812,
) -> dict[str, Any]:
    """Report candidate and discovery-selected audit risk without fitting a model."""
    if role not in {"development", "protected"}:
        raise ValueError("role must be development or protected")
    if isinstance(bootstrap_replicates, bool) or bootstrap_replicates < 1000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    resolved = [Path(path).resolve() for path in paths]
    if not resolved:
        raise ValueError("at least one action branch dataset is required")
    datasets = [GroupedBranchDataset.load(path) for path in resolved]
    protocol_sha = action_qsafe_protocol_sha256()
    reference_kinds = tuple(datasets[0].arrays["candidate_kind"][0].tolist())
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "fall", "candidate_mask", "candidate_requested",
            "candidate_kind", "source_seed", "policy_training_seed")}
    source_summaries = []
    replicas = datasets[0].replica_count
    for path, dataset in zip(resolved, datasets, strict=True):
        validation = dataset.validate()
        collection = dataset.manifest.get("collection_protocol", {})
        if dataset.manifest.get("horizon_steps") != 96 or (
                collection.get("version") != "qsafe.natural_sac_action_branches.v1"):
            raise ValueError("oracle input is not an H96 natural action branch dataset")
        if collection.get("external_force") != "verified_zero" or (
                collection.get("impulse") != "forbidden") or (
                collection.get("model_training_authorized") is not False):
            raise ValueError("oracle input violates force or pre-training gate")
        if collection.get("objective1_protocol_sha256") != protocol_sha:
            raise ValueError("oracle input protocol hash differs from active protocol")
        if dataset.replica_count != replicas or replicas % 2 != 0:
            raise ValueError("all oracle inputs require one common even replica count")
        kinds = tuple(dataset.arrays["candidate_kind"][0].tolist())
        if kinds != reference_kinds or dataset.candidate_count != 24:
            raise ValueError("oracle inputs do not share the K24 action space")
        for name in arrays:
            arrays[name].append(np.asarray(dataset.arrays[name]))
        source_summaries.append({
            "path": str(path), "file_sha256": _sha256(path),
            "groups": dataset.group_count,
            "source_seeds": np.unique(dataset.arrays["source_seed"]).tolist(),
            "actor_seeds": np.unique(
                dataset.arrays["policy_training_seed"]).tolist(),
            "validation": validation,
        })
    combined = {name: np.concatenate(value, axis=0) for name, value in arrays.items()}
    fall = combined["fall"].astype(np.float64)
    mask = combined["candidate_mask"].astype(bool)
    requested = combined["candidate_requested"].astype(np.float64)
    actor_seed = combined["policy_training_seed"].astype(np.int64)
    source_seed = combined["source_seed"].astype(np.int64)
    audit = slice(replicas // 2, replicas)
    selected = _selected_candidate(fall, mask, requested)
    rows = np.arange(len(fall))
    nominal = fall[:, 0, audit].mean(axis=1)
    oracle = fall[rows, selected, audit].mean(axis=1)
    reduction = nominal - oracle
    candidate_rates = []
    for index, kind in enumerate(reference_kinds):
        valid = mask[:, index]
        candidate_rates.append({
            "candidate_index": index,
            "candidate_kind": kind,
            "valid_groups": int(valid.sum()),
            "audit_fall_rate": (
                float(fall[valid, index, audit].mean()) if np.any(valid) else None),
        })
    actor_effects = {
        str(int(actor)): float(reduction[actor_seed == actor].mean())
        for actor in np.unique(actor_seed)}
    source_effects = {
        str(int(source)): float(reduction[source_seed == source].mean())
        for source in np.unique(source_seed)}
    lcb = _bootstrap_lcb(
        reduction, actor_seed, source_seed,
        replicates=int(bootstrap_replicates), seed=int(bootstrap_seed))
    protected_structure = (
        len(actor_effects) >= 2 and len(source_effects) >= 4 and replicas >= 8)
    cross_actor_positive = all(value > 0.0 for value in actor_effects.values())
    source_positive_fraction = float(np.mean([
        value > 0.0 for value in source_effects.values()]))
    passed = bool(
        role == "protected" and protected_structure and lcb > 0.0
        and cross_actor_positive and source_positive_fraction >= 0.75)
    return {
        "schema_version": "qsafe.action_candidate_oracle_report.v1",
        "role": role,
        "objective1_protocol_sha256": protocol_sha,
        "groups": len(fall),
        "candidate_count": len(reference_kinds),
        "replicas": replicas,
        "discovery_replicas": replicas // 2,
        "audit_replicas": replicas // 2,
        "selection": (
            "minimum_discovery_fall_risk_then_minimum_nominal_RMS_deviation"),
        "nominal_audit_fall_rate": float(nominal.mean()),
        "oracle_best_audit_fall_rate": float(oracle.mean()),
        "oracle_absolute_reduction": float(reduction.mean()),
        "oracle_reduction_one_sided_95_lcb": lcb,
        "candidate_audit_fall_rates": candidate_rates,
        "oracle_selected_candidate_counts": {
            reference_kinds[index]: int(np.sum(selected == index))
            for index in np.unique(selected)},
        "actor_seed_effects": actor_effects,
        "source_seed_effects": source_effects,
        "source_positive_fraction": source_positive_fraction,
        "checks": {
            "protected_structure": protected_structure,
            "oracle_reduction_lcb_above_zero": lcb > 0.0,
            "every_actor_direction_positive": cross_actor_positive,
            "at_least_75_percent_source_directions_positive": (
                source_positive_fraction >= 0.75),
        },
        "candidate_oracle_gate_pass": passed,
        "model_training_authorized": passed,
        "selector_training_executed": False,
        "objective1_pass": False,
        "phase2_authorized": False,
        "inputs": source_summaries,
        "bootstrap": {
            "kind": "actor_then_source_then_state_group_cluster_bootstrap",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "confidence": "one_sided_0.95",
        },
    }
