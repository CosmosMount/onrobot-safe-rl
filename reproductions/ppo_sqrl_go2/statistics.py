"""Outcome-blind controls and paired-seed statistical decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .protocol import POSITIVE_AUDIT_SEEDS, Protocol, TARGET_SEEDS


def matched_state_only_rejection(
    state_scores: np.ndarray, action_rejected: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    scores = np.asarray(state_scores, np.float64)
    rejected = np.asarray(action_rejected, bool)
    if scores.ndim != 1 or rejected.ndim != 2 or len(scores) != len(rejected):
        raise ValueError("state scores must be [S] and action decisions [S,K]")
    if not np.all(np.isfinite(scores)):
        raise ValueError("state scores must be finite")
    target_pairs = int(rejected.sum())
    candidates = rejected.shape[1]
    # A state-only rule can reject only complete K-action state rows. Choose
    # the nearest feasible count; deterministic stable sorting resolves ties.
    state_count = int(np.clip(np.rint(target_pairs / candidates), 0, len(scores)))
    order = np.argsort(-scores, kind="stable")
    selected = np.zeros(len(scores), dtype=bool)
    selected[order[:state_count]] = True
    threshold = (
        None if state_count in (0, len(scores)) else
        float((scores[order[state_count - 1]] + scores[order[state_count]]) / 2))
    return np.broadcast_to(selected[:, None], rejected.shape).copy(), threshold


def _rate(outcome: np.ndarray, mask: np.ndarray) -> float | None:
    return float(outcome[mask].mean()) if bool(mask.any()) else None


def risk_enrichment(outcome: np.ndarray, rejected: np.ndarray) -> dict[str, float | None]:
    fall = np.asarray(outcome, bool)
    mask = np.asarray(rejected, bool)
    if fall.shape != mask.shape:
        raise ValueError("fall outcomes and decisions must align")
    high = _rate(fall, mask)
    low = _rate(fall, ~mask)
    return {
        "rejected_fall_probability": high,
        "accepted_fall_probability": low,
        "difference": None if high is None or low is None else high - low,
        "risk_enrichment": (
            None if high is None or low in (None, 0.0) else high / low),
    }


def state_cluster_bootstrap_difference(
    outcome: np.ndarray, rejected: np.ndarray, *, seed: int, replicates: int,
) -> list[float] | None:
    """Bootstrap complete state rows, never candidates, replicas, or falls."""
    fall = np.asarray(outcome, bool)
    mask = np.asarray(rejected, bool)
    if fall.shape != mask.shape or fall.ndim < 2:
        raise ValueError("cluster bootstrap arrays must align by state")
    rows = len(fall)
    flat_fall = fall.reshape(rows, -1)
    flat_mask = mask.reshape(rows, -1)
    high_falls = (flat_fall & flat_mask).sum(1)
    high_count = flat_mask.sum(1)
    low_falls = (flat_fall & ~flat_mask).sum(1)
    low_count = (~flat_mask).sum(1)
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(replicates, np.float64)
    chunk = 2_000
    for start in range(0, replicates, chunk):
        count = min(chunk, replicates - start)
        indices = rng.integers(0, rows, size=(count, rows))
        high_n = high_count[indices].sum(1)
        low_n = low_count[indices].sum(1)
        valid = (high_n > 0) & (low_n > 0)
        block = np.full(count, np.nan)
        block[valid] = (
            high_falls[indices].sum(1)[valid] / high_n[valid]
            - low_falls[indices].sum(1)[valid] / low_n[valid])
        values[start:start + count] = block
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return np.quantile(finite, [0.025, 0.975], method="linear").tolist()


def mechanism_decision(per_seed: Mapping[int, Mapping[str, float | None]]) -> bool:
    positive = sum(
        float(per_seed[seed]["action_difference"] or 0.0) > 0
        for seed in POSITIVE_AUDIT_SEEDS)
    # The collector/report path supplies already-pooled differences to avoid
    # weighting seed summaries equally when accept/reject counts differ.
    pooled = per_seed.get(-1, {})
    return bool(
        positive >= 2
        and float(pooled.get("action_difference") or 0.0) > 0
        and float(pooled.get("action_enrichment") or 0.0)
        > float(pooled.get("state_enrichment") or 0.0))


def paired_bootstrap(differences: Sequence[float], *,
                     protocol: Protocol | None = None) -> dict[str, object]:
    cfg = protocol or Protocol()
    values = np.asarray(differences, np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be a finite non-empty vector")
    rng = np.random.Generator(np.random.PCG64(cfg.bootstrap_seed))
    draws = np.empty(cfg.bootstrap_replicates, np.float64)
    chunk = 10_000
    for start in range(0, len(draws), chunk):
        count = min(chunk, len(draws) - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        draws[start:start + count] = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": np.quantile(draws, [0.025, 0.975], method="linear").tolist(),
        "lcb95": float(np.quantile(draws, 0.05, method="linear")),
        "bootstrap_seed": cfg.bootstrap_seed,
        "bootstrap_replicates": cfg.bootstrap_replicates,
        "resampling_unit": "complete_paired_seed_row",
    }


def target_decision(rows: Sequence[Mapping[str, float]]) -> dict[str, object]:
    if tuple(int(row["seed"]) for row in rows) != TARGET_SEEDS:
        raise ValueError("target rows must be ordered complete seeds 10-15")
    transfer = np.asarray([row["ppo_transfer_falls"] for row in rows], np.float64)
    safe = np.asarray([row["ppo_safe_falls"] for row in rows], np.float64)
    difference = transfer - safe
    positive = int((difference > 0).sum())
    observed = bool(positive >= 4 and safe.sum() < transfer.sum()
                    and difference.mean() > 0)
    return {
        "ppo_sqrl_target_benefit_observed": observed,
        "positive_seeds": positive,
        "ties": int((difference == 0).sum()),
        "negative_seeds": int((difference < 0).sum()),
        "transfer_total_falls": int(transfer.sum()),
        "safe_total_falls": int(safe.sum()),
        "pooled_relative_reduction": (
            None if transfer.sum() == 0 else float((transfer.sum() - safe.sum()) / transfer.sum())),
        "differences": difference.tolist(),
        "bootstrap": paired_bootstrap(difference),
    }
