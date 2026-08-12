"""Validation and pre-training gate for grouped counterfactual branches."""

from __future__ import annotations

from itertools import combinations

import numpy as np


REQUIRED_FIELDS = frozenset({
    "state_id", "episode_id", "split", "risk_stratum", "collector_seed",
    "observation_history", "candidate_index", "candidate_distance",
    "candidate_distance_bin", "critic_action", "absolute_q_target",
    "replica_id", "crn_id", "h96_fall", "first_fall_step",
})


def validate_grouped_branches(data: np.lib.npyio.NpzFile, replicas: int) -> None:
    missing = REQUIRED_FIELDS - set(data.files)
    if missing:
        raise ValueError(f"grouped counterfactual dataset lacks {sorted(missing)}")
    n = len(data["state_id"])
    if data["h96_fall"].shape != (n, 16, replicas) or (
            data["first_fall_step"].shape != (n, 16, replicas)):
        raise ValueError("counterfactual state groups must be complete 16xR")
    if len(set(bytes(value) for value in data["state_id"])) != n:
        raise ValueError("duplicate state identity")
    if len(set(bytes(value) for value in data["episode_id"])) != n:
        raise ValueError("an episode contributed more than one state")
    expected_candidates = np.broadcast_to(np.arange(16), (n, 16))
    if not np.array_equal(data["candidate_index"], expected_candidates):
        raise ValueError("candidate identities are incomplete or reordered")
    expected_replicas = np.broadcast_to(np.arange(1, replicas + 1), (n, 16, replicas))
    if not np.array_equal(data["replica_id"], expected_replicas):
        raise ValueError("replica identities are incomplete or reordered")
    if not np.array_equal(data["critic_action"], data["absolute_q_target"]):
        raise ValueError("critic_action is not the physical PD q-target")
    fall = data["h96_fall"]
    first = data["first_fall_step"]
    if np.any(first[fall] < 1) or np.any(first[fall] > 96) or np.any(first[~fall] != 97):
        raise ValueError("H96 first-fall indexing is inconsistent")
    # Every candidate in a state/replica must use the same CRN; replicas differ.
    crn = data["crn_id"]
    if crn.shape != (n, 16, replicas) or np.any(crn != crn[:, :1, :]):
        raise ValueError("paired CRN differs between candidates")
    if any(len(set(bytes(value) for value in crn[state, 0])) != replicas
           for state in range(n)):
        raise ValueError("replicas reuse the same CRN")


def informativeness_report(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    replicas = data["h96_fall"].shape[2]
    validate_grouped_branches(data, replicas)
    if replicas != 4:
        raise ValueError("development informativeness requires R4")
    risk = data["h96_fall"].mean(axis=2)
    spread = risk.max(axis=1) - risk.min(axis=1)
    non_tie = np.zeros(len(risk), np.int32)
    strong = np.zeros(len(risk), np.int32)
    for state, values in enumerate(risk):
        delta = np.asarray([abs(values[i] - values[j])
                            for i, j in combinations(range(16), 2)])
        non_tie[state] = int(np.count_nonzero(delta > 0))
        strong[state] = int(np.count_nonzero(delta >= 0.5))
    stratum = data["risk_stratum"].astype("U")
    primary = np.isin(stratum, ["boundary", "medium"])
    median = float(np.median(spread[primary]))
    strong_fraction = float(np.mean(strong[primary] > 0))
    nominal = risk[:, 0]
    oracle = risk.min(axis=1)
    report: dict[str, object] = {
        "schema_version": "qsafe.counterfactual_informativeness.v2",
        "states": len(risk), "replicas": replicas,
        "median_empirical_risk_range_boundary_medium": median,
        "states_with_strong_pair_fraction_boundary_medium": strong_fraction,
        "fraction_delta_at_least_0_25": float(np.mean(spread >= 0.25)),
        "fraction_delta_at_least_0_50": float(np.mean(spread >= 0.50)),
        "mean_non_tie_pairs_per_state": float(non_tie.mean()),
        "mean_strong_pairs_per_state": float(strong.mean()),
        "development_empirical_oracle_reduction": float(np.mean(nominal - oracle)),
        "counterfactual_supervision_informative": bool(
            median >= 0.25 and strong_fraction >= 0.30),
        "strata": {}, "collector_seed": {},
    }
    for name in ("boundary", "medium", "normal"):
        mask = stratum == name
        report["strata"][name] = {
            "states": int(mask.sum()),
            "median_risk_range": (
                float(np.median(spread[mask])) if np.any(mask) else None),
            "strong_pair_state_fraction": (
                float(np.mean(strong[mask] > 0)) if np.any(mask) else None),
        }
    for seed in (137, 138):
        mask = data["collector_seed"] == seed
        report["collector_seed"][str(seed)] = {
            "states": int(mask.sum()), "median_risk_range": float(np.median(spread[mask])),
            "strong_pair_state_fraction": float(np.mean(strong[mask] > 0)),
        }
    return report
