"""Machine decision for the three-seed co-training development round."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .protocol import COTRAIN_SEEDS, Protocol


def cotrain_stability(rows: Sequence[Mapping[str, float]], *,
                      protocol: Protocol | None = None) -> dict[str, object]:
    cfg = protocol or Protocol()
    if tuple(int(row["seed"]) for row in rows) != COTRAIN_SEEDS:
        raise ValueError("cotrain rows must be ordered complete seeds 0-2")
    fields = (
        "task_transitions", "safety_transitions", "safety_updates",
        "safety_total_falls", "safety_buffer_retained_falls",
        "final_safe_fraction")
    finite = all(np.isfinite(float(row[field])) for row in rows for field in fields)
    finite = finite and all(bool(row["all_numerics_finite"]) for row in rows)
    exact = all(
        int(row["task_transitions"]) == cfg.pretrain_task_transitions
        and int(row["safety_transitions"]) == cfg.pretrain_safety_transitions
        for row in rows)
    supervised = all(
        int(row["safety_updates"]) > 0
        and int(row["safety_buffer_retained_falls"]) > 0
        for row in rows)
    nondegenerate = all(
        cfg.final_safe_fraction_low < float(row["final_safe_fraction"])
        < cfg.final_safe_fraction_high for row in rows)
    stable = bool(finite and exact and supervised and nondegenerate)
    return {
        "ppo_sqrl_cotrain_stable": stable,
        "finite": finite,
        "exact_budgets": exact,
        "failure_supervision_all_seeds": supervised,
        "nondegenerate_all_seeds": nondegenerate,
    }
