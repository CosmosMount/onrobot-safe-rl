"""Outcome-blind roster selection for supervision-signal diagnostics."""

from __future__ import annotations

import hashlib

import numpy as np


QUOTAS = {
    "train": {
        137: {"boundary": 80, "medium": 40, "normal": 40},
        138: {"boundary": 80, "medium": 40, "normal": 40},
    },
    "calibration": {
        137: {"boundary": 20, "medium": 10, "normal": 10},
        138: {"boundary": 20, "medium": 10, "normal": 10},
    },
}


def _score(state_id: bytes, split: str, seed: int, stratum: str) -> bytes:
    namespace = f"qsafe.counterfactual.signal.v1:{split}:{seed}:{stratum}".encode()
    return hashlib.sha256(namespace + b"\0" + state_id).digest()


def select_diagnostic_rows(
    state_id: np.ndarray,
    split: np.ndarray,
    collector_seed: np.ndarray,
    risk_stratum: np.ndarray,
) -> np.ndarray:
    """Return 400 source rows; the API intentionally has no outcome input."""
    state_id = np.asarray(state_id, "S64")
    split = np.asarray(split).astype("U")
    collector_seed = np.asarray(collector_seed, np.int16)
    risk_stratum = np.asarray(risk_stratum).astype("U")
    n = len(state_id)
    if any(len(value) != n for value in (split, collector_seed, risk_stratum)):
        raise ValueError("development metadata lengths differ")
    if len(set(bytes(value) for value in state_id)) != n:
        raise ValueError("development state identities are not unique")
    selected: list[int] = []
    for role, seed_counts in QUOTAS.items():
        for seed, stratum_counts in seed_counts.items():
            for stratum, required in stratum_counts.items():
                eligible = np.flatnonzero(
                    (split == role) & (collector_seed == seed)
                    & (risk_stratum == stratum))
                if len(eligible) < required:
                    raise RuntimeError(
                        f"insufficient {role}/{seed}/{stratum}: {len(eligible)}")
                ordered = sorted(
                    eligible.tolist(),
                    key=lambda row: _score(bytes(state_id[row]), role, seed, stratum),
                )
                selected.extend(ordered[:required])
    if len(selected) != 400 or len(set(selected)) != 400:
        raise RuntimeError("diagnostic roster is not exactly 400 unique states")
    return np.asarray(selected, np.int32)

