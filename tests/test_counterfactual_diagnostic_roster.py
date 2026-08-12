from __future__ import annotations

from collections import Counter
import inspect

import numpy as np

from safety_data.counterfactual_diagnostic_roster import (
    QUOTAS, select_diagnostic_rows,
)


def test_diagnostic_roster_is_exact_balanced_and_outcome_blind() -> None:
    state, split, seed, stratum = [], [], [], []
    for role in QUOTAS:
        for collector in (137, 138):
            for risk in ("boundary", "medium", "normal"):
                for index in range(200):
                    state.append(f"{role}-{collector}-{risk}-{index}".encode())
                    split.append(role); seed.append(collector); stratum.append(risk)
    rows = select_diagnostic_rows(
        np.asarray(state, "S64"), np.asarray(split), np.asarray(seed),
        np.asarray(stratum))
    assert len(rows) == len(set(rows.tolist())) == 400
    observed = Counter((split[row], seed[row], stratum[row]) for row in rows)
    for role, seed_counts in QUOTAS.items():
        for collector, strata in seed_counts.items():
            for risk, count in strata.items():
                assert observed[role, collector, risk] == count
    assert tuple(inspect.signature(select_diagnostic_rows).parameters) == (
        "state_id", "split", "collector_seed", "risk_stratum")


def test_selection_is_deterministic() -> None:
    state = np.asarray([f"state-{i}" for i in range(2400)], "S64")
    split = np.asarray(["train"] * 1920 + ["calibration"] * 480)
    # Build a full-factorial-like metadata corpus with ample cells.
    seed = np.tile(np.repeat([137, 138], 600), 2)[:2400]
    stratum = np.tile(np.repeat(["boundary", "medium", "normal"], 200), 4)
    # The synthetic split/seed layout does not populate every cell; shuffle in
    # a deterministic interleaving to guarantee coverage.
    tuples = []
    for role, total in (("train", 1600), ("calibration", 800)):
        for index in range(total):
            tuples.append((role, 137 + index % 2,
                           ("boundary", "medium", "normal")[index % 3]))
    split = np.asarray([value[0] for value in tuples])
    seed = np.asarray([value[1] for value in tuples])
    stratum = np.asarray([value[2] for value in tuples])
    first = select_diagnostic_rows(state, split, seed, stratum)
    second = select_diagnostic_rows(state, split, seed, stratum)
    np.testing.assert_array_equal(first, second)
