from __future__ import annotations

from collections import Counter

from safety_data.counterfactual_states import (
    ROLE_COUNTS, assign_episode_disjoint_roster, state_identity,
)


def test_roster_has_exact_balanced_quota_and_episode_isolation() -> None:
    rows = []
    for seed in (137, 138):
        for stratum in ("boundary", "medium", "normal"):
            for index in range(1300):
                episode = f"{seed}-{stratum}-{index}"
                rows.append({
                    "state_id": state_identity(episode, stratum, index),
                    "episode_key": episode, "collector_seed": seed,
                    "risk_stratum": stratum,
                })
    selected = assign_episode_disjoint_roster(rows)
    assert len(selected) == 2400
    assert len({row["episode_key"] for row in selected}) == 2400
    observed = Counter((row["split"], row["risk_stratum"], row["collector_seed"])
                       for row in selected)
    for role, strata in ROLE_COUNTS.items():
        for stratum, total in strata.items():
            assert observed[role, stratum, 137] == total // 2
            assert observed[role, stratum, 138] == total // 2
