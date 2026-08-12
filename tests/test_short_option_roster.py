import numpy as np

from safety_data.short_option_roster import select_fresh_boundary_roster


def test_fresh_roster_is_balanced_and_episode_disjoint() -> None:
    rows = []
    for seed in (137, 138):
        for index in range(340):
            rows.append({
                "state_id": f"state-{seed}-{index}",
                "episode_key": f"episode-{seed}-{index}",
                "collector_seed": seed,
            })
    selected = select_fresh_boundary_roster(rows)
    assert len(selected) == 600
    assert sum(row["collector_seed"] == 137 for row in selected) == 300
    assert len({row["episode_key"] for row in selected}) == 600
