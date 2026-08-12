from __future__ import annotations

from pathlib import Path

import numpy as np

from safety_data.counterfactual_gate import informativeness_report


def test_informativeness_and_crn_contract(tmp_path: Path) -> None:
    n, replicas = 20, 4
    fall = np.zeros((n, 16, replicas), bool)
    fall[:, 8:] = True
    first = np.where(fall, 96, 97).astype(np.int16)
    crn = np.asarray([[[f"{state}-{replica}" for replica in range(replicas)]
                       for _ in range(16)] for state in range(n)], "S64")
    path = tmp_path / "groups.npz"
    np.savez(path,
        state_id=np.asarray([f"s{i}" for i in range(n)], "S64"),
        episode_id=np.asarray([f"e{i}" for i in range(n)], "S64"),
        split=np.asarray(["train"] * n),
        risk_stratum=np.asarray(["boundary"] * n),
        collector_seed=np.asarray([137, 138] * (n // 2)),
        observation_history=np.zeros((n, 5, 46), np.float32),
        candidate_index=np.broadcast_to(np.arange(16), (n, 16)),
        candidate_distance=np.zeros((n, 16), np.float32),
        candidate_distance_bin=np.full((n, 16), "near"),
        critic_action=np.zeros((n, 16, 12), np.float32),
        absolute_q_target=np.zeros((n, 16, 12), np.float32),
        replica_id=np.broadcast_to(np.arange(1, 5), (n, 16, 4)),
        crn_id=crn, h96_fall=fall, first_fall_step=first)
    with np.load(path, allow_pickle=False) as data:
        report = informativeness_report(data)
    assert report["counterfactual_supervision_informative"] is True
    assert report["median_empirical_risk_range_boundary_medium"] == 1.0
