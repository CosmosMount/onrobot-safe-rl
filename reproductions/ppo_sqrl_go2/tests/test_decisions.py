from __future__ import annotations

import numpy as np

from reproductions.ppo_sqrl_go2.protocol import Protocol
from reproductions.ppo_sqrl_go2.stability import cotrain_stability
from reproductions.ppo_sqrl_go2.statistics import (
    matched_state_only_rejection, paired_bootstrap,
    state_cluster_bootstrap_difference, target_decision)


def test_state_only_rejects_complete_rows_at_matched_rate():
    action = np.asarray([
        [1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 0]], bool)
    state, threshold = matched_state_only_rejection(
        np.asarray([0.1, 0.9, 0.2, 0.8]), action)
    assert np.all(state == state[:, :1])
    # Seven action rejections rounds to two complete four-action rows.
    assert state.sum() == 8
    assert state[1, 0] and state[3, 0]


def test_reward_and_velocity_do_not_change_cotrain_gate():
    rows = [{
        "seed": seed, "task_transitions": 30_000_000,
        "safety_transitions": 7_500_000, "safety_updates": 1,
        "safety_total_falls": 1, "safety_buffer_retained_falls": 1,
        "final_safe_fraction": 0.5, "all_numerics_finite": True,
        "reward": -1e30, "velocity": -1e30,
    } for seed in (0, 1, 2)]
    assert cotrain_stability(rows)["ppo_sqrl_cotrain_stable"]


def test_target_gate_directions_ties_and_diagnostics():
    transfer = [10, 10, 10, 10, 10, 10]
    safe = [9, 8, 7, 6, 10, 11]
    rows = [{
        "seed": seed, "ppo_transfer_falls": a, "ppo_safe_falls": b,
        "reward": -1e30, "velocity": 0.0,
    } for seed, a, b in zip(range(10, 16), transfer, safe)]
    result = target_decision(rows)
    assert result["ppo_sqrl_target_benefit_observed"]
    assert result["positive_seeds"] == 4
    assert result["ties"] == 1


def test_bootstrap_is_seed_row_vector_and_repeatable():
    cfg = Protocol(bootstrap_replicates=1000)
    first = paired_bootstrap([1, 2, 3, 4, 5, 6], protocol=cfg)
    second = paired_bootstrap([1, 2, 3, 4, 5, 6], protocol=cfg)
    assert first == second
    assert first["resampling_unit"] == "complete_paired_seed_row"


def test_cluster_bootstrap_resamples_whole_states():
    outcome = np.asarray([[[1, 1], [0, 0]], [[0, 0], [1, 1]]], bool)
    rejected = np.asarray([[[1, 1], [0, 0]], [[1, 1], [0, 0]]], bool)
    interval = state_cluster_bootstrap_difference(
        outcome, rejected, seed=3, replicates=1000)
    assert interval is not None
    assert interval[0] <= 0 <= interval[1]
