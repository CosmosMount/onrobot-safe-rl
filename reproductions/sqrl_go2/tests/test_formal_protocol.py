from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reproductions.sqrl_go2 import formal_protocol
from reproductions.sqrl_go2.diagnostics.development_gate import development_checks
from reproductions.sqrl_go2.diagnostics.formal_results import (
    bootstrap_indices,
    formal_flags,
    paired_summary,
    validate_lineage,
)
from reproductions.sqrl_go2.runners.run_formal import (
    _archive_failed, _is_complete, _jobs,
)
from reproductions.sqrl_go2.runners import common


def test_latin_branch_order_is_frozen_and_balanced():
    expected = (
        ("sac_transfer", "sqrl_mask", "sqrl_full"),
        ("sqrl_mask", "sqrl_full", "sac_transfer"),
        ("sqrl_full", "sac_transfer", "sqrl_mask"),
    )
    for offset, seed in enumerate(formal_protocol.FORMAL_SEEDS):
        assert formal_protocol.branch_order(seed) == expected[offset % 3]
    with pytest.raises(ValueError, match="not in the frozen formal roster"):
        formal_protocol.branch_order(0)


def test_formal_job_roster_has_one_pretrain_then_three_latin_targets_per_seed():
    jobs = _jobs()
    assert len(jobs) == 40
    for index, seed in enumerate(formal_protocol.FORMAL_SEEDS):
        block = jobs[index * 4:(index + 1) * 4]
        assert block[0] == (
            seed, "pretrain_030", None,
            formal_protocol.PRETRAIN_STEPS, "pretrain")
        assert tuple(job[2] for job in block[1:]) == formal_protocol.branch_order(seed)


def test_development_reward_tracking_and_velocity_are_records_only():
    pretrain = {
        str(seed): {
            "complete": True, "tail_falls": 0, "safety_updates": 1,
            "safety_replay_falls": 1, "mask_acceptance_rate": 0.5,
            "no_safe_candidate_rate": 0.5, "finite": True,
            "tail_mean_reward": -1e9,
            "tail_mean_tracking_error": 1e9,
            "mean_forward_velocity": -1e9,
        } for seed in range(3)}
    target = {
        branch: {
            "complete": True, "actor_lineage": True,
            "safety_lineage": True, "finite": True,
            **({"dual_updates": 1} if branch == "sqrl_full" else {}),
        } for branch in formal_protocol.BRANCHES}
    assert all(development_checks(pretrain, target).values())


def test_bootstrap_resamples_ten_complete_seed_rows_and_is_reproducible():
    first = bootstrap_indices()
    second = bootstrap_indices()
    assert first.shape == (100_000, 10)
    np.testing.assert_array_equal(first, second)
    assert first.min() >= 0 and first.max() < 10
    baseline = np.arange(20, 30)
    treatment = baseline - 3
    summary = paired_summary(baseline, treatment, first)
    assert summary["mean_paired_reduction"] == 3.0
    assert summary["mean_bootstrap_95_ci"] == [3.0, 3.0]
    assert summary["mean_one_sided_95_lcb"] == 3.0
    with pytest.raises(ValueError, match="ten complete paired seeds"):
        paired_summary(baseline, treatment, first[:, :9])


def test_zero_baseline_has_no_relative_reduction_definition():
    values = np.zeros(10, dtype=int)
    summary = paired_summary(values, values, bootstrap_indices())
    assert summary["pooled_relative_reduction"] is None
    assert summary["positive_seeds"] == 0


def _comparison(*, mean=1.0, lcb=1.0, positive=10, relative=0.5):
    return {
        "mean_paired_reduction": mean,
        "mean_one_sided_95_lcb": lcb,
        "positive_seeds": positive,
        "pooled_relative_reduction": relative,
    }


def test_primary_secondary_and_lagrangian_flags_use_distinct_frozen_gates():
    comparisons = {
        "sqrl_full_vs_sac_transfer": _comparison(),
        "sqrl_mask_vs_sac_transfer": _comparison(positive=1),
        "sqrl_full_vs_sqrl_mask": _comparison(relative=0.01),
    }
    flags = formal_flags(comparisons, nu_active_seeds=2)
    assert all(flags.values())
    comparisons["sqrl_full_vs_sac_transfer"] = _comparison(positive=7)
    comparisons["sqrl_mask_vs_sac_transfer"] = _comparison(relative=0.29)
    assert not formal_flags(comparisons, 2)["formal_go2_sqrl_reproduced"]
    assert not formal_flags(comparisons, 2)["sqrl_masking_effect_supported"]
    assert not formal_flags(comparisons, 1)["sqrl_lagrangian_effect_supported"]


def test_formal_lock_detects_executable_drift(tmp_path, monkeypatch):
    executable = tmp_path / "implementation.py"
    executable.write_text("frozen = 1\n", encoding="utf-8")
    monkeypatch.setattr(formal_protocol, "executable_paths", lambda: (executable,))
    lock = tmp_path / "lock.json"
    formal_protocol.write_lock(lock)
    formal_protocol.verify_lock(lock)
    executable.write_text("frozen = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lock drifted"):
        formal_protocol.verify_lock(lock)


def test_lineage_rejects_cross_seed_qsafe():
    pretrain = {"actor_sha256": "actor-10", "safety_sha256": "safe-10"}
    target = {
        "initial_actor_sha256": "actor-10",
        "initial_safety_sha256": "safe-10",
    }
    validate_lineage(pretrain, target, seed=10, branch="sqrl_full")
    target["initial_safety_sha256"] = "safe-11"
    with pytest.raises(RuntimeError, match="Q_safe lineage mismatch"):
        validate_lineage(pretrain, target, seed=10, branch="sqrl_full")


def test_completed_run_resume_validation_and_failed_attempt_preservation(tmp_path):
    directory = tmp_path / "seed_10/pretrain_030"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "status": "finished", "phase": "pretrain", "seed": 10,
        "completed_steps": 2, "protocol_id": formal_protocol.PROTOCOL_ID,
    }), encoding="utf-8")
    (directory / "metrics.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    assert _is_complete(
        directory, steps=2, phase="pretrain", seed=10, branch=None)
    archived = _archive_failed(
        tmp_path, directory, "seed_10/pretrain_030", "process_crash")
    assert archived.is_dir() and not directory.exists()
    assert (archived / "manifest.json").is_file()


def test_owned_runtime_creates_fresh_action_mailbox_before_launch(monkeypatch):
    events = []

    class Receiver:
        def __init__(self, key):
            events.append(("receiver", key))

        def bind(self):
            events.append("bind")

        def clear(self):
            events.append("clear")

        def close(self):
            events.append("close")

    class Process:
        pass

    monkeypatch.setattr(common, "SharedMemoryReceiver", Receiver)
    monkeypatch.setattr(
        common.SharedMemoryRingQueue, "unlink_existing",
        lambda key: events.append(("unlink", key)))
    monkeypatch.setattr(common.subprocess, "Popen", lambda command: (
        events.append(("popen", command)) or Process()))
    assert isinstance(common.launch_owned_runtime("target.yaml"), Process)
    assert events[:5] == [
        ("unlink", "go2_runtime_state.ordered"),
        ("receiver", "go2_runtime_action"), "bind", "clear", "close",
    ]
    assert events[5][0] == "popen"
