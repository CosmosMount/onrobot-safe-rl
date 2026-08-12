"""Locked protocol access for the PPO-parallel SQRL data study."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


_PATH = Path(__file__).resolve().parents[1] / "config/qsafe_ppo_sqrl_data_v1.yaml"


def load_ppo_sqrl_protocol() -> dict[str, Any]:
    protocol = yaml.safe_load(_PATH.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get("status") != "active":
        raise ValueError("PPO SQRL protocol is missing or inactive")
    task = protocol["task"]
    critic = protocol["critic"]
    dataset = protocol["ppo_master_dataset"]
    gate = protocol["ppo_branching_gate"]
    checks = {
        "command": task.get("command_mps") == [0.30, 0.0, 0.0],
        "parallelism": task.get("parallel_environments") == 2000,
        "no_forces": task.get("external_push") == "forbidden"
        and task.get("external_impulse") == "forbidden",
        "critic_action": critic.get("critic_action", {}).get("field")
        == "critic_action"
        and critic.get("critic_action", {}).get("equals_absolute_q_target") is True,
        "cost_index": critic.get("cost_index") == "c_t_plus_1",
        "stochastic": dataset.get("action_sampling") == "stochastic",
        "seeds": dataset.get("seeds") == [137, 138],
        "stages": list(dataset.get("stages", {})) == ["early", "boundary", "mature"],
        "nested": dataset.get("nested_aggregate_transition_counts")
        == [1_000_000, 3_000_000, 5_000_000],
        "branching": gate.get("protected_state_groups") == 200
        and gate.get("candidates") == 16
        and gate.get("discovery_replicas") == [1, 2, 3, 4]
        and gate.get("evaluation_replicas") == [5, 6, 7, 8],
        "first_round": protocol.get("first_round", {}).get(
            "allow_new_sac_50k_training") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"PPO SQRL protocol drifted: {failed}")
    return protocol


def ppo_sqrl_protocol_sha256() -> str:
    load_ppo_sqrl_protocol()
    return hashlib.sha256(_PATH.read_bytes()).hexdigest()
