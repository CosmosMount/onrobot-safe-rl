"""Fail-closed loader for the active action-conditioned Objective 1 protocol."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


PROTOCOL_NAME = "objective1_action_conditioned_qsafe_v1"
PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "config" / "qsafe_action_conditioned_objective1_v1.yaml")


def load_action_qsafe_protocol(path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    path = Path(path).resolve()
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get(
            "protocol_schema_version") != 1 or protocol.get(
                "protocol_name") != PROTOCOL_NAME or protocol.get("status") != "active":
        raise ValueError("action-conditioned Objective 1 protocol identity is invalid")
    target = protocol.get("target", {})
    oracle = protocol.get("candidate_oracle_gate", {})
    training = protocol.get("action_qsafe_training", {})
    protected = protocol.get("protected_evidence", {})
    ppo = protocol.get("ppo_source", {})
    get_up = protocol.get("post_fall_get_up", {})
    runtime = protocol.get("runtime_filter", {})
    checks = {
        "revision": protocol.get("protocol_revision") == 2,
        "target_speed": target.get("command_vx_mps") == 0.30,
        "target_horizon": target.get("horizon_policy_steps") == 96,
        "target_observation": target.get("observation") == {
            "frames": 5, "dimension": 46, "deployable_only": True},
        "oracle_first": oracle.get("required_before_model_training") is True,
        "oracle_horizon": oracle.get("horizon_policy_steps") == 96,
        "oracle_no_force": oracle.get("external_force") == "forbidden",
        "oracle_crn": oracle.get("same_state_common_random_numbers") == "required",
        "oracle_no_long_recovery": oracle.get("candidates", {}).get(
            "fixed_long_recovery") == "forbidden",
        "oracle_no_get_up": oracle.get("candidates", {}).get(
            "post_fall_get_up") == "forbidden",
        "early_prefall_admission": oracle.get("admission") == {
            "natural_trajectory_fall_label_required": True,
            "minimum_steps_before_fall_inclusive": 8,
            "maximum_steps_before_fall_inclusive": 64,
            "natural_label_role": "state_admission_only",
            "unexecuted_candidate_label_from_natural_trajectory": "forbidden",
            "candidate_outcome_used_for_admission": False,
        },
        "train_after_oracle": training.get(
            "authorized_only_after_oracle_gate_pass") is True,
        "group_split": training.get("split_unit") == "source_state_group",
        "ppo_no_force": ppo.get("external_force") == "forbidden",
        "ppo_no_recovery": ppo.get("recovery_during_rollout") == "forbidden",
        "ppo_terminal_reset": ppo.get("on_first_fall") == "terminal_and_immediate_reset",
        "get_up_post_fall_only": get_up.get("allowed_only_after_terminal_fall") is True,
        "get_up_not_candidate": get_up.get("allowed_as_pre_fall_candidate") is False,
        "receding_filter": runtime.get("reobserve_and_reselect_each_policy_step") is True,
        "no_default_long_option": runtime.get(
            "persistent_long_option_by_default") == "forbidden",
        "old_results_invalid": protected.get(
            "old_state_only_recovery_results_objective1_eligible") is False,
        "phase2_locked": protected.get(
            "objective2_authorized_before_objective1_pass") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"action-conditioned Objective 1 protocol drift: {failed}")
    return protocol


def action_qsafe_protocol_sha256(path: str | Path = PROTOCOL_PATH) -> str:
    path = Path(path).resolve()
    load_action_qsafe_protocol(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
