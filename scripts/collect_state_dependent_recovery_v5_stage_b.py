#!/usr/bin/env python3
"""Prepare, collect, or finalize one frozen V5 Stage-B evidence role."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from safety_data.policies import load_frozen_droq_policy
from safety_data.recovery_behaviors import build_recovery_behavior_library
from safety_data.state_dependent_recovery_v5 import (
    PROTOCOL_CONTRACT_SHA256,
    PROTOCOL_FILE_SHA256,
    PROTOCOL_PATH,
    load_state_dependent_recovery_v5_protocol,
)
from safety_data.state_dependent_recovery_v5_stage_b import (
    EXECUTION_PROTOCOL_PATH,
    RECOVERY_LIBRARY_FINGERPRINT_SHA256,
    ROLE_ORDER,
    ROLE_SOURCE_SEEDS,
    assignment_for,
    load_stage_b_execution_protocol,
    require_clean_stage_b_generator,
    stage_b_artifact_root,
    validate_stage_a_authorization,
)
from safety_data.state_dependent_recovery_v5_stage_b_collector import (
    collect_preflighted_stage_b_role,
    preflight_stage_b_role_collection,
    production_collection_config,
)
from safety_data.state_dependent_recovery_v5_stage_b_workflow import (
    collect_stage_b_source_once,
    finalize_stage_b_role,
    prepare_stage_b_role,
)
from safety_data.stage_b_paths import compile_stage_b_model_test_commitment
from scripts.collect_closed_loop_recovery_triage import (
    _verify_policy,
    _verify_runtime_contract,
)
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv
from train.state_dependent_recovery_v5_stage_b_actor_bank import (
    actor_identity_for,
    load_actor_bank_manifest,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ACTOR_BANK = (
    _ROOT / "saved" / "qsafe_development"
    / "state_dependent_recovery_v5" / "stage-b"
    / "actor-bank-manifest.json"
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else _ROOT / path


def _load_context(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = _resolve_repo_path(args.protocol)
    supplement_path = _resolve_repo_path(args.supplement)
    if protocol_path.resolve(strict=False) != PROTOCOL_PATH.resolve(
        strict=False
    ) or supplement_path.resolve(strict=False) != EXECUTION_PROTOCOL_PATH.resolve(
        strict=False
    ):
        raise RuntimeError("Stage-B CLI requires the canonical frozen protocols")
    parent = load_state_dependent_recovery_v5_protocol(protocol_path)
    execution = load_stage_b_execution_protocol(supplement_path)
    validate_stage_a_authorization(execution)
    generator_commit = require_clean_stage_b_generator()
    root = stage_b_artifact_root(parent)
    actor_bank_path = _resolve_repo_path(args.actor_bank)
    expected_actor_bank = root / "actor-bank-manifest.json"
    if actor_bank_path.resolve(strict=False) != expected_actor_bank.resolve(
        strict=False
    ):
        raise RuntimeError("Stage-B CLI requires the canonical actor bank")
    actor_bank = load_actor_bank_manifest(
        actor_bank_path,
        expected_bindings={"generator_commit": generator_commit},
    )
    actor_bank_file_sha256 = _sha256(actor_bank_path)
    return {
        "parent": parent,
        "execution": execution,
        "generator_commit": generator_commit,
        "stage_b_root": root,
        "actor_bank": actor_bank,
        "actor_bank_path": actor_bank_path,
        "actor_bank_file_sha256": actor_bank_file_sha256,
    }


def _revalidate_live_context(context: Mapping[str, Any]) -> None:
    """Recheck clean generator and every live actor-bank binding."""
    generator_commit = require_clean_stage_b_generator()
    if generator_commit != context["generator_commit"]:
        raise RuntimeError("Stage-B generator changed after CLI preflight")
    actor_bank_path = Path(context["actor_bank_path"])
    if _sha256(actor_bank_path) != context["actor_bank_file_sha256"]:
        raise RuntimeError("actor-bank manifest bytes changed after CLI preflight")
    observed = load_actor_bank_manifest(
        actor_bank_path,
        expected_bindings={
            "manifest_file_sha256": context["actor_bank_file_sha256"],
            "actor_bank_contract_sha256": context["actor_bank"][
                "actor_bank_contract_sha256"],
            "generator_commit": generator_commit,
        },
    )
    if observed != context["actor_bank"]:
        raise RuntimeError("live actor-bank identity changed after CLI preflight")


def _runtime_preflight(
    context: Mapping[str, Any],
    *,
    role: str,
    source_seed: int,
) -> tuple[Any, dict[str, Any]]:
    parent = context["parent"]
    actor_bank = context["actor_bank"]
    assignment = assignment_for(role, source_seed)
    identity = actor_identity_for(
        actor_bank,
        role=role,
        actor_seed=assignment.actor_training_seed,
        checkpoint_step=assignment.checkpoint_step,
    )
    config_path = _resolve_repo_path(parent["policy_config"]["path"])
    if str(config_path) != str(Path(actor_bank["training_config_binding"]["path"])):
        raise RuntimeError("actor-bank training config path differs from parent")
    target = parent["target"]
    mature_entry = parent["mature_recovery_policy"]
    mature_checkpoint = _resolve_repo_path(mature_entry["checkpoint"])
    source_checkpoint = Path(identity["checkpoint_path"])

    torch.set_num_threads(1)
    robot_cfg, train_cfg, _ = load_app_config(config_path)
    if not np.isclose(
        float(robot_cfg.move_speed),
        float(target["command_speed_mps"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("runtime command speed differs from Stage-B target")
    if not np.isclose(
        float(robot_cfg.success_orientation_rad),
        float(target["failure"]["max_abs_roll_pitch_rad"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("runtime failure tilt differs from Stage-B target")
    robot_cfg = replace(
        robot_cfg,
        fallen_orientation_rad=robot_cfg.success_orientation_rad,
    )
    early_policy = load_frozen_droq_policy(
        source_checkpoint,
        config_path,
        observation_dim=robot_cfg.obs_dim,
        action_dim=robot_cfg.num_joints,
        training_step=assignment.checkpoint_step,
        device="cpu",
    )
    mature_policy = load_frozen_droq_policy(
        mature_checkpoint,
        config_path,
        observation_dim=robot_cfg.obs_dim,
        action_dim=robot_cfg.num_joints,
        training_step=int(mature_entry["training_step"]),
        device="cpu",
    )
    _verify_policy(early_policy, identity, "Stage-B source")
    _verify_policy(mature_policy, mature_entry, "Stage-B mature recovery")
    env = MujocoSnapshotEnv(
        Path(str(target["model_mjcf"])),
        robot_cfg,
        policy_frequency=train_cfg.control_frequency,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter,
    )
    _verify_runtime_contract(env, robot_cfg, train_cfg, parent)
    recovery_program = build_recovery_behavior_library(
        mature_policy, env.action_applier)
    if recovery_program.fingerprint() != RECOVERY_LIBRARY_FINGERPRINT_SHA256:
        raise RuntimeError("Stage-B recovery library fingerprint differs")

    collection = parent["collection"]
    impulse = collection["source_impulse"]
    pre_screen = collection["proposal_pre_screen"]
    config = production_collection_config(
        role=role,
        source_seed=source_seed,
        max_episode_steps=int(collection["max_episode_steps"]),
        max_trajectories=int(collection["max_source_trajectories_per_seed"]),
        proposal_cooldown_steps=int(
            collection["rejected_proposal_cooldown_policy_steps"]),
        settle_seconds=float(collection["settle_seconds"]),
        source_impulse_interval_steps=int(impulse["interval_policy_steps"]),
        source_linear_std_mps=float(impulse["linear_std_mps"]),
        source_angular_std_radps=float(impulse["angular_std_radps"]),
        proposal_min_tilt_rad=float(pre_screen["min_tilt_rad_inclusive"]),
        proposal_max_height_m=float(pre_screen["max_height_m_inclusive"]),
    )
    prepared = preflight_stage_b_role_collection(
        role=role,
        env=env,
        early_policy=early_policy,
        recovery_program=recovery_program,
        policy_set_manifest=actor_bank,
        config=config,
        generator_commit=context["generator_commit"],
        parent_protocol_sha256=PROTOCOL_FILE_SHA256,
        parent_protocol_contract_sha256=PROTOCOL_CONTRACT_SHA256,
        production_contract=True,
    )
    return prepared, identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROTOCOL_PATH))
    parser.add_argument("--supplement", default=str(EXECUTION_PROTOCOL_PATH))
    parser.add_argument("--actor-bank", default=str(_DEFAULT_ACTOR_BANK))
    subparsers = parser.add_subparsers(dest="operation", required=True)

    prepare = subparsers.add_parser("prepare-role")
    prepare.add_argument("--role", choices=ROLE_ORDER, required=True)

    collect = subparsers.add_parser("collect-source")
    collect.add_argument("--role", choices=ROLE_ORDER, required=True)
    collect.add_argument("--source-seed", type=int, required=True)
    collect.add_argument("--expected-role-attempt-sha256", required=True)

    finalize = subparsers.add_parser("finalize-role")
    finalize.add_argument("--role", choices=ROLE_ORDER, required=True)
    finalize.add_argument("--expected-role-attempt-sha256", required=True)

    commit = subparsers.add_parser(
        "commit-model-test",
        help="commit the canonical outcome-free Model-Test report only",
    )
    commit.add_argument("--expected-producer-attempt-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = _load_context(args)
    if args.operation == "commit-model-test":
        _revalidate_live_context(context)
        stage_b_root = Path(context["stage_b_root"])
        result = compile_stage_b_model_test_commitment(
            report_path=stage_b_root / "model-test" / "report.json",
            commitment_path=stage_b_root / "model-test-committed.json",
            expected_producer_attempt_sha256=(
                args.expected_producer_attempt_sha256),
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    common = {
        "stage_b_root": context["stage_b_root"],
        "role": args.role,
        "generator_commit": context["generator_commit"],
        "actor_bank_manifest": context["actor_bank"],
        "actor_bank_manifest_file_sha256": context[
            "actor_bank_file_sha256"],
    }
    _revalidate_live_context(context)
    if args.operation == "prepare-role":
        result = prepare_stage_b_role(**common)
    elif args.operation == "collect-source":
        if args.source_seed not in ROLE_SOURCE_SEEDS[args.role]:
            raise RuntimeError("source seed is outside the frozen role roster")
        prepared, identity = _runtime_preflight(
            context, role=args.role, source_seed=args.source_seed)

        def progress(record: Mapping[str, Any]) -> None:
            print(json.dumps(record, sort_keys=True), flush=True)

        def collect_and_revalidate(callback):
            value = collect_preflighted_stage_b_role(
                preflight=prepared, progress=callback)
            _revalidate_live_context(context)
            return value

        result = collect_stage_b_source_once(
            **common,
            source_seed=args.source_seed,
            actor_identity=identity,
            expected_role_attempt_sha256=(
                args.expected_role_attempt_sha256),
            simulator_fingerprint=prepared.env.simulator_fingerprint(),
            recovery_library_fingerprint_sha256=(
                prepared.recovery_program_binding["fingerprint_sha256"]),
            collect=collect_and_revalidate,
            progress_sink=progress,
        )
    elif args.operation == "finalize-role":
        result = finalize_stage_b_role(
            **common,
            expected_role_attempt_sha256=(
                args.expected_role_attempt_sha256),
        )
    else:  # pragma: no cover - argparse is exhaustive.
        raise AssertionError("unreachable Stage-B operation")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
