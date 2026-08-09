#!/usr/bin/env python3
"""Collect one locked v3 closed-loop recovery triage source shard."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Mapping

import numpy as np
import torch
import yaml

from safety_data.closed_loop_recovery_collector import (
    ClosedLoopRecoveryCollectionConfig,
    canonical_protocol_sha256,
    collect_preflighted_closed_loop_recovery_triage,
    preflight_closed_loop_recovery_collection,
)
from safety_data.closed_loop_recovery_triage import (
    validate_closed_loop_recovery_protocol,
)
from safety_data.paths import (
    ProtectedEvidencePathError,
    assert_development_path,
    require_v3_audit_consumed_or_safe_input,
)
from safety_data.policies import load_frozen_droq_policy
from safety_data.recovery_behaviors import (
    RecoveryBehaviorConfig,
    build_recovery_behavior_library,
)
from scripts.collect_native_grouped_qsafe import (
    _git_commit,
    _prepare_staged_outputs,
    _publish_staged_outputs,
    _sha256,
)
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL_PATH = (
    _ROOT / "config" / "qsafe_closed_loop_recovery_triage_v3.yaml")
_PROTOCOL_NAME = "objective1_closed_loop_recovery_triage_v3"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_protocol() -> dict:
    path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(_PROTOCOL_PATH))
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get(
            "protocol_schema_version") != 1 or protocol.get(
                "protocol_name") != _PROTOCOL_NAME:
        raise ValueError("canonical v3 recovery protocol is invalid")
    validate_closed_loop_recovery_protocol(protocol)
    if protocol.get("scope") != (
            "conditional_development_mechanism_triage_only") or protocol.get(
                "claim_eligible") is not False:
        raise ValueError("v3 scope or claim eligibility has drifted")
    target = protocol.get("target")
    collection = protocol.get("collection")
    policies = protocol.get("early_task_policies")
    if not isinstance(target, dict) or not isinstance(collection, dict) or (
            not isinstance(policies, list)):
        raise ValueError("v3 target, collection, or policy contract is missing")
    expected_collection = {
        "groups_per_source_seed": 64,
        "total_groups": 384,
        "total_candidate_replicas": 128,
        "total_candidate_branch_rollouts": 442368,
    }
    for name, expected in expected_collection.items():
        if collection.get(name) != expected:
            raise ValueError(f"v3 collection.{name} has drifted")
    if target.get("horizon_policy_steps") != 96 or target.get(
            "command_speed_mps") != 0.30 or target.get("failure", {}).get(
                "max_abs_roll_pitch_rad") != 0.523599 or target.get(
                    "policy_hz") != 50 or target.get("low_level_hz") != 500:
        raise ValueError(
            "v3 horizon, speed, failure threshold, or control timing has drifted")
    if collection.get("candidates") != RecoveryBehaviorConfig().manifest_protocol():
        raise ValueError("implemented K9 behavior manifest differs from protocol")
    flattened_seeds = [
        int(seed) for policy in policies for seed in policy.get("source_seeds", [])]
    if flattened_seeds != [7801, 7802, 7811, 7812, 7821, 7822]:
        raise ValueError("v3 source-seed order has drifted")
    if [int(policy.get("training_step", -1)) for policy in policies] != [
            25438, 50030, 100359]:
        raise ValueError("v3 early policy ages have drifted")
    admission = collection.get("admission", {})
    partition = collection.get("replica_partition", {})
    if admission.get("replicas") != 32 or admission.get(
            "accept_min_falls_inclusive") != 6 or admission.get(
                "accept_max_falls_inclusive") != 26 or partition.get(
                    "discovery_replicas") != 64 or partition.get(
                        "audit_replicas") != 64:
        raise ValueError("v3 admission or D/A replica contract has drifted")
    expected_physical = {
        "attempt_shard_filename_template": (
            "source-{source_seed}.attempt-started.json"),
        "admission_shard_filename_template": "source-{source_seed}.admission.npz",
        "admission_privileged_shard_filename_template": (
            "source-{source_seed}.admission.privileged.npz"),
        "discovery_shard_filename_template": "source-{source_seed}.discovery.npz",
        "discovery_privileged_shard_filename_template": (
            "source-{source_seed}.discovery.privileged.npz"),
        "audit_shard_filename_template": "source-{source_seed}.audit.npz",
        "audit_privileged_shard_filename_template": (
            "source-{source_seed}.audit.privileged.npz"),
        "collection_report_shard_filename_template": (
            "source-{source_seed}.collection-report.json"),
        "audit_merge_before_selection": "forbidden",
    }
    for name, expected in expected_physical.items():
        if collection.get(name) != expected:
            raise ValueError(f"v3 collection.{name} has drifted")
    if collection.get("settle_seconds") != 0.04 or collection.get(
            "settle_policy_steps") != 2:
        raise ValueError("v3 reset settling contract has drifted")
    return protocol


def _policy_for_seed(protocol: Mapping[str, object], source_seed: int) -> dict:
    matches = [
        dict(policy) for policy in protocol["early_task_policies"]
        if source_seed in list(map(int, policy["source_seeds"]))
    ]
    if len(matches) != 1:
        raise ValueError("source seed does not map to exactly one early policy")
    return matches[0]
def _verify_policy(policy: object, expected: dict, role: str) -> None:
    manifest = policy.manifest()
    checks = {
        "training_step": "training_step",
        "actor_sha256": "actor_sha256",
        "actor_state_dict_sha256": "actor_state_dict_sha256",
        "policy_fingerprint_sha256": "policy_fingerprint_sha256",
        "checkpoint_fingerprint_sha256": "checkpoint_fingerprint_sha256",
    }
    for expected_name, manifest_name in checks.items():
        if manifest.get(manifest_name) != expected.get(expected_name):
            raise ValueError(
                f"loaded {role} policy {manifest_name} differs from protocol")


def _policy_set_manifest(protocol: dict) -> dict:
    config = protocol["policy_config"]
    return {
        "type": "locked_early_sac_policy_age_set_v3",
        "policy_training_seed": int(config["policy_training_seed"]),
        "config_sha256": str(config["config_sha256"]),
        "policies": [
            {
                name: copy.deepcopy(policy[name])
                for name in (
                    "training_step", "source_seeds", "actor_sha256",
                    "actor_state_dict_sha256", "policy_fingerprint_sha256",
                    "checkpoint_fingerprint_sha256",
                )
            }
            for policy in protocol["early_task_policies"]
        ],
    }


def _verify_runtime_contract(
    env: object,
    robot_cfg: object,
    train_cfg: object,
    protocol: dict,
) -> None:
    """Fail closed if simulator/controller timing differs from the protocol."""
    target = protocol["target"]
    candidates = protocol["collection"]["candidates"]
    policy_hz = float(target["policy_hz"])
    low_level_hz = float(target["low_level_hz"])
    kp = float(candidates["kp"])
    kd = float(candidates["kd"])
    if not np.isclose(float(train_cfg.control_frequency), policy_hz):
        raise ValueError("training policy frequency differs from v3 protocol")
    if train_cfg.max_joint_delta is not None or bool(
            train_cfg.use_action_filter):
        raise ValueError("v3 requires no max-delta and no action filter")
    if not np.allclose(np.asarray(robot_cfg.kp, dtype=float), kp) or not (
            np.allclose(np.asarray(robot_cfg.kd, dtype=float), kd)):
        raise ValueError("loaded robot gains differ from v3 protocol")
    if not np.isclose(float(env.policy_frequency), policy_hz) or not np.isclose(
            float(env.model.opt.timestep), 1.0 / low_level_hz):
        raise ValueError("simulator policy/low-level timing differs from v3 protocol")
    expected_substeps = int(round(low_level_hz / policy_hz))
    if int(env.substeps) != expected_substeps or not np.allclose(
            np.asarray(env.kp, dtype=float), kp) or not np.allclose(
                np.asarray(env.kd, dtype=float), kd):
        raise ValueError("simulator substeps or controller gains differ from v3")
    settle_seconds = float(protocol["collection"]["settle_seconds"])
    if int(round(settle_seconds * policy_hz)) != int(
            protocol["collection"]["settle_policy_steps"]):
        raise ValueError("v3 settle seconds do not map to locked policy steps")
    if int(robot_cfg.num_joints) != 12 or int(robot_cfg.obs_dim) != 46:
        raise ValueError("v3 requires the locked 12D action and 46D observation")
    simulator = env.simulator_fingerprint()
    if simulator.get("mjcf_xml_sha256") != target.get(
            "model_mjcf_dependency_sha256"):
        raise ValueError("external MJCF dependency digest differs from v3 protocol")
    if simulator.get("failure_measurement") != {
            "height_reference": "base_link_body_origin_world_z",
            "cadence": "post_policy_step_after_all_low_level_substeps",
            "low_level_substeps_per_policy_step": 10,
    }:
        raise ValueError("simulator failure measurement semantics differ from v3")
    applier = env.action_applier
    observed_action_contract = {
        "q_target_semantic": "absolute_joint_position_sent",
        "init_qpos": np.asarray(applier.init_qpos, dtype=float).tolist(),
        "action_offset": np.asarray(
            applier.action_offset, dtype=float).tolist(),
        "joint_min": np.asarray(applier.joint_min, dtype=float).tolist(),
        "joint_max": np.asarray(applier.joint_max, dtype=float).tolist(),
        "projection": (
            "clip_normalized_then_joint_bounds_then_slew_then_filter"),
    }
    if observed_action_contract != target.get("action_application_contract"):
        raise ValueError("resolved runtime action projection differs from v3")


def _cohort_lock(
    path: Path,
    *,
    generator_commit: str,
    protocol_sha256: str,
    protocol_contract_sha256: str,
    protocol: dict,
) -> dict:
    expected = {
        "schema_version": "qsafe.closed_loop_recovery.cohort_lock.v3",
        "protocol_name": _PROTOCOL_NAME,
        "protocol_file_sha256": protocol_sha256,
        "protocol_contract_sha256": protocol_contract_sha256,
        "generator_commit": generator_commit,
        "source_seed_policy_step": {
            str(seed): int(policy["training_step"])
            for policy in protocol["early_task_policies"]
            for seed in policy["source_seeds"]
        },
        "outcome_state": "no_analysis_before_all_shards_complete",
    }
    rendered = json.dumps(expected, indent=2, sort_keys=True) + "\n"
    expected_bytes = rendered.encode("utf-8")
    staged = _prepare_staged_outputs((path,))
    staging = staged[0][0]
    try:
        staging.write_bytes(expected_bytes)
        try:
            _publish_staged_outputs(staged)
        except FileExistsError:
            pass
    finally:
        staging.unlink(missing_ok=True)
    try:
        path = assert_development_path(
            require_v3_audit_consumed_or_safe_input(path))
    except ProtectedEvidencePathError as exc:
        raise RuntimeError(
            "v3 cohort lock must be a regular non-symlink file") from exc
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError("v3 cohort lock publication did not persist") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("v3 cohort lock must be a regular non-symlink file")
    observed_bytes = path.read_bytes()
    try:
        observed = json.loads(observed_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("v3 cohort lock is not canonical JSON") from exc
    if observed_bytes != expected_bytes or observed != expected:
        raise RuntimeError("v3 cohort is bound to another protocol or commit")
    return observed


def _outputs(root: Path, source_seed: int) -> dict[str, Path]:
    prefix = f"source-{source_seed}"
    values = {
        "attempt": root / f"{prefix}.attempt-started.json",
        "admission": root / f"{prefix}.admission.npz",
        "admission_privileged": root / f"{prefix}.admission.privileged.npz",
        "discovery": root / f"{prefix}.discovery.npz",
        "discovery_privileged": root / f"{prefix}.discovery.privileged.npz",
        "audit": root / f"{prefix}.audit.npz",
        "audit_privileged": root / f"{prefix}.audit.privileged.npz",
        "report": root / f"{prefix}.collection-report.json",
    }
    # Audit destinations are intentionally absent from this preflight probe.
    # They are created only by the final atomic hard-link publication, whose
    # EEXIST result provides no-clobber without inspecting a pre-existing audit
    # shard.  In particular, a repeated invocation must stop at the attempt
    # marker without touching either audit pathname.
    preflight_names = (
        "attempt", "admission", "admission_privileged", "discovery",
        "discovery_privileged", "report",
    )
    existing = []
    for name in preflight_names:
        path = values[name]
        if os.path.lexists(os.fspath(path)):
            existing.append(path)
            # The attempt marker is the earliest one-shot boundary.  Once it
            # exists, do not probe any later artifact, even a non-audit one.
            if name == "attempt":
                break
    if existing:
        raise FileExistsError(f"refusing to overwrite v3 shard outputs: {existing}")
    return values


def _start_attempt(
    path: Path,
    *,
    source_seed: int,
    policy_training_step: int,
    generator_commit: str,
    protocol_sha256: str,
    protocol_contract_sha256: str,
    cohort_lock_sha256: str,
) -> dict:
    """Persist one conservative collection-consumption marker before stepping."""
    contract = {
        "schema_version": "qsafe.closed_loop_recovery.attempt.v3",
        "protocol_name": _PROTOCOL_NAME,
        "protocol_file_sha256": protocol_sha256,
        "protocol_contract_sha256": protocol_contract_sha256,
        "generator_commit": generator_commit,
        "cohort_lock_sha256": cohort_lock_sha256,
        "source_seed": int(source_seed),
        "policy_training_step": int(policy_training_step),
        "started_at_unix_ns": time.time_ns(),
        "state": "started_outcome_may_have_been_generated",
        "restart_authorized": False,
        "candidate_outcomes_summarized": False,
    }
    staged = _prepare_staged_outputs((path,))
    staging = staged[0][0]
    try:
        staging.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        _publish_staged_outputs(staged)
    finally:
        staging.unlink(missing_ok=True)
    return contract


def _require_same_clean_commit(expected: str, phase: str) -> None:
    if _git_commit() != expected:
        raise RuntimeError(f"generator commit changed {phase}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-seed", required=True, type=int)
    parser.add_argument("--progress-every", type=int, default=4)
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("progress interval must be positive")

    protocol = _load_protocol()
    policy_entry = _policy_for_seed(protocol, args.source_seed)
    collection = protocol["collection"]
    target = protocol["target"]
    policy_config = protocol["policy_config"]
    lexical_artifact_root = Path(os.path.abspath(
        _ROOT / str(collection["artifact_root"])))
    artifact_root = assert_development_path(
        require_v3_audit_consumed_or_safe_input(lexical_artifact_root))
    if artifact_root != lexical_artifact_root:
        raise RuntimeError(
            "v3 artifact root may not contain any symlinked path component")
    if os.path.lexists(os.fspath(lexical_artifact_root)):
        root_metadata = lexical_artifact_root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(
                root_metadata.st_mode):
            raise RuntimeError(
                "v3 artifact root must be a real directory when it exists")
    outputs = _outputs(artifact_root, args.source_seed)
    generator_commit = _git_commit()
    protocol_sha256 = _file_sha256(_PROTOCOL_PATH)
    protocol_contract_sha256 = canonical_protocol_sha256(protocol)
    lock_path = artifact_root / str(collection["cohort_lock_filename"])

    config_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(
            _ROOT / str(policy_config["path"])))
    early_checkpoint = assert_development_path(
        require_v3_audit_consumed_or_safe_input(
            _ROOT / str(policy_entry["checkpoint"])))
    mature_entry = protocol["mature_recovery_policy"]
    mature_checkpoint = assert_development_path(
        require_v3_audit_consumed_or_safe_input(
            _ROOT / str(mature_entry["checkpoint"])))
    model_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(str(target["model_mjcf"])))
    torch.set_num_threads(1)
    robot_cfg, train_cfg, _ = load_app_config(config_path)
    if not np.isclose(
            float(robot_cfg.move_speed), float(target["command_speed_mps"]),
            rtol=0.0, atol=1e-12):
        raise ValueError("loaded command speed differs from v3 protocol")
    if not np.isclose(
            float(robot_cfg.success_orientation_rad),
            float(target["failure"]["max_abs_roll_pitch_rad"]),
            rtol=0.0, atol=1e-12):
        raise ValueError("success orientation threshold differs from v3 protocol")
    robot_cfg = replace(
        robot_cfg,
        fallen_orientation_rad=robot_cfg.success_orientation_rad,
    )
    early_policy = load_frozen_droq_policy(
        early_checkpoint,
        config_path,
        observation_dim=robot_cfg.obs_dim,
        action_dim=robot_cfg.num_joints,
        training_step=int(policy_entry["training_step"]),
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
    _verify_policy(early_policy, policy_entry, "early")
    _verify_policy(mature_policy, mature_entry, "mature")
    env = MujocoSnapshotEnv(
        model_path,
        robot_cfg,
        policy_frequency=train_cfg.control_frequency,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter,
    )
    _verify_runtime_contract(env, robot_cfg, train_cfg, protocol)
    recovery_program = build_recovery_behavior_library(
        mature_policy, env.action_applier)
    if recovery_program.manifest_protocol() != collection["candidates"]:
        raise RuntimeError("runtime K9 behavior library differs from protocol")

    source_impulse = collection["source_impulse"]
    admission = collection["admission"]
    partition = collection["replica_partition"]
    config = ClosedLoopRecoveryCollectionConfig(
        source_seed=args.source_seed,
        policy_training_step=int(policy_entry["training_step"]),
        policy_training_seed=int(policy_config["policy_training_seed"]),
        target_groups=int(collection["groups_per_source_seed"]),
        horizon_steps=int(target["horizon_policy_steps"]),
        admission_replicas=int(admission["replicas"]),
        admission_min_falls=int(admission["accept_min_falls_inclusive"]),
        admission_max_falls=int(admission["accept_max_falls_inclusive"]),
        discovery_replicas=int(partition["discovery_replicas"]),
        audit_replicas=int(partition["audit_replicas"]),
        max_episode_steps=int(collection["max_episode_steps"]),
        max_proposals=int(collection["max_proposals_per_source_seed"]),
        max_trajectories=int(collection["max_source_trajectories_per_seed"]),
        proposal_cooldown_steps=int(
            collection["rejected_proposal_cooldown_policy_steps"]),
        settle_seconds=float(collection["settle_seconds"]),
        source_impulse_interval_steps=int(
            source_impulse["interval_policy_steps"]),
        source_linear_std_mps=float(source_impulse["linear_std_mps"]),
        source_angular_std_radps=float(source_impulse["angular_std_radps"]),
        proposal_min_tilt_rad=float(
            collection["proposal_pre_screen"]["min_tilt_rad_inclusive"]),
        proposal_max_height_m=float(
            collection["proposal_pre_screen"]["max_height_m_inclusive"]),
    )
    policy_set_manifest = _policy_set_manifest(protocol)
    prepared = preflight_closed_loop_recovery_collection(
        env=env,
        early_policy=early_policy,
        recovery_program=recovery_program,
        policy_set_manifest=policy_set_manifest,
        config=config,
        generator_commit=generator_commit,
        protocol_sha256=protocol_sha256,
        protocol_contract_sha256=protocol_contract_sha256,
    )
    _require_same_clean_commit(generator_commit, "during v3 preflight")
    # Publish the cohort only after every deterministic policy/simulator/schema
    # preflight has passed.  A later attempt marker therefore means that the
    # next operation may genuinely generate simulator outcomes.
    cohort = _cohort_lock(
        lock_path,
        generator_commit=generator_commit,
        protocol_sha256=protocol_sha256,
        protocol_contract_sha256=protocol_contract_sha256,
        protocol=protocol,
    )
    _require_same_clean_commit(
        generator_commit, "before v3 attempt consumption")
    attempt = _start_attempt(
        outputs["attempt"],
        source_seed=args.source_seed,
        policy_training_step=int(policy_entry["training_step"]),
        generator_commit=generator_commit,
        protocol_sha256=protocol_sha256,
        protocol_contract_sha256=protocol_contract_sha256,
        cohort_lock_sha256=_file_sha256(lock_path),
    )
    def progress(value: dict) -> None:
        if value["groups"] == 1 or value["groups"] % args.progress_every == 0 or (
                value["groups"] == config.target_groups):
            print(json.dumps(
                {"event": "closed_loop_v3_collection_progress", **value},
                sort_keys=True), flush=True)

    result = collect_preflighted_closed_loop_recovery_triage(
        preflight=prepared,
        progress=progress,
    )
    if _git_commit() != generator_commit:
        raise RuntimeError("generator commit changed during v3 collection")

    destination_order = (
        "admission", "admission_privileged", "discovery",
        "discovery_privileged", "audit", "audit_privileged", "report")
    staged = _prepare_staged_outputs(tuple(
        outputs[name] for name in destination_order))
    staging = {
        name: pair[0] for name, pair in zip(destination_order, staged, strict=True)}
    try:
        result.admission.save(staging["admission"])
        result.admission_privileged.save(
            staging["admission_privileged"], result.admission)
        result.discovery.save(staging["discovery"])
        result.discovery_privileged.save(staging["discovery_privileged"])
        result.audit.save(staging["audit"])
        result.audit_privileged.save(staging["audit_privileged"])

        # Do not reopen either candidate-outcome file here.  Their save methods
        # have already enforced the structural schema and bound content hashes;
        # the report exposes only dimensions and hashes, never outcome-derived
        # summaries such as mixed-outcome fractions.
        admission_validation = result.admission.validate()
        admission_privileged_validation = (
            result.admission_privileged.validate(result.admission))
        validations = {
            "admission": admission_validation,
            "admission_privileged": admission_privileged_validation,
            "discovery": {
                "groups": result.discovery.group_count,
                "max_candidates": result.discovery.candidate_count,
                "replicas": result.discovery.replica_count,
                "horizon_steps": result.discovery.horizon_steps,
                "content_sha256": result.discovery.manifest["content_sha256"],
            },
            "discovery_privileged": {
                "groups": len(result.discovery_privileged.group_id),
                "content_sha256": result.discovery_privileged.manifest[
                    "content_sha256"],
            },
            "audit": {
                "groups": result.audit.group_count,
                "max_candidates": result.audit.candidate_count,
                "replicas": result.audit.replica_count,
                "horizon_steps": result.audit.horizon_steps,
                "content_sha256": result.audit.manifest["content_sha256"],
            },
            "audit_privileged": {
                "groups": len(result.audit_privileged.group_id),
                "content_sha256": result.audit_privileged.manifest[
                    "content_sha256"],
            },
        }
        for role in ("discovery", "audit"):
            report = validations[role]
            if report["groups"] != 64 or report["max_candidates"] != 9 or (
                    report["replicas"] != 64):
                raise RuntimeError(f"persisted {role} shard dimensions drifted")
        if validations["admission"]["accepted"] != 64:
            raise RuntimeError("persisted admission ledger accepted count drifted")
        if not np.array_equal(
                result.discovery["group_id"], result.audit["group_id"]):
            raise RuntimeError("discovery/audit identities differ before publication")
        artifacts = {
            "admission": result.admission,
            "admission_privileged": result.admission_privileged,
            "discovery": result.discovery,
            "discovery_privileged": result.discovery_privileged,
            "audit": result.audit,
            "audit_privileged": result.audit_privileged,
        }
        report = {
            "schema_version": "qsafe.closed_loop_recovery_collection_report.v3",
            "protocol_name": _PROTOCOL_NAME,
            "protocol_path": str(_PROTOCOL_PATH),
            "protocol_file_sha256": protocol_sha256,
            "protocol_contract_sha256": protocol_contract_sha256,
            "cohort_lock": str(lock_path),
            "cohort_lock_sha256": _file_sha256(lock_path),
            "cohort_contract": cohort,
            "attempt_marker": str(outputs["attempt"]),
            "attempt_marker_sha256": _file_sha256(outputs["attempt"]),
            "attempt_contract": attempt,
            "development_only": True,
            "claim_eligible": False,
            "source_seed": args.source_seed,
            "policy_training_step": int(policy_entry["training_step"]),
            "generator_commit": generator_commit,
            "generator_worktree_clean": True,
            "outputs": {
                name: {
                    "path": str(outputs[name]),
                    "file_sha256": _sha256(staging[name]),
                    "content_sha256": artifacts[name].manifest["content_sha256"],
                }
                for name in destination_order if name != "report"
            },
            "validations": validations,
            "source_steps": result.source_steps,
            "trajectories": result.trajectories,
            "proposals": result.proposals,
            "candidate_outcomes_summarized": False,
            "selection_lock_created": False,
            "audit_opened_for_analysis": False,
            "model_training_authorized": False,
            "phase2_authorized": False,
        }
        staging["report"].write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        if _git_commit() != generator_commit:
            raise RuntimeError("generator worktree changed before v3 publication")
        _publish_staged_outputs(staged)
    finally:
        for path, _ in staged:
            path.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
