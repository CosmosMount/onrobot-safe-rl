#!/usr/bin/env python3
"""Collect one locked Objective-1 recovery-option triage shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from safety_data.collector import (
    GaussianImpulseSchedule,
    NativeCollectionConfig,
    collect_native_groups,
)
from safety_data.paths import assert_development_path
from safety_data.policies import load_frozen_droq_policy
from safety_data.recovery_options import RecoveryOptionCandidateConfig
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView
from scripts.collect_native_grouped_qsafe import (
    _git_commit,
    _prepare_staged_outputs,
    _publish_staged_outputs,
    _sha256,
)
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_PROTOCOL_PATH = (
    _REPOSITORY_ROOT / "config" / "qsafe_recovery_option_triage_v2.yaml"
).resolve()
_PROTOCOL_NAME = "objective1_recovery_option_triage_v2"
_PROFILE_SCOPE = "development_mechanism_triage_only"
_EVIDENCE_LIMIT = (
    "development causal-headroom triage only; cannot pass Objective 1, "
    "authorize selector/online evaluation, or unlock Phase 2")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bind_cohort_lock(
    path: Path,
    *,
    generator_commit: str,
    protocol_sha256: str,
    source_seeds: list[int],
) -> dict:
    """Atomically bind every triage shard to one code/protocol cohort."""
    expected = {
        "schema_version": "qsafe.recovery_option_triage.cohort_lock.v1",
        "protocol_name": _PROTOCOL_NAME,
        "protocol_file_sha256": protocol_sha256,
        "generator_commit": generator_commit,
        "source_seeds": source_seeds,
    }
    staged = _prepare_staged_outputs((path,))
    staging = staged[0][0]
    try:
        staging.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            _publish_staged_outputs(staged)
        except FileExistsError:
            pass
    finally:
        staging.unlink(missing_ok=True)
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("triage cohort lock is unreadable") from exc
    if observed != expected:
        raise RuntimeError(
            "triage cohort is already bound to a different protocol/commit")
    return expected


def _load_locked_protocol() -> dict:
    path = assert_development_path(_CANONICAL_PROTOCOL_PATH)
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict) or protocol.get(
            "protocol_schema_version") != 1 or protocol.get(
            "protocol_name") != _PROTOCOL_NAME:
        raise ValueError("canonical recovery-option triage protocol is invalid")
    if protocol.get("scope") != _PROFILE_SCOPE or protocol.get(
            "claim_eligible") is not False:
        raise ValueError("triage scope or claim eligibility has drifted")
    collection = protocol.get("collection")
    if not isinstance(collection, dict):
        raise ValueError("triage protocol has no collection contract")
    expected = {
        "source_seeds": [7601, 7602, 7603],
        "groups_per_source_seed": 128,
        "total_groups": 384,
        "total_replicas": 64,
        "total_h32_branches": 712704,
    }
    for name, value in expected.items():
        if collection.get(name) != value:
            raise ValueError(f"triage collection.{name} has drifted")
    target = protocol.get("target")
    if not isinstance(target, dict) or target.get(
            "horizon_policy_steps") != 32 or target.get(
            "command_speed_mps") != 0.30:
        raise ValueError("triage target horizon or speed has drifted")
    frozen = protocol.get("frozen_policy")
    if not isinstance(frozen, dict) or frozen.get("training_step") != 500000:
        raise ValueError("triage frozen-policy contract has drifted")
    if collection.get("candidates") != (
            RecoveryOptionCandidateConfig().manifest_protocol()):
        raise ValueError(
            "implemented K29 candidate contract differs from canonical protocol")
    return protocol


def _output_bundle(
    args: argparse.Namespace,
    *,
    artifact_root: Path,
) -> tuple[Path, Path, Path]:
    output = assert_development_path(args.output)
    privileged = assert_development_path(
        args.privileged_output or output.with_name(
            f"{output.stem}.privileged.npz"))
    report = assert_development_path(
        args.report or output.with_name(f"{output.stem}.report.json"))
    if output.suffix != ".npz" or privileged.suffix != ".npz":
        raise ValueError("dataset outputs must use .npz")
    if report.suffix != ".json":
        raise ValueError("collection report must use .json")
    if len({output, privileged, report}) != 3:
        raise ValueError("deployable, privileged, and report outputs must differ")
    root = artifact_root.resolve()
    for path in (output, privileged, report):
        if not path.resolve().is_relative_to(root):
            raise ValueError(
                f"triage output must be below locked artifact root {root}")
    existing = [path for path in (output, privileged, report) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    return output, privileged, report


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--privileged-output")
    parser.add_argument("--report")
    parser.add_argument("--progress-every", type=int, default=4)
    args = parser.parse_args()

    if args.progress_every <= 0:
        parser.error("progress interval must be positive")
    protocol = _load_locked_protocol()
    collection = protocol["collection"]
    target = protocol["target"]
    frozen = protocol["frozen_policy"]
    source_seeds = [int(value) for value in collection["source_seeds"]]
    if args.source_seed not in source_seeds:
        parser.error(
            f"source seed must be one of the preregistered values {source_seeds}")
    artifact_root = assert_development_path(
        _REPOSITORY_ROOT / str(collection["artifact_root"]))
    output, privileged_output, report_output = _output_bundle(
        args, artifact_root=artifact_root)

    # Bind the long-running generator to one clean commit before loading the
    # actor or observing any branch outcome.
    generator_commit = _git_commit()
    protocol_sha256 = _file_sha256(_CANONICAL_PROTOCOL_PATH)
    cohort_lock_path = assert_development_path(
        artifact_root / str(collection["cohort_lock_filename"]))
    cohort_lock = _bind_cohort_lock(
        cohort_lock_path,
        generator_commit=generator_commit,
        protocol_sha256=protocol_sha256,
        source_seeds=source_seeds,
    )
    config_path = assert_development_path(
        _REPOSITORY_ROOT / str(frozen["config"]))
    checkpoint_path = assert_development_path(
        _REPOSITORY_ROOT / str(frozen["checkpoint"]))
    model_path = assert_development_path(str(target["model_mjcf"]))
    torch.set_num_threads(1)
    robot_cfg, train_cfg, _ = load_app_config(config_path)
    if not np.isclose(float(robot_cfg.move_speed), float(
            target["command_speed_mps"]), rtol=0.0, atol=1e-12):
        raise ValueError("runtime config move_speed differs from triage protocol")
    policy = load_frozen_droq_policy(
        checkpoint_path,
        config_path,
        observation_dim=robot_cfg.obs_dim,
        action_dim=robot_cfg.num_joints,
        training_step=int(frozen["training_step"]),
        device="cpu",
    )
    policy_manifest = policy.manifest()
    policy_checks = {
        "actor_fingerprint_sha256": "policy_fingerprint_sha256",
        "actor_state_dict_sha256": "actor_state_dict_sha256",
        "config_sha256": "config_sha256",
    }
    for protocol_key, manifest_key in policy_checks.items():
        if policy_manifest.get(manifest_key) != frozen[protocol_key]:
            raise ValueError(
                f"loaded actor {manifest_key} differs from triage protocol")
    env = MujocoSnapshotEnv(
        model_path,
        robot_cfg,
        policy_frequency=train_cfg.control_frequency,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter,
    )

    source_impulse = collection["source_impulse"]
    branch_impulse = collection["branch_impulse"]
    partition = collection["replica_partition"]
    candidate_config = RecoveryOptionCandidateConfig()
    branch_disturbance = GaussianImpulseSchedule(
        policy_steps=tuple(map(int, branch_impulse["policy_steps"])),
        linear_std_mps=float(branch_impulse["linear_std_mps"]),
        angular_std_radps=float(branch_impulse["angular_std_radps"]),
    )
    groups = _integer(
        collection["groups_per_source_seed"], "groups_per_source_seed")
    acceptance = float(collection["natural_acceptance_probability"])
    maximum_source_steps = max(
        groups, 20 * int(np.ceil(groups / acceptance)))
    collection_config = NativeCollectionConfig(
        split="recovery_option_triage_v2",
        target_groups=groups,
        source_seed=args.source_seed,
        policy_training_seed=42,
        horizon_steps=int(target["horizon_policy_steps"]),
        replicas=int(collection["total_replicas"]),
        natural_acceptance_probability=acceptance,
        max_episode_steps=int(collection["max_episode_steps"]),
        max_groups_per_trajectory=int(
            collection["max_groups_per_trajectory"]),
        max_source_steps=maximum_source_steps,
        settle_seconds=float(collection["settle_seconds"]),
        source_impulse_interval_steps=int(
            source_impulse["interval_policy_steps"]),
        source_linear_std_mps=float(source_impulse["linear_std_mps"]),
        source_angular_std_radps=float(source_impulse["angular_std_radps"]),
        discovery_replicas=int(partition["discovery_replicas"]),
        audit_replicas=int(partition["audit_replicas"]),
        profile_name=_PROTOCOL_NAME,
        profile_scope=_PROFILE_SCOPE,
        evidence_limit=_EVIDENCE_LIMIT,
    )
    started = time.monotonic()

    def progress(value):
        if value["groups"] == 1 or value["groups"] % args.progress_every == 0 or (
                value["groups"] == groups):
            print(json.dumps(
                {"event": "triage_collection_progress", **value},
                sort_keys=True))

    result = collect_native_groups(
        env=env,
        source_policy=policy,
        continuation_policy=policy,
        candidate_config=candidate_config,
        branch_disturbance=branch_disturbance,
        config=collection_config,
        generator_commit=generator_commit,
        progress=progress,
    )
    elapsed = time.monotonic() - started
    if _git_commit() != generator_commit:
        raise RuntimeError("generator commit changed during triage collection")
    if result.dataset.manifest["candidate_protocol"] != collection["candidates"]:
        raise RuntimeError("collected candidate protocol differs from preregistration")
    collected_protocol = result.dataset.manifest["collection_protocol"]
    if collected_protocol.get("replica_partition") != partition or (
            collected_protocol.get("profile_name") != _PROTOCOL_NAME) or (
            collected_protocol.get("scope") != _PROFILE_SCOPE):
        raise RuntimeError("collected profile/partition differs from preregistration")

    staged = _prepare_staged_outputs((
        output, privileged_output, report_output))
    dataset_staging, privileged_staging, report_staging = (
        item[0] for item in staged)
    try:
        result.dataset.save(dataset_staging)
        result.privileged.save(privileged_staging)
        persisted = GroupedBranchDataset.load(dataset_staging)
        persisted_privileged = PrivilegedBranchView.load(
            privileged_staging, deployable=persisted)
        validation = persisted.validate()
        privileged_validation = persisted_privileged.validate(persisted)
        if validation["groups"] != groups or validation["replicas"] != 64 or (
                validation["max_candidates"] != 29) or validation[
                    "replica_partition"] != partition:
            raise RuntimeError("persisted triage shard failed locked dimensions")
        report = {
            "schema_version": "qsafe.recovery_option_triage_collection.v1",
            "protocol_name": _PROTOCOL_NAME,
            "protocol_path": str(_CANONICAL_PROTOCOL_PATH),
            "protocol_file_sha256": protocol_sha256,
            "cohort_lock": str(cohort_lock_path),
            "cohort_lock_sha256": _file_sha256(cohort_lock_path),
            "cohort_contract": cohort_lock,
            "development_only": True,
            "claim_eligible": False,
            "source_seed": args.source_seed,
            "generator_commit": generator_commit,
            "generator_worktree_clean": True,
            "dataset": str(output),
            "dataset_sha256": _sha256(dataset_staging),
            "dataset_content_sha256": persisted.manifest["content_sha256"],
            "privileged": str(privileged_output),
            "privileged_sha256": _sha256(privileged_staging),
            "privileged_content_sha256": persisted_privileged.manifest[
                "content_sha256"],
            "validation": validation,
            "privileged_validation": privileged_validation,
            "source_steps": result.source_steps,
            "episodes": result.episodes,
            "near_failure_groups": result.near_failure_groups,
            "randomly_accepted_groups": result.randomly_accepted_groups,
            "skipped_candidate_support_groups": (
                result.skipped_candidate_support_groups),
            "elapsed_seconds": elapsed,
            "groups_per_second": groups / max(
                elapsed, np.finfo(float).tiny),
            "triage_gate_evaluated": False,
            "selector_calibration_authorized": False,
            "paired_closed_loop_authorized": False,
            "online_training_authorized": False,
            "phase2_authorized": False,
        }
        report_staging.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if _git_commit() != generator_commit:
            raise RuntimeError("generator commit/worktree changed before publish")
        _publish_staged_outputs(staged)
    finally:
        for staging, _ in staged:
            staging.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
