#!/usr/bin/env python3
"""Collect a native MuJoCo grouped Q_safe development dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from safety_data.candidates import EvidenceCandidateConfig
from safety_data.collector import (
    GaussianImpulseSchedule,
    NativeCollectionConfig,
    collect_native_groups,
)
from safety_data.paths import (
    assert_development_path,
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)
from safety_data.policies import load_frozen_droq_policy
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    commit = subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT),
         "status", "--porcelain=v1", "-z"], check=True,
        capture_output=True)
    if status.stdout:
        raise RuntimeError(
            "native evidence collection requires a clean git worktree so "
            "generator_commit identifies the code that actually ran")
    return commit


def _staging_path(destination: Path) -> Path:
    """Reserve a same-filesystem hidden path without touching destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.staging-",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(descriptor)
    staging = Path(raw_path)
    staging.unlink()
    return staging


def _publish_staged_outputs(
    staged_outputs: tuple[tuple[Path, Path], ...],
) -> None:
    """Publish a complete bundle without overwriting raced destinations.

    Each staged file lives beside its destination, so a hard link is an
    atomic, no-overwrite publication.  The report is supplied last and acts
    as the completion marker.  If publication raises, links created by this
    call are rolled back.
    """
    published: list[tuple[Path, Path]] = []
    try:
        for staging, destination in staged_outputs:
            os.link(staging, destination)
            published.append((staging, destination))
    except BaseException:
        for staging, destination in reversed(published):
            # Do not remove a path another process may have replaced between
            # our link and rollback.
            try:
                if os.path.samefile(staging, destination):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for staging, _ in staged_outputs:
            staging.unlink(missing_ok=True)


def _prepare_staged_outputs(
    destinations: tuple[Path, ...],
) -> tuple[tuple[Path, Path], ...]:
    prepared: list[tuple[Path, Path]] = []
    try:
        for destination in destinations:
            prepared.append((_staging_path(destination), destination))
    except BaseException:
        for staging, _ in prepared:
            staging.unlink(missing_ok=True)
        raise
    return tuple(prepared)


def _steps(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "branch impulse steps must be comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one branch impulse step is required")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/go2_50hz_sqrl_paper_sac_pretrain.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model",
        default=("/home/xyz/code/unitree_mujoco/"
                 "unitree_robots/go2/scene_empty.xml"),
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--source-seed", type=int, required=True)
    parser.add_argument("--policy-training-seed", type=int, default=42)
    parser.add_argument("--training-step", type=int)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--discovery-replicas", type=int)
    parser.add_argument("--audit-replicas", type=int)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--accept-probability", type=float, default=0.50)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--max-groups-per-trajectory", type=int, default=5)
    parser.add_argument("--max-source-steps", type=int)
    parser.add_argument("--settle-seconds", type=float, default=0.05)
    parser.add_argument("--source-impulse-interval", type=int, default=10)
    parser.add_argument("--source-linear-std", type=float, default=1.0)
    parser.add_argument("--source-angular-std", type=float, default=4.0)
    parser.add_argument("--branch-impulse-steps", type=_steps, default=(8, 16))
    parser.add_argument("--branch-linear-std", type=float, default=1.0)
    parser.add_argument("--branch-angular-std", type=float, default=4.0)
    parser.add_argument("--actor-local-rms", type=float, default=0.50)
    parser.add_argument("--perturbation-rms", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--privileged-output")
    parser.add_argument("--report")
    args = parser.parse_args()

    config_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.config))
    checkpoint_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.checkpoint))
    model_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.model))
    output = assert_development_path(assert_safe_evidence_output(args.output))
    privileged_output = assert_development_path(assert_safe_evidence_output(
        args.privileged_output or output.with_name(
            f"{output.stem}.privileged.npz")))
    report_output = assert_development_path(assert_safe_evidence_output(
        args.report or output.with_name(f"{output.stem}.report.json")))
    if output.suffix != ".npz" or privileged_output.suffix != ".npz":
        parser.error("dataset outputs must use .npz")
    if report_output.suffix != ".json":
        parser.error("collection report must use .json")
    if len({output, privileged_output, report_output}) != 3:
        parser.error("deployable, privileged and report outputs must be distinct")
    existing = [path for path in (output, privileged_output, report_output)
                if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")
    if args.torch_threads <= 0 or args.progress_every <= 0:
        parser.error("torch threads and progress interval must be positive")
    if not 0.0 < args.accept_probability <= 1.0:
        parser.error("accept probability must lie in (0,1]")
    if (args.discovery_replicas is None) != (args.audit_replicas is None):
        parser.error(
            "discovery and audit replica counts must be supplied together")
    if args.discovery_replicas is not None and (
            args.discovery_replicas <= 0 or args.audit_replicas <= 0 or
            args.discovery_replicas + args.audit_replicas != args.replicas):
        parser.error(
            "positive discovery plus audit replica counts must equal --replicas")
    candidate_config = EvidenceCandidateConfig(
        actor_sample_max_delta_rms=args.actor_local_rms,
        perturbation_radius_rms=args.perturbation_rms,
    )
    branch_disturbance = GaussianImpulseSchedule(
        policy_steps=tuple(args.branch_impulse_steps),
        linear_std_mps=args.branch_linear_std,
        angular_std_radps=args.branch_angular_std,
    )
    if any(step >= args.horizon for step in branch_disturbance.policy_steps):
        parser.error("branch impulse steps must be below the rollout horizon")
    maximum_source_steps = args.max_source_steps
    if maximum_source_steps is None:
        expected = int(np.ceil(args.groups / args.accept_probability))
        maximum_source_steps = max(args.groups, 20 * expected)
    collection_config = NativeCollectionConfig(
        split=args.split,
        target_groups=args.groups,
        source_seed=args.source_seed,
        policy_training_seed=args.policy_training_seed,
        horizon_steps=args.horizon,
        replicas=args.replicas,
        natural_acceptance_probability=args.accept_probability,
        max_episode_steps=args.max_episode_steps,
        max_groups_per_trajectory=args.max_groups_per_trajectory,
        max_source_steps=maximum_source_steps,
        settle_seconds=args.settle_seconds,
        source_impulse_interval_steps=args.source_impulse_interval,
        source_linear_std_mps=args.source_linear_std,
        source_angular_std_radps=args.source_angular_std,
        discovery_replicas=args.discovery_replicas,
        audit_replicas=args.audit_replicas,
    )
    # Fail before loading a checkpoint or allocating MuJoCo when the artifact
    # could not truthfully name the code revision that generated it.
    commit = _git_commit()
    torch.set_num_threads(args.torch_threads)
    robot_cfg, train_cfg, _ = load_app_config(config_path)
    policy = load_frozen_droq_policy(
        checkpoint_path,
        config_path,
        observation_dim=robot_cfg.obs_dim,
        action_dim=robot_cfg.num_joints,
        training_step=args.training_step,
        device=args.device,
    )
    env = MujocoSnapshotEnv(
        model_path,
        robot_cfg,
        policy_frequency=train_cfg.control_frequency,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter,
    )
    started = time.monotonic()

    def progress(value):
        if value["groups"] == 1 or value["groups"] % args.progress_every == 0 or (
                value["groups"] == args.groups):
            print(json.dumps({"event": "collection_progress", **value}, sort_keys=True))

    result = collect_native_groups(
        env=env,
        source_policy=policy,
        continuation_policy=policy,
        candidate_config=candidate_config,
        branch_disturbance=branch_disturbance,
        config=collection_config,
        generator_commit=commit,
        progress=progress,
    )
    elapsed = time.monotonic() - started
    staged = _prepare_staged_outputs((
        output,
        privileged_output,
        # The report is the bundle completion marker and must publish last.
        report_output,
    ))
    dataset_staging, privileged_staging, report_staging = (
        item[0] for item in staged)
    try:
        result.dataset.save(dataset_staging)
        result.privileged.save(privileged_staging)
        # Reload exact on-disk bytes and re-check privileged identity alignment
        # before any destination becomes visible.
        persisted_dataset = GroupedBranchDataset.load(dataset_staging)
        persisted_privileged = PrivilegedBranchView.load(
            privileged_staging, deployable=persisted_dataset)
        validation = persisted_dataset.validate()
        privileged_validation = persisted_privileged.validate(
            persisted_dataset)
        report = {
            "schema_version": "qsafe.native_collection_report.v2",
            "development_only": True,
            "profile_name": result.dataset.manifest["collection_protocol"]
            ["profile_name"],
            "scope": result.dataset.manifest["collection_protocol"]["scope"],
            "evidence_limit": result.dataset.manifest["collection_protocol"]
            ["evidence_limit"],
            "generator_commit": commit,
            "generator_worktree_clean": True,
            "dataset": str(output),
            "dataset_sha256": _sha256(dataset_staging),
            "dataset_content_sha256": persisted_dataset.manifest[
                "content_sha256"],
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
            # This operational, pre-outcome skip count stays outside the
            # dataset manifest so already-collected shard causal contracts
            # remain merge-compatible.
            "skipped_candidate_support_groups": (
                result.skipped_candidate_support_groups),
            "elapsed_seconds": elapsed,
            "groups_per_second": result.dataset.group_count / max(
                elapsed, np.finfo(float).tiny),
            "phase1_data_gate_pass": False,
            "phase1_data_gate_note": (
                "individual strong-impulse development shard only; combine "
                "disjoint source seeds and audit all preregistered thresholds; "
                "natural closed-loop and online evidence remain required"),
            "phase2_authorized": False,
        }
        report_staging.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _publish_staged_outputs(staged)
    finally:
        for staging, _ in staged:
            staging.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
