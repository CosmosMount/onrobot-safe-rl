#!/usr/bin/env python3
"""Collect H96 same-state candidate labels from natural SAC snapshots."""

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

from safety_data.natural_sac_action_branches import (
    build_risk_stratified_plan,
    collect_natural_action_groups,
    validate_natural_source_manifest,
)
from safety_data.action_qsafe_protocol import (
    PROTOCOL_PATH,
    action_qsafe_protocol_sha256,
    load_action_qsafe_protocol,
)
from safety_data.natural_sac_calibration import CalibratedStateRiskPredictor
from safety_data.policies import load_frozen_droq_policy
from safety_data.schema import GroupedBranchDataset
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"], cwd=_ROOT,
        check=True, capture_output=True).stdout
    if status:
        raise RuntimeError("natural action collection requires a clean worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()


def _publish_no_clobber(staging: Path, output: Path) -> None:
    try:
        os.link(staging, output)
    finally:
        staging.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-data", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--state-risk-model", required=True)
    parser.add_argument("--groups", type=int, required=True)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source_data).resolve()
    source_manifest_path = Path(args.source_manifest).resolve()
    model = Path(args.state_risk_model).resolve()
    output = Path(args.output).resolve()
    report_path = output.with_suffix(".report.json")
    if output.suffix != ".npz":
        parser.error("output must use .npz")
    if args.groups <= 0 or args.replicas <= 0 or args.progress_every <= 0:
        parser.error("groups, replicas, and progress interval must be positive")
    if output.exists() or report_path.exists():
        raise FileExistsError("natural action output was already published")
    commit = _clean_commit()
    load_action_qsafe_protocol(PROTOCOL_PATH)
    protocol_sha256 = action_qsafe_protocol_sha256(PROTOCOL_PATH)
    manifest = validate_natural_source_manifest(
        source_manifest_path, source, sha256=_sha256)
    robot, train, _ = load_app_config(manifest["config_path"])
    if not np.isclose(robot.move_speed, 0.30) or train.use_action_filter:
        raise ValueError("natural action runtime differs from Objective 1 target")
    torch.set_num_threads(1)
    policy = load_frozen_droq_policy(
        manifest["actor_manifest"]["actor_path"], manifest["config_path"],
        observation_dim=robot.obs_dim, action_dim=robot.num_joints,
        training_step=int(manifest["actor_training_step"]), device="cpu")
    env = MujocoSnapshotEnv(
        manifest["model_path"], robot,
        policy_frequency=train.control_frequency,
        max_joint_delta=train.max_joint_delta, use_action_filter=False)
    predictor = CalibratedStateRiskPredictor(model, device=args.device)
    started = time.monotonic()
    with np.load(source, allow_pickle=False) as arrays:
        risk, uncertainty = predictor(arrays["observation_history"])
        plan = build_risk_stratified_plan(
            identities=arrays["identity"], state_risk=risk,
            state_uncertainty=uncertainty,
            groups=args.groups)

        def progress(value):
            if value["groups"] == 1 or value["groups"] % args.progress_every == 0:
                print(json.dumps({"event": "natural_action_progress", **value},
                                 sort_keys=True), flush=True)

        with policy.inference_session() as sample:
            dataset = collect_natural_action_groups(
                arrays=arrays, source_manifest=manifest, plan=plan,
                env=env, policy=policy, sample_action=sample,
                generator_commit=commit, replicas=args.replicas,
                protocol_sha256=protocol_sha256,
                progress=progress)
    elapsed = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{output.name}.staging-", suffix=".npz", dir=output.parent)
    os.close(descriptor)
    staging = Path(raw); staging.unlink()
    dataset.save(staging)
    persisted = GroupedBranchDataset.load(staging)
    validation = persisted.validate()
    _publish_no_clobber(staging, output)
    report = {
        "schema_version": "qsafe.natural_sac_action_branch_report.v1",
        "generator_commit": commit,
        "generator_worktree_clean": True,
        "objective1_protocol": str(PROTOCOL_PATH),
        "objective1_protocol_sha256": protocol_sha256,
        "source_data": str(source),
        "source_data_sha256": _sha256(source),
        "source_seed": int(manifest["source_seed"]),
        "actor_seed": int(manifest["actor_seed"]),
        "actor_training_step": int(manifest["actor_training_step"]),
        "state_risk_model": str(model),
        "state_risk_model_sha256": _sha256(model),
        "output": str(output),
        "output_sha256": _sha256(output),
        "groups": persisted.group_count,
        "replicas": int(args.replicas),
        "candidate_count": persisted.candidate_count,
        "horizon_policy_steps": 96,
        "external_force": "verified_zero",
        "branch_outcome_used_for_selection": False,
        "elapsed_seconds": elapsed,
        "validation": validation,
        "objective1_claim_eligible": False,
        "phase2_authorized": False,
    }
    temporary_report = report_path.with_name(
        f".{report_path.name}.tmp-{os.getpid()}")
    temporary_report.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _publish_no_clobber(temporary_report, report_path)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
