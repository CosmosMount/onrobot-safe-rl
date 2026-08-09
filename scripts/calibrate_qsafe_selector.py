#!/usr/bin/env python3
"""Lock a repeated Q_safe selector on the declared calibration split only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch
import yaml

from rl.qsafe.artifact import load_qsafe_artifact
from rl.qsafe.calibration import (
    SelectorCalibrationInputs,
    SelectorCalibrationSpec,
    calibrate_selector,
)
from rl.qsafe.data import TorchGroupedView
from safety_data.paths import (
    assert_development_path,
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)
from safety_data.policies import load_frozen_droq_policy
from safety_data.reward_q import load_frozen_droq_reward_q
from safety_data.schema import GroupedBranchDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_git_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"], check=True,
        capture_output=True)
    if status.stdout:
        raise RuntimeError(
            "selector calibration requires a clean git worktree")
    return commit


def _member_predictions(
    artifact,
    view: TorchGroupedView,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    members = len(artifact.ensemble.members)
    output = np.empty(
        (members, view.group_count, view.dataset.candidate_count),
        dtype=np.float32)
    indices = view.all_indices()
    artifact.ensemble.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            batch = view.batch(selected, device)
            prediction = artifact.ensemble.predict(
                batch.observation_history,
                batch.nominal_action,
                batch.candidate_action,
                batch.privileged_state,
            )
            values = prediction.member_risk.detach().cpu().numpy()
            if values.shape != (
                    members, len(selected), view.dataset.candidate_count):
                raise RuntimeError("Q_safe ensemble returned an invalid shape")
            output[:, selected, :] = values
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Q_safe ensemble returned non-finite calibration risk")
    return output


def _reward_values(reward_q, dataset: GroupedBranchDataset) -> np.ndarray:
    history = np.asarray(dataset["obs_history"], dtype=np.float32)
    requested = np.asarray(dataset["candidate_requested"], dtype=np.float32)
    mask = np.asarray(dataset["candidate_mask"], dtype=bool)
    values = np.empty(mask.shape, dtype=np.float32)
    for group in range(dataset.group_count):
        # Dense invalid slots must not poison critic inference.  They retain
        # nominal reward-Q and remain excluded by the original support mask.
        actions = np.where(
            mask[group, :, None],
            requested[group],
            requested[group, :1],
        )
        values[group] = reward_q.conservative_values(
            history[group, -1], actions)
        values[group, ~mask[group]] = values[group, 0]
    return values


def _publish_json(output: Path, value: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector calibration: {output}")
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{output.name}.staging-", suffix=".json", dir=output.parent)
    os.close(descriptor)
    staging = Path(raw_path)
    try:
        staging.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        # Reparse the exact staged bytes before an atomic no-clobber link.
        json.loads(staging.read_text(encoding="utf-8"))
        os.link(staging, output)
    finally:
        staging.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Agent directory containing the matched actor.pt and critic.pt")
    parser.add_argument(
        "--config", default="config/go2_50hz_sqrl_paper_sac_pretrain.yaml")
    parser.add_argument(
        "--protocol", default="config/qsafe_evidence_protocol.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()

    # Guard the lexical spelling before any resolver or loader can follow a
    # final-component alias into a locked audit artifact.
    artifact_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.artifact))
    calibration_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.calibration))
    checkpoint = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.checkpoint))
    config_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.config))
    protocol_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.protocol))
    output = assert_development_path(assert_safe_evidence_output(args.output))
    if output.suffix != ".json":
        parser.error("selector calibration output must use .json")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector calibration: {output}")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    commit = _clean_git_commit()

    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != 2:
        raise ValueError("selector calibration requires evidence protocol version 2")
    spec = SelectorCalibrationSpec.from_protocol(
        protocol["phase1"]["selector_calibration"])
    dataset = GroupedBranchDataset.load(calibration_path)
    artifact = load_qsafe_artifact(artifact_path, device=device)
    provenance = artifact.manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Q_safe artifact has no training provenance")
    expected_hash = provenance.get("dataset_content_sha256", {}).get(
        "calibration")
    if expected_hash != dataset.manifest.get("content_sha256"):
        raise ValueError(
            "calibration dataset does not match the artifact's declared split")
    if artifact.network_config.privileged_dim != 0 or (
            artifact.manifest.get("feature_view") != "deployable"):
        raise ValueError("selector calibration requires a deployable Q_safe artifact")

    actor = load_frozen_droq_policy(
        checkpoint,
        config_path,
        observation_dim=46,
        action_dim=12,
        device=device,
    )
    reward_q = load_frozen_droq_reward_q(
        checkpoint,
        config_path,
        observation_dim=46,
        action_dim=12,
        device=device,
    )
    continuation = provenance.get("continuation_policy_contract")
    actor_manifest = actor.manifest()
    reward_manifest = reward_q.manifest()
    if not isinstance(continuation, dict) or not continuation.get("verified"):
        raise ValueError("Q_safe artifact continuation policy is unverified")
    for key in ("training_step", "config_sha256", "observation_dim", "action_dim"):
        if continuation.get(key) != reward_manifest.get(key):
            raise ValueError(
                f"reward critic disagrees with continuation policy field {key!r}")
    for key in (
        "policy_fingerprint_sha256", "actor_state_dict_sha256",
        "config_sha256", "training_step", "observation_dim",
        "actor_observation_dim", "action_dim",
    ):
        if continuation.get(key) != actor_manifest.get(key):
            raise ValueError(
                f"runtime actor disagrees with continuation policy field {key!r}")
    if Path(actor_manifest["actor_path"]).resolve().parent != Path(
            reward_manifest["critic_path"]).resolve().parent:
        raise ValueError("runtime actor and reward critic are not one checkpoint bundle")

    view = TorchGroupedView(
        dataset,
        artifact.normalization,
        action_view=artifact.action_view,
        view_role="calibration",
    )
    member_risk = _member_predictions(
        artifact, view, device=device, batch_size=args.batch_size)
    reward_values = _reward_values(reward_q, dataset)
    inputs = SelectorCalibrationInputs(
        member_risk=member_risk,
        empirical_risk=np.asarray(dataset["fall"], dtype=np.float64).mean(axis=2),
        requested=np.asarray(dataset["candidate_requested"]),
        executed=np.asarray(dataset["candidate_executed"]),
        q_target=np.asarray(dataset["candidate_q_target"]),
        reward_q=reward_values,
        candidate_mask=np.asarray(dataset["candidate_mask"]),
        acceptance_probability=np.asarray(dataset["acceptance_probability"]),
        trajectory_id=np.asarray(dataset["trajectory_id"]),
        source_seed=np.asarray(dataset["source_seed"]),
    )
    result = calibrate_selector(
        inputs,
        spec,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    report = {
        "schema_version": "qsafe.selector_calibration.v1",
        "development_only": True,
        "protocol_name": protocol["protocol_name"],
        "generator_commit": commit,
        "artifact": str(artifact_path),
        "artifact_manifest_sha256": _sha256(artifact_path / "manifest.json"),
        "calibration_dataset": str(calibration_path),
        "calibration_content_sha256": dataset.manifest["content_sha256"],
        "actor_policy_fingerprint_sha256": actor.fingerprint(),
        "reward_q_fingerprint_sha256": reward_q.fingerprint(),
        "result": result.to_dict(),
        "artifact_common_model_gates_pass": bool(
            provenance.get("claim_eligible", False)),
        "paired_evaluation_authorized": bool(
            provenance.get("claim_eligible", False) and result.feasible),
        "phase2_authorized": False,
    }
    _publish_json(output, report)
    print(json.dumps({
        "output": str(output),
        "feasible": result.feasible,
        "selector_config": (
            None if result.selector_config is None
            else result.to_dict()["selector_config"]),
        "paired_evaluation_authorized": report["paired_evaluation_authorized"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
