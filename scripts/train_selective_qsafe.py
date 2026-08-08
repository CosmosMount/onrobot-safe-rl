#!/usr/bin/env python3
"""Train and audit a grouped Selective Advantage Q_safe ensemble."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch
import yaml

from rl.qsafe.artifact import save_qsafe_artifact
from rl.qsafe.data import NormalizationStats, TorchGroupedView
from rl.qsafe.loss import QSafeLossConfig
from rl.qsafe.network import QSafeNetworkConfig
from rl.qsafe.training import (
    QSafeTrainingConfig,
    predict_qsafe_ensemble,
    train_qsafe_ensemble,
)
from safety_data.metrics import evaluate_predictions
from safety_data.paths import assert_development_path
from safety_data.schema import (
    GroupedBranchDataset,
    PrivilegedBranchView,
    audit_split_disjointness,
)


_CAUSAL_MANIFEST_KEYS = (
    "simulator_fingerprint",
    "candidate_protocol",
    "fall_definition",
    "observation_contract",
    "action_application_contract",
    "state_hash_contract",
    "collection_protocol",
)
_POLICY_BINDING_KEYS = (
    "policy_fingerprint_sha256",
    "actor_state_dict_sha256",
    "config_sha256",
    "training_step",
    "observation_dim",
    "actor_observation_dim",
    "action_dim",
)


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _policy_binding_contract(
    manifest: Any,
    *,
    role: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"dataset {role} policy manifest must be nonempty")
    verified = all(key in manifest for key in _POLICY_BINDING_KEYS)
    if verified:
        contract = {
            key: copy.deepcopy(manifest[key]) for key in _POLICY_BINDING_KEYS
        }
        for key in (
            "policy_fingerprint_sha256", "actor_state_dict_sha256",
            "config_sha256",
        ):
            digest = contract[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest):
                raise ValueError(
                    f"dataset {role} policy {key} is invalid")
        if isinstance(contract["training_step"], bool) or not isinstance(
                contract["training_step"], int) or contract["training_step"] < 0:
            raise ValueError(f"dataset {role} policy training_step is invalid")
        if contract["observation_dim"] != 46 or contract["action_dim"] != 12:
            raise ValueError(
                f"dataset {role} policy has an incompatible observation/action contract")
        if isinstance(contract["actor_observation_dim"], bool) or not isinstance(
                contract["actor_observation_dim"], int) or not (
                    1 <= contract["actor_observation_dim"] <= 46):
            raise ValueError(
                f"dataset {role} actor_observation_dim is invalid")
        contract["verified"] = True
        return contract
    # Legacy/synthetic fixtures remain trainable only as explicitly unbound
    # diagnostics.  They can never become a deployable claim artifact.
    return {
        "verified": False,
        "legacy_manifest_sha256": _canonical_sha256(manifest),
    }


def _dataset_causal_contract(
    dataset: GroupedBranchDataset,
) -> dict[str, Any]:
    manifest = dataset.manifest
    contract = {
        key: copy.deepcopy(manifest.get(key))
        for key in _CAUSAL_MANIFEST_KEYS
    }
    contract["horizon_steps"] = int(manifest["horizon_steps"])
    contract["source_policy"] = _policy_binding_contract(
        manifest.get("source_policy"), role="source")
    contract["continuation_policy"] = _policy_binding_contract(
        manifest.get("continuation_policy"), role="continuation")
    return contract


def _require_causal_split_compatibility(
    datasets: tuple[GroupedBranchDataset, ...],
) -> tuple[dict[str, Any], list[str]]:
    contracts = [_dataset_causal_contract(dataset) for dataset in datasets]
    reference = contracts[0]
    for index, contract in enumerate(contracts[1:], start=1):
        if contract != reference:
            raise ValueError(
                "train/calibration/test causal dataset contracts differ; "
                f"split index {index} cannot share one Q_safe artifact")
    commits = [str(dataset.manifest["generator_commit"]) for dataset in datasets]
    return reference, commits


def _finite_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_privileged(
    path: str | None,
    dataset: GroupedBranchDataset,
) -> PrivilegedBranchView | None:
    return None if path is None else PrivilegedBranchView.load(
        path, deployable=dataset)


def _data_gate(
    dataset: GroupedBranchDataset,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    report = dataset.validate()
    mask = np.asarray(dataset["candidate_mask"], dtype=bool)
    checks = {
        "independent_groups": dataset.group_count
        >= int(thresholds["min_independent_groups"]),
        "trajectory_clusters": len(np.unique(dataset["trajectory_id"]))
        >= int(thresholds["min_independent_trajectory_clusters"]),
        "source_seeds": len(np.unique(dataset["source_seed"]))
        >= int(thresholds["min_source_seeds"]),
        "candidates_per_group": int(mask.sum(axis=1).min())
        >= int(thresholds["min_candidates_per_group"]),
        "replicas_per_candidate": dataset.replica_count
        >= int(thresholds["min_replicas_per_candidate"]),
        "mixed_outcomes": float(report["mixed_outcome_fraction"])
        >= float(thresholds["min_mixed_outcome_fraction"]),
        "duplicate_groups": float(report["duplicate_state_fraction"])
        <= float(thresholds["max_duplicate_group_fraction"]),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "observed": {
            "independent_groups": dataset.group_count,
            "trajectory_clusters": int(len(np.unique(dataset["trajectory_id"]))),
            "source_seeds": int(len(np.unique(dataset["source_seed"]))),
            "min_candidates_per_group": int(mask.sum(axis=1).min()),
            "replicas_per_candidate": dataset.replica_count,
            "mixed_outcome_fraction": float(report["mixed_outcome_fraction"]),
            "duplicate_group_fraction": float(report["duplicate_state_fraction"]),
        },
    }


def _model_gate(metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    pair_ci_low = float(metrics["pair_accuracy_ci95"][0])
    top1_ci_low = float(metrics["top1_reduction_ci95"][0])
    checks = {
        "pair_accuracy": float(metrics["pair_accuracy_group_macro"])
        >= float(thresholds["min_pair_accuracy"]),
        "pair_accuracy_ci_low": pair_ci_low
        >= float(thresholds["min_pair_accuracy_ci_low"]),
        "strong_pair_accuracy": float(metrics["strong_pair_accuracy_group_macro"])
        >= float(thresholds["min_strong_pair_accuracy"]),
        "top1_absolute_reduction": float(metrics["top1_absolute_reduction"])
        >= float(thresholds["min_top1_absolute_reduction"]),
        "top1_reduction_ci_low": top1_ci_low
        > float(thresholds["min_top1_reduction_ci_low"]),
        "oracle_gap_capture": float(metrics["oracle_gap_capture"])
        >= float(thresholds["min_oracle_gap_capture"]),
        "ece": float(metrics["ece_equal_mass"])
        <= float(thresholds["max_ece"]),
    }
    return {"pass": all(checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--train-privileged")
    parser.add_argument("--calibration-privileged")
    parser.add_argument("--test-privileged")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol", default="config/qsafe_evidence_protocol.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--calibration-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument(
        "--action-mode",
        choices=("selective_advantage", "pointwise", "state_only"),
        default="selective_advantage",
    )
    parser.add_argument("--frame-hidden-dim", type=int, default=128)
    parser.add_argument("--state-hidden-dim", type=int, default=128)
    parser.add_argument("--action-hidden-dim", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()

    privileged_paths = (
        args.train_privileged,
        args.calibration_privileged,
        args.test_privileged,
    )
    if any(privileged_paths) and not all(privileged_paths):
        parser.error("privileged diagnostics require all three split views")

    protocol_path = assert_development_path(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if int(protocol.get("protocol_version", -1)) != 2:
        raise ValueError("this trainer requires evidence protocol version 2")
    output = assert_development_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")

    train_data = GroupedBranchDataset.load(args.train)
    calibration_data = GroupedBranchDataset.load(args.calibration)
    test_data = GroupedBranchDataset.load(args.test)
    split_audit = audit_split_disjointness(
        [train_data, calibration_data, test_data])
    causal_contract, dataset_generator_commits = (
        _require_causal_split_compatibility(
            (train_data, calibration_data, test_data)))
    train_privileged = _load_privileged(args.train_privileged, train_data)
    calibration_privileged = _load_privileged(
        args.calibration_privileged, calibration_data)
    test_privileged = _load_privileged(args.test_privileged, test_data)

    speeds = [
        float(np.asarray(dataset["command_vx"], dtype=np.float64)[0])
        for dataset in (train_data, calibration_data, test_data)
    ]
    if max(speeds) - min(speeds) > 1e-6:
        raise ValueError(
            "train/calibration/test must target one command speed because the "
            "deployable 46D observation does not contain the command")

    normalization = NormalizationStats.fit(train_data, train_privileged)
    train_view = TorchGroupedView(
        train_data, normalization, train_privileged)
    calibration_view = TorchGroupedView(
        calibration_data, normalization, calibration_privileged)
    test_view = TorchGroupedView(
        test_data, normalization, test_privileged)
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device)
    network_config = QSafeNetworkConfig(
        action_dim=train_view.action_dim,
        frame_hidden_dim=args.frame_hidden_dim,
        state_hidden_dim=args.state_hidden_dim,
        action_hidden_dim=args.action_hidden_dim,
        privileged_dim=train_view.privileged_dim,
        action_mode=args.action_mode,
    )
    training_config = QSafeTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ensemble_members=args.ensemble_members,
        seed=args.seed,
        device=device,
        calibration_steps=args.calibration_steps,
    )
    loss_config = QSafeLossConfig()
    trained = train_qsafe_ensemble(
        train_view,
        network_config,
        training_config,
        loss_config,
        calibration_view,
    )
    calibration_prediction = predict_qsafe_ensemble(
        trained, calibration_view, device=device)
    test_prediction = predict_qsafe_ensemble(
        trained, test_view, device=device)
    calibration_metrics = evaluate_predictions(
        calibration_data,
        calibration_prediction,
        strong_pair_gap=float(
            protocol["phase1"]["model_gate"]["strong_pair_min_empirical_risk_gap"]),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.seed + 81,
    )
    test_metrics = evaluate_predictions(
        test_data,
        test_prediction,
        strong_pair_gap=float(
            protocol["phase1"]["model_gate"]["strong_pair_min_empirical_risk_gap"]),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.seed + 82,
    )
    data_gate = _data_gate(train_data, protocol["phase1"]["data_gate"])
    model_gate = _model_gate(test_metrics, protocol["phase1"]["model_gate"])
    deployable = train_privileged is None
    runtime_binding_verified = bool(
        causal_contract["source_policy"]["verified"]
        and causal_contract["continuation_policy"]["verified"])
    claim_eligible = bool(
        deployable
        and runtime_binding_verified
        and data_gate["pass"]
        and model_gate["pass"])
    provenance = {
        "generator_commit": _git_commit(),
        "protocol_path": str(protocol_path),
        "protocol_name": protocol["protocol_name"],
        "command_vx": speeds[0],
        "action_feature_contract": {
            "view": train_view.action_view,
            "components_in_order": [
                "requested", "executed", "q_target"],
            "total_width": train_view.action_dim,
        },
        "split_audit": split_audit,
        "dataset_generator_commits": dataset_generator_commits,
        "dataset_causal_contract": causal_contract,
        "dataset_causal_contract_sha256": _canonical_sha256(causal_contract),
        "source_policy_contract": causal_contract["source_policy"],
        "continuation_policy_contract": causal_contract[
            "continuation_policy"],
        "runtime_binding_verified": runtime_binding_verified,
        "dataset_content_sha256": {
            "train": train_data.manifest["content_sha256"],
            "calibration": calibration_data.manifest["content_sha256"],
            "test": test_data.manifest["content_sha256"],
        },
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
        "data_gate": data_gate,
        "model_gate": model_gate,
        "claim_eligible": claim_eligible,
        "claim_ineligible_reason": (
            None if claim_eligible else
            "privileged diagnostics cannot support a deployment claim"
            if not deployable else
            "source/continuation policy runtime binding is not verified"
            if not runtime_binding_verified else
            "one or more preregistered data/model gates failed"),
    }
    save_qsafe_artifact(
        output,
        trained,
        normalization,
        network_config,
        training_config,
        loss_config,
        provenance=provenance,
        array_attachments={
            "calibration_predictions.npy": calibration_prediction,
            "test_predictions.npy": test_prediction,
        },
    )
    print(json.dumps(_finite_json({
        "artifact": str(output),
        "claim_eligible": claim_eligible,
        "data_gate": data_gate,
        "model_gate": model_gate,
        "test_metrics": test_metrics,
    }), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
