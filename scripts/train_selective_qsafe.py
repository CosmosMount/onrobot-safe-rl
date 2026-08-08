#!/usr/bin/env python3
"""Train and audit a grouped Selective Advantage Q_safe ensemble."""

from __future__ import annotations

import argparse
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
    claim_eligible = bool(deployable and data_gate["pass"] and model_gate["pass"])
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
