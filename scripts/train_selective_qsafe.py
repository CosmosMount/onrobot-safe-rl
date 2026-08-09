#!/usr/bin/env python3
"""Train and audit a grouped Selective Advantage Q_safe ensemble."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
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
_TRAINING_REGISTRY_SCHEMA_VERSION = "qsafe.model_training_registry.v1"
_HELDOUT_CONSUMPTION_UNIT = "preregistered_run_id_and_test_file_sha256"
_DIAGNOSTIC_CLAIM_POLICY = "no_multiple_comparison_claim"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_LEDGER_ROOT_VALUE = (
    "saved/qsafe_development/ledgers/qsafe_model_test_once_v1")
_CANONICAL_PROTOCOL_PATH = (
    _REPOSITORY_ROOT / "config" / "qsafe_evidence_protocol.yaml"
).resolve()
_LOCKED_HYPERPARAMETERS = (
    "epochs",
    "batch_size",
    "ensemble_members",
    "learning_rate",
    "weight_decay",
    "seed",
    "frame_hidden_dim",
    "state_hidden_dim",
    "action_hidden_dim",
    "calibration_steps",
    "bootstrap_replicates",
    "gradient_clip_norm",
)
_INTEGER_HYPERPARAMETERS = {
    "epochs", "batch_size", "ensemble_members", "seed",
    "frame_hidden_dim", "state_hidden_dim", "action_hidden_dim",
    "calibration_steps", "bootstrap_replicates",
}
_NONNEGATIVE_HYPERPARAMETERS = {"seed", "weight_decay"}
_EXPECTED_RUN_CONTRACTS = {
    "primary_selective_deployable": {
        "claim_role": "primary",
        "claim_eligible": True,
        "feature_view": "deployable",
        "action_feature_view": "application_concat",
        "action_mode": "selective_advantage",
    },
    "diagnostic_pointwise_deployable": {
        "claim_role": "diagnostic",
        "claim_eligible": False,
        "feature_view": "deployable",
        "action_feature_view": "application_concat",
        "action_mode": "pointwise",
    },
    "diagnostic_state_only_deployable": {
        "claim_role": "diagnostic",
        "claim_eligible": False,
        "feature_view": "deployable",
        "action_feature_view": "application_concat",
        "action_mode": "state_only",
    },
    "diagnostic_privileged_selective": {
        "claim_role": "diagnostic",
        "claim_eligible": False,
        "feature_view": "privileged",
        "action_feature_view": "application_concat",
        "action_mode": "selective_advantage",
    },
}


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


def _locked_training_run(
    protocol: dict[str, Any],
    run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    phase1 = protocol.get("phase1")
    training = phase1.get("model_training") if isinstance(phase1, dict) else None
    if not isinstance(training, dict) or training.get(
            "registry_schema_version") != _TRAINING_REGISTRY_SCHEMA_VERSION:
        raise ValueError("protocol has no supported Phase 1 model-training registry")
    heldout = training.get("heldout_consumption")
    if not isinstance(heldout, dict) or set(heldout) != {
        "unit", "consumptions_per_key", "ledger_root",
        "diagnostic_claim_policy",
    }:
        raise ValueError("held-out consumption policy is incomplete or has drifted")
    if heldout.get("unit") != _HELDOUT_CONSUMPTION_UNIT or heldout.get(
            "consumptions_per_key") != 1 or heldout.get(
                "diagnostic_claim_policy") != _DIAGNOSTIC_CLAIM_POLICY:
        raise ValueError("held-out once-only or diagnostic claim policy has drifted")
    ledger_value = heldout.get("ledger_root")
    if ledger_value != _CANONICAL_LEDGER_ROOT_VALUE:
        raise ValueError("held-out ledger_root has drifted from its canonical path")
    ledger_root = assert_development_path(_REPOSITORY_ROOT / ledger_value)

    runs = training.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(_EXPECTED_RUN_CONTRACTS):
        raise ValueError("model-training registry must contain exactly four locked runs")
    for registered_id, expected_contract in _EXPECTED_RUN_CONTRACTS.items():
        value = runs.get(registered_id)
        if not isinstance(value, dict) or set(value) != {
            *expected_contract, "hyperparameters",
        }:
            raise ValueError(f"training run {registered_id!r} has schema drift")
        for name, expected in expected_contract.items():
            if value.get(name) != expected:
                raise ValueError(
                    f"training run {registered_id!r} changes locked field {name!r}")
        hyperparameters = value.get("hyperparameters")
        if not isinstance(hyperparameters, dict) or set(hyperparameters) != set(
                _LOCKED_HYPERPARAMETERS):
            raise ValueError(
                f"training run {registered_id!r} has hyperparameter schema drift")
        for name, item in hyperparameters.items():
            if isinstance(item, bool):
                raise ValueError(
                    f"training run {registered_id!r} hyperparameter {name!r} is invalid")
            if name in _INTEGER_HYPERPARAMETERS:
                valid_type = isinstance(item, int)
            else:
                valid_type = isinstance(item, (int, float)) and np.isfinite(item)
            if not valid_type:
                raise ValueError(
                    f"training run {registered_id!r} hyperparameter {name!r} is invalid")
            valid_range = item >= 0 if name in _NONNEGATIVE_HYPERPARAMETERS else item > 0
            if not valid_range:
                raise ValueError(
                    f"training run {registered_id!r} hyperparameter {name!r} is invalid")
    if run_id not in runs:
        raise ValueError(f"run_id {run_id!r} is not preregistered")
    return copy.deepcopy(runs[run_id]), copy.deepcopy(heldout), ledger_root


def _validate_cli_for_run(
    args: argparse.Namespace,
    run: dict[str, Any],
    privileged_paths: tuple[str | None, str | None, str | None],
) -> None:
    if any(privileged_paths) and not all(privileged_paths):
        raise ValueError("privileged diagnostics require all three split views")
    expected_privileged = run["feature_view"] == "privileged"
    supplied_privileged = all(privileged_paths)
    if supplied_privileged != expected_privileged:
        raise ValueError(
            f"run_id {args.run_id!r} feature-view drift: expected "
            f"{run['feature_view']!r}")
    if args.action_mode != run["action_mode"]:
        raise ValueError(
            f"run_id {args.run_id!r} action-mode drift: got "
            f"{args.action_mode!r}, expected {run['action_mode']!r}")
    if run["action_feature_view"] != "application_concat":
        raise ValueError("trainer supports only the locked application_concat action view")
    expected = run["hyperparameters"]
    for name in _LOCKED_HYPERPARAMETERS:
        actual = getattr(args, name)
        if actual != expected[name]:
            raise ValueError(
                f"run_id {args.run_id!r} parameter drift for {name}: "
                f"got {actual!r}, expected {expected[name]!r}")


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_status() -> bytes:
    return subprocess.run(
        [
            "git", "-C", str(_REPOSITORY_ROOT),
            "status", "--porcelain=v1", "-z",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
    ).stdout


def _require_clean_git_state(
    *,
    phase: str,
    expected_commit: str | None = None,
) -> str:
    commit = _git_commit()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            f"git HEAD changed during Q_safe training at {phase}: "
            f"{expected_commit} -> {commit}")
    if _git_status():
        raise RuntimeError(
            f"Q_safe training requires a clean git worktree at {phase}")
    return commit


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _consume_heldout_once(
    *,
    test_path: Path,
    test_file_sha256: str,
    ledger_root: Path,
    run_id: str,
    protocol_name: str,
    git_commit: str,
) -> dict[str, str]:
    if len(test_file_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in test_file_sha256):
        raise ValueError("held-out test file SHA-256 is invalid")
    root = assert_development_path(ledger_root)
    root.mkdir(parents=True, exist_ok=True)
    marker = assert_development_path(
        root / f"{run_id}.{test_file_sha256}.json")
    payload = {
        "schema_version": "qsafe.heldout_consumption.v1",
        "consumption_unit": _HELDOUT_CONSUMPTION_UNIT,
        "consumptions_for_key": 1,
        "protocol_name": protocol_name,
        "run_id": run_id,
        "test_file_sha256": test_file_sha256,
        "test_path": str(test_path),
        "git_commit": git_commit,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(marker, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "held-out test file was already consumed for this preregistered "
            f"run_id: {marker}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial marker intentionally remains consumed: retrying after any
        # ambiguity would violate the exactly-once policy.
        raise
    return {
        "marker_path": str(marker),
        "marker_sha256": _file_sha256(marker),
        "test_file_sha256": test_file_sha256,
        "run_id": run_id,
    }


def _load_heldout_once(
    test_path: Path,
    *,
    ledger_root: Path,
    run_id: str,
    protocol_name: str,
    git_commit: str,
) -> tuple[GroupedBranchDataset, dict[str, str]]:
    source = assert_development_path(test_path)
    test_file_sha256 = _file_sha256(source)
    consumption = _consume_heldout_once(
        test_path=source,
        test_file_sha256=test_file_sha256,
        ledger_root=ledger_root,
        run_id=run_id,
        protocol_name=protocol_name,
        git_commit=git_commit,
    )
    dataset = GroupedBranchDataset.load(source)
    if _file_sha256(source) != test_file_sha256:
        raise RuntimeError("held-out test file changed while it was being loaded")
    return dataset, consumption


def _claim_decision(
    run: dict[str, Any],
    *,
    deployable: bool,
    runtime_binding_verified: bool,
    data_gate_pass: bool,
    model_gate_pass: bool,
) -> tuple[bool, str | None]:
    if not bool(run["claim_eligible"]):
        return (
            False,
            "preregistered diagnostic run is ineligible for a primary claim; "
            "no multiple-comparison claim is permitted",
        )
    if not deployable:
        return False, "privileged diagnostics cannot support a deployment claim"
    if not runtime_binding_verified:
        return False, "source/continuation policy runtime binding is not verified"
    if not data_gate_pass or not model_gate_pass:
        return False, "one or more preregistered data/model gates failed"
    return True, None


def _load_privileged(
    path: str | Path | None,
    dataset: GroupedBranchDataset,
) -> PrivilegedBranchView | None:
    return None if path is None else PrivilegedBranchView.load(
        path, deployable=dataset)


def _require_preheldout_model_authorization(
    train_dataset: GroupedBranchDataset,
) -> None:
    """Reject triage-only option labels before touching held-out state."""
    if "candidate_option_steps" in train_dataset.arrays:
        raise ValueError(
            "recovery-option triage data are not authorized for model training; "
            "a passed independent-replica triage and a fresh duration-aware "
            "model protocol are required before held-out consumption")


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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--train-privileged")
    parser.add_argument("--calibration-privileged")
    parser.add_argument("--test-privileged")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol", default=str(_CANONICAL_PROTOCOL_PATH))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ensemble-members", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
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

    protocol_path = assert_development_path(args.protocol)
    if protocol_path != _CANONICAL_PROTOCOL_PATH:
        raise ValueError(
            "Q_safe training requires the repository's canonical locked protocol")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if int(protocol.get("protocol_version", -1)) != 2:
        raise ValueError("this trainer requires evidence protocol version 2")
    run, heldout_policy, ledger_root = _locked_training_run(
        protocol, args.run_id)
    _validate_cli_for_run(args, run, privileged_paths)
    output = assert_development_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    training_commit = _require_clean_git_state(phase="before training")

    train_path = assert_development_path(args.train)
    calibration_path = assert_development_path(args.calibration)
    test_path = assert_development_path(args.test)
    train_privileged_path = (
        None if args.train_privileged is None
        else assert_development_path(args.train_privileged))
    calibration_privileged_path = (
        None if args.calibration_privileged is None
        else assert_development_path(args.calibration_privileged))
    test_privileged_path = (
        None if args.test_privileged is None
        else assert_development_path(args.test_privileged))
    train_data = GroupedBranchDataset.load(train_path)
    _require_preheldout_model_authorization(train_data)
    calibration_data = GroupedBranchDataset.load(calibration_path)
    train_privileged = _load_privileged(train_privileged_path, train_data)
    calibration_privileged = _load_privileged(
        calibration_privileged_path, calibration_data)
    test_data, heldout_consumption = _load_heldout_once(
        test_path,
        ledger_root=ledger_root,
        run_id=args.run_id,
        protocol_name=str(protocol["protocol_name"]),
        git_commit=training_commit,
    )
    test_privileged = _load_privileged(test_privileged_path, test_data)
    split_audit = audit_split_disjointness(
        [train_data, calibration_data, test_data])
    causal_contract, dataset_generator_commits = (
        _require_causal_split_compatibility(
            (train_data, calibration_data, test_data)))

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
    expected_feature_view = str(run["feature_view"])
    for role, view in (
        ("train", train_view),
        ("calibration", calibration_view),
        ("test", test_view),
    ):
        if view.feature_view != expected_feature_view:
            raise ValueError(
                f"run_id {args.run_id!r} {role} feature-view drift: "
                f"got {view.feature_view!r}, expected {expected_feature_view!r}")
        if view.action_view != run["action_feature_view"]:
            raise ValueError(
                f"run_id {args.run_id!r} {role} action-feature-view drift")
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
        gradient_clip_norm=args.gradient_clip_norm,
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
    observed_model_gate = _model_gate(
        test_metrics, protocol["phase1"]["model_gate"])
    model_gate_for_claim = bool(run["claim_eligible"])
    model_gate = {
        "pass": bool(observed_model_gate["pass"] and model_gate_for_claim),
        "observed_pass": bool(observed_model_gate["pass"]),
        "claim_evaluated": model_gate_for_claim,
        "checks": observed_model_gate["checks"],
    }
    deployable = train_privileged is None
    runtime_binding_verified = bool(
        causal_contract["source_policy"]["verified"]
        and causal_contract["continuation_policy"]["verified"])
    claim_eligible, claim_ineligible_reason = _claim_decision(
        run,
        deployable=deployable,
        runtime_binding_verified=runtime_binding_verified,
        data_gate_pass=bool(data_gate["pass"]),
        model_gate_pass=bool(model_gate["pass"]),
    )
    _require_clean_git_state(
        phase="after training",
        expected_commit=training_commit,
    )
    provenance = {
        "generator_commit": training_commit,
        "generator_worktree_clean": True,
        "generator_commit_stable": True,
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": _file_sha256(protocol_path),
        "protocol_name": protocol["protocol_name"],
        "training_run_id": args.run_id,
        "training_run_contract": run,
        "training_run_contract_sha256": _canonical_sha256(run),
        "heldout_consumption_policy": heldout_policy,
        "heldout_consumption": heldout_consumption,
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
        "claim_ineligible_reason": claim_ineligible_reason,
        "diagnostic_no_multiple_comparison_claim": bool(
            run["claim_role"] == "diagnostic"),
    }

    def final_git_check() -> None:
        _require_clean_git_state(
            phase="before artifact publication",
            expected_commit=training_commit,
        )

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
        pre_publish_check=final_git_check,
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
