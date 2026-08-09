from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import yaml

from rl.qsafe.artifact import load_qsafe_artifact
from scripts import train_selective_qsafe
from tests.test_safety_data import synthetic_dataset


class TrainSelectiveQSafeCliTest(unittest.TestCase):
    @staticmethod
    def _args(run_id: str, run: dict, **changes) -> argparse.Namespace:
        values = {
            "run_id": run_id,
            "action_mode": run["action_mode"],
            **run["hyperparameters"],
        }
        values.update(changes)
        return argparse.Namespace(**values)

    def test_grouped_three_split_training_writes_auditable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, offset in (
                ("train", 0), ("calibration", 1), ("evaluation", 2),
            ):
                dataset, _ = synthetic_dataset(
                    split=f"development_{name}", offset=offset)
                paths.append(dataset.save(root / f"{name}.npz"))
            protocol = yaml.safe_load(Path(
                "config/qsafe_evidence_protocol.yaml").read_text(
                    encoding="utf-8"))
            run = copy.deepcopy(protocol["phase1"]["model_training"]["runs"][
                "primary_selective_deployable"])
            run["hyperparameters"].update(
                epochs=1,
                batch_size=2,
                ensemble_members=2,
                calibration_steps=1,
                frame_hidden_dim=8,
                state_hidden_dim=8,
                action_hidden_dim=8,
                bootstrap_replicates=20,
            )
            heldout = copy.deepcopy(protocol["phase1"]["model_training"][
                "heldout_consumption"])
            heldout["ledger_root"] = str(root / "ledger")
            output = root / "artifact"
            argv = [
                "train_selective_qsafe",
                "--run-id", "primary_selective_deployable",
                "--train", str(paths[0]),
                "--calibration", str(paths[1]),
                "--test", str(paths[2]),
                "--output", str(output),
                "--epochs", "1",
                "--batch-size", "2",
                "--ensemble-members", "2",
                "--calibration-steps", "1",
                "--frame-hidden-dim", "8",
                "--state-hidden-dim", "8",
                "--action-hidden-dim", "8",
                "--bootstrap-replicates", "20",
                "--device", "cpu",
            ]
            with mock.patch("sys.argv", argv), mock.patch.object(
                    train_selective_qsafe,
                    "_locked_training_run",
                    return_value=(run, heldout, root / "ledger")), mock.patch.object(
                    train_selective_qsafe,
                    "_require_clean_git_state",
                    return_value="d" * 40):
                self.assertEqual(train_selective_qsafe.main(), 0)
            artifact = load_qsafe_artifact(output)
            self.assertEqual(artifact.network_config.action_dim, 36)
            self.assertEqual(artifact.action_view, "application_concat")
            self.assertEqual(
                artifact.action_components,
                ("requested", "executed", "q_target"),
            )
            provenance = artifact.manifest["provenance"]
            self.assertFalse(provenance["claim_eligible"])
            self.assertFalse(provenance["runtime_binding_verified"])
            self.assertIn(
                "runtime binding", provenance["claim_ineligible_reason"])
            self.assertIn("dataset_causal_contract_sha256", provenance)
            self.assertEqual(provenance["split_audit"]["pairs_checked"], 3)
            self.assertEqual(
                provenance["training_run_id"],
                "primary_selective_deployable")
            self.assertEqual(
                provenance["heldout_consumption"]["run_id"],
                "primary_selective_deployable")
            self.assertTrue(Path(
                provenance["heldout_consumption"]["marker_path"]).is_file())
            prediction = np.load(
                output / "test_predictions.npy", allow_pickle=False)
            self.assertEqual(prediction.shape, (4, 3))
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("test_predictions.npy", manifest["component_sha256"])

    def test_protocol_locks_exact_four_run_matrix_and_defaults(self):
        protocol = yaml.safe_load(Path(
            "config/qsafe_evidence_protocol.yaml").read_text(encoding="utf-8"))
        training = protocol["phase1"]["model_training"]
        self.assertEqual(
            set(training["runs"]),
            set(train_selective_qsafe._EXPECTED_RUN_CONTRACTS))
        expected_hyperparameters = {
            "epochs": 100,
            "batch_size": 64,
            "ensemble_members": 5,
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "seed": 20260809,
            "frame_hidden_dim": 128,
            "state_hidden_dim": 128,
            "action_hidden_dim": 128,
            "calibration_steps": 100,
            "bootstrap_replicates": 2000,
            "gradient_clip_norm": 5.0,
        }
        for run_id, contract in train_selective_qsafe._EXPECTED_RUN_CONTRACTS.items():
            run, heldout, _ = train_selective_qsafe._locked_training_run(
                protocol, run_id)
            self.assertEqual(run["hyperparameters"], expected_hyperparameters)
            for name, expected in contract.items():
                self.assertEqual(run[name], expected)
            self.assertEqual(heldout["consumptions_per_key"], 1)

    def test_recovery_option_data_refuses_before_heldout_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = [
                "train_selective_qsafe",
                "--run-id", "primary_selective_deployable",
                "--train", str(root / "train.npz"),
                "--calibration", str(root / "calibration.npz"),
                "--test", str(root / "evaluation.npz"),
                "--output", str(root / "artifact"),
                "--device", "cpu",
            ]
            recovery = SimpleNamespace(
                arrays={"candidate_option_steps": np.ones((1, 29), np.int8)})
            with mock.patch("sys.argv", argv), mock.patch.object(
                    train_selective_qsafe.GroupedBranchDataset,
                    "load", return_value=recovery) as load, mock.patch.object(
                    train_selective_qsafe,
                    "_require_clean_git_state", return_value="d" * 40), mock.patch.object(
                    train_selective_qsafe,
                    "_load_heldout_once") as heldout:
                with self.assertRaisesRegex(
                        ValueError, "not authorized for model training"):
                    train_selective_qsafe.main()
            self.assertEqual(load.call_count, 1)
            heldout.assert_not_called()

    def test_ledger_and_git_checks_are_anchored_to_repository(self):
        protocol = yaml.safe_load(Path(
            "config/qsafe_evidence_protocol.yaml").read_text(encoding="utf-8"))
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                _, _, ledger = train_selective_qsafe._locked_training_run(
                    protocol, "primary_selective_deployable")
            finally:
                os.chdir(original)
        self.assertEqual(
            ledger,
            train_selective_qsafe._REPOSITORY_ROOT
            / train_selective_qsafe._CANONICAL_LEDGER_ROOT_VALUE)

        completed = mock.Mock(stdout=b"")
        with mock.patch.object(
                train_selective_qsafe.subprocess,
                "run", return_value=completed) as invoked:
            train_selective_qsafe._git_status()
        command = invoked.call_args.args[0]
        self.assertEqual(command[:3], [
            "git", "-C", str(train_selective_qsafe._REPOSITORY_ROOT)])
        self.assertIn("--untracked-files=all", command)
        self.assertNotIn("--ignored", command)

    def test_parameter_and_feature_view_drift_fail_closed(self):
        protocol = yaml.safe_load(Path(
            "config/qsafe_evidence_protocol.yaml").read_text(encoding="utf-8"))
        primary, _, _ = train_selective_qsafe._locked_training_run(
            protocol, "primary_selective_deployable")
        args = self._args("primary_selective_deployable", primary)
        train_selective_qsafe._validate_cli_for_run(
            args, primary, (None, None, None))
        drifted = copy.copy(args)
        drifted.epochs = 99
        with self.assertRaisesRegex(ValueError, "parameter drift for epochs"):
            train_selective_qsafe._validate_cli_for_run(
                drifted, primary, (None, None, None))
        drifted = copy.copy(args)
        drifted.action_mode = "pointwise"
        with self.assertRaisesRegex(ValueError, "action-mode drift"):
            train_selective_qsafe._validate_cli_for_run(
                drifted, primary, (None, None, None))
        with self.assertRaisesRegex(ValueError, "feature-view drift"):
            train_selective_qsafe._validate_cli_for_run(
                args, primary, ("a", "b", "c"))

        privileged, _, _ = train_selective_qsafe._locked_training_run(
            protocol, "diagnostic_privileged_selective")
        privileged_args = self._args(
            "diagnostic_privileged_selective", privileged)
        with self.assertRaisesRegex(ValueError, "feature-view drift"):
            train_selective_qsafe._validate_cli_for_run(
                privileged_args, privileged, (None, None, None))

    def test_diagnostic_run_can_never_be_claim_eligible(self):
        protocol = yaml.safe_load(Path(
            "config/qsafe_evidence_protocol.yaml").read_text(encoding="utf-8"))
        diagnostic, _, _ = train_selective_qsafe._locked_training_run(
            protocol, "diagnostic_pointwise_deployable")
        eligible, reason = train_selective_qsafe._claim_decision(
            diagnostic,
            deployable=True,
            runtime_binding_verified=True,
            data_gate_pass=True,
            model_gate_pass=True,
        )
        self.assertFalse(eligible)
        self.assertIn("no multiple-comparison claim", reason)

    def test_dirty_or_changed_head_fails_closed(self):
        with mock.patch.object(
                train_selective_qsafe, "_git_commit", return_value="a" * 40), (
                mock.patch.object(
                    train_selective_qsafe, "_git_status", return_value=b"?? note\0")):
            with self.assertRaisesRegex(RuntimeError, "clean git worktree"):
                train_selective_qsafe._require_clean_git_state(
                    phase="before training")
        with mock.patch.object(
                train_selective_qsafe, "_git_commit", return_value="b" * 40), (
                mock.patch.object(
                    train_selective_qsafe, "_git_status", return_value=b"")):
            with self.assertRaisesRegex(RuntimeError, "HEAD changed"):
                train_selective_qsafe._require_clean_git_state(
                    phase="after training", expected_commit="a" * 40)

    def test_one_shot_marker_precedes_load_and_duplicate_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "heldout-bytes.bin"
            source.write_bytes(b"synthetic held-out bytes")
            ledger = root / "ledger"
            sentinel = object()
            observed_loads = []

            def load_after_marker(path):
                observed_loads.append(Path(path))
                self.assertEqual(Path(path), source.resolve())
                markers = list(ledger.glob("*.json"))
                self.assertEqual(len(markers), len(observed_loads))
                return sentinel

            with mock.patch.object(
                    train_selective_qsafe.GroupedBranchDataset,
                    "load", side_effect=load_after_marker) as loader:
                dataset, consumption = train_selective_qsafe._load_heldout_once(
                    source,
                    ledger_root=ledger,
                    run_id="primary_selective_deployable",
                    protocol_name="synthetic_protocol",
                    git_commit="a" * 40,
                )
                self.assertIs(dataset, sentinel)
                self.assertEqual(loader.call_count, 1)
                marker = Path(consumption["marker_path"])
                self.assertTrue(marker.is_file())
                self.assertEqual(
                    consumption["marker_sha256"],
                    train_selective_qsafe._file_sha256(marker))
                with self.assertRaisesRegex(RuntimeError, "already consumed"):
                    train_selective_qsafe._load_heldout_once(
                        source,
                        ledger_root=ledger,
                        run_id="primary_selective_deployable",
                        protocol_name="synthetic_protocol",
                        git_commit="a" * 40,
                    )
                self.assertEqual(loader.call_count, 1)
                _, second_run = train_selective_qsafe._load_heldout_once(
                    source,
                    ledger_root=ledger,
                    run_id="diagnostic_pointwise_deployable",
                    protocol_name="synthetic_protocol",
                    git_commit="a" * 40,
                )
                self.assertEqual(loader.call_count, 2)
                self.assertNotEqual(
                    consumption["marker_path"], second_run["marker_path"])

    def test_heldout_symlink_alias_is_rejected_before_hash_or_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "source-7801.audit.npz"
            audit.write_bytes(b"must-not-read")
            alias = root / "evaluation.npz"
            alias.symlink_to(audit)
            with mock.patch.object(
                    train_selective_qsafe, "_file_sha256",
                    side_effect=AssertionError("alias must not be hashed")):
                with self.assertRaisesRegex(
                        PermissionError, "refuse symlink inputs"):
                    train_selective_qsafe._load_heldout_once(
                        alias,
                        ledger_root=root / "ledger",
                        run_id="primary_selective_deployable",
                        protocol_name="synthetic_protocol",
                        git_commit="a" * 40,
                    )
            self.assertFalse((root / "ledger").exists())

    def test_causal_contract_drift_between_splits_fails_closed(self):
        datasets = [
            synthetic_dataset(
                split=f"development_{name}", offset=index)[0]
            for index, name in enumerate(("train", "calibration", "test"))
        ]
        datasets[1].manifest["candidate_protocol"] = {
            **datasets[1].manifest["candidate_protocol"],
            "generator": "drifted_after_train",
        }
        with self.assertRaisesRegex(ValueError, "causal dataset contracts differ"):
            train_selective_qsafe._require_causal_split_compatibility(
                tuple(datasets))

    def test_modern_policy_binding_records_behavior_not_local_path(self):
        manifest = {
            "policy_fingerprint_sha256": "a" * 64,
            "actor_state_dict_sha256": "b" * 64,
            "config_sha256": "c" * 64,
            "training_step": 500_000,
            "observation_dim": 46,
            "actor_observation_dim": 46,
            "action_dim": 12,
            "actor_path": "/machine-specific/checkpoint/actor.pt",
            "device": "cuda:7",
        }
        contract = train_selective_qsafe._policy_binding_contract(
            manifest, role="continuation")
        self.assertTrue(contract["verified"])
        self.assertNotIn("actor_path", contract)
        self.assertNotIn("device", contract)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            train_selective_qsafe._policy_binding_contract(
                {**manifest, "policy_fingerprint_sha256": "short"},
                role="continuation")


if __name__ == "__main__":
    unittest.main()
