from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from rl.qsafe.artifact import load_qsafe_artifact, save_qsafe_artifact
from rl.qsafe.data import NormalizationStats
from rl.qsafe.loss import QSafeLossConfig
from rl.qsafe.network import QSafeEnsemble, QSafeNetworkConfig, SelectiveAdvantageQSafe
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
    bind_recovery_program_manifest,
    make_recovery_program_feature_manifest,
)
from rl.qsafe.recovery_selector import (
    RecoveryConformalOffsets,
    RecoverySelectorBundle,
    RecoverySelectorConfig,
)
from safety_data.recovery_behaviors import RecoveryBehaviorLibrary
from rl.qsafe.training import (
    RECOVERY_PROGRAM_V4_LOSS_CONFIG,
    RECOVERY_PROGRAM_V4_MEMBER_SEED_STRIDE,
    RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
    RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
    QSafeTrainingConfig,
    TrainedQSafeEnsemble,
    TrainedQSafeMember,
)
from tests.test_qsafe_recovery_behaviors import _MaturePolicy, _applier


def full_recovery_binding():
    library = RecoveryBehaviorLibrary(_MaturePolicy(), _applier())
    return bind_recovery_program_manifest(library.manifest())


def frozen_selector_bundle():
    return RecoverySelectorBundle.create(
        offsets=RecoveryConformalOffsets(
            nominal_lower=0.03,
            risk_upper=np.asarray([0.0] + [0.04] * 8),
            benefit_lower=np.asarray([0.0] + [0.05] * 8),
            calibration_report_sha256="b" * 64,
        ),
        selector_config=RecoverySelectorConfig(
            nominal_risk_lcb_trigger=0.50,
            min_benefit_lcb=0.08,
            max_risk_ucb=0.55,
            max_epistemic_std=0.20,
            max_action_delta_rms=0.50,
            max_q_target_delta_rms=0.25,
        ),
        probability_calibration_report_sha256="a" * 64,
        uncertainty_calibration_report_sha256="b" * 64,
        selector_search_report_sha256="c" * 64,
    )


def canonical_manifest_sha256(path: Path) -> str:
    manifest = json.loads(
        (path / "manifest.json").read_text(encoding="utf-8"))
    payload = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class QSafeArtifactTest(unittest.TestCase):
    def _trained(self):
        torch.manual_seed(7)
        config = QSafeNetworkConfig(
            frame_hidden_dim=8,
            state_hidden_dim=8,
            action_hidden_dim=8,
        )
        members = []
        for index in range(2):
            model = SelectiveAdvantageQSafe(config)
            members.append(TrainedQSafeMember(
                model=model,
                seed=100 + index,
                bootstrap_trajectories=[f"trajectory-{index}"],
                epoch_loss=[1.0, 0.5],
                temperature=1.0 + 0.1 * index,
            ))
        trained = TrainedQSafeEnsemble(
            ensemble=QSafeEnsemble(
                [member.model for member in members],
                [member.temperature for member in members],
            ),
            members=members,
            action_view="requested",
            action_dim=12,
        )
        return config, trained

    def test_round_trip_and_refuses_overwrite(self):
        config, trained = self._trained()
        normalization = NormalizationStats(
            np.zeros(46, dtype=np.float32),
            np.ones(46, dtype=np.float32),
        )
        training = QSafeTrainingConfig(
            epochs=2, ensemble_members=2, calibration_steps=0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "development-artifact"
            save_qsafe_artifact(
                output,
                trained,
                normalization,
                config,
                training,
                QSafeLossConfig(),
                provenance={"train_content_sha256": "a" * 64},
            )
            loaded = load_qsafe_artifact(output)
            observation = torch.randn(3, 5, 46)
            nominal = torch.randn(3, 12)
            candidate = torch.randn(3, 4, 12)
            with torch.no_grad():
                expected = trained.ensemble.predict(
                    observation, nominal, candidate).member_risk
                actual = loaded.ensemble.predict(
                    observation, nominal, candidate).member_risk
            torch.testing.assert_close(actual, expected)
            self.assertEqual(loaded.network_config, config)
            self.assertEqual(loaded.action_view, "requested")
            self.assertEqual(loaded.action_components, ("requested",))
            with self.assertRaises(FileExistsError):
                save_qsafe_artifact(
                    output,
                    trained,
                    normalization,
                    config,
                    training,
                    QSafeLossConfig(),
                    provenance={},
                )

    def test_hash_mismatch_is_detected_before_loading_weights(self):
        config, trained = self._trained()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "development-artifact"
            save_qsafe_artifact(
                output,
                trained,
                NormalizationStats(np.zeros(46), np.ones(46)),
                config,
                QSafeTrainingConfig(
                    epochs=1, ensemble_members=2, calibration_steps=0),
                QSafeLossConfig(),
                provenance={},
            )
            with (output / "member_00.pt").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_qsafe_artifact(output)

    def test_normalization_must_be_covered_by_component_hash_manifest(self):
        config, trained = self._trained()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "development-artifact"
            save_qsafe_artifact(
                output,
                trained,
                NormalizationStats(np.zeros(46), np.ones(46)),
                config,
                QSafeTrainingConfig(
                    epochs=1, ensemble_members=2, calibration_steps=0),
                QSafeLossConfig(),
                provenance={},
            )
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["component_sha256"]["normalization.npz"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                    ValueError, "normalization.*component hash"):
                load_qsafe_artifact(output)

    def test_save_requires_exact_member_objects_and_temperatures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, wrong_model = self._trained()
            wrong_model.ensemble.members[0] = SelectiveAdvantageQSafe(config)
            with self.assertRaisesRegex(ValueError, "model object differs"):
                save_qsafe_artifact(
                    root / "wrong-model",
                    wrong_model,
                    NormalizationStats(np.zeros(46), np.ones(46)),
                    config,
                    QSafeTrainingConfig(
                        epochs=1, ensemble_members=2, calibration_steps=0),
                    QSafeLossConfig(),
                    provenance={},
                )

            config, wrong_temperature = self._trained()
            wrong_temperature.members[0].temperature = 1.25
            with self.assertRaisesRegex(
                    ValueError, "temperatures differ"):
                save_qsafe_artifact(
                    root / "wrong-temperature",
                    wrong_temperature,
                    NormalizationStats(np.zeros(46), np.ones(46)),
                    config,
                    QSafeTrainingConfig(
                        epochs=1, ensemble_members=2, calibration_steps=0),
                    QSafeLossConfig(),
                    provenance={},
                )

    def test_failed_pre_publish_check_leaves_no_artifact(self):
        config, trained = self._trained()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "development-artifact"
            guard = mock.Mock(side_effect=RuntimeError("git state changed"))
            with self.assertRaisesRegex(RuntimeError, "git state changed"):
                save_qsafe_artifact(
                    output,
                    trained,
                    NormalizationStats(np.zeros(46), np.ones(46)),
                    config,
                    QSafeTrainingConfig(
                        epochs=1, ensemble_members=2, calibration_steps=0),
                    QSafeLossConfig(),
                    provenance={},
                    pre_publish_check=guard,
                )
            guard.assert_called_once_with()
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".development-artifact.tmp-*")), [])

    def test_action_feature_order_is_part_of_the_load_contract(self):
        config, trained = self._trained()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "development-artifact"
            save_qsafe_artifact(
                output,
                trained,
                NormalizationStats(np.zeros(46), np.ones(46)),
                config,
                QSafeTrainingConfig(
                    epochs=1, ensemble_members=2, calibration_steps=0),
                QSafeLossConfig(),
                provenance={},
            )
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["action_feature_contract"]["components_in_order"] = [
                "executed"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "action feature contract"):
                load_qsafe_artifact(output)

    def test_manifest_symlink_alias_is_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            artifact.mkdir()
            audit = root / "source-7801.audit.npz"
            audit.write_bytes(b"must-not-read")
            (artifact / "manifest.json").symlink_to(audit)
            with mock.patch.object(
                    Path, "read_text",
                    side_effect=AssertionError("audit alias must not be read")):
                with self.assertRaisesRegex(
                        PermissionError, "refuse symlink inputs"):
                    load_qsafe_artifact(artifact)

    def test_recovery_program_artifact_round_trip_binds_full_contract(self):
        recovery_binding = full_recovery_binding()
        selector_bundle = frozen_selector_bundle()
        library_fingerprint = recovery_binding["fingerprint_sha256"]
        feature_contract = make_recovery_program_feature_manifest(
            library_fingerprint)
        config = RECOVERY_PROGRAM_V4_NETWORK_CONFIG
        training_config = RECOVERY_PROGRAM_V4_TRAINING_CONFIG
        loss_config = RECOVERY_PROGRAM_V4_LOSS_CONFIG
        normalization = NormalizationStats(
            np.zeros(46),
            np.ones(46),
            fit_content_sha256="9" * 64,
            fit_split="stage_b_fit",
        )
        members = []
        for index in range(training_config.ensemble_members):
            model = SelectiveAdvantageQSafe(config)
            members.append(TrainedQSafeMember(
                model=model,
                seed=(training_config.seed
                      + RECOVERY_PROGRAM_V4_MEMBER_SEED_STRIDE * index),
                bootstrap_trajectories=[f"recovery-{index}"],
                epoch_loss=[0.5] * training_config.epochs,
            ))
        trained = TrainedQSafeEnsemble(
            ensemble=QSafeEnsemble([member.model for member in members]),
            members=members,
            action_view="recovery_program_v1",
            action_dim=RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
            recovery_program_binding=copy.deepcopy(recovery_binding),
            recovery_program_feature_manifest=copy.deepcopy(feature_contract),
            recovery_program_feature_contract_sha256=feature_contract[
                "feature_contract_sha256"],
            recovery_library_fingerprint_sha256=library_fingerprint,
            normalization=normalization,
            command_vx=0.30,
            privileged_dim=0,
            train_split="stage_b_fit",
            network_config=config,
            training_config=training_config,
            loss_config=loss_config,
        )
        provenance = {
            "command_vx": 0.30,
            "recovery_program": recovery_binding,
            "recovery_program_feature_contract": feature_contract,
            "recovery_selector_bundle": selector_bundle.to_dict(),
            "recovery_selector_bundle_sha256": selector_bundle.bundle_sha256,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "recovery-program-artifact"
            trained.members[0].epoch_loss[0] = -0.1
            with self.assertRaisesRegex(ValueError, "epoch metadata"):
                save_qsafe_artifact(
                    Path(directory) / "negative-epoch-loss",
                    trained,
                    normalization,
                    config,
                    training_config,
                    loss_config,
                    provenance=provenance,
                    recovery_selector_bundle=selector_bundle,
                )
            trained.members[0].epoch_loss[0] = 0.5

            original_temperature = trained.members[0].temperature
            bad_temperature = math.exp(4.0) + 1.0
            trained.members[0].temperature = bad_temperature
            trained.ensemble.temperatures[0] = bad_temperature
            with self.assertRaisesRegex(
                    ValueError, r"exp\(-4\).*exp\(4\)"):
                save_qsafe_artifact(
                    Path(directory) / "out-of-range-temperature",
                    trained,
                    normalization,
                    config,
                    training_config,
                    loss_config,
                    provenance=provenance,
                    recovery_selector_bundle=selector_bundle,
                )
            trained.members[0].temperature = original_temperature
            trained.ensemble.temperatures[0] = original_temperature

            save_qsafe_artifact(
                output,
                trained,
                normalization,
                config,
                training_config,
                loss_config,
                provenance=provenance,
                recovery_selector_bundle=selector_bundle,
            )
            expected_manifest_sha256 = canonical_manifest_sha256(output)
            with self.assertRaisesRegex(
                    ValueError, "requires expected_manifest_sha256"):
                load_qsafe_artifact(output)
            with self.assertRaisesRegex(
                    ValueError, "manifest authorization mismatch"):
                load_qsafe_artifact(
                    output, expected_manifest_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "exact lowercase SHA-256"):
                load_qsafe_artifact(
                    output,
                    expected_manifest_sha256=expected_manifest_sha256.upper(),
                )
            loaded = load_qsafe_artifact(
                output,
                expected_manifest_sha256=expected_manifest_sha256,
            )
            self.assertEqual(loaded.action_view, "recovery_program_v1")
            self.assertEqual(
                loaded.authorized_manifest_sha256,
                expected_manifest_sha256,
            )
            self.assertEqual(
                loaded.claim_identity_sha256,
                expected_manifest_sha256,
            )
            self.assertEqual(
                loaded.network_config.action_dim,
                RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
            )
            self.assertEqual(
                loaded.manifest["action_feature_contract"][
                    "feature_contract_sha256"],
                feature_contract["feature_contract_sha256"],
            )
            self.assertEqual(
                loaded.manifest["action_feature_contract"][
                    "recovery_selector_bundle_sha256"],
                selector_bundle.bundle_sha256,
            )
            self.assertEqual(
                loaded.normalization.fit_content_sha256,
                normalization.fit_content_sha256,
            )
            self.assertEqual(
                loaded.normalization.fit_split, normalization.fit_split)

            substituted = copy.deepcopy(recovery_binding)
            substituted["fingerprint_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                    ValueError, "differs from the training view"):
                save_qsafe_artifact(
                    Path(directory) / "substituted-provenance",
                    trained,
                    normalization,
                    config,
                    training_config,
                    loss_config,
                    provenance={
                        "recovery_program": substituted,
                        "recovery_program_feature_contract": feature_contract,
                        "recovery_selector_bundle": selector_bundle.to_dict(),
                        "recovery_selector_bundle_sha256": (
                            selector_bundle.bundle_sha256),
                    },
                    recovery_selector_bundle=selector_bundle,
                )

            with self.assertRaisesRegex(
                    TypeError, "requires RecoverySelectorBundle"):
                save_qsafe_artifact(
                    Path(directory) / "missing-selector-object",
                    trained,
                    normalization,
                    config,
                    training_config,
                    loss_config,
                    provenance=provenance,
                )

            substituted_selector = copy.deepcopy(provenance)
            substituted_selector["recovery_selector_bundle_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                    ValueError, "selector provenance differs"):
                save_qsafe_artifact(
                    Path(directory) / "substituted-selector",
                    trained,
                    normalization,
                    config,
                    training_config,
                    loss_config,
                    provenance=substituted_selector,
                    recovery_selector_bundle=selector_bundle,
                )

            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            original_manifest = copy.deepcopy(manifest)
            manifest["training_config"]["epochs"] = 99
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact V4"):
                load_qsafe_artifact(
                    output,
                    expected_manifest_sha256=canonical_manifest_sha256(output),
                )

            manifest = copy.deepcopy(original_manifest)
            manifest["provenance"]["recovery_program_feature_contract"][
                "behavior_steps"][1] = 11
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature contract"):
                load_qsafe_artifact(
                    output,
                    expected_manifest_sha256=canonical_manifest_sha256(output),
                )

            for field, bad_value, message in (
                ("temperature", math.exp(4.0) + 1.0, "temperature"),
                ("epoch_loss", [-0.1] * training_config.epochs, "epoch"),
            ):
                with self.subTest(recovery_member_field=field):
                    manifest = copy.deepcopy(original_manifest)
                    manifest["members"][0][field] = bad_value
                    manifest_path.write_text(
                        json.dumps(manifest), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        load_qsafe_artifact(
                            output,
                            expected_manifest_sha256=(
                                canonical_manifest_sha256(output)),
                        )


if __name__ == "__main__":
    unittest.main()
