from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from rl.qsafe.artifact import load_qsafe_artifact, save_qsafe_artifact
from rl.qsafe.data import NormalizationStats
from rl.qsafe.loss import QSafeLossConfig
from rl.qsafe.network import QSafeEnsemble, QSafeNetworkConfig, SelectiveAdvantageQSafe
from rl.qsafe.training import (
    QSafeTrainingConfig,
    TrainedQSafeEnsemble,
    TrainedQSafeMember,
)


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


if __name__ == "__main__":
    unittest.main()
