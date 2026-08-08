from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from rl.qsafe.artifact import load_qsafe_artifact
from scripts import train_selective_qsafe
from tests.test_safety_data import synthetic_dataset


class TrainSelectiveQSafeCliTest(unittest.TestCase):
    def test_grouped_three_split_training_writes_auditable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, offset in (("train", 0), ("calibration", 1), ("test", 2)):
                dataset, _ = synthetic_dataset(
                    split=f"development_{name}", offset=offset)
                paths.append(dataset.save(root / f"{name}.npz"))
            output = root / "artifact"
            argv = [
                "train_selective_qsafe",
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
            with mock.patch("sys.argv", argv):
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
            self.assertEqual(provenance["split_audit"]["pairs_checked"], 3)
            prediction = np.load(
                output / "test_predictions.npy", allow_pickle=False)
            self.assertEqual(prediction.shape, (4, 3))
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("test_predictions.npy", manifest["component_sha256"])


if __name__ == "__main__":
    unittest.main()
