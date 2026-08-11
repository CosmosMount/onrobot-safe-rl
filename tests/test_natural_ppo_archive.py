from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safety_data.natural_ppo_archive import (
    checkpoint_age_bucket,
    deterministic_pairs,
    validate_and_match_archive,
)


class NaturalPpoArchiveTest(unittest.TestCase):
    def test_checkpoint_age_buckets_are_registered_boundaries(self):
        self.assertEqual(checkpoint_age_bucket(0), 0)
        self.assertEqual(checkpoint_age_bucket(999_999), 0)
        self.assertEqual(checkpoint_age_bucket(1_000_000), 1)
        self.assertEqual(checkpoint_age_bucket(29_999_999), 5)
        with self.assertRaises(ValueError):
            checkpoint_age_bucket(30_000_001)

    def test_matching_is_sorted_without_replacement(self):
        pairs = deterministic_pairs(
            [("fall-b", 2, "s"), ("fall-a", 1, "s")],
            [("normal-z", "s"), ("normal-a", "s")],
        )
        self.assertEqual(pairs, [
            ("fall-a", 1, "normal-a", "s"),
            ("fall-b", 2, "normal-z", "s"),
        ])
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            deterministic_pairs([("a", 1, "s"), ("b", 1, "s")],
                                [("n", "s")])

    def test_complete_archive_is_validated_and_matched(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "archive"
            normal_root = root / "normals"
            normal_root.mkdir(parents=True)
            fall_path = root / "falls-000000.npz"
            normal_path = normal_root / "normals-000000.npz"
            mask = np.zeros((1, 65), dtype=bool)
            mask[0, :2] = True
            availability = np.zeros((1, 7), dtype=bool)
            availability[0, :2] = True
            indices = np.full((1, 7), -1, dtype=np.int16)
            indices[0, :2] = [1, 0]
            commands = np.zeros((1, 65, 3), dtype=np.float32)
            commands[0, :2, 0] = 0.3
            policy_steps = np.zeros((1, 65), dtype=np.int64)
            policy_steps[0, :2] = [10, 11]
            steps_to_fall = np.zeros((1, 65), dtype=np.int16)
            steps_to_fall[0, :2] = [2, 1]
            randomization = {
                "randomized_geom_friction": np.asarray(
                    [[[[0.8, 0.0, 0.0]]]], dtype=np.float32),
                "randomized_body_ipos": np.zeros((1, 1, 3), dtype=np.float32),
                "randomized_encoder_bias": np.zeros((1, 12), dtype=np.float32),
            }
            np.savez_compressed(
                fall_path,
                identity=np.asarray([b"fall"], dtype="S64"),
                trajectory_length=np.asarray([2], dtype=np.int16),
                trajectory_mask=mask,
                trajectory_command=commands,
                prefall_availability=availability,
                prefall_trajectory_index=indices,
                trajectory_policy_step=policy_steps,
                trajectory_steps_to_fall=steps_to_fall,
                trajectory_fall_within_96_steps=mask,
                **randomization,
            )
            np.savez_compressed(
                normal_path,
                identity=np.asarray([b"normal-a", b"normal-b"], dtype="S64"),
                qualification_future_nonterminal_steps=np.asarray(
                    [96, 96], dtype=np.int16),
                fall_within_96_steps=np.asarray([False, False]),
                outcome_horizon_policy_steps=np.asarray([96, 96], dtype=np.int16),
                command=np.asarray([[0.3, 0.0, 0.0]] * 2, dtype=np.float32),
                policy_step=np.asarray([20, 21], dtype=np.int64),
                randomized_geom_friction=np.repeat(
                    randomization["randomized_geom_friction"], 2, axis=0),
                randomized_body_ipos=np.repeat(
                    randomization["randomized_body_ipos"], 2, axis=0),
                randomized_encoder_bias=np.repeat(
                    randomization["randomized_encoder_bias"], 2, axis=0),
            )

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            (normal_root / "manifest.json").write_text(json.dumps({
                "schema_version": "qsafe.mjlab_natural_normals.v2",
                "event_count": 2,
                "minimum_future_nonterminal_steps": 96,
                "shards": [{
                    "path": normal_path.name,
                    "sha256": digest(normal_path),
                    "event_count": 2,
                }],
            }), encoding="utf-8")
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": "qsafe.mjlab_natural_falls.v2",
                "event_count": 1,
                "prefall_offsets": [1, 2, 4, 8, 16, 32, 64],
                "external_force": "verified_zero",
                "direct_qsafe_supervision": {
                    "state_risk": True,
                    "executed_action_risk_under_ppo_continuation": True,
                    "counterfactual_recovery_action_risk": False,
                    "horizon_policy_steps": 96,
                },
                "shards": [{
                    "path": fall_path.name,
                    "sha256": digest(fall_path),
                    "event_count": 1,
                }],
                "provenance": {
                    "normal_manifest": "normals/manifest.json",
                    "independent_fall_episodes": 1,
                },
            }), encoding="utf-8")
            output = root / "matched.npz"
            report = validate_and_match_archive(root, output)
            self.assertEqual(report["available_prefall_states"], 2)
            self.assertEqual(report["matched_normal_states"], 2)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".report.json").is_file())


if __name__ == "__main__":
    unittest.main()
