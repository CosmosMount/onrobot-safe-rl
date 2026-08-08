from __future__ import annotations

import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stdout

import numpy as np

from safety_data.merge import merge_grouped_shards, merge_privileged_shards
from safety_data.schema import PRIVILEGED_SCHEMA_VERSION, PrivilegedBranchView
from scripts.merge_grouped_qsafe_shards import _publish_no_clobber, main
from tests.test_safety_data import synthetic_dataset


def privileged(dataset, offset: int) -> PrivilegedBranchView:
    return PrivilegedBranchView(
        manifest={
            "schema_version": PRIVILEGED_SCHEMA_VERSION,
            "feature_view": "privileged_diagnostic_only",
            "split": dataset.manifest["split"],
            "generator_commit": dataset.manifest["generator_commit"],
            "deployable_content_sha256": dataset.validate(
                verify_hash=False)["content_sha256"],
            "feature_extraction_contract": "synthetic-height-tilt-v1",
        },
        group_id=dataset["group_id"].copy(),
        state_hash=dataset["state_hash"].copy(),
        features=(np.arange(dataset.group_count * 2).reshape(-1, 2)
                  + offset).astype(np.float32),
        feature_names=np.asarray(["height", "tilt"]),
    )


class GroupedShardMergeTest(unittest.TestCase):
    def shards(self, directory: Path):
        datasets = []
        views = []
        for shard, offset in enumerate((0, 1)):
            dataset, _ = synthetic_dataset(
                split="development_merged", offset=offset)
            dataset.manifest["collection_protocol"] = {
                "version": "synthetic-native-v1"}
            view = privileged(dataset, 100 * shard)
            dataset_path = dataset.save(directory / f"shard-{shard}.npz")
            view_path = view.save(directory / f"shard-{shard}.privileged.npz")
            loaded = type(dataset).load(dataset_path)
            datasets.append(loaded)
            views.append(type(view).load(view_path, deployable=loaded))
        return datasets, views

    def test_merge_preserves_shard_order_and_privileged_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets, views = self.shards(Path(directory))
            combined = merge_grouped_shards(datasets)
            combined_view = merge_privileged_shards(
                views, datasets, combined)
        report = combined.validate()
        self.assertEqual(report["groups"], 8)
        self.assertEqual(len(combined.manifest["shards"]), 2)
        self.assertNotIn("path", combined.manifest["shards"][0])
        self.assertEqual(
            combined.manifest["shards"][0]["content_sha256"],
            datasets[0].manifest["content_sha256"],
        )
        self.assertEqual(report["unique_source_seeds"], 8)
        np.testing.assert_array_equal(
            combined["group_id"][:4], datasets[0]["group_id"])
        np.testing.assert_array_equal(
            combined_view.group_id, combined["group_id"])
        combined_view.validate(combined)
        self.assertEqual(
            combined_view.manifest["shards"][1][
                "deployable_content_sha256"],
            datasets[1].manifest["content_sha256"],
        )

    def test_source_seed_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets, _ = self.shards(Path(directory))
            second, _ = synthetic_dataset(
                split="development_merged", offset=3)
            second.manifest["collection_protocol"] = {
                "version": "synthetic-native-v1"}
            second.arrays["source_seed"][:] = datasets[0]["source_seed"]
            second = type(second).load(
                second.save(Path(directory) / "overlap.npz"))
            with self.assertRaisesRegex(ValueError, "source_seed overlaps"):
                merge_grouped_shards([datasets[0], second])

    def test_new_manifest_fields_are_locked_without_an_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets, _ = self.shards(Path(directory))
            changed = datasets[1]
            changed.manifest["new_labeling_contract"] = {
                "failure_boundary": "different"}
            changed.save(Path(directory) / "changed-contract.npz")
            changed = type(changed).load(
                Path(directory) / "changed-contract.npz")
            with self.assertRaisesRegex(ValueError, "causal manifest contract"):
                merge_grouped_shards([datasets[0], changed])

    def test_merge_requires_verified_leaf_hashes_and_stable_numeric_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets, _ = self.shards(Path(directory))
            unhashed, _ = synthetic_dataset(
                split="development_merged", offset=4)
            unhashed.manifest["collection_protocol"] = {
                "version": "synthetic-native-v1"}
            with self.assertRaisesRegex(ValueError, "verified.*content_sha256"):
                merge_grouped_shards([datasets[0], unhashed])

            changed = datasets[1]
            changed.arrays["source_seed"] = changed["source_seed"].astype(
                np.int32)
            changed.save(Path(directory) / "changed-dtype.npz")
            changed = type(changed).load(Path(directory) / "changed-dtype.npz")
            with self.assertRaisesRegex(ValueError, "changes dtype"):
                merge_grouped_shards([datasets[0], changed])

    def test_privileged_link_and_combined_provenance_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets, views = self.shards(Path(directory))
            combined = merge_grouped_shards(datasets)

            bad_view = privileged(datasets[1], offset=200)
            bad_view.manifest["deployable_content_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                    ValueError, "deployable_content_sha256"):
                merge_privileged_shards(
                    [views[0], bad_view], datasets, combined)

            combined.manifest["shards"][0]["content_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "provenance"):
                merge_privileged_shards(views, datasets, combined)

    def test_privileged_feature_contract_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets, views = self.shards(Path(directory))
            changed = views[1]
            changed.manifest["feature_extraction_contract"] = "different-v2"
            changed.save(Path(directory) / "changed-privileged.npz")
            changed = type(changed).load(
                Path(directory) / "changed-privileged.npz",
                deployable=datasets[1],
            )
            combined = merge_grouped_shards(datasets)
            with self.assertRaisesRegex(ValueError, "causal feature contract"):
                merge_privileged_shards(
                    [views[0], changed], datasets, combined)

    def test_no_clobber_publication_rolls_back_its_earlier_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_stage = root / ".first-stage"
            second_stage = root / ".second-stage"
            first_stage.write_bytes(b"new-first")
            second_stage.write_bytes(b"new-second")
            first_output = root / "first.npz"
            occupied_output = root / "occupied.json"
            occupied_output.write_bytes(b"keep-me")
            with self.assertRaises(FileExistsError):
                _publish_no_clobber([
                    (first_stage, first_output),
                    (second_stage, occupied_output),
                ])
            self.assertFalse(first_output.exists())
            self.assertEqual(occupied_output.read_bytes(), b"keep-me")
            self.assertFalse(first_stage.exists())
            self.assertFalse(second_stage.exists())

    def test_invalid_protocol_leaves_no_partial_cli_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets, _ = self.shards(root)
            output = root / "merged.npz"
            report = root / "merged.report.json"
            protocol = root / "invalid-protocol.json"
            protocol.write_text(json.dumps({"phase1": {}}), encoding="utf-8")
            argv = [
                "merge_grouped_qsafe_shards.py",
                *(str(dataset.path) for dataset in datasets),
                "--output", str(output),
                "--report", str(report),
                "--protocol", str(protocol),
            ]
            with mock.patch("sys.argv", argv):
                with self.assertRaises(KeyError):
                    main()
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())

    def test_cli_publishes_complete_hashed_triplet_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets, views = self.shards(root)
            output = root / "merged.npz"
            privileged_output = root / "merged.privileged.npz"
            report_output = root / "merged.report.json"
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({
                "phase1": {"data_gate": {
                    "min_independent_groups": 1,
                    "min_independent_trajectory_clusters": 1,
                    "min_source_seeds": 1,
                    "min_candidates_per_group": 2,
                    "min_replicas_per_candidate": 1,
                    "min_mixed_outcome_fraction": 0.0,
                    "max_duplicate_group_fraction": 0.0,
                }},
            }), encoding="utf-8")
            argv = [
                "merge_grouped_qsafe_shards.py",
                *(str(dataset.path) for dataset in datasets),
                "--privileged-shards",
                *(str(view.path) for view in views),
                "--output", str(output),
                "--privileged-output", str(privileged_output),
                "--report", str(report_output),
                "--protocol", str(protocol),
            ]
            with mock.patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            merged = type(datasets[0]).load(output)
            merged_view = type(views[0]).load(
                privileged_output, deployable=merged)
            report = json.loads(report_output.read_text(encoding="utf-8"))
            self.assertEqual(
                report["publication_contract"],
                "atomic_no_clobber_report_last_v1",
            )
            self.assertEqual(report["validation"]["groups"], 8)
            self.assertEqual(
                merged_view.manifest["deployable_content_sha256"],
                merged.manifest["content_sha256"],
            )

            before = {
                path: path.read_bytes()
                for path in (output, privileged_output, report_output)
            }
            with mock.patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    main()
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
