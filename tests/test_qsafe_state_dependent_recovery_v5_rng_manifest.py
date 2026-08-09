from __future__ import annotations

from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

from safety_data import closed_loop_recovery_collector as collector
from safety_data import state_dependent_recovery_v5 as v5
from safety_data.schema import GroupedBranchDataset
import scripts.merge_state_dependent_recovery_v5 as merge_v5
from tests.test_safety_data import synthetic_dataset


_VERSION = "qsafe.state_dependent_recovery.collection.v5_stage_a"
_SPLIT = "state_dependent_recovery_v5_stage_a_discovery"


def _discovery_manifest() -> dict:
    return {
        "collection_protocol": {
            "version": _VERSION,
            "role": "discovery",
            "seed_derivation": v5.expected_v5_seed_manifest(),
        },
        "split": _SPLIT,
    }


class StateDependentRecoveryV5RngManifestTest(unittest.TestCase):
    def test_collector_manifest_survives_real_dataset_serialization_for_merger(self):
        config = collector.ClosedLoopRecoveryCollectionConfig(
            source_seed=8901,
            policy_training_step=25_438,
            seed_domain=v5.SEED_DOMAIN,
            seed_role_tags=v5.SEED_ROLE_TAGS,
            seed_algorithm=v5.SEED_ALGORITHM,
            dataset_split_prefix="state_dependent_recovery_v5_stage_a",
            collection_protocol_version=_VERSION,
        )
        dataset, _ = synthetic_dataset(split=_SPLIT)
        dataset.manifest["collection_protocol"] = (
            collector._common_collection_manifest(
                role="discovery",
                config=config,
                protocol_sha256="a" * 64,
                protocol_contract_sha256="b" * 64,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = dataset.save(Path(directory) / "synthetic-discovery.npz")
            loaded = GroupedBranchDataset.load(path)
        merge_v5._require_exact_v5_discovery_rng_split(loaded.manifest)
        self.assertEqual(
            loaded.manifest["collection_protocol"]["seed_derivation"],
            v5.expected_v5_seed_manifest(),
        )

    def test_writer_canonical_and_merge_use_one_exact_four_field_manifest(self):
        config = collector.ClosedLoopRecoveryCollectionConfig(
            source_seed=8901,
            policy_training_step=25_438,
            seed_domain=v5.SEED_DOMAIN,
            seed_role_tags=v5.SEED_ROLE_TAGS,
            seed_algorithm=v5.SEED_ALGORITHM,
            dataset_split_prefix="state_dependent_recovery_v5_stage_a",
            collection_protocol_version=_VERSION,
        )
        common_builder = collector.seed_derivation_manifest
        with mock.patch.object(
                collector, "seed_derivation_manifest",
                wraps=common_builder) as writer_builder:
            written = collector._common_collection_manifest(
                role="discovery",
                config=config,
                protocol_sha256="a" * 64,
                protocol_contract_sha256="b" * 64,
            )["seed_derivation"]
        writer_builder.assert_called_once_with(
            seed_domain=v5.SEED_DOMAIN,
            seed_role_tags=v5.SEED_ROLE_TAGS,
            seed_algorithm=v5.SEED_ALGORITHM,
        )
        common = collector.seed_derivation_manifest(
            seed_domain=v5.SEED_DOMAIN,
            seed_role_tags=v5.SEED_ROLE_TAGS,
            seed_algorithm=v5.SEED_ALGORITHM,
        )
        with mock.patch.object(
                v5, "seed_derivation_manifest",
                wraps=common_builder) as expected_builder:
            canonical = v5.expected_v5_seed_manifest()
        expected_builder.assert_called_once_with(
            seed_domain=v5.SEED_DOMAIN,
            seed_role_tags=v5.SEED_ROLE_TAGS,
            seed_algorithm=v5.SEED_ALGORITHM,
        )

        self.assertEqual(
            set(canonical),
            {"domain_hex", "role_tags", "algorithm", "stream_mapping"},
        )
        self.assertEqual(written, common)
        self.assertEqual(common, canonical)
        self.assertIs(
            v5.seed_derivation_manifest,
            collector.seed_derivation_manifest,
        )
        self.assertIs(
            merge_v5.validate_v5_outcome_manifest,
            v5.validate_v5_outcome_manifest,
        )
        with mock.patch.object(
                merge_v5, "validate_v5_outcome_manifest",
                wraps=v5.validate_v5_outcome_manifest) as merge_validator:
            manifest = _discovery_manifest()
            merge_v5._require_exact_v5_discovery_rng_split(manifest)
        merge_validator.assert_called_once_with(manifest, "discovery")

        canonical["role_tags"]["audit"] = -1
        self.assertNotEqual(canonical, v5.expected_v5_seed_manifest())

        noninjective = collector.seed_derivation_manifest(
            seed_domain=b"generic_seed_domain\0",
            seed_role_tags={"fit": 1},
            seed_algorithm="generic_seed_algorithm_v1",
        )
        self.assertEqual(set(noninjective), {
            "domain_hex", "role_tags", "algorithm"})
        self.assertNotIn("stream_mapping", noninjective)

    def test_missing_or_altered_rng_fields_and_split_are_rejected_exactly(self):
        def missing(field: str):
            manifest = _discovery_manifest()
            del manifest["collection_protocol"]["seed_derivation"][field]
            return manifest

        def altered(field: str, value):
            manifest = _discovery_manifest()
            manifest["collection_protocol"]["seed_derivation"][field] = value
            return manifest

        missing_split = _discovery_manifest()
        del missing_split["split"]
        altered_split = _discovery_manifest()
        altered_split["split"] = _SPLIT + "_drift"
        extra_rng_field = _discovery_manifest()
        extra_rng_field["collection_protocol"]["seed_derivation"][
            "unexpected"] = True
        missing_version = _discovery_manifest()
        del missing_version["collection_protocol"]["version"]
        altered_version = _discovery_manifest()
        altered_version["collection_protocol"]["version"] = _VERSION + "_drift"
        cases = {
            "missing_version": missing_version,
            "altered_version": altered_version,
            "missing_algorithm": missing("algorithm"),
            "altered_algorithm": altered("algorithm", "sha256_low63_v1"),
            "missing_stream_mapping": missing("stream_mapping"),
            "altered_stream_mapping": altered("stream_mapping", {}),
            "missing_domain": missing("domain_hex"),
            "altered_domain": altered("domain_hex", "00"),
            "missing_tags": missing("role_tags"),
            "altered_tags": altered("role_tags", {"audit": 140}),
            "extra_rng_field": extra_rng_field,
            "missing_split": missing_split,
            "altered_split": altered_split,
        }
        for name, manifest in cases.items():
            if "version" in name:
                expected = "V5 collection version"
            elif "split" in name:
                expected = "exact V5 split"
            else:
                expected = "exact V5 RNG manifest"
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, expected):
                merge_v5._require_exact_v5_discovery_rng_split(manifest)

        audit = _discovery_manifest()
        audit["collection_protocol"]["role"] = "audit"
        audit["split"] = "state_dependent_recovery_v5_stage_a_audit"
        v5.validate_v5_outcome_manifest(audit, "audit")
        audit["collection_protocol"]["seed_derivation"].pop("algorithm")
        with self.assertRaisesRegex(ValueError, "exact V5 RNG manifest"):
            v5.validate_v5_outcome_manifest(audit, "audit")

        wrong_role = _discovery_manifest()
        wrong_role["collection_protocol"]["role"] = "audit"
        with self.assertRaisesRegex(ValueError, "collection role"):
            merge_v5._require_exact_v5_discovery_rng_split(wrong_role)

    def test_merge_proceeds_past_valid_rng_predicate_and_never_publishes_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "discovery": root / "merged.npz",
                "discovery_privileged": root / "merged-privileged.npz",
                "discovery_report": root / "merge-report.json",
                "discovery_leaves": [
                    root / "source-8901.discovery.npz"],
                "discovery_privileged_leaves": [
                    root / "source-8901.discovery.privileged.npz"],
            }
            protocol = {"collection": {
                "discovery_filename": paths["discovery"].name,
                "discovery_privileged_filename": paths[
                    "discovery_privileged"].name,
                "discovery_merge_report_filename": paths[
                    "discovery_report"].name,
            }}
            readiness = {"role_commitments": {
                "discovery": [{"path": str(paths["discovery_leaves"][0])}],
                "discovery_privileged": [{
                    "path": str(paths["discovery_privileged_leaves"][0])}],
            }}

            def run(manifest: dict) -> ValueError:
                # group_count=0 is the next independent merge predicate.  A
                # valid RNG manifest must reach it; an invalid one must not.
                dataset = types.SimpleNamespace(
                    manifest=manifest,
                    group_count=0,
                    candidate_count=9,
                    replica_count=64,
                    horizon_steps=96,
                )
                publication = mock.Mock()
                with mock.patch.object(
                        merge_v5, "_canonical_non_audit",
                        side_effect=lambda path, **kwargs: path), \
                        mock.patch.object(
                            merge_v5.GroupedBranchDataset, "load",
                            return_value=dataset), \
                        mock.patch.object(
                            merge_v5.PrivilegedBranchView, "load",
                            return_value=types.SimpleNamespace()), \
                        mock.patch.object(
                            merge_v5, "_publish_no_clobber", publication):
                    with self.assertRaises(ValueError) as raised:
                        merge_v5._merge_discovery(
                            protocol=protocol,
                            paths=paths,
                            readiness=readiness,
                            commit="d" * 40,
                        )
                publication.assert_not_called()
                return raised.exception

            valid_error = run(_discovery_manifest())
            self.assertIn("exact G/K/R/H", str(valid_error))

            invalid = _discovery_manifest()
            invalid["collection_protocol"]["seed_derivation"][
                "algorithm"] += "_drift"
            invalid_error = run(invalid)
            self.assertIn("exact V5 RNG manifest", str(invalid_error))


if __name__ == "__main__":
    unittest.main()
