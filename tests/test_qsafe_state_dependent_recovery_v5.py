from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

import safety_data.closed_loop_recovery_collector as collector
import safety_data.closed_loop_recovery_triage as v3
import safety_data.state_dependent_recovery_v5 as v5
from safety_data.paths import (
    ProtectedEvidencePathError,
    require_v3_audit_consumed_or_safe_input,
)
from safety_data.state_dependent_recovery_v5 import (
    FROZEN_V4_SCIENCE_CONTRACT_SHA256,
    PROTOCOL_CONTRACT_SHA256,
    PROTOCOL_FILE_SHA256,
    SEED_ALGORITHM,
    SEED_DOMAIN,
    SEED_DOMAIN_PREFIX_LOW15,
    SEED_ROLE_TAGS,
    SOURCE_SEEDS,
    StateDependentRecoveryV5Error,
    evaluate_state_dependent_stage_a,
    load_state_dependent_recovery_v5_protocol,
    v5_seed,
    validate_state_dependent_recovery_v5_protocol,
)
import scripts.collect_closed_loop_recovery_triage as v3_collect_script
import scripts.collect_native_grouped_qsafe as native_collect_script
import scripts.collect_state_dependent_recovery_v5 as v5_collect_script
import scripts.merge_grouped_qsafe_shards as generic_merge_script


def _reference_seed(
    domain: bytes,
    tags: dict[str, int],
    source_seed: int,
    identity: int,
    role: str,
    namespace: int,
    index: int,
) -> int:
    digest = hashlib.sha256(domain)
    digest.update(int(tags[role]).to_bytes(4, "little"))
    for value in (source_seed, identity, namespace, index):
        digest.update(int(value).to_bytes(16, "little", signed=False))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def _reference_v5_seed(
    source_seed: int,
    identity: int,
    role_tag: int,
    namespace: int,
    index: int,
) -> int:
    packed = SEED_DOMAIN_PREFIX_LOW15
    for width, value in (
        (14, source_seed),
        (8, role_tag),
        (18, identity),
        (2, namespace),
        (6, index),
    ):
        packed = (packed << width) | value
    return (1 << 63) | packed


def _fall_array(counts: list[int]) -> np.ndarray:
    result = np.zeros((384, 9, 64), dtype=np.uint8)
    for candidate, count in enumerate(counts):
        result[:, candidate, :count] = 1
    return result


def _exact_bootstrap(
    metrics: np.ndarray,
    source_seed: np.ndarray,
    age_strata: dict[int, tuple[int, ...]],
    *,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    del age_strata, seed, chunk_size
    estimate = np.asarray([
        v3._equal_seed_mean(metrics[:, column], source_seed, SOURCE_SEEDS)
        for column in range(metrics.shape[1])
    ])
    return np.broadcast_to(estimate, (replicates, len(estimate))).copy()


class StateDependentRecoveryV5ProtocolTest(unittest.TestCase):
    def test_canonical_protocol_locks_complete_stage_chain(self):
        protocol = load_state_dependent_recovery_v5_protocol()
        result = validate_state_dependent_recovery_v5_protocol(protocol)
        self.assertEqual(
            result["protocol_contract_sha256"], PROTOCOL_CONTRACT_SHA256)
        self.assertEqual(
            hashlib.sha256(v5.PROTOCOL_PATH.read_bytes()).hexdigest(),
            PROTOCOL_FILE_SHA256,
        )
        self.assertEqual(
            collector.canonical_protocol_sha256(
                v5._frozen_science_contract(protocol)),
            FROZEN_V4_SCIENCE_CONTRACT_SHA256,
        )
        terminal_record = Path(
            "QSAFE_STATE_DEPENDENT_RECOVERY_V4_TECHNICAL_FAILURE.md")
        self.assertEqual(
            hashlib.sha256(terminal_record.read_bytes()).hexdigest(),
            "cb3b7648cedb78210694df52788aa3e8adc52e165d8d325e951db095c787546e",
        )
        self.assertEqual(result["actor_training_seed"], 42)
        self.assertEqual(
            result["claim_scope"],
            "seed42_fixed_actor_conditional_mechanism_only",
        )
        self.assertEqual(protocol["stage_B"]["feature_contract"][
            "model_action_width"], 82)
        self.assertEqual(protocol["stage_B"]["actor_training_seeds"], {
            "fit": [43, 44, 45, 46],
            "probability_calibration": [47, 48],
            "uncertainty_calibration": [49, 50],
            "selector_calibration": [51, 52],
            "model_test": [53, 54, 55, 56],
        })
        self.assertEqual(protocol["stage_C"]["total_groups"], 1200)
        self.assertEqual(
            protocol["stage_D"]["training_seeds"], list(range(201, 225)))
        self.assertEqual(
            protocol["stage_D"]["exact_sign_flip"]["assignments_exact"],
            2 ** 24)
        self.assertEqual(
            protocol["stage_C"]["bootstrap"]["seed"], 20_260_813)
        self.assertEqual(
            protocol["stage_B"]["calibration"]["selector_search"][
                "bootstrap_seed"],
            20_260_811,
        )
        self.assertEqual(
            protocol["authorization_compiler"]["naked_boolean_authorization_inputs"],
            "forbidden",
        )
        self.assertEqual(
            protocol["persistent_option_runtime"]["max_option_starts_per_episode"],
            1,
        )
        self.assertEqual(
            protocol["protection"]["forbidden_legacy_machine_protocol"],
            "config/qsafe_evidence_protocol.yaml",
        )
        self.assertFalse(result["stage_B_training_triggered"])
        self.assertFalse(result["phase2_authorized"])

    def test_v5_entrypoint_rejects_v3_and_any_gate_drift(self):
        old_protocol = yaml.safe_load(Path(
            "config/qsafe_closed_loop_recovery_triage_v3.yaml").read_text(
                encoding="utf-8"))
        with self.assertRaisesRegex(
                StateDependentRecoveryV5Error, "protocol_name"):
            validate_state_dependent_recovery_v5_protocol(old_protocol)
        protocol = load_state_dependent_recovery_v5_protocol()
        protocol["triage_gates"]["stage_A_primary"][
            "min_audit_absolute_reduction"] = 0.099
        with self.assertRaisesRegex(
                StateDependentRecoveryV5Error, "complete canonical"):
            validate_state_dependent_recovery_v5_protocol(protocol)

    def test_v5_loader_fails_closed_on_raw_protocol_hash_drift(self):
        with mock.patch.object(v5, "PROTOCOL_FILE_SHA256", "0" * 64), \
                self.assertRaisesRegex(
                    StateDependentRecoveryV5Error, "raw SHA-256"):
            load_state_dependent_recovery_v5_protocol()

    def test_head_has_no_v4_compatibility_entrypoint(self):
        for path in (
            "safety_data/state_dependent_recovery_v4.py",
            "scripts/collect_state_dependent_recovery_v4.py",
            "scripts/merge_state_dependent_recovery_v4.py",
            "scripts/audit_state_dependent_recovery_v4.py",
            "config/qsafe_state_dependent_recovery_v4.yaml",
        ):
            with self.subTest(path=path):
                self.assertFalse(Path(path).exists())

    def test_collector_git_binding_is_anchored_when_cwd_changes(self):
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                native_collect_script.subprocess, "run") as run:
            run.side_effect = [
                mock.Mock(stdout="generator-commit\n"),
                mock.Mock(stdout=b""),
            ]
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(
                    native_collect_script._git_commit(), "generator-commit")
            finally:
                os.chdir(previous)
        for call in run.call_args_list:
            self.assertEqual(
                call.args[0][:3], ["git", "-C", str(repository_root)])

    def test_v5_seed_domain_is_exact_and_v3_defaults_are_bit_identical(self):
        tags = dict(SEED_ROLE_TAGS)
        expected = _reference_v5_seed(8901, 3, tags["audit"], 2, 7)
        self.assertEqual(expected, 12_836_256_290_346_894_215)
        self.assertEqual(v5_seed(8901, 3, "audit", 2, 7), expected)
        self.assertNotEqual(SEED_DOMAIN_PREFIX_LOW15, 18_561)
        self.assertTrue(set(SOURCE_SEEDS).isdisjoint({
            8401, 8402, 8411, 8412, 8421, 8422,
        }))

        v3_domain = b"qsafe_closed_loop_v3_seed\0"
        v3_tags = {
            "source_reset": 10,
            "source_impulse": 11,
            "source_action": 12,
            "admission": 20,
            "discovery": 30,
            "audit": 40,
        }
        expected_v3 = _reference_seed(
            v3_domain, v3_tags, 7801, 3, "audit", 2, 7)
        self.assertEqual(
            collector._derived_seed(7801, 3, "audit", 2, 7), expected_v3)
        self.assertNotEqual(
            v5_seed(7801, 3, "audit", 2, 7), expected_v3)
        self.assertEqual(v5_seed(8901, 3, "audit", 2, 7) >> 63, 1)
        self.assertEqual(expected_v3 >> 63, 0)
        for role in v3_tags:
            for identity, namespace, index in (
                (0, 0, 0),
                (204_799, 3, 63),
            ):
                self.assertEqual(
                    collector._derived_seed(
                        7801, identity, role, namespace, index),
                    _reference_seed(
                        v3_domain, v3_tags, 7801, identity, role,
                        namespace, index),
                )

        encoded: dict[int, tuple[int, str, int, int, int]] = {}
        for source_seed in SOURCE_SEEDS:
            for role, tag in SEED_ROLE_TAGS:
                identities = (
                    (0, 2047, (1 << 18) - 1)
                    if role == "source_reset" else
                    (0, 204799, (1 << 18) - 1)
                    if role in ("source_impulse", "source_action") else
                    (0, 4095, (1 << 18) - 1)
                )
                for identity in identities:
                    for namespace in range(4):
                        for index in (0, 31, 63):
                            seed = v5_seed(
                                source_seed, identity, role, namespace, index)
                            self.assertEqual(
                                seed,
                                _reference_v5_seed(
                                    source_seed, identity, tag, namespace, index),
                            )
                            key = (
                                source_seed, role, identity, namespace, index)
                            self.assertNotIn(seed, encoded)
                            encoded[seed] = key
        for args, field in (
            ((1 << 14, 0, "audit", 0, 0), "source_seed"),
            ((8901, 1 << 18, "audit", 0, 0), "identity"),
            ((8901, 0, "audit", 4, 0), "namespace"),
            ((8901, 0, "audit", 0, 64), "index"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, field):
                v5_seed(*args)

        domains: dict[str, set[int]] = {}
        for role in ("admission", "discovery", "audit"):
            bundle, randomness = collector.role_randomness(
                source_seed=8901,
                proposal_index=3,
                replicas=8,
                role=role,
                seed_domain=SEED_DOMAIN,
                role_tags=SEED_ROLE_TAGS,
                seed_algorithm=SEED_ALGORITHM,
            )
            domains[role] = set(np.concatenate((
                bundle.crn_id,
                bundle.rollout_seed,
                bundle.perturbation_seed,
                np.asarray([randomness.candidate_seed], dtype=np.uint64),
            )).tolist())
            for namespace, observed in (
                (0, bundle.crn_id),
                (1, bundle.rollout_seed),
                (2, bundle.perturbation_seed),
            ):
                np.testing.assert_array_equal(
                    observed,
                    np.asarray([
                        v5_seed(8901, 3, role, namespace, index)
                        for index in range(8)
                    ], dtype=np.uint64),
                )
            self.assertEqual(
                randomness.candidate_seed,
                v5_seed(8901, 3, role, 3, 0),
            )
        self.assertFalse(domains["admission"] & domains["discovery"])
        self.assertFalse(domains["admission"] & domains["audit"])
        self.assertFalse(domains["discovery"] & domains["audit"])

    def test_v5_collection_config_rejects_cap_overflow_at_preflight(self):
        cases = (
            (
                "source_seed",
                lambda: v5_collect_script._v5_collection_config(
                    source_seed=1 << 14, policy_training_step=25_438),
            ),
            (
                "max_proposals",
                lambda: v5_collect_script._v5_collection_config(
                    source_seed=8901, policy_training_step=25_438,
                    max_proposals=(1 << 18) + 1),
            ),
            (
                "trajectory/step",
                lambda: v5_collect_script._v5_collection_config(
                    source_seed=8901, policy_training_step=25_438,
                    max_trajectories=2622, max_episode_steps=100),
            ),
            (
                "replica count",
                lambda: v5_collect_script._v5_collection_config(
                    source_seed=8901, policy_training_step=25_438,
                    discovery_replicas=65),
            ),
            (
                "role_tag",
                lambda: collector.ClosedLoopRecoveryCollectionConfig(
                    source_seed=8901,
                    policy_training_step=25_438,
                    seed_domain=SEED_DOMAIN,
                    seed_role_tags=tuple(
                        (name, 256 if name == "audit" else tag)
                        for name, tag in SEED_ROLE_TAGS
                    ),
                    seed_algorithm=SEED_ALGORITHM,
                ),
            ),
        )
        for message, build in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message):
                build()

    def test_v5_preflight_requires_and_preserves_absent_artifact_root(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            protocol_path = repository / "config" / "protocol.yaml"
            artifact_root = repository / "saved" / "v5"
            protocol = {"collection": {"artifact_root": "saved/v5"}}
            with mock.patch.object(
                    v5_collect_script, "PROTOCOL_PATH", protocol_path), \
                    mock.patch.object(
                        v5_collect_script, "_load_protocol",
                        return_value=protocol):
                v5_collect_script._require_absent_v5_root_for_preflight(
                    ["--source-seed", "8901", "--preflight-only"])
                self.assertFalse(artifact_root.exists())
                artifact_root.mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError, "root to be absent"):
                    v5_collect_script._require_absent_v5_root_for_preflight(
                        ["--source-seed", "8901", "--preflight-only"])

            with mock.patch.object(
                    v5_collect_script, "_load_protocol",
                    side_effect=AssertionError("non-preflight loaded protocol")):
                v5_collect_script._require_absent_v5_root_for_preflight(
                    ["--source-seed", "8901"])

    def test_v5_preflight_main_rechecks_root_and_cli_forbids_abbreviation(self):
        protocol = {"collection": {
            "artifact_root": "saved/qsafe_development/synthetic-v5-root"}}
        with mock.patch.object(
                v5_collect_script, "_load_protocol", return_value=protocol), \
                mock.patch.object(
                    v5_collect_script.os.path, "lexists",
                    side_effect=[False, True]), \
                mock.patch.object(
                    v5_collect_script._v3_collector, "main", return_value=0), \
                mock.patch(
                    "sys.argv", ["collect_state_dependent_recovery_v5.py",
                                 "--source-seed", "8901",
                                 "--preflight-only"]):
            with self.assertRaisesRegex(RuntimeError, "root to be absent"):
                v5_collect_script.main()

        with mock.patch(
                "sys.argv", ["collect_closed_loop_recovery_triage.py",
                             "--source-seed", "8901", "--preflight"]), \
                self.assertRaises(SystemExit) as raised:
            v3_collect_script.main()
        self.assertEqual(raised.exception.code, 2)

    def test_v5_preflight_rechecks_root_when_delegated_main_raises(self):
        protocol = {"collection": {
            "artifact_root": "saved/qsafe_development/synthetic-v5-root"}}
        with mock.patch.object(
                v5_collect_script, "_load_protocol", return_value=protocol), \
                mock.patch.object(
                    v5_collect_script.os.path, "lexists",
                    side_effect=[False, True]), \
                mock.patch.object(
                    v5_collect_script._v3_collector, "main",
                    side_effect=ValueError("synthetic delegated failure")), \
                mock.patch(
                    "sys.argv", ["collect_state_dependent_recovery_v5.py",
                                 "--source-seed", "8901",
                                 "--preflight-only"]):
            with self.assertRaisesRegex(RuntimeError, "root to be absent"):
                v5_collect_script.main()

    def test_collector_binding_sets_v5_seed_split_and_version_then_restores(self):
        original_name = v3_collect_script._PROTOCOL_NAME
        original_class = v3_collect_script.ClosedLoopRecoveryCollectionConfig
        with v5_collect_script._v5_entrypoint_binding():
            self.assertEqual(v3_collect_script._PROTOCOL_NAME, v5.PROTOCOL_NAME)
            config = v3_collect_script.ClosedLoopRecoveryCollectionConfig(
                source_seed=8901,
                policy_training_step=25_438,
            )
            self.assertEqual(config.seed_domain, SEED_DOMAIN)
            self.assertEqual(config.seed_role_tags, SEED_ROLE_TAGS)
            self.assertEqual(config.seed_algorithm, SEED_ALGORITHM)
            self.assertEqual(
                config.dataset_split_prefix,
                "state_dependent_recovery_v5_stage_a",
            )
            self.assertTrue(
                config.explicit_filter_settings_in_action_contract)
            manifest = collector._common_collection_manifest(
                role="discovery",
                config=config,
                protocol_sha256="a" * 64,
                protocol_contract_sha256="b" * 64,
            )
            self.assertEqual(manifest["seed_derivation"], {
                "domain_hex": SEED_DOMAIN.hex(),
                "role_tags": dict(SEED_ROLE_TAGS),
                "algorithm": SEED_ALGORITHM,
                "stream_mapping": copy.deepcopy(
                    collector._INJECTIVE_V4_STREAM_MAPPING),
            })
        self.assertEqual(v3_collect_script._PROTOCOL_NAME, original_name)
        self.assertIs(
            v3_collect_script.ClosedLoopRecoveryCollectionConfig,
            original_class,
        )
        default = collector.ClosedLoopRecoveryCollectionConfig(
            source_seed=7801,
            policy_training_step=25_438,
        )
        default_manifest = collector._common_collection_manifest(
            role="discovery",
            config=default,
            protocol_sha256="a" * 64,
            protocol_contract_sha256="b" * 64,
        )
        self.assertNotIn("seed_derivation", default_manifest)
        self.assertEqual(
            default_manifest["version"],
            collector.COLLECTION_PROTOCOL_VERSION,
        )
        self.assertEqual(
            collector.canonical_protocol_sha256(default_manifest),
            "937111ded6b5fab1c1760bff0142decd9da4d2fdb5d937f2cd2f0796386945f6",
        )

    def test_v5_collector_rejects_policy_config_raw_hash_drift(self):
        with mock.patch.object(
                v5_collect_script._v3_collector,
                "_file_sha256",
                return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "policy config raw SHA-256"):
                v5_collect_script._load_protocol()


class StateDependentRecoveryV5StatisticsTest(unittest.TestCase):
    def setUp(self):
        self.protocol = load_state_dependent_recovery_v5_protocol()
        self.source_seed = np.repeat(
            np.asarray(SOURCE_SEEDS, dtype=np.int64), 64)
        self.discovery = _fall_array([32, 8, 8, 36, 40, 44, 48, 52, 56])
        self.audit = _fall_array([32, 8, 24, 36, 40, 44, 48, 52, 56])

    def _evaluate(
        self,
        discovery: np.ndarray | None = None,
        audit: np.ndarray | None = None,
    ) -> dict:
        with mock.patch.object(
                v5._v3, "_hierarchical_bootstrap",
                side_effect=_exact_bootstrap):
            return evaluate_state_dependent_stage_a(
                protocol=self.protocol,
                discovery_fall=self.discovery if discovery is None else discovery,
                audit_fall=self.audit if audit is None else audit,
                source_seed=self.source_seed,
            )

    def test_uniform_discovery_ties_are_primary_and_column_equivariant(self):
        first = self._evaluate()
        self.assertAlmostEqual(first["audit_absolute_reduction"], 0.25)
        self.assertTrue(first["pass"])
        self.assertEqual(
            first["primary_rule"],
            "per_state_all_exact_discovery_minima_uniform_expectation",
        )
        self.assertAlmostEqual(first["discovery_minimizer_tie_fraction"], 1.0)

        order = np.asarray([0, 2, 1, 3, 4, 5, 6, 7, 8])
        permuted = self._evaluate(
            discovery=self.discovery[:, order],
            audit=self.audit[:, order],
        )
        self.assertAlmostEqual(
            permuted["audit_absolute_reduction"],
            first["audit_absolute_reduction"],
        )
        self.assertAlmostEqual(
            permuted["one_sided_95_lcb"], first["one_sided_95_lcb"])

    def test_all_six_seed_direction_is_mandatory(self):
        audit = self.audit.copy()
        first_seed = self.source_seed == SOURCE_SEEDS[0]
        audit[first_seed, 1:3, :] = 0
        audit[first_seed, 1:3, :40] = 1
        result = self._evaluate(audit=audit)
        self.assertGreaterEqual(result["audit_absolute_reduction"], 0.10)
        self.assertFalse(result["checks"]["all_six_source_seeds_positive"])
        self.assertFalse(result["pass"])

    def test_binary_and_exact_shape_contracts_fail_closed(self):
        with self.assertRaisesRegex(
                StateDependentRecoveryV5Error, "exact shape"):
            evaluate_state_dependent_stage_a(
                protocol=self.protocol,
                discovery_fall=self.discovery[:1],
                audit_fall=self.audit[:1],
                source_seed=self.source_seed[:1],
            )
        malformed = self.discovery.astype(np.int16)
        malformed[0, 0, 0] = 2
        with self.assertRaisesRegex(
                StateDependentRecoveryV5Error, "binary"):
            evaluate_state_dependent_stage_a(
                protocol=self.protocol,
                discovery_fall=malformed,
                audit_fall=self.audit,
                source_seed=self.source_seed,
            )


class StateDependentRecoveryV5FirewallTest(unittest.TestCase):
    def test_clean_binding_helper_anchors_git_and_hashes_raw_protocol(self):
        commit = "d" * 40
        with mock.patch.object(
                v5.subprocess,
                "run",
                side_effect=(mock.Mock(stdout=commit + "\n"),
                             mock.Mock(stdout=b"")),
        ) as run:
            observed = v5._require_clean_head_protocol_binding()
        self.assertEqual(observed, (
            commit,
            hashlib.sha256(v5.PROTOCOL_PATH.read_bytes()).hexdigest(),
        ))
        for call in run.call_args_list:
            command = call.args[0]
            self.assertEqual(command[:3], ["git", "-C", str(v5._REPOSITORY_ROOT)])

    def test_selection_core_rejects_readiness_outside_clean_binding(self):
        protocol = load_state_dependent_recovery_v5_protocol()
        spec = v5._validate_protocol(protocol)
        clean_commit = "d" * 40
        protocol_file_sha256 = "a" * 64
        cases = (
            (
                {"generator_commit": "e" * 40,
                 "protocol_file_sha256": protocol_file_sha256},
                "current clean HEAD",
            ),
            (
                {"generator_commit": clean_commit,
                 "protocol_file_sha256": "b" * 64},
                "canonical protocol file",
            ),
        )
        artifact_path = lambda value, **kwargs: Path(value)
        for readiness, message in cases:
            with self.subTest(message=message), mock.patch.object(
                    v5, "_validate_protocol", return_value=spec), \
                    mock.patch.object(
                        v5, "_require_clean_head_protocol_binding",
                        return_value=(clean_commit, protocol_file_sha256)), \
                    mock.patch.object(
                        v5._v3, "_artifact_path", side_effect=artifact_path), \
                    mock.patch.object(
                        v5._v3, "_collection_readiness",
                        return_value=readiness), \
                    mock.patch.object(
                        v5, "_validate_discovery_seed_contract_before_lock",
                        side_effect=AssertionError("discovery outcome opened")):
                with self.assertRaisesRegex(
                        StateDependentRecoveryV5Error, message):
                    v5.create_state_dependent_selection_lock(
                        protocol=protocol,
                        admission_path="admission-ledger-deployable.npz",
                        discovery_path="discovery-g384.npz",
                        collection_report_paths=[],
                        selection_lock_path="selection-lock.json",
                    )

    def test_uninformative_discovery_publishes_failure_report_without_audit(self):
        protocol = load_state_dependent_recovery_v5_protocol()
        base_spec = v5._validate_protocol(protocol)
        clean_commit = "d" * 40
        protocol_file_sha256 = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_protocol = copy.deepcopy(protocol)
            test_protocol["collection"]["artifact_root"] = str(root)
            spec = copy.deepcopy(base_spec)
            spec["protocol"] = test_protocol
            spec["collection"] = test_protocol["collection"]
            lock_result = {
                "generator_commit": clean_commit,
                "protocol_file_sha256": protocol_file_sha256,
                "selection_semantics": copy.deepcopy(v5.SELECTION_SEMANTICS),
                "selection_lock_sha256": "b" * 64,
                "audit_identifier": "c" * 64,
                "audit_authorized": False,
                "data_gate": {
                    "structural_contract_pass": True,
                    "pass": False,
                    "discovery_informativeness": {
                        "pass": False,
                        "checks": {
                            "overall_nominal_risk_inclusive": False,
                            "each_policy_age_nominal_risk_inclusive": True,
                        },
                    },
                },
            }
            readiness = {
                "generator_commit": clean_commit,
                "protocol_file_sha256": protocol_file_sha256,
            }
            artifact_path = lambda value, **kwargs: Path(value)
            with mock.patch.object(
                    v5, "_validate_protocol", return_value=spec), \
                    mock.patch.object(
                        v5, "_require_clean_head_protocol_binding",
                        return_value=(clean_commit, protocol_file_sha256)), \
                    mock.patch.object(
                        v5._v3, "_artifact_path", side_effect=artifact_path), \
                    mock.patch.object(
                        v5._v3, "_collection_readiness",
                        return_value=readiness), \
                    mock.patch.object(
                        v5, "_validate_discovery_seed_contract_before_lock"), \
                    mock.patch.object(
                        v5._v3, "create_selection_lock",
                        return_value=lock_result), \
                    mock.patch.object(
                        v5._v3, "_load_audit_shards_after_consumption",
                        side_effect=AssertionError("audit outcome opened")):
                result = v5.create_state_dependent_selection_lock(
                    protocol=test_protocol,
                    admission_path=root / "admission-ledger-deployable.npz",
                    discovery_path=root / "discovery-g384.npz",
                    collection_report_paths=[],
                    selection_lock_path=root / "selection-lock.json",
                )

            report_path = root / test_protocol["collection"][
                "triage_report_filename"]
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(result["stage_A_failure_report"], str(report_path))
            self.assertEqual(result["decision"], "no_model_training")
            self.assertFalse(report["data_gate"]["pass"])
            self.assertFalse(report["stage_A_primary"]["tested"])
            self.assertIsNone(report["stage_A_primary"]["pass"])
            self.assertEqual(report["decision"], "no_model_training")
            self.assertFalse(report["audit_opened_for_analysis"])
            self.assertFalse(report["audit_consumed"])
            for field in (
                    "stage_B_authorized", "model_training_authorized",
                    "model_training_triggered", "paired_closed_loop_authorized",
                    "online_training_authorized", "objective1_pass",
                    "phase2_authorized"):
                self.assertFalse(report[field], field)

    def test_audit_core_rejects_caller_binding_before_lock_probe(self):
        protocol = load_state_dependent_recovery_v5_protocol()
        spec = v5._validate_protocol(protocol)
        clean_commit = "d" * 40
        protocol_file_sha256 = "a" * 64
        cases = (
            ("e" * 40, protocol_file_sha256, "current clean HEAD"),
            (clean_commit, "b" * 64, "canonical raw V5 protocol"),
        )
        for expected_commit, expected_protocol, message in cases:
            with self.subTest(message=message), mock.patch.object(
                    v5, "_validate_protocol", return_value=spec), \
                    mock.patch.object(
                        v5, "_require_clean_head_protocol_binding",
                        return_value=(clean_commit, protocol_file_sha256)), \
                    mock.patch.object(
                        v5._v3, "_artifact_path",
                        side_effect=AssertionError("selection lock probed")):
                with self.assertRaisesRegex(
                        StateDependentRecoveryV5Error, message):
                    v5.consume_and_evaluate_state_dependent_audit(
                        protocol=protocol,
                        selection_lock_path="selection-lock.json",
                        expected_selection_lock_sha256="c" * 64,
                        audit_paths=[],
                        audit_consumed_path="audit-consumed.json",
                        expected_generator_commit=expected_commit,
                        expected_protocol_file_sha256=expected_protocol,
                    )

    def test_generic_merger_rejects_v5_audit_name_lexically(self):
        path = Path("/tmp/does-not-exist/source-8901.audit.npz")
        with mock.patch.object(
                Path, "lstat", side_effect=AssertionError("filesystem probe")), \
                mock.patch("builtins.open", side_effect=AssertionError("open")):
            with self.assertRaisesRegex(ValueError, "locked audit"):
                generic_merge_script._reject_locked_audit_basenames([path])
        with self.assertRaisesRegex(ValueError, "locked audit"):
            generic_merge_script._reject_locked_v3_audit_basenames([path])

    def test_generic_merger_rejects_v5_audit_in_every_cli_path_preprobe(self):
        root = Path("/tmp/qsafe-v5-lexical-firewall-does-not-exist")
        normal_a = root / "normal-a.npz"
        normal_b = root / "normal-b.npz"
        normal_output = root / "normal-output.npz"
        normal_protocol = root / "normal-protocol.yaml"
        audit = root / "source-8901.audit.npz"
        privileged_audit = root / "source-8901.audit.privileged.npz"
        cases = {
            "shard": [
                "merge_grouped_qsafe_shards.py", str(audit), str(normal_b),
                "--output", str(normal_output),
                "--protocol", str(normal_protocol),
            ],
            "privileged_shard": [
                "merge_grouped_qsafe_shards.py", str(normal_a), str(normal_b),
                "--privileged-shards", str(privileged_audit),
                str(root / "normal-privileged.npz"),
                "--output", str(normal_output),
                "--privileged-output", str(root / "normal-priv-output.npz"),
                "--protocol", str(normal_protocol),
            ],
            "collection_report": [
                "merge_grouped_qsafe_shards.py", str(normal_a), str(normal_b),
                "--collection-reports", str(audit),
                "--output", str(normal_output),
                "--protocol", str(normal_protocol),
            ],
            "output": [
                "merge_grouped_qsafe_shards.py", str(normal_a), str(normal_b),
                "--output", str(audit),
                "--protocol", str(normal_protocol),
            ],
            "privileged_output": [
                "merge_grouped_qsafe_shards.py", str(normal_a), str(normal_b),
                "--privileged-shards", str(root / "normal-priv-a.npz"),
                str(root / "normal-priv-b.npz"),
                "--output", str(normal_output),
                "--privileged-output", str(privileged_audit),
                "--protocol", str(normal_protocol),
            ],
            "report": [
                "merge_grouped_qsafe_shards.py", str(normal_a), str(normal_b),
                "--output", str(normal_output),
                "--report", str(audit),
                "--protocol", str(normal_protocol),
            ],
            "protocol": [
                "merge_grouped_qsafe_shards.py", str(normal_a), str(normal_b),
                "--output", str(normal_output),
                "--protocol", str(audit),
            ],
        }
        for name, argv in cases.items():
            with self.subTest(name=name), mock.patch("sys.argv", argv), \
                    mock.patch.object(
                        generic_merge_script.os.path,
                        "lexists",
                        side_effect=AssertionError("filesystem probe"),
                    ), mock.patch.object(
                        generic_merge_script,
                        "require_v3_audit_consumed_or_safe_input",
                        side_effect=AssertionError("input guard reached"),
                    ), mock.patch.object(
                        generic_merge_script,
                        "assert_development_path",
                        side_effect=AssertionError("resolve reached"),
                    ):
                with self.assertRaisesRegex(ValueError, "locked audit"):
                    generic_merge_script.main()

    def test_generic_loader_guard_requires_bound_v5_marker_and_rejects_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "source-8901.audit.npz"
            payload = b"synthetic-v5-audit"
            audit.write_bytes(payload)
            original_lstat = Path.lstat

            def forbid_final_probe(path, *args, **kwargs):
                if path == audit:
                    raise AssertionError("audit final component probed premarker")
                return original_lstat(path, *args, **kwargs)

            with mock.patch.object(Path, "lstat", new=forbid_final_probe):
                with self.assertRaises(ProtectedEvidencePathError):
                    require_v3_audit_consumed_or_safe_input(audit)

            hardlink_alias = root / "audit-hardlink-alias.npz"
            os.link(audit, hardlink_alias)
            with self.assertRaisesRegex(
                    ProtectedEvidencePathError, "hard-linked"):
                require_v3_audit_consumed_or_safe_input(hardlink_alias)
            hardlink_alias.unlink()
            self.assertEqual(
                require_v3_audit_consumed_or_safe_input(root), root)

            protocol_contract = "1" * 64
            protocol_file = "2" * 64
            audit_identifier = "3" * 64
            lock = {
                "protocol_name": v5.PROTOCOL_NAME,
                "protocol_contract_sha256": protocol_contract,
                "protocol_file_sha256": protocol_file,
                "audit_identifier": audit_identifier,
                "audit_authorized": True,
                "collection_readiness_manifest": {
                    "role_commitments": {
                        "audit": [{
                            "path": str(audit),
                            "file_sha256": hashlib.sha256(payload).hexdigest(),
                        }],
                    },
                },
            }
            lock_path = root / "selection-lock.json"
            lock_path.write_text(
                json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8")
            marker = {
                "schema_version": v5.AUDIT_CONSUMED_SCHEMA_VERSION,
                "protocol_name": v5.PROTOCOL_NAME,
                "protocol_contract_sha256": protocol_contract,
                "protocol_file_sha256": protocol_file,
                "selection_lock_sha256": hashlib.sha256(
                    lock_path.read_bytes()).hexdigest(),
                "audit_identifier": audit_identifier,
                "created_at_utc": "2026-08-09T00:00:00+00:00",
                "status": "irreversibly_consumed_before_outcome_read",
            }
            (root / "audit-consumed.json").write_text(
                json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
            self.assertEqual(
                require_v3_audit_consumed_or_safe_input(audit), audit)

            alias = root / "audit-alias.npz"
            alias.symlink_to(audit.name)
            with self.assertRaisesRegex(
                    ProtectedEvidencePathError, "symlink"):
                require_v3_audit_consumed_or_safe_input(alias)
            ancestor_alias = root / "alias-dir"
            real = root / "real-dir"
            real.mkdir()
            ancestor_alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(
                    ProtectedEvidencePathError, "symlinked ancestor"):
                require_v3_audit_consumed_or_safe_input(
                    ancestor_alias / "source-8901.audit.npz")

    def test_audit_path_check_is_lexical_and_does_not_probe_outcomes(self):
        protocol = load_state_dependent_recovery_v5_protocol()
        spec = v5._validate_protocol(protocol)
        root = Path.cwd() / protocol["collection"]["artifact_root"]
        paths = [root / f"source-{seed}.audit.npz" for seed in SOURCE_SEEDS]
        lock = {
            "expected_audit_shards": [
                {
                    "ordinal": ordinal,
                    "source_seed": seed,
                    "path": str(path),
                    "file_sha256": hashlib.sha256(
                        f"file-{seed}".encode()).hexdigest(),
                    "content_sha256": hashlib.sha256(
                        f"content-{seed}".encode()).hexdigest(),
                }
                for ordinal, (seed, path) in enumerate(zip(
                    SOURCE_SEEDS, paths, strict=True))
            ],
        }
        with mock.patch.object(
                Path, "lstat", side_effect=AssertionError("filesystem probe")), \
                mock.patch("builtins.open", side_effect=AssertionError("open")):
            observed = v5._locked_audit_paths_before_consumption(
                paths, protocol=protocol, spec=spec, lock=lock)
        self.assertEqual(observed, paths)
        bad = list(paths)
        bad[0] = Path("/tmp/formal-hidden/source-8901.audit.npz")
        with self.assertRaisesRegex(
                StateDependentRecoveryV5Error, "denied"):
            v5._locked_audit_paths_before_consumption(
                bad, protocol=protocol, spec=spec, lock=lock)

    def test_marker_persists_when_first_post_marker_audit_load_fails(self):
        protocol = load_state_dependent_recovery_v5_protocol()
        base_spec = v5._validate_protocol(protocol)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_protocol = copy.deepcopy(protocol)
            test_protocol["collection"]["artifact_root"] = str(root)
            spec = copy.deepcopy(base_spec)
            spec["protocol"] = test_protocol
            spec["collection"] = test_protocol["collection"]
            audit_paths = [
                root / f"source-{seed}.audit.npz" for seed in SOURCE_SEEDS]
            selection_hash = "b" * 64
            protocol_file_hash = "a" * 64
            lock = {
                "protocol_file_sha256": protocol_file_hash,
                "generator_commit": "abc1234",
                "audit_identifier": "c" * 64,
                "expected_audit_shards": [
                    {
                        "ordinal": ordinal,
                        "source_seed": seed,
                        "path": str(path),
                        "file_sha256": hashlib.sha256(
                            f"file-{seed}".encode()).hexdigest(),
                        "content_sha256": hashlib.sha256(
                            f"content-{seed}".encode()).hexdigest(),
                    }
                    for ordinal, (seed, path) in enumerate(zip(
                        SOURCE_SEEDS, audit_paths, strict=True))
                ],
                "selection_semantics": copy.deepcopy(v5.SELECTION_SEMANTICS),
            }
            marker_path = root / "audit-consumed.json"

            def fail_after_marker(
                paths,
                observed_spec,
                *,
                manifest_validator,
            ):
                del paths, observed_spec
                self.assertTrue(marker_path.is_file())
                self.assertIs(
                    manifest_validator, v5.validate_v5_outcome_manifest)
                audit_manifest = {
                    "collection_protocol": {
                        "role": "audit",
                        "version": v5.COLLECTION_PROTOCOL_VERSION,
                        "seed_derivation": v5.expected_v5_seed_manifest(),
                    },
                    "split": f"{v5.DATASET_SPLIT_PREFIX}_audit",
                }
                manifest_validator(audit_manifest, "audit")
                audit_manifest["collection_protocol"]["seed_derivation"].pop(
                    "stream_mapping")
                with self.assertRaisesRegex(
                        StateDependentRecoveryV5Error,
                        "exact V5 RNG manifest"):
                    manifest_validator(audit_manifest, "audit")
                raise v3.ClosedLoopRecoveryTriageError("synthetic audit stop")

            artifact_path = lambda value, **kwargs: Path(value)
            with mock.patch.object(v5, "_validate_protocol", return_value=spec), \
                    mock.patch.object(
                        v5, "_require_clean_head_protocol_binding",
                        return_value=("abc1234", protocol_file_hash)), \
                    mock.patch.object(v5._v3, "_artifact_path",
                                      side_effect=artifact_path), \
                    mock.patch.object(v5._v3, "_read_selection_lock",
                                      return_value=lock), \
                    mock.patch.object(
                        v5._v3, "_load_audit_shards_after_consumption",
                        side_effect=fail_after_marker,
                    ) as loader:
                with self.assertRaisesRegex(
                        StateDependentRecoveryV5Error, "synthetic audit stop"):
                    v5.consume_and_evaluate_state_dependent_audit(
                        protocol=test_protocol,
                        selection_lock_path=root / "selection-lock.json",
                        expected_selection_lock_sha256=selection_hash,
                        audit_paths=audit_paths,
                        audit_consumed_path=marker_path,
                        expected_generator_commit="abc1234",
                        expected_protocol_file_sha256=protocol_file_hash,
                    )
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    marker["status"],
                    "irreversibly_consumed_before_outcome_read",
                )
                with self.assertRaisesRegex(
                        StateDependentRecoveryV5Error, "already been consumed"):
                    v5.consume_and_evaluate_state_dependent_audit(
                        protocol=test_protocol,
                        selection_lock_path=root / "selection-lock.json",
                        expected_selection_lock_sha256=selection_hash,
                        audit_paths=audit_paths,
                        audit_consumed_path=marker_path,
                        expected_generator_commit="abc1234",
                        expected_protocol_file_sha256=protocol_file_hash,
                    )
                self.assertEqual(loader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
