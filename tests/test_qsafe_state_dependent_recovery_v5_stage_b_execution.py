from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from safety_data import state_dependent_recovery_v5_stage_b as stage_b
from safety_data.schema import GroupedBranchDataset


class StageBExecutionProtocolTest(unittest.TestCase):
    def test_canonical_protocol_and_roster_are_frozen(self) -> None:
        protocol = stage_b.load_stage_b_execution_protocol()
        self.assertEqual(
            protocol["execution_protocol_file_sha256"],
            stage_b.EXECUTION_PROTOCOL_FILE_SHA256,
        )
        self.assertEqual(
            protocol["execution_protocol_contract_sha256"],
            stage_b.EXECUTION_PROTOCOL_CONTRACT_SHA256,
        )
        rows = stage_b.source_assignments()
        self.assertEqual(len(rows), 21)
        self.assertEqual(len({row.source_seed for row in rows}), 21)
        self.assertEqual({row.actor_training_seed for row in rows}, set(range(43, 50)))
        self.assertEqual(
            sum(row.groups for row in rows if row.role == "fit"), 768
        )
        self.assertEqual(
            sum(row.groups for row in rows if row.role == "model_test"), 384
        )
        self.assertEqual(
            sum(row.groups * 9 * row.label_replicas for row in rows),
            608_256,
        )

    def test_actor_source_age_mapping_is_exact(self) -> None:
        expected = {
            ("fit", 8501): (43, 25_000, 128, 32),
            ("fit", 8512): (44, 50_000, 128, 32),
            ("probability_calibration", 8621): (45, 100_000, 64, 32),
            ("model_test", 8702): (49, 25_000, 64, 64),
            ("model_test", 8722): (49, 100_000, 64, 64),
        }
        for (role, source_seed), frozen in expected.items():
            row = stage_b.assignment_for(role, source_seed)
            self.assertEqual(
                (
                    row.actor_training_seed,
                    row.checkpoint_step,
                    row.groups,
                    row.label_replicas,
                ),
                frozen,
            )
        with self.assertRaises(stage_b.StageBExecutionError):
            stage_b.assignment_for("fit", 8701)

    def test_raw_mutation_and_duplicate_yaml_fail_closed(self) -> None:
        raw = stage_b.EXECUTION_PROTOCOL_PATH.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.yaml"
            changed.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(
                stage_b.StageBExecutionError, "file SHA-256"
            ):
                stage_b.load_stage_b_execution_protocol(changed)

            duplicate = Path(directory) / "duplicate.yaml"
            duplicate.write_bytes(raw + b"protocol_name: duplicate\n")
            with self.assertRaisesRegex(
                stage_b.StageBExecutionError, "duplicate key"
            ):
                stage_b.load_stage_b_execution_protocol(
                    duplicate, enforce_canonical_hash=False
                )

    def test_semantic_mutation_fails_even_without_hash_enforcement(self) -> None:
        raw = stage_b.EXECUTION_PROTOCOL_PATH.read_text(encoding="utf-8")
        changed_raw = raw.replace(
            "checkpoint_steps_exact: [25000, 50000, 100000]",
            "checkpoint_steps_exact: [25001, 50000, 100000]",
        )
        self.assertNotEqual(raw, changed_raw)
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.yaml"
            changed.write_text(changed_raw, encoding="utf-8")
            with self.assertRaisesRegex(
                stage_b.StageBExecutionError, "checkpoint_steps_exact"
            ):
                stage_b.load_stage_b_execution_protocol(
                    changed, enforce_canonical_hash=False
                )


class StageBSeedContractTest(unittest.TestCase):
    def test_golden_vectors(self) -> None:
        cases = (
            (
                dict(role="fit", partition="admission", source_seed=8501,
                     stream_role="source_reset", identity=0, namespace=0,
                     index=0),
                13_528_677_859_094_167_552,
            ),
            (
                dict(role="fit", partition="admission", source_seed=8501,
                     stream_role="admission", identity=17, namespace=2,
                     index=31),
                13_528_677_859_765_260_703,
            ),
            (
                dict(role="fit", partition="label", source_seed=8501,
                     stream_role="label", identity=17, namespace=3,
                     index=0),
                10_393_609_569_833_062_848,
            ),
            (
                dict(role="model_test", partition="label", source_seed=8724,
                     stream_role="label", identity=4095, namespace=2,
                     index=63),
                9_533_988_822_070_591_423,
            ),
        )
        for kwargs, expected in cases:
            self.assertEqual(stage_b.stage_b_seed(**kwargs), expected)

    def test_ten_domains_and_branch_streams_are_disjoint(self) -> None:
        report = stage_b.validate_role_seed_disjointness()
        self.assertTrue(report["pass"])
        self.assertTrue(report["domain_prefixes_pairwise_distinct"])
        self.assertEqual(len(report["domain_prefix_low15"]), 10)
        self.assertEqual(report["representative_stream_collisions"], {})

        admission = stage_b.branch_randomness(
            role="model_test",
            partition="admission",
            source_seed=8701,
            proposal_index=5,
            replicas=32,
        )
        labels = stage_b.branch_randomness(
            role="model_test",
            partition="label",
            source_seed=8701,
            proposal_index=5,
            replicas=64,
        )
        admission_values = set(np.concatenate([
            np.asarray(admission[name]).reshape(-1)
            for name in ("crn_id", "rollout_seed", "perturbation_seed",
                         "candidate_seed")
        ]).tolist())
        label_values = set(np.concatenate([
            np.asarray(labels[name]).reshape(-1)
            for name in ("crn_id", "rollout_seed", "perturbation_seed",
                         "candidate_seed")
        ]).tolist())
        self.assertFalse(admission_values & label_values)

    def test_seed_caps_and_partition_roles_fail_closed(self) -> None:
        base = dict(
            role="fit",
            partition="admission",
            source_seed=8501,
            stream_role="admission",
            identity=0,
            namespace=0,
            index=0,
        )
        for change in (
            {"source_seed": 1 << 14},
            {"identity": 1 << 18},
            {"namespace": 4},
            {"index": 64},
        ):
            with self.assertRaises(stage_b.StageBExecutionError):
                stage_b.stage_b_seed(**(base | change))
        with self.assertRaises(stage_b.StageBExecutionError):
            stage_b.stage_b_seed(**(base | {"stream_role": "label"}))
        with self.assertRaises(stage_b.StageBExecutionError):
            stage_b.stage_b_seed(**(
                base | {"partition": "label", "stream_role": "admission"}
            ))


class StageBPreflightTest(unittest.TestCase):
    def test_preflight_does_not_open_model_test(self) -> None:
        with mock.patch.object(
            stage_b, "validate_stage_a_authorization", return_value={}
        ) as authorization:
            result = stage_b.preflight_stage_b_execution()
        authorization.assert_called_once()
        self.assertTrue(result["pass"])
        self.assertFalse(result["model_test_opened"])
        self.assertEqual(result["source_assignments"], 21)
        self.assertEqual(result["candidate_label_rollouts"], 608_256)
        payload = dict(result)
        digest = payload.pop("preflight_sha256")
        self.assertEqual(digest, stage_b.canonical_sha256(payload))


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _synthetic_actor_bank() -> dict[str, object]:
    identities = []
    for role in stage_b.ROLE_ORDER:
        for actor_seed in stage_b.ROLE_ACTOR_SEEDS[role]:
            for step in stage_b.CHECKPOINT_STEPS:
                prefix = f"{role}:{actor_seed}:{step}"
                identities.append({
                    "role": role,
                    "actor_training_seed": actor_seed,
                    "checkpoint_step": step,
                    "actor_checkpoint_sha256": _digest(prefix + ":actor"),
                    "actor_state_dict_sha256": _digest(prefix + ":state"),
                    "policy_fingerprint_sha256": _digest(prefix + ":policy"),
                    "checkpoint_fingerprint_sha256": _digest(
                        prefix + ":checkpoint"
                    ),
                })
    return {"identities": identities}


def _identity_only_role_dataset(
    role: str,
    actor_bank: dict[str, object],
    *,
    seed_offset: int,
) -> GroupedBranchDataset:
    actor_lookup = {
        (
            int(item["actor_training_seed"]),
            int(item["checkpoint_step"]),
        ): str(item["policy_fingerprint_sha256"])
        for item in actor_bank["identities"]
        if item["role"] == role
    }
    rows = [row for row in stage_b.source_assignments() if row.role == role]
    actor_seed: list[int] = []
    source_seed: list[int] = []
    policy_source: list[str] = []
    for row in rows:
        actor_seed.extend([row.actor_training_seed] * row.groups)
        source_seed.extend([row.source_seed] * row.groups)
        policy_source.extend([
            actor_lookup[(row.actor_training_seed, row.checkpoint_step)]
        ] * row.groups)
    groups = len(actor_seed)
    base = seed_offset * 10_000_000
    arrays = {
        "group_id": np.asarray([f"{role}:group:{i}" for i in range(groups)]),
        "state_hash": np.asarray([_digest(f"{role}:state:{i}") for i in range(groups)]),
        "trajectory_fingerprint_sha256": np.asarray([
            _digest(f"{role}:trajectory-snapshot:{i}")
            for i in range(groups)
        ]),
        "trajectory_id": np.asarray([
            f"{role}:trajectory:{i}" for i in range(groups)
        ]),
        "policy_training_seed": np.asarray(actor_seed, dtype=np.int64),
        "source_seed": np.asarray(source_seed, dtype=np.int64),
        "policy_source": np.asarray(policy_source),
        "crn_id": np.arange(base, base + groups * 2, dtype=np.uint64).reshape(
            groups, 2
        ),
        "rollout_seed": np.arange(
            base + 2_000_000, base + 2_000_000 + groups * 2,
            dtype=np.uint64,
        ).reshape(groups, 2),
        "perturbation_seed": np.arange(
            base + 4_000_000, base + 4_000_000 + groups * 2,
            dtype=np.uint64,
        ).reshape(groups, 2),
        "candidate_seed": np.arange(
            base + 6_000_000, base + 6_000_000 + groups, dtype=np.uint64
        ),
    }
    return GroupedBranchDataset(
        manifest={"collection_protocol": {
            "role": role,
            "trajectory_fingerprint_array": stage_b.TRAJECTORY_FINGERPRINT_ARRAY,
            "trajectory_fingerprint_contract": stage_b.TRAJECTORY_FINGERPRINT_CONTRACT,
        }}, arrays=arrays
    )


class StageBSplitDisjointnessTest(unittest.TestCase):
    def test_all_ten_pairs_pass_without_outcome_columns(self) -> None:
        actor_bank = _synthetic_actor_bank()
        datasets = {
            role: _identity_only_role_dataset(role, actor_bank, seed_offset=index + 1)
            for index, role in enumerate(stage_b.ROLE_ORDER)
        }
        report = stage_b.compile_split_disjointness(
            role_datasets=datasets,
            actor_bank_manifest=actor_bank,
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["pairs_checked"], 10)
        self.assertFalse(report["outcome_columns_read"])
        self.assertTrue(all(pair["pass"] for pair in report["pairs"]))
        payload = dict(report)
        digest = payload.pop("report_sha256")
        self.assertEqual(digest, stage_b.canonical_sha256(payload))

    def test_cross_role_state_or_seed_collision_fails(self) -> None:
        actor_bank = _synthetic_actor_bank()
        datasets = {
            role: _identity_only_role_dataset(role, actor_bank, seed_offset=index + 1)
            for index, role in enumerate(stage_b.ROLE_ORDER)
        }
        datasets["model_test"].arrays["state_hash"][0] = datasets["fit"][
            "state_hash"
        ][0]
        with self.assertRaisesRegex(stage_b.StageBExecutionError, "overlap"):
            stage_b.compile_split_disjointness(
                role_datasets=datasets,
                actor_bank_manifest=actor_bank,
            )

    def test_all_ten_partition_rng_domains_are_persisted_and_disjoint(self) -> None:
        actor_bank = _synthetic_actor_bank()
        datasets = {
            role: _identity_only_role_dataset(role, actor_bank, seed_offset=index + 1)
            for index, role in enumerate(stage_b.ROLE_ORDER)
        }
        admissions = {
            role: SimpleNamespace(arrays={
                "admission_crn_id": np.asarray([1_000_000_000 + index], dtype=np.uint64),
                "admission_rollout_seed": np.asarray([1_100_000_000 + index], dtype=np.uint64),
                "admission_perturbation_seed": np.asarray([1_200_000_000 + index], dtype=np.uint64),
                "admission_candidate_seed": np.asarray([1_300_000_000 + index], dtype=np.uint64),
            })
            for index, role in enumerate(stage_b.ROLE_ORDER)
        }
        report = stage_b.compile_partition_rng_disjointness(
            role_admissions=admissions,
            role_labels=datasets,
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["pairs_checked"], 45)
        self.assertFalse(report["outcome_columns_read"])


if __name__ == "__main__":
    unittest.main()
