from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import safety_data.state_dependent_recovery_v5_stage_b_workflow as workflow
import safety_data.paths as guarded_paths
import safety_data.stage_b_paths as guarded_stage_b_paths
import safety_data.state_dependent_recovery_v5_stage_b as stage_b_contract
from safety_data.closed_loop_recovery_collector import (
    ADMISSION_PRIVILEGED_SCHEMA_VERSION,
    ADMISSION_SCHEMA_VERSION,
    FALL_HEIGHT_REFERENCE,
    FALL_SAMPLING_CADENCE,
    FIRST_FAILURE_STEP_SEMANTICS,
    AdmissionLedger,
    AdmissionPrivilegedView,
)
from safety_data.schema import (
    CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS,
    CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS,
    CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION,
    PRIVILEGED_SCHEMA_VERSION,
    SCHEMA_VERSION,
    GroupedBranchDataset,
    PrivilegedBranchView,
)
from safety_data.stage_b_paths import stage_b_model_test_producer_read_scope
from safety_data.state_dependent_recovery_v5_stage_b import (
    CHECKPOINT_STEPS,
    ROLE_ACTOR_SEEDS,
    ROLE_ORDER,
    assignment_for,
    branch_randomness,
    canonical_sha256,
)
from safety_data.state_dependent_recovery_v5_stage_b_collector import (
    StageBRoleCollectionResult,
)
from train.state_dependent_recovery_v5_stage_b_actor_bank import (
    ACTOR_BANK_SCHEMA_VERSION,
)


_GENERATOR = "e" * 40
_ACTOR_BANK_FILE_SHA256 = "b" * 64
_INIT = np.asarray([0.05, 0.70, -1.40] * 4, dtype=np.float32)
_OFFSET = np.asarray([0.20, 0.40, 0.40] * 4, dtype=np.float32)
_LOWER = np.asarray([-1.05, -1.57, -2.72] * 4, dtype=np.float32)
_UPPER = np.asarray([1.05, 3.49, -0.84] * 4, dtype=np.float32)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _actor_bank() -> dict[str, object]:
    identities = []
    for role in ROLE_ORDER:
        for actor_seed in ROLE_ACTOR_SEEDS[role]:
            for checkpoint_step in CHECKPOINT_STEPS:
                prefix = f"{role}:{actor_seed}:{checkpoint_step}"
                actor_hash = _digest(prefix + ":actor")
                identities.append({
                    "role": role,
                    "actor_training_seed": actor_seed,
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_path": f"/tmp/{prefix}/agent",
                    "actor_checkpoint_sha256": actor_hash,
                    "actor_sha256": actor_hash,
                    "actor_state_dict_sha256": _digest(prefix + ":state"),
                    "policy_fingerprint_sha256": _digest(prefix + ":policy"),
                    "checkpoint_fingerprint_sha256": _digest(
                        prefix + ":checkpoint"),
                    "policy_config_sha256": _digest("config"),
                    "generator_commit": _GENERATOR,
                    "run_contract_sha256": _digest(prefix + ":run"),
                    "snapshot_manifest_file_sha256": _digest(
                        prefix + ":snapshot"),
                })
    basis: dict[str, object] = {
        "schema_version": ACTOR_BANK_SCHEMA_VERSION,
        "protocol_binding": {
            "path": "protocol", "file_sha256": _digest("protocol"),
            "contract_sha256": _digest("protocol-contract")},
        "execution_supplement_binding": {
            "path": "execution", "file_sha256": _digest("execution"),
            "contract_sha256": _digest("execution-contract")},
        "stage_a_report_binding": {
            "path": "stage-a", "file_sha256": _digest("stage-a")},
        "training_config_binding": {
            "path": "config", "file_sha256": _digest("config")},
        "actor_bank_attempt_binding": {
            "path": "attempt", "file_sha256": _digest("attempt"),
            "contract_sha256": _digest("attempt-contract")},
        "generator_commit": _GENERATOR,
        "actor_training_seeds": {
            role: list(ROLE_ACTOR_SEEDS[role]) for role in ROLE_ORDER},
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "checkpoint_semantics": (
            "after_transition_and_scheduled_update_before_next_transition"),
        "identity_count": 42,
        "expected_identity_count": 42,
        "actor_inclusion_rule": (
            "all_preregistered_seed_step_identities_no_filtering"),
        "return_or_fall_filtering": "forbidden",
        "checkpoint_substitution": "forbidden",
        "nearby_checkpoint_substitution": "forbidden",
        "policy_only": True,
        "identities": identities,
    }
    return basis | {"actor_bank_contract_sha256": canonical_sha256(basis)}


def _fall_definition() -> dict[str, object]:
    return {
        "max_abs_roll_pitch_rad": 0.523599,
        "min_base_height_m": 0.18,
        "tilt_comparator": "greater_than_or_equal",
        "height_comparator": "strict_less_than",
        "height_reference": FALL_HEIGHT_REFERENCE,
        "sampling_cadence": FALL_SAMPLING_CADENCE,
        "within_policy_hold_crossings": "not_observed",
        "first_failure_step_semantics": FIRST_FAILURE_STEP_SEMANTICS,
    }


def _action_contract() -> dict[str, object]:
    return {
        "q_target_semantic": "absolute_joint_position_sent",
        "init_qpos": _INIT.tolist(),
        "action_offset": _OFFSET.tolist(),
        "joint_min": _LOWER.tolist(),
        "joint_max": _UPPER.tolist(),
    }


def _source_result(
    *,
    role: str,
    source_seed: int,
    actor_bank: dict[str, object],
) -> StageBRoleCollectionResult:
    assignment = assignment_for(role, source_seed)
    identity = next(
        item for item in actor_bank["identities"]
        if item["role"] == role
        and item["actor_training_seed"] == assignment.actor_training_seed
        and item["checkpoint_step"] == assignment.checkpoint_step
    )
    group_id = f"stage-b-{role}:source-{source_seed}:trajectory-0:step-0"
    state_hash = _digest(f"state:{role}:{source_seed}")
    trajectory_id = f"stage-b-{role}:source-{source_seed}:trajectory-0"
    admission_rng = branch_randomness(
        role=role,
        partition="admission",
        source_seed=source_seed,
        proposal_index=0,
        replicas=32,
    )
    admission_fall = np.zeros((1, 32), dtype=bool)
    admission_fall[:, :6] = True
    admission_manifest = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "feature_view": "deployable_admission",
        "generator_commit": _GENERATOR,
        "protocol_sha256": "1" * 64,
        "protocol_contract_sha256": "2" * 64,
        "fall_definition": _fall_definition(),
        "simulator_fingerprint": {"backend": "synthetic-stage-b"},
        "source_policy": actor_bank,
        "continuation_policy": actor_bank,
        "action_application_contract": _action_contract(),
        "source_seed": source_seed,
        "policy_training_step": assignment.checkpoint_step,
        "admission_replicas": 32,
        "horizon_steps": 96,
        "accept_min_falls_inclusive": 6,
        "accept_max_falls_inclusive": 26,
        "all_proposals_recorded": True,
        "candidate_outcomes_used_for_admission": False,
        "stage_b_role": role,
    }
    history = np.zeros((1, 5, 46), dtype=np.float32)
    history[..., -12:] = _INIT
    admission = AdmissionLedger(admission_manifest, {
        "proposal_id": np.asarray([group_id]),
        "proposal_index": np.asarray([0], dtype=np.int64),
        "state_hash": np.asarray([state_hash]),
        "trajectory_id": np.asarray([trajectory_id]),
        "episode_id": np.asarray([source_seed], dtype=np.int64),
        "episode_step": np.asarray([0], dtype=np.int32),
        "source_seed": np.asarray([source_seed], dtype=np.int64),
        "policy_training_step": np.asarray(
            [assignment.checkpoint_step], dtype=np.int64),
        "policy_source": np.asarray([
            identity["policy_fingerprint_sha256"]]),
        "obs_history": history.copy(),
        "admission_crn_id": admission_rng["crn_id"][None, :],
        "admission_rollout_seed": admission_rng["rollout_seed"][None, :],
        "admission_perturbation_seed": (
            admission_rng["perturbation_seed"][None, :]),
        "admission_candidate_seed": np.asarray(
            [admission_rng["candidate_seed"]], dtype=np.uint64),
        "fall": admission_fall,
        "first_failure_step": np.where(
            admission_fall, 1, 97).astype(np.int16),
        "accepted": np.asarray([True]),
        "accepted_group_index": np.asarray([0], dtype=np.int64),
        "decision_reason": np.asarray(["accepted_6_to_26_of_32"]),
    })
    admission_hash = admission.validate(verify_hash=False)["content_sha256"]
    admission.manifest["content_sha256"] = admission_hash
    admission_privileged = AdmissionPrivilegedView(
        manifest={
            "schema_version": ADMISSION_PRIVILEGED_SCHEMA_VERSION,
            "feature_view": "privileged_admission_diagnostic_only",
            "generator_commit": _GENERATOR,
            "protocol_sha256": "1" * 64,
            "protocol_contract_sha256": "2" * 64,
            "deployable_content_sha256": admission_hash,
        },
        proposal_id=np.asarray([group_id]),
        state_hash=np.asarray([state_hash]),
        initial_tilt_rad=np.asarray([0.2], dtype=np.float32),
        initial_height_m=np.asarray([0.3], dtype=np.float32),
        max_tilt_rad=np.full((1, 32), 0.2, dtype=np.float32),
        min_height_m=np.full((1, 32), 0.3, dtype=np.float32),
    )
    admission_privileged.manifest["content_sha256"] = (
        admission_privileged.validate(
            admission, verify_hash=False)["content_sha256"])

    label_replicas = workflow.LABEL_REPLICAS[role]
    label_rng = branch_randomness(
        role=role,
        partition="label",
        source_seed=source_seed,
        proposal_index=0,
        replicas=label_replicas,
    )
    kinds = np.asarray(CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS)
    candidates = len(kinds)
    requested = np.zeros((1, candidates, 12), dtype=np.float32)
    q_target = np.broadcast_to(
        _INIT, (1, candidates, 12)).astype(np.float32).copy()
    fall = np.zeros((1, candidates, label_replicas), dtype=bool)
    for candidate in range(candidates):
        fall[0, candidate, :candidate + 1] = True
    arrays = {
        "group_id": np.asarray([group_id]),
        "state_hash": np.asarray([state_hash]),
        "trajectory_fingerprint_sha256": np.asarray([
            _digest(f"trajectory-snapshot:{role}:{source_seed}")
        ]),
        "trajectory_id": np.asarray([trajectory_id]),
        "episode_id": np.asarray([source_seed], dtype=np.int64),
        "episode_step": np.asarray([0], dtype=np.int32),
        "policy_training_seed": np.asarray(
            [assignment.actor_training_seed], dtype=np.int64),
        "source_seed": np.asarray([source_seed], dtype=np.int64),
        "policy_source": np.asarray([
            identity["policy_fingerprint_sha256"]]),
        "command_vx": np.asarray([0.3], dtype=np.float32),
        "acceptance_probability": np.asarray([1.0], dtype=np.float64),
        "obs_history": history.copy(),
        "q_send_history": np.broadcast_to(
            _INIT, (1, 5, 12)).astype(np.float32).copy(),
        "nominal_action_requested": np.zeros((1, 12), dtype=np.float32),
        "candidate_requested": requested,
        "candidate_executed": requested.copy(),
        "candidate_q_target": q_target,
        "candidate_kind": kinds[None, :],
        "candidate_mask": np.ones((1, candidates), dtype=bool),
        "candidate_behavior_steps": np.asarray(
            CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS,
            dtype=np.int16,
        )[None, :],
        "fall": fall,
        "first_failure_step": np.where(fall, 1, 97).astype(np.int16),
        "max_tilt_rad": np.where(fall, 0.7, 0.2).astype(np.float32),
        "min_height_m": np.where(fall, 0.1, 0.3).astype(np.float32),
        "crn_id": label_rng["crn_id"][None, :],
        "rollout_seed": label_rng["rollout_seed"][None, :],
        "perturbation_seed": label_rng["perturbation_seed"][None, :],
        "candidate_seed": np.asarray(
            [label_rng["candidate_seed"]], dtype=np.uint64),
    }
    label_manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": f"state_dependent_recovery_v5_stage_b_{role}_label",
        "feature_view": "deployable",
        "horizon_steps": 96,
        "generator_commit": _GENERATOR,
        "simulator_fingerprint": {"backend": "synthetic-stage-b"},
        "source_policy": actor_bank,
        "continuation_policy": actor_bank,
        "candidate_protocol": {
            "protocol_version": CLOSED_LOOP_RECOVERY_CANDIDATE_PROTOCOL_VERSION,
            "count": 9,
            "ordered_names": list(CLOSED_LOOP_RECOVERY_CANDIDATE_KINDS),
            "behavior_steps_array": "candidate_behavior_steps",
            "behavior_override_steps": list(
                CLOSED_LOOP_RECOVERY_BEHAVIOR_STEPS),
        },
        "fall_definition": _fall_definition(),
        "observation_contract": {
            "frames": 5,
            "dimension": 46,
            "tail_semantic": "previous_absolute_action_q_target",
        },
        "action_application_contract": _action_contract(),
        "state_hash_contract": "sha256_compound_snapshot_v1",
        "collection_protocol": {
            "role": role,
            "partition": "label",
            "trajectory_fingerprint_array": (
                stage_b_contract.TRAJECTORY_FINGERPRINT_ARRAY),
            "trajectory_fingerprint_contract": (
                stage_b_contract.TRAJECTORY_FINGERPRINT_CONTRACT),
        },
    }
    labels = GroupedBranchDataset(label_manifest, arrays)
    label_hash = labels.validate(verify_hash=False)["content_sha256"]
    labels.manifest["content_sha256"] = label_hash
    labels_privileged = PrivilegedBranchView(
        manifest={
            "schema_version": PRIVILEGED_SCHEMA_VERSION,
            "feature_view": "privileged_diagnostic_only",
            "split": label_manifest["split"],
            "generator_commit": _GENERATOR,
            "deployable_content_sha256": label_hash,
        },
        group_id=np.asarray([group_id]),
        state_hash=np.asarray([state_hash]),
        features=np.asarray([[0.2, 0.3]], dtype=np.float32),
        feature_names=np.asarray(["tilt", "height"]),
    )
    labels_privileged.manifest["content_sha256"] = (
        labels_privileged.validate(
            labels, verify_hash=False)["content_sha256"])
    return StageBRoleCollectionResult(
        role=role,
        admission=admission,
        admission_privileged=admission_privileged,
        labels=labels,
        labels_privileged=labels_privileged,
        source_steps=10 + source_seed,
        trajectories=1,
        proposals=1,
    )


class StageBRoleWorkflowTest(unittest.TestCase):
    def test_fit_attempt_requires_complete_later_role_assignment_bank(self):
        def reseal(value: dict[str, object]) -> dict[str, object]:
            basis = dict(value)
            basis.pop("actor_bank_contract_sha256", None)
            return basis | {
                "actor_bank_contract_sha256": canonical_sha256(basis),
            }

        cases: dict[str, dict[str, object]] = {}

        missing = json.loads(json.dumps(_actor_bank()))
        missing["identities"].pop()
        missing["identity_count"] = 41
        cases["missing model-test identity"] = reseal(missing)

        duplicated = json.loads(json.dumps(_actor_bank()))
        duplicated["identities"][-1]["checkpoint_step"] = CHECKPOINT_STEPS[-2]
        cases["mutated model-test identity"] = reseal(duplicated)

        for name, actor_bank in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "stage-b"
                with self.assertRaisesRegex(
                    workflow.StageBExecutionError,
                    "42 identities|model_test.*incomplete|exact frozen order",
                ):
                    workflow.prepare_stage_b_role(
                        stage_b_root=root,
                        role="fit",
                        generator_commit=_GENERATOR,
                        actor_bank_manifest=actor_bank,
                        actor_bank_manifest_file_sha256=(
                            _ACTOR_BANK_FILE_SHA256
                        ),
                        created_at_utc="2026-08-10T00:00:00+00:00",
                    )
                self.assertFalse((root / "fit/attempt-started.json").exists())

    def test_two_source_synthetic_role_is_report_last_and_outcome_free(self):
        actor_bank = _actor_bank()
        source_roster = dict(workflow.ROLE_SOURCE_SEEDS)
        source_roster["fit"] = (8501, 8502)
        group_counts = dict(workflow.GROUPS_PER_SOURCE)
        group_counts["fit"] = 1
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            workflow, "ROLE_SOURCE_SEEDS", source_roster
        ), mock.patch.object(workflow, "GROUPS_PER_SOURCE", group_counts):
            root = Path(directory) / "stage-b"
            attempt = workflow.prepare_stage_b_role(
                stage_b_root=root,
                role="fit",
                generator_commit=_GENERATOR,
                actor_bank_manifest=actor_bank,
                actor_bank_manifest_file_sha256=_ACTOR_BANK_FILE_SHA256,
                created_at_utc="2026-08-10T00:00:00+00:00",
            )
            attempt_sha = attempt["attempt_sha256"]
            for source_seed in (8501, 8502):
                assignment = assignment_for("fit", source_seed)
                identity = next(
                    item for item in actor_bank["identities"]
                    if item["role"] == "fit"
                    and item["actor_training_seed"]
                    == assignment.actor_training_seed
                    and item["checkpoint_step"]
                    == assignment.checkpoint_step
                )
                result = _source_result(
                    role="fit",
                    source_seed=source_seed,
                    actor_bank=actor_bank,
                )

                def collect(progress, value=result, seed=source_seed):
                    self.assertTrue((
                        root / "fit"
                        / f"source-{seed}.attempt-started.json").is_file())
                    progress({"groups": 1, "target_groups": 1})
                    return value

                source_report = workflow.collect_stage_b_source_once(
                    stage_b_root=root,
                    role="fit",
                    source_seed=source_seed,
                    generator_commit=_GENERATOR,
                    actor_identity=identity,
                    actor_bank_manifest=actor_bank,
                    actor_bank_manifest_file_sha256=(
                        _ACTOR_BANK_FILE_SHA256),
                    expected_role_attempt_sha256=attempt_sha,
                    simulator_fingerprint={"backend": "synthetic-stage-b"},
                    recovery_library_fingerprint_sha256="c" * 64,
                    collect=collect,
                    created_at_utc="2026-08-10T00:01:00+00:00",
                )
                self.assertEqual(
                    source_report["candidate_outcome_summary"], "forbidden")

            role_report = workflow.finalize_stage_b_role(
                stage_b_root=root,
                role="fit",
                generator_commit=_GENERATOR,
                actor_bank_manifest=actor_bank,
                actor_bank_manifest_file_sha256=_ACTOR_BANK_FILE_SHA256,
                expected_role_attempt_sha256=attempt_sha,
                created_at_utc="2026-08-10T00:02:00+00:00",
            )
            self.assertEqual(role_report["groups"], 2)
            self.assertEqual(role_report["status"], "complete_evidence_hashes_only")
            rendered = (root / "fit/report.json").read_text(encoding="utf-8")
            self.assertNotIn("fall_rate", rendered)
            self.assertNotIn("candidate_risk", rendered)
            with self.assertRaises(Exception):
                workflow.finalize_stage_b_role(
                    stage_b_root=root,
                    role="fit",
                    generator_commit=_GENERATOR,
                    actor_bank_manifest=actor_bank,
                    actor_bank_manifest_file_sha256=(
                        _ACTOR_BANK_FILE_SHA256),
                    expected_role_attempt_sha256=attempt_sha,
                )

    def test_model_test_stale_leaf_consumes_attempt_and_fails_closed(self):
        actor_bank = _actor_bank()
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "safety_data.stage_b_paths.require_clean_stage_b_generator",
            return_value=_GENERATOR,
        ):
            root = Path(directory) / "stage-b"
            stale = root / "model-test/source-8701.labels-r64.npz"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale\n")
            with self.assertRaisesRegex(Exception, "already exists"):
                workflow.prepare_stage_b_role(
                    stage_b_root=root,
                    role="model_test",
                    generator_commit=_GENERATOR,
                    actor_bank_manifest=actor_bank,
                    actor_bank_manifest_file_sha256=(
                        _ACTOR_BANK_FILE_SHA256),
                    created_at_utc="2026-08-10T00:00:00+00:00",
                )
            attempt = root / "model-test/attempt-started.json"
            self.assertTrue(attempt.is_file())
            with self.assertRaises(Exception):
                workflow.prepare_stage_b_role(
                    stage_b_root=root,
                    role="model_test",
                    generator_commit=_GENERATOR,
                    actor_bank_manifest=actor_bank,
                    actor_bank_manifest_file_sha256=(
                        _ACTOR_BANK_FILE_SHA256),
                )

    def test_model_test_report_publication_immediately_revokes_producer(self):
        actor_bank = _actor_bank()
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "safety_data.stage_b_paths.require_clean_stage_b_generator",
            return_value=_GENERATOR,
        ):
            root = Path(directory) / "stage-b"
            attempt = workflow.prepare_stage_b_role(
                stage_b_root=root,
                role="model_test",
                generator_commit=_GENERATOR,
                actor_bank_manifest=actor_bank,
                actor_bank_manifest_file_sha256=_ACTOR_BANK_FILE_SHA256,
                created_at_utc="2026-08-10T00:00:00+00:00",
            )
            leaf = root / "model-test/source-8701.labels-r64.npz"
            with stage_b_model_test_producer_read_scope(
                attempt_path=root / "model-test/attempt-started.json",
                expected_attempt_sha256=attempt["attempt_sha256"],
                evidence_paths=[leaf],
            ):
                (root / "model-test/report.json").write_bytes(
                    workflow._canonical_json({"published": True}))
            with self.assertRaisesRegex(Exception, "revoked by report"):
                with stage_b_model_test_producer_read_scope(
                    attempt_path=root / "model-test/attempt-started.json",
                    expected_attempt_sha256=attempt["attempt_sha256"],
                    evidence_paths=[leaf],
                ):
                    pass

    def test_minimized_five_role_model_test_finalize_binds_split_before_report(
        self,
    ) -> None:
        """Exercise the full producer-scope split proof without real outcomes."""
        actor_bank = _actor_bank()
        mini_actor_seeds = {
            role: (ROLE_ACTOR_SEEDS[role][0],) for role in ROLE_ORDER
        }
        mini_steps = (25_000, 50_000)
        mini_sources = {
            "fit": (8501, 8511),
            "probability_calibration": (8601, 8611),
            "uncertainty_calibration": (8631, 8641),
            "selector_calibration": (8661, 8671),
            "model_test": (8701, 8711),
        }
        mini_groups = {role: 1 for role in ROLE_ORDER}

        def minimized_source_assignments():
            return tuple(
                stage_b_contract.StageBSourceAssignment(
                    role=role,
                    actor_training_seed=mini_actor_seeds[role][0],
                    checkpoint_step=checkpoint_step,
                    source_seed=source_seed,
                    groups=1,
                    admission_replicas=stage_b_contract.ADMISSION_REPLICAS,
                    label_replicas=stage_b_contract.LABEL_REPLICAS[role],
                )
                for role in ROLE_ORDER
                for checkpoint_step, source_seed in zip(
                    mini_steps, mini_sources[role], strict=True)
            )

        def minimized_actor_identities(manifest):
            return {
                role: [
                    item for item in manifest["identities"]
                    if item["role"] == role
                    and item["actor_training_seed"] == mini_actor_seeds[role][0]
                    and item["checkpoint_step"] in mini_steps
                ]
                for role in ROLE_ORDER
            }

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory) / "stage-b"
            stack.enter_context(mock.patch.object(
                workflow, "ROLE_SOURCE_SEEDS", mini_sources))
            stack.enter_context(mock.patch.object(
                workflow, "GROUPS_PER_SOURCE", mini_groups))
            stack.enter_context(mock.patch.object(
                stage_b_contract, "ROLE_SOURCE_SEEDS", mini_sources))
            stack.enter_context(mock.patch.object(
                stage_b_contract, "ROLE_ACTOR_SEEDS", mini_actor_seeds))
            stack.enter_context(mock.patch.object(
                stage_b_contract, "CHECKPOINT_STEPS", mini_steps))
            stack.enter_context(mock.patch.object(
                stage_b_contract, "GROUPS_PER_SOURCE", mini_groups))
            stack.enter_context(mock.patch.object(
                stage_b_contract,
                "source_assignments",
                side_effect=minimized_source_assignments,
            ))
            stack.enter_context(mock.patch.object(
                guarded_paths, "_STAGE_B_ROLE_SOURCE_SEEDS", mini_sources))
            stack.enter_context(mock.patch.object(
                guarded_stage_b_paths,
                "_STAGE_B_ROLE_SOURCE_SEEDS",
                mini_sources,
            ))
            expected_model_artifacts = (
                guarded_paths._stage_b_expected_model_test_artifacts())
            stack.enter_context(mock.patch.object(
                guarded_paths,
                "_STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS",
                expected_model_artifacts,
            ))
            stack.enter_context(mock.patch.object(
                guarded_stage_b_paths,
                "_STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS",
                expected_model_artifacts,
            ))
            stack.enter_context(mock.patch.object(
                workflow,
                "_STAGE_B_EXPECTED_MODEL_TEST_ARTIFACTS",
                expected_model_artifacts,
            ))
            stack.enter_context(mock.patch.object(
                stage_b_contract,
                "_actor_identities_by_role",
                side_effect=minimized_actor_identities,
            ))
            stack.enter_context(mock.patch.object(
                workflow,
                "_validate_actor_bank",
                side_effect=lambda manifest, **_: json.loads(
                    json.dumps(manifest)),
            ))
            stack.enter_context(mock.patch.object(
                guarded_stage_b_paths,
                "require_clean_stage_b_generator",
                return_value=_GENERATOR,
            ))

            original_compile = workflow.compile_split_disjointness
            captured_content: dict[str, str] = {}

            def compile_and_capture(*, role_datasets, actor_bank_manifest):
                captured_content.update({
                    role: dataset.validate()["content_sha256"]
                    for role, dataset in role_datasets.items()
                })
                return original_compile(
                    role_datasets=role_datasets,
                    actor_bank_manifest=actor_bank_manifest,
                )

            stack.enter_context(mock.patch.object(
                workflow,
                "compile_split_disjointness",
                side_effect=compile_and_capture,
            ))
            original_publish = workflow._publish_staged
            publication_orders: list[list[Path]] = []
            captured_raw: dict[str, str] = {}

            def publish_and_capture(staged, **kwargs):
                destinations = [Path(destination) for _, destination in staged]
                publication_orders.append(destinations)
                directory_to_role = {
                    role.replace("_", "-"): role for role in ROLE_ORDER
                }
                for source, destination in staged:
                    destination = Path(destination)
                    if destination.name in (
                        "labels-r32-deployable.npz",
                        "labels-r64-deployable.npz",
                    ):
                        role = directory_to_role[destination.parent.name]
                        captured_raw[role] = hashlib.sha256(
                            Path(source).read_bytes()).hexdigest()
                return original_publish(staged, **kwargs)

            stack.enter_context(mock.patch.object(
                workflow, "_publish_staged", side_effect=publish_and_capture))

            attempts: dict[str, str] = {}
            role_reports: dict[str, dict[str, object]] = {}
            for role in ROLE_ORDER:
                attempt = workflow.prepare_stage_b_role(
                    stage_b_root=root,
                    role=role,
                    generator_commit=_GENERATOR,
                    actor_bank_manifest=actor_bank,
                    actor_bank_manifest_file_sha256=(
                        _ACTOR_BANK_FILE_SHA256),
                    created_at_utc="2026-08-10T00:00:00+00:00",
                )
                attempts[role] = str(attempt["attempt_sha256"])
                for source_seed in mini_sources[role]:
                    assignment = assignment_for(role, source_seed)
                    identity = next(
                        item for item in actor_bank["identities"]
                        if item["role"] == role
                        and item["actor_training_seed"]
                        == assignment.actor_training_seed
                        and item["checkpoint_step"]
                        == assignment.checkpoint_step
                    )
                    synthetic = _source_result(
                        role=role,
                        source_seed=source_seed,
                        actor_bank=actor_bank,
                    )
                    workflow.collect_stage_b_source_once(
                        stage_b_root=root,
                        role=role,
                        source_seed=source_seed,
                        generator_commit=_GENERATOR,
                        actor_identity=identity,
                        actor_bank_manifest=actor_bank,
                        actor_bank_manifest_file_sha256=(
                            _ACTOR_BANK_FILE_SHA256),
                        expected_role_attempt_sha256=attempts[role],
                        simulator_fingerprint={
                            "backend": "synthetic-stage-b"},
                        recovery_library_fingerprint_sha256="c" * 64,
                        collect=lambda progress, value=synthetic: (
                            progress({"groups": 1, "target_groups": 1})
                            or value
                        ),
                        created_at_utc="2026-08-10T00:01:00+00:00",
                    )
                role_reports[role] = workflow.finalize_stage_b_role(
                    stage_b_root=root,
                    role=role,
                    generator_commit=_GENERATOR,
                    actor_bank_manifest=actor_bank,
                    actor_bank_manifest_file_sha256=(
                        _ACTOR_BANK_FILE_SHA256),
                    expected_role_attempt_sha256=attempts[role],
                    created_at_utc="2026-08-10T00:02:00+00:00",
                )

            split_path = root / "stage-b-split-disjointness-report.json"
            model_report_path = root / "model-test/report.json"
            split = json.loads(split_path.read_text(encoding="utf-8"))
            top_fields = {
                "schema_version",
                *workflow._frozen_identity(_GENERATOR),
                "actor_bank_manifest_file_sha256",
                "actor_bank_contract_sha256",
                "role_order",
                "role_aggregate_labels",
                "identity_proof",
                "model_test_source",
                "outcome_columns_read",
                "pass",
                "report_sha256",
            }
            self.assertEqual(set(split), top_fields)
            split_basis = dict(split)
            split_self_hash = split_basis.pop("report_sha256")
            self.assertEqual(
                split_self_hash, canonical_sha256(split_basis))
            proof = split["identity_proof"]
            self.assertEqual(set(proof), {
                "schema_version", "dimensions", "identity_array_fields", "roles", "pairs_checked",
                "pairs", "outcome_columns_read", "pass", "report_sha256",
            })
            proof_basis = dict(proof)
            proof_self_hash = proof_basis.pop("report_sha256")
            self.assertEqual(proof_self_hash, canonical_sha256(proof_basis))
            self.assertEqual(proof["pairs_checked"], 10)
            self.assertEqual(len(proof["pairs"]), 10)
            for pair in proof["pairs"]:
                self.assertTrue(pair["pass"])
                self.assertTrue(all(
                    count == 0
                    for count in pair["collision_counts"].values()))

            bindings = split["role_aggregate_labels"]
            self.assertEqual(
                [item["role"] for item in bindings], list(ROLE_ORDER))
            self.assertEqual(set(captured_raw), set(ROLE_ORDER))
            self.assertEqual(set(captured_content), set(ROLE_ORDER))
            for item in bindings:
                self.assertEqual(set(item), {
                    "role", "path", "file_sha256", "content_sha256",
                    "groups", "role_report_file_sha256",
                })
                role = item["role"]
                self.assertEqual(item["file_sha256"], captured_raw[role])
                self.assertEqual(
                    item["content_sha256"], captured_content[role])
                self.assertEqual(item["groups"], 2)
                if role == "model_test":
                    self.assertIsNone(item["role_report_file_sha256"])
                    committed_label = next(
                        artifact for artifact in role_reports[role][
                            "evidence_artifacts"]
                        if artifact["kind"] == "label")
                    self.assertEqual(
                        committed_label["sha256"], item["file_sha256"])
                else:
                    self.assertEqual(
                        len(item["role_report_file_sha256"]), 64)

            model_publications = [
                order for order in publication_orders
                if order and order[-1] == model_report_path
            ]
            self.assertEqual(len(model_publications), 1)
            order = model_publications[0]
            self.assertEqual(order[-1], model_report_path)
            self.assertLess(order.index(split_path), order.index(model_report_path))
            manifest = json.loads((
                root / "model-test/collection-manifest.json"
            ).read_text(encoding="utf-8"))
            self.assertTrue(manifest["split_disjointness_report"][
                "published_before_model_test_role_report"])

            with self.assertRaisesRegex(Exception, "revoked by report"):
                with stage_b_model_test_producer_read_scope(
                    attempt_path=root / "model-test/attempt-started.json",
                    expected_attempt_sha256=attempts["model_test"],
                    evidence_paths=[
                        root / "model-test/labels-r64-deployable.npz"],
                ):
                    pass


class StageBRoleCliTest(unittest.TestCase):
    def test_prepare_role_cli_routes_only_frozen_context(self):
        import scripts.collect_state_dependent_recovery_v5_stage_b as cli

        context = {
            "stage_b_root": Path("/tmp/stage-b"),
            "generator_commit": _GENERATOR,
            "actor_bank": _actor_bank(),
            "actor_bank_file_sha256": _ACTOR_BANK_FILE_SHA256,
        }
        with mock.patch.object(
            cli, "_load_context", return_value=context
        ), mock.patch.object(
            cli, "_revalidate_live_context"
        ), mock.patch.object(
                cli,
                "prepare_stage_b_role",
                return_value={"attempt_sha256": "d" * 64},
        ) as prepare, mock.patch("builtins.print"):
            self.assertEqual(cli.main(["prepare-role", "--role", "fit"]), 0)
        prepare.assert_called_once_with(
            stage_b_root=context["stage_b_root"],
            role="fit",
            generator_commit=_GENERATOR,
            actor_bank_manifest=context["actor_bank"],
            actor_bank_manifest_file_sha256=_ACTOR_BANK_FILE_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
