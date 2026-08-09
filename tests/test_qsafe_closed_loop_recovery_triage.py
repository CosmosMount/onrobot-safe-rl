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

import safety_data.closed_loop_recovery_triage as triage
from safety_data.closed_loop_recovery_collector import (
    ADMISSION_SCHEMA_VERSION,
    AdmissionLedger,
)
from safety_data.closed_loop_recovery_triage import (
    ClosedLoopRecoveryTriageError,
    V3_BEHAVIOR_STEPS,
    V3_CANDIDATE_NAMES,
    canonical_sha256,
    consume_and_evaluate_audit,
    create_selection_lock,
    validate_closed_loop_recovery_protocol,
    validate_collection_readiness,
)
from safety_data.schema import GroupedBranchDataset, SCHEMA_VERSION


PROTOCOL_PATH = Path("config/qsafe_closed_loop_recovery_triage_v3.yaml")
SOURCE_SEEDS = (7801, 7802, 7811, 7812, 7821, 7822)
POLICY_AGES = (25438, 25438, 50030, 50030, 100359, 100359)
GENERATOR_COMMIT = "abc1234"


def _fingerprint(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _protocol(root: Path) -> dict:
    value = copy.deepcopy(yaml.safe_load(
        PROTOCOL_PATH.read_text(encoding="utf-8")))
    collection = value["collection"]
    collection["artifact_root"] = str(root)
    collection["groups_per_source_seed"] = 2
    collection["total_groups"] = 12
    collection["admission"]["replicas"] = 4
    collection["admission"]["accept_min_falls_inclusive"] = 1
    collection["admission"]["accept_max_falls_inclusive"] = 3
    collection["admission"]["accepted_empirical_risk_interval"] = [0.25, 0.75]
    collection["replica_partition"]["discovery_replicas"] = 20
    collection["replica_partition"]["audit_replicas"] = 20
    collection["replica_partition"]["discovery_indices"] = {
        "start_inclusive": 0,
        "stop_exclusive": 20,
    }
    collection["replica_partition"]["audit_indices"] = {
        "start_inclusive": 20,
        "stop_exclusive": 40,
    }
    collection["total_candidate_replicas"] = 40
    collection["total_candidate_branch_rollouts"] = 12 * 9 * 40
    data = value["triage_gates"]["data"]
    data["independent_groups_exact"] = 12
    data["unique_source_trajectories_exact"] = 12
    data["groups_per_required_source_seed_exact"] = 2
    data["admission_replicas_exact"] = 4
    data["admission_falls_inclusive"] = [1, 3]
    data["discovery_replicas_exact"] = 20
    data["audit_replicas_exact"] = 20
    return value


def _fall_definition(protocol: dict) -> dict:
    failure = protocol["target"]["failure"]
    return {
        name: failure[name]
        for name in (
            "max_abs_roll_pitch_rad",
            "min_base_height_m",
            "tilt_comparator",
            "height_comparator",
            "height_reference",
            "sampling_cadence",
            "within_policy_hold_crossings",
            "first_failure_step_semantics",
        )
    }


def _identities() -> dict[str, np.ndarray]:
    source_seed = np.repeat(np.asarray(SOURCE_SEEDS, dtype=np.int64), 2)
    age_map = dict(zip(SOURCE_SEEDS, POLICY_AGES))
    policy_age = np.asarray(
        [age_map[int(seed)] for seed in source_seed], dtype=np.int64)
    return {
        "group_id": np.asarray([
            f"group-{index:02d}" for index in range(12)]),
        "state_fingerprint": np.asarray([
            _fingerprint(f"state-{index}") for index in range(12)]),
        "trajectory_fingerprint": np.asarray([
            _fingerprint(f"trajectory-{index}") for index in range(12)]),
        "source_seed": source_seed,
        "policy_age": policy_age,
    }


def _policy_fingerprints(protocol: dict) -> dict[int, str]:
    return {
        int(seed): str(policy["policy_fingerprint_sha256"])
        for policy in protocol["early_task_policies"]
        for seed in policy["source_seeds"]
    }


def _policy_bundle(protocol: dict) -> dict:
    config = protocol["policy_config"]
    return {
        "type": "locked_early_sac_policy_age_set_v3",
        "policy_training_seed": int(config["policy_training_seed"]),
        "config_sha256": str(config["config_sha256"]),
        "policies": [{
            name: copy.deepcopy(policy[name])
            for name in (
                "training_step", "source_seeds", "actor_sha256",
                "actor_state_dict_sha256", "policy_fingerprint_sha256",
                "checkpoint_fingerprint_sha256",
            )
        } for policy in protocol["early_task_policies"]],
    }


def _simulator_fingerprint(protocol: dict) -> dict:
    return {
        "backend": "mujoco",
        "mujoco_version": "test",
        "model_path": protocol["target"]["model_mjcf"],
        "mjcf_xml_sha256": protocol["target"][
            "model_mjcf_dependency_sha256"],
        "timestep_s": 0.002,
        "policy_frequency_hz": 50.0,
        "substeps": 10,
        "failure_measurement": {
            "height_reference": "base_link_body_origin_world_z",
            "cadence": "post_policy_step_after_all_low_level_substeps",
            "low_level_substeps_per_policy_step": 10,
        },
        "kp": [60.0] * 12,
        "kd": [5.0] * 12,
        "actuator_ctrl_low": [-100.0] * 12,
        "actuator_ctrl_high": [100.0] * 12,
        "action_filter": None,
        "max_joint_delta": None,
    }


def _recovery_program_binding(protocol: dict) -> dict:
    mature = protocol["mature_recovery_policy"]
    candidates = protocol["collection"]["candidates"]
    action = protocol["target"]["action_application_contract"]
    manifest = {
        "candidate_protocol": copy.deepcopy(candidates),
        "candidate_protocol_sha256": canonical_sha256(candidates),
        "mature_policy_identity": {
            "training_step": int(mature["training_step"]),
            "config_sha256": protocol["policy_config"]["config_sha256"],
            "actor_sha256": mature["actor_sha256"],
            "actor_state_dict_sha256": mature["actor_state_dict_sha256"],
            "policy_fingerprint_sha256": mature[
                "policy_fingerprint_sha256"],
            "checkpoint_fingerprint_sha256": mature[
                "checkpoint_fingerprint_sha256"],
            "observation_dim": 46,
            "actor_observation_dim": 46,
            "action_dim": 12,
        },
        "action_projection": {
            name: copy.deepcopy(action[name])
            for name in ("init_qpos", "action_offset", "joint_min", "joint_max")
        } | {"max_joint_delta": None, "use_action_filter": False},
        "input_boundary": "corrected_deployable_5x46_only",
        "privileged_inputs": "forbidden",
    }
    return {
        "manifest": manifest,
        "fingerprint_sha256": canonical_sha256(manifest),
    }


def _write_admission(
    root: Path,
    protocol: dict,
    *,
    v4_tagged_seeds: bool = False,
) -> Path:
    path = root / protocol["collection"]["admission_deployable_filename"]
    identities = _identities()
    fall = np.zeros((12, 4), dtype=bool)
    fall[:, :2] = True
    arrays = {
        "proposal_id": identities["group_id"].copy(),
        "proposal_index": np.arange(12, dtype=np.int64),
        "state_hash": identities["state_fingerprint"].copy(),
        "trajectory_id": identities["trajectory_fingerprint"].copy(),
        "episode_id": np.arange(12, dtype=np.int64),
        "episode_step": np.arange(12, dtype=np.int64),
        "source_seed": identities["source_seed"].copy(),
        "policy_training_step": identities["policy_age"].copy(),
        "policy_source": np.asarray([
            _policy_fingerprints(protocol)[int(seed)]
            for seed in identities["source_seed"]]),
        "obs_history": np.zeros((12, 5, 46), dtype=np.float32),
        "admission_crn_id": np.arange(
            100_000, 100_000 + 12 * 4, dtype=np.uint64).reshape(12, 4),
        "admission_rollout_seed": np.arange(
            200_000, 200_000 + 12 * 4, dtype=np.uint64).reshape(12, 4),
        "admission_perturbation_seed": np.arange(
            300_000, 300_000 + 12 * 4, dtype=np.uint64).reshape(12, 4),
        "fall": fall,
        "first_failure_step": np.where(fall, 1, 97).astype(np.int16),
        "accepted": np.ones(12, dtype=bool),
        "accepted_group_index": np.arange(12, dtype=np.int64),
        "decision_reason": np.asarray(["accepted_1_to_3_of_4"] * 12),
    }
    if v4_tagged_seeds:
        for name in (
                "admission_crn_id", "admission_rollout_seed",
                "admission_perturbation_seed"):
            arrays[name] |= np.uint64(1 << 63)
    ledger = AdmissionLedger(
        manifest={
            "schema_version": ADMISSION_SCHEMA_VERSION,
            "feature_view": "deployable_admission",
            "generator_commit": GENERATOR_COMMIT,
            "protocol_sha256": canonical_sha256(protocol),
            "protocol_contract_sha256": canonical_sha256(protocol),
            "fall_definition": _fall_definition(protocol),
            "simulator_fingerprint": _simulator_fingerprint(protocol),
            "source_policy": _policy_bundle(protocol),
            "continuation_policy": _policy_bundle(protocol),
            "action_application_contract": copy.deepcopy(
                protocol["target"]["action_application_contract"]),
            "source_seeds": list(SOURCE_SEEDS),
            "policy_training_steps": list(POLICY_AGES),
            "shards": [{
                "ordinal": ordinal,
                "source_seed": seed,
                "policy_training_step": age,
                "proposals": 2,
                "accepted": 2,
                "content_sha256": _fingerprint(f"admission-shard-{seed}"),
            } for ordinal, (seed, age) in enumerate(zip(
                SOURCE_SEEDS, POLICY_AGES, strict=True))],
            "admission_replicas": 4,
            "horizon_steps": 96,
            "accept_min_falls_inclusive": 1,
            "accept_max_falls_inclusive": 3,
            "all_proposals_recorded": True,
            "candidate_outcomes_used_for_admission": False,
        },
        arrays=arrays,
    )
    staging = ledger.save(root / "admission-ledger-staging.npz")
    staging.rename(path)
    return path


def _risk_to_fall(risk: np.ndarray, replicas: int = 20) -> np.ndarray:
    risk = np.asarray(risk, dtype=np.float64)
    counts = np.rint(risk * replicas).astype(np.int64)
    if not np.allclose(counts / replicas, risk, rtol=0.0, atol=1e-12):
        raise AssertionError("test risks must lie on the replica grid")
    fall = np.zeros((*risk.shape, replicas), dtype=np.int8)
    for group in range(risk.shape[0]):
        for candidate in range(risk.shape[1]):
            fall[group, candidate, :counts[group, candidate]] = 1
    return fall


def _informative_discovery_risk() -> np.ndarray:
    risk = np.empty((12, 9), dtype=np.float64)
    risk[:, 0] = 0.70
    risk[:, 1] = 0.30
    risk[:, 4] = 0.40
    risk[:, 5] = 0.45
    risk[:, 6] = 0.50
    risk[:, 7] = 0.55
    risk[:, 8] = 0.60
    for group in range(12):
        if group % 2 == 0:
            risk[group, 2] = 0.00
            risk[group, 3] = 0.70
        else:
            risk[group, 2] = 0.65
            risk[group, 3] = 0.00
    return risk


def _grouped_dataset(
    protocol: dict,
    role: str,
    risk: np.ndarray,
    *,
    group_slice: slice = slice(None),
    v4_tagged_seeds: bool = False,
) -> GroupedBranchDataset:
    identities = {
        name: values[group_slice]
        for name, values in _identities().items()
    }
    risk = np.asarray(risk, dtype=np.float64)
    groups = risk.shape[0]
    fall = _risk_to_fall(risk)
    action_contract = protocol["target"]["action_application_contract"]
    init_q = np.asarray(action_contract["init_qpos"], dtype=np.float32)
    q_send = np.broadcast_to(init_q, (groups, 5, 12)).copy()
    observation = np.zeros((groups, 5, 46), dtype=np.float32)
    observation[..., -12:] = q_send
    requested = np.zeros((groups, 9, 12), dtype=np.float32)
    q_target = np.broadcast_to(init_q, (groups, 9, 12)).copy()
    base = 4_000_000 if role == "discovery" else 10_000_000
    arrays = {
        "group_id": identities["group_id"].copy(),
        "state_hash": identities["state_fingerprint"].copy(),
        "trajectory_id": identities["trajectory_fingerprint"].copy(),
        "episode_id": np.arange(groups, dtype=np.int64) + int(
            identities["source_seed"][0]) * 100,
        "episode_step": np.arange(groups, dtype=np.int64),
        "policy_training_seed": np.full(groups, 42, dtype=np.int64),
        "source_seed": identities["source_seed"].copy(),
        "policy_source": np.asarray([
            _policy_fingerprints(protocol)[int(seed)]
            for seed in identities["source_seed"]]),
        "command_vx": np.full(groups, 0.30, dtype=np.float32),
        "acceptance_probability": np.ones(groups, dtype=np.float32),
        "obs_history": observation,
        "q_send_history": q_send,
        "nominal_action_requested": np.zeros((groups, 12), dtype=np.float32),
        "candidate_requested": requested,
        "candidate_executed": requested.copy(),
        "candidate_q_target": q_target,
        "candidate_kind": np.repeat(
            np.asarray(V3_CANDIDATE_NAMES)[None, :], groups, axis=0),
        "candidate_mask": np.ones((groups, 9), dtype=bool),
        "candidate_behavior_steps": np.repeat(
            np.asarray(V3_BEHAVIOR_STEPS, dtype=np.int64)[None, :],
            groups,
            axis=0,
        ),
        "fall": fall,
        "first_failure_step": np.where(fall, 1, 97).astype(np.int16),
        "max_tilt_rad": np.where(fall, 0.60, 0.10).astype(np.float32),
        "min_height_m": np.full(fall.shape, 0.30, dtype=np.float32),
        "crn_id": np.arange(
            base, base + groups * 20, dtype=np.uint64).reshape(groups, 20),
        "rollout_seed": np.arange(
            base + 1_000_000,
            base + 1_000_000 + groups * 20,
            dtype=np.uint64,
        ).reshape(groups, 20),
        "perturbation_seed": np.arange(
            base + 2_000_000,
            base + 2_000_000 + groups * 20,
            dtype=np.uint64,
        ).reshape(groups, 20),
        "candidate_seed": np.arange(
            base + 2_500_000,
            base + 2_500_000 + groups,
            dtype=np.uint64,
        ),
    }
    if v4_tagged_seeds:
        for name in (
                "crn_id", "rollout_seed", "perturbation_seed",
                "candidate_seed"):
            arrays[name] |= np.uint64(1 << 63)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split": f"closed_loop_recovery_v3_{role}",
        "feature_view": "deployable",
        "horizon_steps": 96,
        "generator_commit": GENERATOR_COMMIT,
        "simulator_fingerprint": _simulator_fingerprint(protocol),
        "source_policy": _policy_bundle(protocol),
        "continuation_policy": _policy_bundle(protocol),
        "candidate_protocol": copy.deepcopy(protocol["collection"]["candidates"]),
        "fall_definition": _fall_definition(protocol),
        "observation_contract": {
            "frames": 5,
            "dimension": 46,
            "tail_semantic": "previous_absolute_action_q_target",
        },
        "action_application_contract": copy.deepcopy(action_contract),
        "state_hash_contract": "sha256_compound_snapshot_v1",
        "collection_protocol": {
            "version": "qsafe.closed_loop_recovery_collection.v3",
            "role": role,
            "protocol_sha256": canonical_sha256(protocol),
            "protocol_contract_sha256": canonical_sha256(protocol),
            "physical_replica_role_files": True,
        },
        "recovery_program": _recovery_program_binding(protocol),
    }
    return GroupedBranchDataset(manifest, arrays)


def _write_discovery(
    root: Path,
    protocol: dict,
    risk: np.ndarray | None = None,
    *,
    v4_tagged_seeds: bool = False,
) -> Path:
    path = root / protocol["collection"]["discovery_filename"]
    if risk is None:
        risk = _informative_discovery_risk()
    dataset = _grouped_dataset(
        protocol, "discovery", risk,
        v4_tagged_seeds=v4_tagged_seeds)
    dataset.arrays["preassigned_audit_crn_id"] = np.arange(
        7_000_000, 7_000_000 + 12 * 20, dtype=np.uint64).reshape(12, 20)
    dataset.arrays["preassigned_audit_rollout_seed"] = np.arange(
        8_000_000, 8_000_000 + 12 * 20, dtype=np.uint64).reshape(12, 20)
    dataset.arrays["preassigned_audit_perturbation_seed"] = np.arange(
        9_000_000, 9_000_000 + 12 * 20, dtype=np.uint64).reshape(12, 20)
    dataset.arrays["preassigned_audit_candidate_seed"] = np.arange(
        9_500_000, 9_500_000 + 12, dtype=np.uint64)
    if v4_tagged_seeds:
        for name in (
                "preassigned_audit_crn_id",
                "preassigned_audit_rollout_seed",
                "preassigned_audit_perturbation_seed",
                "preassigned_audit_candidate_seed"):
            dataset.arrays[name] |= np.uint64(1 << 63)
    dataset.manifest["shards"] = [{
        "ordinal": ordinal,
        "content_sha256": _fingerprint(f"discovery-shard-{seed}"),
        "generator_commit": GENERATOR_COMMIT,
        "groups": 2,
        "source_seeds": [seed],
    } for ordinal, seed in enumerate(SOURCE_SEEDS)]
    staging = dataset.save(root / "discovery-g384-staging.npz")
    staging.rename(path)
    return path


def _write_audit(
    root: Path,
    protocol: dict,
    risk: np.ndarray,
    *,
    wrong_seed: bool = False,
    v4_tagged_seeds: bool = False,
) -> tuple[list[Path], list[dict[str, str]]]:
    paths: list[Path] = []
    commitments: list[dict[str, str]] = []
    for ordinal, seed in enumerate(SOURCE_SEEDS):
        selected = slice(2 * ordinal, 2 * ordinal + 2)
        dataset = _grouped_dataset(
            protocol, "audit", risk[selected], group_slice=selected,
            v4_tagged_seeds=v4_tagged_seeds)
        for name, base in (
            ("crn_id", 7_000_000),
            ("rollout_seed", 8_000_000),
            ("perturbation_seed", 9_000_000),
        ):
            start = base + 2 * ordinal * 20
            dataset.arrays[name] = np.arange(
                start, start + 2 * 20, dtype=np.uint64).reshape(2, 20)
        dataset.arrays["candidate_seed"] = np.arange(
            9_500_000 + 2 * ordinal,
            9_500_000 + 2 * ordinal + 2,
            dtype=np.uint64,
        )
        if v4_tagged_seeds:
            for name in (
                    "crn_id", "rollout_seed", "perturbation_seed",
                    "candidate_seed"):
                dataset.arrays[name] |= np.uint64(1 << 63)
        if wrong_seed and ordinal == 0:
            dataset.arrays["rollout_seed"][0, 0] += 999_999
        path = root / f"source-{seed}.audit.npz"
        staging = dataset.save(root / f"source-{seed}.audit-staging.npz")
        staging.rename(path)
        paths.append(path)
        commitments.append({
            "file_sha256": _file_sha256(path),
            "content_sha256": str(dataset.manifest["content_sha256"]),
        })
    return paths, commitments


def _audit_paths(root: Path) -> list[Path]:
    return [root / f"source-{seed}.audit.npz" for seed in SOURCE_SEEDS]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_collection_reports(
    root: Path,
    protocol: dict,
    audit_paths: list[Path],
    audit_commitments: list[dict[str, str]],
) -> list[Path]:
    result: list[Path] = []
    protocol_sha256 = canonical_sha256(protocol)
    cohort_path = root / protocol["collection"]["cohort_lock_filename"]
    cohort = {
        "schema_version": "qsafe.closed_loop_recovery.cohort_lock.v3",
        "protocol_name": protocol["protocol_name"],
        "protocol_file_sha256": protocol_sha256,
        "protocol_contract_sha256": protocol_sha256,
        "generator_commit": GENERATOR_COMMIT,
        "source_seed_policy_step": {
            str(seed): age
            for seed, age in zip(SOURCE_SEEDS, POLICY_AGES, strict=True)
        },
        "outcome_state": "no_analysis_before_all_shards_complete",
    }
    cohort_path.write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cohort_sha256 = _file_sha256(cohort_path)
    for ordinal, (seed, age, audit_path, audit_commitment) in enumerate(zip(
            SOURCE_SEEDS, POLICY_AGES, audit_paths, audit_commitments,
            strict=True)):
        role_paths = {
            "admission": root / f"source-{seed}.admission.npz",
            "admission_privileged": (
                root / f"source-{seed}.admission.privileged.npz"),
            "discovery": root / f"source-{seed}.discovery.npz",
            "discovery_privileged": (
                root / f"source-{seed}.discovery.privileged.npz"),
            "audit": audit_path,
            "audit_privileged": root / f"source-{seed}.audit.privileged.npz",
        }
        role_content_hashes = {
            "admission": _fingerprint(f"admission-shard-{seed}"),
            "admission_privileged": _fingerprint(
                f"admission-privileged-shard-{seed}"),
            "discovery": _fingerprint(f"discovery-shard-{seed}"),
            "discovery_privileged": _fingerprint(
                f"discovery-privileged-shard-{seed}"),
            "audit": audit_commitment["content_sha256"],
            "audit_privileged": _fingerprint(
                f"audit-privileged-shard-{seed}"),
        }
        outputs = {
            role: {
                "path": str(role_path),
                "file_sha256": (
                    audit_commitment["file_sha256"] if role == "audit"
                    else _fingerprint(f"{role}-file-{seed}")),
                "content_sha256": role_content_hashes[role],
            }
            for role, role_path in role_paths.items()
        }
        validations = {
            "admission": {
                "proposals": 2,
                "accepted": 2,
                "content_sha256": role_content_hashes["admission"],
            },
            "admission_privileged": {
                "proposals": 2,
                "content_sha256": role_content_hashes[
                    "admission_privileged"],
            },
            "discovery": {
                "groups": 2,
                "max_candidates": 9,
                "replicas": 20,
                "horizon_steps": 96,
                "content_sha256": role_content_hashes["discovery"],
            },
            "discovery_privileged": {
                "groups": 2,
                "content_sha256": role_content_hashes[
                    "discovery_privileged"],
            },
            "audit": {
                "groups": 2,
                "max_candidates": 9,
                "replicas": 20,
                "horizon_steps": 96,
                "content_sha256": role_content_hashes["audit"],
            },
            "audit_privileged": {
                "groups": 2,
                "content_sha256": role_content_hashes["audit_privileged"],
            },
        }
        attempt_path = root / f"source-{seed}.attempt-started.json"
        attempt = {
            "schema_version": "qsafe.closed_loop_recovery.attempt.v3",
            "protocol_name": protocol["protocol_name"],
            "protocol_file_sha256": protocol_sha256,
            "protocol_contract_sha256": protocol_sha256,
            "generator_commit": GENERATOR_COMMIT,
            "cohort_lock_sha256": cohort_sha256,
            "source_seed": seed,
            "policy_training_step": age,
            "started_at_unix_ns": 1_000_000 + ordinal,
            "state": "started_outcome_may_have_been_generated",
            "restart_authorized": False,
            "candidate_outcomes_summarized": False,
        }
        attempt_path.write_text(
            json.dumps(attempt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        attempt_sha256 = _file_sha256(attempt_path)
        report = {
            "schema_version": (
                "qsafe.closed_loop_recovery_collection_report.v3"),
            "protocol_name": protocol["protocol_name"],
            "protocol_path": str(PROTOCOL_PATH),
            "protocol_file_sha256": protocol_sha256,
            "protocol_contract_sha256": protocol_sha256,
            "cohort_lock": str(cohort_path),
            "cohort_lock_sha256": cohort_sha256,
            "cohort_contract": cohort,
            "attempt_marker": str(attempt_path),
            "attempt_marker_sha256": attempt_sha256,
            "attempt_contract": attempt,
            "development_only": True,
            "claim_eligible": False,
            "source_seed": seed,
            "policy_training_step": age,
            "generator_commit": GENERATOR_COMMIT,
            "generator_worktree_clean": True,
            "outputs": outputs,
            "validations": validations,
            "source_steps": 10,
            "trajectories": 2,
            "proposals": 2,
            "candidate_outcomes_summarized": False,
            "selection_lock_created": False,
            "audit_opened_for_analysis": False,
            "model_training_authorized": False,
            "phase2_authorized": False,
        }
        path = root / f"source-{seed}.collection-report.json"
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result.append(path)
    admission_path = root / protocol["collection"][
        "admission_deployable_filename"]
    discovery_path = root / protocol["collection"]["discovery_filename"]
    if admission_path.is_file() and discovery_path.is_file():
        _write_merge_completion_reports(
            root,
            protocol,
            result,
            admission_path=admission_path,
            discovery_path=discovery_path,
        )
    return result


def _write_merge_completion_reports(
    root: Path,
    protocol: dict,
    collection_reports: list[Path],
    *,
    admission_path: Path,
    discovery_path: Path,
) -> None:
    readiness = validate_collection_readiness(
        protocol=protocol,
        collection_report_paths=collection_reports,
    )
    admission = AdmissionLedger.load(admission_path)
    discovery = GroupedBranchDataset.load(discovery_path)
    admission_validation = admission.validate()
    discovery_validation = discovery.validate()
    admission_inputs = []
    for commitment, record in zip(
            readiness["role_commitments"]["admission"],
            readiness["manifest"]["source_records"],
            strict=True):
        validation = record["validations"]["admission"]
        admission_inputs.append({
            "path": commitment["path"],
            "file_sha256": commitment["file_sha256"],
            "content_sha256": commitment["content_sha256"],
            "proposals": validation["proposals"],
            "accepted": validation["accepted"],
        })
    admission_privileged_content = _fingerprint(
        "merged-admission-privileged-content")
    admission_report = {
        "schema_version": "qsafe.closed_loop_admission_merge_report.v3",
        "protocol_file_sha256": readiness["protocol_file_sha256"],
        "protocol_contract_sha256": canonical_sha256(protocol),
        "merge_commit": GENERATOR_COMMIT,
        "collection_readiness_sha256": readiness["readiness_sha256"],
        "source_seed_order": list(SOURCE_SEEDS),
        "inputs": admission_inputs,
        "output": str(admission_path),
        "output_file_sha256": _file_sha256(admission_path),
        "output_content_sha256": admission.manifest["content_sha256"],
        "privileged_output": str(
            root / protocol["collection"]["admission_privileged_filename"]),
        "privileged_file_sha256": _fingerprint(
            "merged-admission-privileged-file"),
        "privileged_content_sha256": admission_privileged_content,
        "validation": admission_validation,
        "privileged_validation": {
            "proposals": admission_validation["proposals"],
            "content_sha256": admission_privileged_content,
        },
        "candidate_outcomes_opened": False,
        "audit_opened": False,
        "model_training_authorized": False,
        "phase2_authorized": False,
    }
    admission_report_path = root / protocol["collection"][
        "admission_merge_report_filename"]
    admission_report_path.write_text(
        json.dumps(admission_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    discovery_inputs = [{
        "path": commitment["path"],
        "file_sha256": commitment["file_sha256"],
        "content_sha256": commitment["content_sha256"],
        "generator_commit": GENERATOR_COMMIT,
        "groups": 2,
        "source_seeds": [seed],
    } for seed, commitment in zip(
        SOURCE_SEEDS,
        readiness["role_commitments"]["discovery"],
        strict=True,
    )]
    discovery_privileged_inputs = [{
        "path": privileged["path"],
        "file_sha256": privileged["file_sha256"],
        "content_sha256": privileged["content_sha256"],
        "generator_commit": GENERATOR_COMMIT,
        "deployable_content_sha256": deployable["content_sha256"],
    } for privileged, deployable in zip(
        readiness["role_commitments"]["discovery_privileged"],
        readiness["role_commitments"]["discovery"],
        strict=True,
    )]
    exact_gate = {
        "pass": True,
        "checks": {
            "physical_role_discovery": True,
            "independent_groups_exact": True,
            "trajectory_clusters_exact": True,
            "source_seed_order_and_counts_exact": True,
            "candidates_exact": True,
            "candidate_kind_exact": True,
            "candidate_behavior_steps_exact": True,
            "candidate_protocol_exact": True,
            "discovery_replicas_exact": True,
            "horizon_exact": True,
            "discovery_seed_shape_exact": True,
            "audit_seed_preassignment_shape_exact": True,
            "audit_seed_preassignment_unique": True,
            "discovery_audit_seed_domains_disjoint": True,
            "audit_merge_forbidden": True,
        },
    }
    discovery_privileged_content = _fingerprint(
        "merged-discovery-privileged-content")
    discovery_report = {
        "schema_version": "qsafe.grouped_merge_report.v3",
        "development_only": True,
        "publication_contract": "atomic_no_clobber_report_last_v1",
        "merge_tool_commit": GENERATOR_COMMIT,
        "merge_tool_worktree_clean": True,
        "merge_tool_commit_stable": True,
        "output": str(discovery_path),
        "output_sha256": _file_sha256(discovery_path),
        "output_content_sha256": discovery.manifest["content_sha256"],
        "privileged_output": str(
            root / protocol["collection"]["discovery_privileged_filename"]),
        "privileged_sha256": _fingerprint(
            "merged-discovery-privileged-file"),
        "privileged_content_sha256": discovery_privileged_content,
        "input_shards": discovery_inputs,
        "input_privileged_shards": discovery_privileged_inputs,
        "validation": discovery_validation,
        "data_gate_role": "closed_loop_recovery_triage",
        "collection_data_gate": exact_gate,
        "phase1_data_gate": exact_gate,
        "phase2_authorized": False,
        "collection_readiness_sha256": readiness["readiness_sha256"],
    }
    discovery_report_path = root / protocol["collection"][
        "discovery_merge_report_filename"]
    discovery_report_path.write_text(
        json.dumps(discovery_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_readiness(
    root: Path,
    protocol: dict,
    audit_risk: np.ndarray | None = None,
    *,
    wrong_seed: bool = False,
    v4_tagged_seeds: bool = False,
) -> tuple[list[Path], list[Path]]:
    if audit_risk is None:
        audit_risk = _informative_discovery_risk()
    audit_paths, audit_commitments = _write_audit(
        root, protocol, audit_risk, wrong_seed=wrong_seed,
        v4_tagged_seeds=v4_tagged_seeds)
    return audit_paths, _write_collection_reports(
        root, protocol, audit_paths, audit_commitments)


def _formal_bootstrap_patch(test_replicates: int = 96):
    """Run few draws while asserting the formal API requested locked values."""
    original = triage._hierarchical_bootstrap

    def fast_bootstrap(
        group_metrics,
        source_seed,
        age_strata,
        *,
        replicates: int,
        seed: int,
        chunk_size: int,
    ):
        if (replicates, seed, chunk_size) != (50_000, 20_260_809, 512):
            raise AssertionError("formal bootstrap invocation drifted")
        return original(
            group_metrics,
            source_seed,
            age_strata,
            replicates=test_replicates,
            seed=seed,
            chunk_size=chunk_size,
        )

    return mock.patch.object(
        triage, "_hierarchical_bootstrap", side_effect=fast_bootstrap)


class ClosedLoopRecoveryProtocolTest(unittest.TestCase):
    def test_public_validator_locks_complete_canonical_protocol(self):
        protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
        validation = validate_closed_loop_recovery_protocol(protocol)
        self.assertEqual(
            validation["protocol_contract_sha256"],
            "07f530c582df38a1ff685fa0f8c0546f01eebb8cb9ec9573911e6f6076a59c3b",
        )
        self.assertEqual(validation["groups"], 384)
        self.assertEqual(validation["discovery_replicas"], 64)
        self.assertEqual(validation["audit_replicas"], 64)
        self.assertEqual(validation["horizon_policy_steps"], 96)
        self.assertEqual(validation["bootstrap_replicates"], 50_000)

        drifted = copy.deepcopy(protocol)
        drifted["target"]["command_speed_mps"] = 0.31
        with self.assertRaisesRegex(
                ClosedLoopRecoveryTriageError, "complete canonical"):
            validate_closed_loop_recovery_protocol(drifted)

    def test_hierarchical_bootstrap_pcg64_draw_order_golden(self):
        source_seed = np.repeat(np.asarray(SOURCE_SEEDS, dtype=np.int64), 2)
        metrics = np.arange(36, dtype=np.float64).reshape(12, 3) / 10.0
        age_strata = {
            25_438: (7801, 7802),
            50_030: (7811, 7812),
            100_359: (7821, 7822),
        }
        draws = triage._hierarchical_bootstrap(
            metrics,
            source_seed,
            age_strata,
            replicates=515,
            seed=20_260_809,
            chunk_size=512,
        )
        digest = hashlib.sha256(np.asarray(
            draws, dtype="<f8").tobytes(order="C")).hexdigest()
        self.assertEqual(
            digest,
            "a6315f279f8d97a7ff9d5ed500165c4cad0d571122f05924d943f144d68691cc",
        )

    def test_direct_audit_loader_requires_marker_before_final_lstat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.multiple(
                    triage,
                    _V3_GROUPS_PER_SEED=2,
                    _V3_ADMISSION_REPLICAS=4,
                    _V3_ADMISSION_MIN_FALLS=1,
                    _V3_ADMISSION_MAX_FALLS=3,
                    _V3_DISCOVERY_REPLICAS=20,
                    _V3_AUDIT_REPLICAS=20):
                spec = triage._validate_protocol(_protocol(root))
            audit = root / "source-7801.audit.npz"
            original_lstat = Path.lstat

            def reject_audit_probe(path: Path):
                if path == audit:
                    raise AssertionError("audit final component was probed")
                return original_lstat(path)

            with mock.patch.object(Path, "lstat", reject_audit_probe):
                with self.assertRaisesRegex(
                        ClosedLoopRecoveryTriageError,
                        "audit-consumed marker"):
                    triage._load_outcome_npz(
                        audit, "audit", spec,
                        expected_groups=2,
                        expected_source_seed=7801,
                    )


class _ScaledV3TestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        patcher = mock.patch.multiple(
            triage,
            _V3_GROUPS_PER_SEED=2,
            _V3_ADMISSION_REPLICAS=4,
            _V3_ADMISSION_MIN_FALLS=1,
            _V3_ADMISSION_MAX_FALLS=3,
            _V3_DISCOVERY_REPLICAS=20,
            _V3_AUDIT_REPLICAS=20,
        )
        patcher.start()
        self.addCleanup(patcher.stop)


class ClosedLoopRecoverySelectionLockTest(_ScaledV3TestCase):
    def test_relative_artifact_root_is_repository_relative_after_chdir(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(
                dir=repository, prefix=".qsafe-v3-") as directory, \
                tempfile.TemporaryDirectory(prefix="qsafe-away-") as away:
            root = Path(directory)
            protocol = _protocol(root)
            protocol["collection"]["artifact_root"] = str(
                root.relative_to(repository))
            admission = _write_admission(root, protocol)
            discovery = _write_discovery(root, protocol)
            _, reports = _prepare_readiness(root, protocol)
            selection_semantics = {
                "schema_version": "qsafe.test.selection_semantics.v1",
                "primary_selection": "synthetic_test_only",
            }
            previous = Path.cwd()
            try:
                os.chdir(away)
                readiness = validate_collection_readiness(
                    protocol=protocol,
                    collection_report_paths=reports,
                )
                lock = create_selection_lock(
                    protocol=protocol,
                    admission_path=admission,
                    discovery_path=discovery,
                    collection_report_paths=reports,
                    selection_lock_path=(root / protocol["collection"][
                        "selection_lock_filename"]),
                    selection_semantics=selection_semantics,
                )
            finally:
                os.chdir(previous)
            self.assertEqual(
                readiness["manifest"]["artifact_root"], str(root.resolve()))
            self.assertTrue(lock["audit_authorized"])
            self.assertEqual(lock["selection_semantics"], selection_semantics)
            persisted = json.loads((root / protocol["collection"][
                "selection_lock_filename"]).read_text(encoding="utf-8"))
            self.assertEqual(
                persisted["selection_semantics"], selection_semantics)

    def test_selection_requires_untampered_merge_completion_reports(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            admission = _write_admission(root, protocol)
            discovery = _write_discovery(root, protocol)
            _, reports = _prepare_readiness(root, protocol)
            merge_report_path = root / protocol["collection"][
                "discovery_merge_report_filename"]
            merge_report = json.loads(
                merge_report_path.read_text(encoding="utf-8"))
            merge_report["collection_data_gate"]["checks"][
                "horizon_exact"] = False
            merge_report["collection_data_gate"]["pass"] = False
            merge_report["phase1_data_gate"] = copy.deepcopy(
                merge_report["collection_data_gate"])
            merge_report_path.write_text(
                json.dumps(merge_report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            lock_path = root / protocol["collection"]["selection_lock_filename"]
            with mock.patch.object(
                    triage, "_load_outcome_npz", wraps=triage._load_outcome_npz,
            ) as outcome_loader, self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "exact data gate"):
                create_selection_lock(
                    protocol=protocol,
                    admission_path=admission,
                    discovery_path=discovery,
                    collection_report_paths=reports,
                    selection_lock_path=lock_path,
                )
            outcome_loader.assert_not_called()
            self.assertFalse(lock_path.exists())

    def test_public_readiness_and_selection_never_touch_role_npz_files(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            admission = _write_admission(root, protocol)
            discovery = _write_discovery(root, protocol)
            audit, reports = _prepare_readiness(root, protocol)

            readiness = validate_collection_readiness(
                protocol=protocol,
                collection_report_paths=reports,
            )
            self.assertEqual(
                set(readiness["role_commitments"]),
                {
                    "admission",
                    "admission_privileged",
                    "discovery",
                    "discovery_privileged",
                    "audit",
                    "audit_privileged",
                },
            )
            self.assertEqual(len(readiness["role_commitments"]["audit"]), 6)
            self.assertEqual(
                canonical_sha256(readiness["manifest"]),
                readiness["readiness_sha256"],
            )

            # Reports and control JSON are sufficient at selection time.  The
            # role files promised by the reports, especially audit, remain
            # unopened and may even be offline without affecting selection.
            for path in audit:
                path.unlink()
            lock = create_selection_lock(
                protocol=protocol,
                admission_path=admission,
                discovery_path=discovery,
                collection_report_paths=reports,
                selection_lock_path=(
                    root / protocol["collection"]["selection_lock_filename"]),
            )
            self.assertEqual(
                lock["expected_audit_shards"],
                readiness["role_commitments"]["audit"],
            )

    def test_report_symlink_and_extra_outcome_field_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            _, reports = _prepare_readiness(root, protocol)
            target = root / "source-7801.report-target.json"
            reports[0].rename(target)
            reports[0].symlink_to(target)
            with self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "symlink"):
                validate_collection_readiness(
                    protocol=protocol,
                    collection_report_paths=reports,
                )

        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            _, reports = _prepare_readiness(root, protocol)
            value = json.loads(reports[0].read_text(encoding="utf-8"))
            value["audit_risk"] = 0.5
            reports[0].write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "extra or missing"):
                validate_collection_readiness(
                    protocol=protocol,
                    collection_report_paths=reports,
                )

    def test_merged_leaf_hash_must_match_completion_report(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            admission_path = _write_admission(root, protocol)
            admission = AdmissionLedger.load(admission_path)
            admission.manifest["shards"][0]["content_sha256"] = _fingerprint(
                "different-admission-leaf")
            staging = admission.save(root / "tampered-admission-staging.npz")
            staging.replace(admission_path)
            discovery = _write_discovery(root, protocol)
            _, reports = _prepare_readiness(root, protocol)
            lock_path = root / protocol["collection"]["selection_lock_filename"]
            with self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "merged admission leaves"):
                create_selection_lock(
                    protocol=protocol,
                    admission_path=admission_path,
                    discovery_path=discovery,
                    collection_report_paths=reports,
                    selection_lock_path=lock_path,
                )
            self.assertFalse(lock_path.exists())

    def test_locks_nonnominal_global_and_uniform_per_group_ties(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            admission = _write_admission(root, protocol)
            risk = _informative_discovery_risk()
            # One local exact tie must remain a uniform set in the lock.
            risk[0, 4] = risk[0, 2]
            discovery = _write_discovery(root, protocol, risk)
            _, reports = _prepare_readiness(root, protocol)
            lock_path = root / protocol["collection"]["selection_lock_filename"]

            lock = create_selection_lock(
                protocol=protocol,
                admission_path=admission,
                discovery_path=discovery,
                collection_report_paths=reports,
                selection_lock_path=lock_path,
            )

            self.assertTrue(lock_path.is_file())
            self.assertTrue(lock["audit_authorized"])
            self.assertEqual(
                lock["selected_global_candidate"]["candidate_index"], 1)
            self.assertEqual(
                lock["selected_global_candidate"]["selection_scope"],
                "eight_nonnominal_candidates",
            )
            self.assertEqual(
                lock["group_selection"][0]["discovery_minimizer_indices"],
                [2, 4],
            )
            self.assertEqual(
                lock["group_selection"][0]["uniform_weights"], [0.5, 0.5])
            self.assertRegex(lock["selection_lock_sha256"], r"^[0-9a-f]{64}$")
            with self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "reuse or clobber"):
                create_selection_lock(
                    protocol=protocol,
                    admission_path=admission,
                    discovery_path=discovery,
                    collection_report_paths=reports,
                    selection_lock_path=lock_path,
                )

    def test_global_exact_tie_uses_locked_candidate_order(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            admission = _write_admission(root, protocol)
            risk = _informative_discovery_risk()
            risk[:, 2] = 0.30  # exact aggregate tie between k1 and k2
            discovery = _write_discovery(root, protocol, risk)
            _, reports = _prepare_readiness(root, protocol)
            lock = create_selection_lock(
                protocol=protocol,
                admission_path=admission,
                discovery_path=discovery,
                collection_report_paths=reports,
                selection_lock_path=(
                    root / protocol["collection"]["selection_lock_filename"]),
            )
            self.assertEqual(
                lock["selected_global_candidate"]["candidate_index"], 1)

    def test_uninformative_discovery_is_locked_but_cannot_open_audit(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            admission = _write_admission(root, protocol)
            risk = np.full((12, 9), 0.05, dtype=np.float64)
            discovery = _write_discovery(root, protocol, risk)
            _, reports = _prepare_readiness(root, protocol)
            lock_path = root / protocol["collection"]["selection_lock_filename"]
            lock = create_selection_lock(
                protocol=protocol,
                admission_path=admission,
                discovery_path=discovery,
                collection_report_paths=reports,
                selection_lock_path=lock_path,
            )
            self.assertFalse(lock["data_gate"]["pass"])
            self.assertFalse(lock["audit_authorized"])
            consumed = root / protocol["collection"]["audit_consumed_filename"]
            with self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "did not authorize"):
                consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock[
                        "selection_lock_sha256"],
                    audit_paths=_audit_paths(root),
                    audit_consumed_path=consumed,
                )
            self.assertFalse(consumed.exists())


class ClosedLoopRecoveryAuditTest(_ScaledV3TestCase):
    def _locked_cohort(
        self,
        root: Path,
        audit_risk: np.ndarray | None = None,
        *,
        wrong_seed: bool = False,
    ) -> tuple[dict, Path, dict, list[Path]]:
        protocol = _protocol(root)
        admission = _write_admission(root, protocol)
        discovery = _write_discovery(root, protocol)
        audit, reports = _prepare_readiness(
            root, protocol, audit_risk, wrong_seed=wrong_seed)
        lock_path = root / protocol["collection"]["selection_lock_filename"]
        lock = create_selection_lock(
            protocol=protocol,
            admission_path=admission,
            discovery_path=discovery,
            collection_report_paths=reports,
            selection_lock_path=lock_path,
        )
        return protocol, lock_path, lock, audit

    def test_v4_tagged_uint64_seeds_round_trip_through_lock_and_audit(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            admission = _write_admission(
                root, protocol, v4_tagged_seeds=True)
            discovery = _write_discovery(
                root, protocol, v4_tagged_seeds=True)
            audit, reports = _prepare_readiness(
                root, protocol, v4_tagged_seeds=True)
            lock_path = root / protocol["collection"][
                "selection_lock_filename"]
            lock = create_selection_lock(
                protocol=protocol,
                admission_path=admission,
                discovery_path=discovery,
                collection_report_paths=reports,
                selection_lock_path=lock_path,
            )
            persisted = json.loads(lock_path.read_text(encoding="utf-8"))
            first = persisted["replica_partition"][0]
            for name in (
                    "admission_crn_ids", "admission_rollout_seeds",
                    "admission_perturbation_seeds", "discovery_crn_ids",
                    "discovery_rollout_seeds",
                    "discovery_perturbation_seeds", "audit_crn_ids",
                    "audit_rollout_seeds", "audit_perturbation_seeds"):
                self.assertTrue(all(value >= 1 << 63 for value in first[name]))
            self.assertGreaterEqual(first["discovery_candidate_seed"], 1 << 63)
            self.assertGreaterEqual(first["audit_candidate_seed"], 1 << 63)

            with _formal_bootstrap_patch():
                result = consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock[
                        "selection_lock_sha256"],
                    audit_paths=audit,
                    audit_consumed_path=(
                        root / protocol["collection"][
                            "audit_consumed_filename"]),
                )
            consumed = root / protocol["collection"][
                "audit_consumed_filename"]
            self.assertTrue(consumed.is_file())
            self.assertRegex(
                result["audit_consumed_marker_sha256"], r"^[0-9a-f]{64}$")

    def test_control_symlinks_never_resolve_an_audit_target_premarker(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol = _protocol(root)
            audit_target = root / "source-7801.audit.npz"
            lock_path = root / protocol["collection"]["selection_lock_filename"]
            lock_path.symlink_to(audit_target)
            consumed = root / protocol["collection"]["audit_consumed_filename"]
            original_resolve = Path.resolve

            def guarded_resolve(path, *args, **kwargs):
                if path.name == audit_target.name:
                    raise AssertionError("audit target resolved before marker")
                return original_resolve(path, *args, **kwargs)

            with mock.patch.object(Path, "resolve", new=guarded_resolve):
                with self.assertRaisesRegex(
                        ClosedLoopRecoveryTriageError, "symlink"):
                    consume_and_evaluate_audit(
                        protocol=protocol,
                        selection_lock_path=lock_path,
                        expected_selection_lock_sha256="a" * 64,
                        audit_paths=_audit_paths(root),
                        audit_consumed_path=consumed,
                    )
            self.assertFalse(os.path.lexists(consumed))

    def test_every_audit_path_probe_occurs_after_consumed_marker(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol, lock_path, lock, audit = self._locked_cohort(root)
            consumed = root / protocol["collection"]["audit_consumed_filename"]
            audit_names = {path.name for path in audit}
            original_lstat = Path.lstat
            original_resolve = Path.resolve
            original_hash = triage._sha256_file
            original_load = GroupedBranchDataset.load

            def marker_exists() -> bool:
                return os.path.lexists(os.fspath(consumed))

            def guarded_lstat(path, *args, **kwargs):
                if path.name in audit_names:
                    self.assertTrue(marker_exists())
                return original_lstat(path, *args, **kwargs)

            def guarded_resolve(path, *args, **kwargs):
                if path.name in audit_names:
                    self.assertTrue(marker_exists())
                return original_resolve(path, *args, **kwargs)

            def guarded_hash(path):
                if Path(path).name in audit_names:
                    self.assertTrue(marker_exists())
                return original_hash(path)

            def guarded_load(path):
                if Path(path).name in audit_names:
                    self.assertTrue(marker_exists())
                return original_load(path)

            with mock.patch.object(Path, "lstat", new=guarded_lstat), \
                    mock.patch.object(Path, "resolve", new=guarded_resolve), \
                    mock.patch.object(
                        triage, "_sha256_file", side_effect=guarded_hash), \
                    mock.patch.object(
                        GroupedBranchDataset, "load", side_effect=guarded_load), \
                    _formal_bootstrap_patch(24):
                consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock[
                        "selection_lock_sha256"],
                    audit_paths=audit,
                    audit_consumed_path=consumed,
                )
            self.assertTrue(consumed.is_file())

    def test_bootstrap_override_surface_does_not_exist(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol, lock_path, lock, audit = self._locked_cohort(root)
            consumed = root / protocol["collection"]["audit_consumed_filename"]
            with self.assertRaisesRegex(TypeError, "bootstrap_replicates"):
                consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock[
                        "selection_lock_sha256"],
                    audit_paths=audit,
                    audit_consumed_path=consumed,
                    bootstrap_replicates=8,
                )
            self.assertFalse(consumed.exists())

    def test_primary_and_conditional_pass_with_equal_seed_effects(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol, lock_path, lock, audit = self._locked_cohort(root)
            consumed = root / protocol["collection"]["audit_consumed_filename"]
            original_loader = triage._load_outcome_npz

            def marker_guard(path, role, spec, **kwargs):
                if role == "audit":
                    self.assertTrue(
                        consumed.is_file(),
                        "audit outcome was opened before the consumed marker",
                    )
                return original_loader(path, role, spec, **kwargs)

            with mock.patch.object(
                    triage, "_load_outcome_npz", side_effect=marker_guard), (
                    _formal_bootstrap_patch()):
                report = consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock[
                        "selection_lock_sha256"],
                    audit_paths=audit,
                    audit_consumed_path=consumed,
                )

            self.assertAlmostEqual(
                report["primary_global_backup"]["audit_absolute_reduction"],
                0.40,
            )
            self.assertTrue(report["primary_global_backup"]["pass"])
            self.assertAlmostEqual(
                report["conditional_state_dependent"][
                    "audit_absolute_reduction"],
                0.70,
            )
            self.assertAlmostEqual(
                report["conditional_state_dependent"][
                    "incremental_reduction_over_global"],
                0.30,
            )
            self.assertTrue(report["conditional_state_dependent"]["pass"])
            self.assertEqual(
                report["decision"],
                "preregister_fresh_option_ranking_qsafe_protocol",
            )
            self.assertFalse(report["bootstrap"]["override_used"])
            self.assertEqual(report["bootstrap"]["replicates_used"], 50_000)
            self.assertEqual(report["bootstrap"]["seed_used"], 20_260_809)
            self.assertEqual(report["bootstrap"]["chunk_size"], 512)
            self.assertEqual(report["bootstrap"]["quantile_method"], "linear")
            self.assertFalse(report["model_training_authorized"])
            self.assertFalse(report["phase2_authorized"])
            with self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "clobber"):
                consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock[
                        "selection_lock_sha256"],
                    audit_paths=audit,
                    audit_consumed_path=consumed,
                )

    def test_primary_failure_can_fire_nine_effect_no_headroom_band(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            # Every K9 arm has exactly the same audit outcome.  All fixed and
            # locked-rule effects and their simultaneous UCBs are exactly zero.
            audit_risk = np.full((12, 9), 0.50, dtype=np.float64)
            protocol, lock_path, lock, audit = self._locked_cohort(
                root, audit_risk)
            consumed = root / protocol["collection"]["audit_consumed_filename"]
            with _formal_bootstrap_patch(48):
                report = consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock[
                        "selection_lock_sha256"],
                    audit_paths=audit,
                    audit_consumed_path=consumed,
                )

            self.assertFalse(report["primary_global_backup"]["pass"])
            self.assertFalse(
                report["conditional_state_dependent"]["tested"])
            self.assertIsNone(
                report["conditional_state_dependent"]["pass"])
            self.assertIsNone(
                report["conditional_state_dependent"]["checks"])
            self.assertEqual(
                len(report["no_headroom"]["simultaneous_one_sided_95_ucb"]),
                9,
            )
            self.assertTrue(report["no_headroom"]["fires"])
            self.assertEqual(
                report["decision"],
                "redesign_recovery_library_before_model_training",
            )

    def test_exact_lock_hash_is_required_before_consumption(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol, lock_path, lock, audit = self._locked_cohort(root)
            consumed = root / protocol["collection"]["audit_consumed_filename"]
            with self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "required hash"):
                consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256="0" * 64,
                    audit_paths=audit,
                    audit_consumed_path=consumed,
                )
            self.assertFalse(consumed.exists())

    def test_bad_audit_binding_stays_consumed_after_failure(self):
        with tempfile.TemporaryDirectory(prefix="qsafe-v3-") as directory:
            root = Path(directory)
            protocol, lock_path, lock, audit = self._locked_cohort(
                root, wrong_seed=True)
            consumed = root / protocol["collection"]["audit_consumed_filename"]
            with self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "seeds differ"):
                consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock[
                        "selection_lock_sha256"],
                    audit_paths=audit,
                    audit_consumed_path=consumed,
                )
            self.assertTrue(consumed.is_file())
            with self.assertRaisesRegex(
                    ClosedLoopRecoveryTriageError, "clobber"):
                consume_and_evaluate_audit(
                    protocol=protocol,
                    selection_lock_path=lock_path,
                    expected_selection_lock_sha256=lock[
                        "selection_lock_sha256"],
                    audit_paths=audit,
                    audit_consumed_path=consumed,
                )


if __name__ == "__main__":
    unittest.main()
