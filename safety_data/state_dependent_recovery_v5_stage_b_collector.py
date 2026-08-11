"""Single-label, role-isolated collector for V5 Stage B.

Unlike the Stage-A triage collector, this collector produces exactly one
candidate-label partition after a physically separate nominal admission
partition.  It never creates an unregistered second outcome split.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from rl.qsafe.recovery_program import make_recovery_program_feature_manifest
from safety_data.closed_loop_recovery_collector import (
    AdmissionLedger,
    AdmissionPrivilegedView,
    ClosedLoopRecoveryCollectionConfig,
    _action_application_contract,
    _fall_definition,
    _finalize_admission,
    evaluate_nominal_admission,
    preflight_closed_loop_recovery_collection,
)
from safety_data.collector import (
    CollectedGroup,
    GroupIdentity,
    GroupRandomness,
    GroupedBranchAssembler,
    PRIVILEGED_FEATURE_NAMES,
    privileged_features,
)
from safety_data.native import ReplicaSeedBundle, evaluate_same_state_group
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView
from safety_data.state_dependent_recovery_v5 import SEED_ROLE_TAGS
from safety_data.state_dependent_recovery_v5_stage_b import (
    ADMISSION_REPLICAS,
    CANDIDATES,
    EXECUTION_PROTOCOL_CONTRACT_SHA256,
    EXECUTION_PROTOCOL_FILE_SHA256,
    GROUPS_PER_SOURCE,
    HORIZON_POLICY_STEPS,
    LABEL_REPLICAS,
    RECOVERY_LIBRARY_FINGERPRINT_SHA256,
    StageBExecutionError,
    TRAJECTORY_FINGERPRINT_ARRAY,
    TRAJECTORY_FINGERPRINT_CONTRACT,
    assignment_for,
    branch_randomness,
    canonical_sha256,
    stage_b_seed,
)


COLLECTION_PROTOCOL_VERSION = (
    "qsafe.state_dependent_recovery_v5.stage_b_role_collection.v1"
)
DATASET_SPLIT_PREFIX = "state_dependent_recovery_v5_stage_b"
SEED_ALGORITHM = "high_bit_then_domain_low15_then_14_8_18_2_6_bitpack_v1"


@dataclass(frozen=True)
class StageBRoleCollectionPreflight:
    role: str
    env: Any
    early_policy: Any
    recovery_program: Any
    config: ClosedLoopRecoveryCollectionConfig
    generator_commit: str
    parent_protocol_sha256: str
    parent_protocol_contract_sha256: str
    policy_set_manifest: dict[str, Any]
    source_fingerprint: str
    recovery_program_binding: dict[str, Any]
    label_assembler: GroupedBranchAssembler
    production_contract: bool


@dataclass(frozen=True)
class StageBRoleCollectionResult:
    role: str
    admission: AdmissionLedger
    admission_privileged: AdmissionPrivilegedView
    labels: GroupedBranchDataset
    labels_privileged: PrivilegedBranchView
    source_steps: int
    trajectories: int
    proposals: int


def _domain(role: str, partition: str) -> bytes:
    return (
        f"qsafe_state_dependent_recovery_v4_stage_b_{role}_{partition}\0"
    ).encode("ascii")


def production_collection_config(
    *,
    role: str,
    source_seed: int,
    max_episode_steps: int,
    max_trajectories: int,
    proposal_cooldown_steps: int,
    settle_seconds: float,
    source_impulse_interval_steps: int,
    source_linear_std_mps: float,
    source_angular_std_radps: float,
    proposal_min_tilt_rad: float,
    proposal_max_height_m: float,
) -> ClosedLoopRecoveryCollectionConfig:
    assignment = assignment_for(role, source_seed)
    return ClosedLoopRecoveryCollectionConfig(
        source_seed=source_seed,
        policy_training_step=assignment.checkpoint_step,
        policy_training_seed=assignment.actor_training_seed,
        target_groups=GROUPS_PER_SOURCE[role],
        horizon_steps=HORIZON_POLICY_STEPS,
        admission_replicas=ADMISSION_REPLICAS,
        admission_min_falls=6,
        admission_max_falls=26,
        discovery_replicas=LABEL_REPLICAS[role],
        audit_replicas=LABEL_REPLICAS[role],
        max_episode_steps=max_episode_steps,
        max_proposals=4096,
        max_trajectories=max_trajectories,
        proposal_cooldown_steps=proposal_cooldown_steps,
        settle_seconds=settle_seconds,
        source_impulse_interval_steps=source_impulse_interval_steps,
        source_linear_std_mps=source_linear_std_mps,
        source_angular_std_radps=source_angular_std_radps,
        proposal_min_tilt_rad=proposal_min_tilt_rad,
        proposal_max_height_m=proposal_max_height_m,
        seed_domain=_domain(role, "admission"),
        seed_role_tags=SEED_ROLE_TAGS,
        seed_algorithm=SEED_ALGORITHM,
        dataset_split_prefix=f"{DATASET_SPLIT_PREFIX}_{role}",
        collection_protocol_version=COLLECTION_PROTOCOL_VERSION,
        trajectory_id_prefix=f"stage-b-{role.replace('_', '-')}",
        explicit_filter_settings_in_action_contract=True,
    )


def _validate_production_config(
    role: str,
    config: ClosedLoopRecoveryCollectionConfig,
) -> None:
    assignment = assignment_for(role, int(config.source_seed))
    checks = {
        "policy_training_seed": assignment.actor_training_seed,
        "policy_training_step": assignment.checkpoint_step,
        "target_groups": assignment.groups,
        "horizon_steps": HORIZON_POLICY_STEPS,
        "admission_replicas": ADMISSION_REPLICAS,
        "admission_min_falls": 6,
        "admission_max_falls": 26,
        "discovery_replicas": assignment.label_replicas,
        "audit_replicas": assignment.label_replicas,
        "max_proposals": 4096,
        "seed_domain": _domain(role, "admission"),
        "seed_role_tags": SEED_ROLE_TAGS,
        "seed_algorithm": SEED_ALGORITHM,
        "collection_protocol_version": COLLECTION_PROTOCOL_VERSION,
    }
    for name, expected in checks.items():
        if getattr(config, name) != expected:
            raise StageBExecutionError(
                f"production Stage-B collector config {name} has drifted"
            )


def _label_collection_protocol(
    *,
    role: str,
    config: ClosedLoopRecoveryCollectionConfig,
) -> dict[str, Any]:
    domain = _domain(role, "label")
    manifest = {
        "version": COLLECTION_PROTOCOL_VERSION,
        "role": role,
        "partition": "label",
        "scope": "conditional_development_stage_b_model_learning_only",
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256
        ),
        "selection_timing": "admission_before_candidate_outcomes",
        "admission_outcomes_used_as_labels": False,
        "label_replicas": int(config.discovery_replicas),
        "horizon_steps": int(config.horizon_steps),
        "candidate_count": CANDIDATES,
        "sampling_strata": {
            "admission_positive_conditional": {
                "predicate": "locked_nominal_admission_positive",
                "acceptance_probability": 1.0,
            },
        },
        "acceptance_probability_field_semantics": (
            "unit_analysis_weight_within_conditional_cohort_not_source_stream_"
            "inclusion_probability"
        ),
        "natural_incidence_claim": False,
        "max_groups_per_trajectory": 1,
        "trajectory_fingerprint_array": TRAJECTORY_FINGERPRINT_ARRAY,
        "trajectory_fingerprint_contract": TRAJECTORY_FINGERPRINT_CONTRACT,
        "physical_admission_and_label_files": True,
        "seed_derivation": {
            "algorithm": SEED_ALGORITHM,
            "domain_hex": domain.hex(),
            "domain_sha256_prefix_low15": (
                int.from_bytes(hashlib.sha256(domain).digest()[:2], "little")
                & 0x7FFF
            ),
            "role_tag": 130,
            "stream_mapping": {
                "crn_id": {"namespace": 0, "index": "replica_index"},
                "rollout_seed": {"namespace": 1, "index": "replica_index"},
                "perturbation_seed": {
                    "namespace": 2, "index": "replica_index"
                },
                "candidate_seed": {"namespace": 3, "index": 0},
            },
        },
        "branch_disturbance": "zero",
        "candidate_outcomes_summarized_by_collector": False,
    }
    return manifest | {"contract_sha256": canonical_sha256(manifest)}


def preflight_stage_b_role_collection(
    *,
    role: str,
    env: Any,
    early_policy: Any,
    recovery_program: Any,
    policy_set_manifest: Mapping[str, Any],
    config: ClosedLoopRecoveryCollectionConfig,
    generator_commit: str,
    parent_protocol_sha256: str,
    parent_protocol_contract_sha256: str,
    production_contract: bool = True,
) -> StageBRoleCollectionPreflight:
    """Build every deterministic binding without generating an outcome."""
    if production_contract:
        _validate_production_config(role, config)
    prepared = preflight_closed_loop_recovery_collection(
        env=env,
        early_policy=early_policy,
        recovery_program=recovery_program,
        policy_set_manifest=policy_set_manifest,
        config=config,
        generator_commit=generator_commit,
        protocol_sha256=parent_protocol_sha256,
        protocol_contract_sha256=parent_protocol_contract_sha256,
    )
    if prepared.recovery_program_binding["fingerprint_sha256"] != (
        RECOVERY_LIBRARY_FINGERPRINT_SHA256
    ) and production_contract:
        raise StageBExecutionError("Stage-B recovery library fingerprint drifted")
    label_assembler = GroupedBranchAssembler(
        split=f"{DATASET_SPLIT_PREFIX}_{role}_label",
        horizon_steps=config.horizon_steps,
        generator_commit=generator_commit,
        simulator_fingerprint=env.simulator_fingerprint(),
        source_policy=policy_set_manifest,
        continuation_policy=policy_set_manifest,
        candidate_protocol=prepared.candidate_protocol,
        fall_definition=_fall_definition(env),
        action_application_contract=_action_application_contract(
            env, include_filter_settings=True
        ),
        collection_protocol=_label_collection_protocol(role=role, config=config),
        recovery_program=prepared.recovery_program_binding,
        privileged_feature_names=PRIVILEGED_FEATURE_NAMES,
    )
    label_assembler.manifest["recovery_program_feature_contract"] = (
        make_recovery_program_feature_manifest(
            prepared.recovery_program_binding["fingerprint_sha256"]
        )
        if production_contract
        else {
            "schema_version": "synthetic_stage_b_feature_contract",
            "feature_contract_sha256": "0" * 64,
        }
    )
    return StageBRoleCollectionPreflight(
        role=role,
        env=env,
        early_policy=early_policy,
        recovery_program=recovery_program,
        config=config,
        generator_commit=generator_commit,
        parent_protocol_sha256=parent_protocol_sha256,
        parent_protocol_contract_sha256=parent_protocol_contract_sha256,
        policy_set_manifest=copy.deepcopy(dict(policy_set_manifest)),
        source_fingerprint=prepared.source_fingerprint,
        recovery_program_binding=copy.deepcopy(
            prepared.recovery_program_binding
        ),
        label_assembler=label_assembler,
        production_contract=production_contract,
    )


def _replica_bundle(values: Mapping[str, Any]) -> ReplicaSeedBundle:
    return ReplicaSeedBundle(
        crn_id=np.asarray(values["crn_id"], dtype=np.uint64),
        rollout_seed=np.asarray(values["rollout_seed"], dtype=np.uint64),
        perturbation_seed=np.asarray(
            values["perturbation_seed"], dtype=np.uint64
        ),
    )


def _group_randomness(values: Mapping[str, Any]) -> GroupRandomness:
    return GroupRandomness(
        crn_id=np.asarray(values["crn_id"], dtype=np.uint64),
        rollout_seed=np.asarray(values["rollout_seed"], dtype=np.uint64),
        perturbation_seed=np.asarray(
            values["perturbation_seed"], dtype=np.uint64
        ),
        candidate_seed=int(values["candidate_seed"]),
    )


def collect_preflighted_stage_b_role(
    *,
    preflight: StageBRoleCollectionPreflight,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> StageBRoleCollectionResult:
    """Consume one source once and create admission plus one label dataset."""
    if not isinstance(preflight, StageBRoleCollectionPreflight):
        raise TypeError("preflight must be a Stage-B role preflight")
    env = preflight.env
    early_policy = preflight.early_policy
    recovery_program = preflight.recovery_program
    config = preflight.config
    role = preflight.role
    assembler = preflight.label_assembler
    if assembler.group_count != 0:
        raise StageBExecutionError("Stage-B label assembler was already consumed")

    admission_rows: list[dict[str, Any]] = []
    admission_privileged_rows: list[dict[str, Any]] = []
    source_steps = 0
    episode_number = 0
    episode_step = 0
    last_proposal_step = -config.proposal_cooldown_steps
    trajectory_fingerprint_sha256: str | None = None

    def source_seed(stream_role: str, identity: int, namespace: int = 0) -> int:
        return stage_b_seed(
            role=role,
            partition="admission",
            source_seed=int(config.source_seed),
            stream_role=stream_role,
            identity=identity,
            namespace=namespace,
            index=0,
        )

    def reset() -> None:
        nonlocal episode_step, last_proposal_step
        nonlocal trajectory_fingerprint_sha256
        if episode_number >= config.max_trajectories:
            raise RuntimeError(
                "collector exhausted max_trajectories before target groups"
            )
        env.reset_standing(
            settle_seconds=config.settle_seconds,
            rng=np.random.default_rng(source_seed("source_reset", episode_number)),
        )
        # This physical/content cluster identity is captured before the first
        # source impulse, observation, or policy action.  It deliberately
        # excludes role/source/episode labels and all admission/label outcomes.
        trajectory_fingerprint_sha256 = env.capture().compound_sha256()
        episode_step = 0
        last_proposal_step = -config.proposal_cooldown_steps

    reset()
    while assembler.group_count < config.target_groups:
        if len(admission_rows) >= config.max_proposals:
            raise RuntimeError("collector exhausted max_proposals before target groups")
        absolute_episode_step = (
            episode_number * config.max_episode_steps + episode_step
        )
        if episode_step > 0 and episode_step % (
            config.source_impulse_interval_steps
        ) == 0:
            impulse_rng = np.random.default_rng(
                source_seed("source_impulse", absolute_episode_step)
            )
            env.apply_base_velocity_impulse(
                linear_velocity_delta=impulse_rng.normal(
                    0.0, config.source_linear_std_mps, size=3
                ),
                angular_velocity_delta=impulse_rng.normal(
                    0.0, config.source_angular_std_radps, size=3
                ),
            )
        history = env.record_observation()
        observation = history[-1]
        source_action = early_policy.sample_action(
            observation,
            np.random.default_rng(
                source_seed("source_action", absolute_episode_step)
            ),
        )
        measurement = env.measurement()
        if measurement.failure:
            episode_number += 1
            reset()
            continue
        cooldown_ready = (
            episode_step - last_proposal_step >= config.proposal_cooldown_steps
        )
        pre_screen = (
            float(measurement.tilt_rad) >= config.proposal_min_tilt_rad
            or float(measurement.height_m) <= config.proposal_max_height_m
        )
        accepted = False
        if cooldown_ready and pre_screen:
            proposal_index = len(admission_rows)
            last_proposal_step = episode_step
            snapshot = env.capture()
            if env.measurement().failure:
                raise RuntimeError("proposal snapshot is already a failure")
            state_hash = snapshot.compound_sha256()
            if trajectory_fingerprint_sha256 is None:
                raise AssertionError("source trajectory fingerprint was not set")
            trajectory_id = (
                f"{config.trajectory_id_prefix}:source-{config.source_seed}:"
                f"trajectory-{episode_number}"
            )
            proposal_id = f"{trajectory_id}:step-{episode_step}"
            admission_values = branch_randomness(
                role=role,
                partition="admission",
                source_seed=int(config.source_seed),
                proposal_index=proposal_index,
                replicas=int(config.admission_replicas),
            )
            nominal = early_policy.deterministic_action(observation)
            admission = evaluate_nominal_admission(
                env=env,
                snapshot=snapshot,
                nominal_first_action=nominal,
                seeds=_replica_bundle(admission_values),
                horizon_steps=config.horizon_steps,
                continuation_policy=early_policy,
            )
            fall_count = int(np.count_nonzero(admission.fall))
            accepted = bool(
                config.admission_min_falls <= fall_count
                <= config.admission_max_falls
            )
            accepted_group_index = assembler.group_count if accepted else -1
            episode_id = source_seed("source_reset", episode_number, namespace=1)
            admission_rows.append({
                "proposal_id": proposal_id,
                "proposal_index": proposal_index,
                "state_hash": state_hash,
                "trajectory_id": trajectory_id,
                "episode_id": episode_id,
                "episode_step": episode_step,
                "source_seed": config.source_seed,
                "policy_training_step": config.policy_training_step,
                "policy_source": preflight.source_fingerprint,
                "obs_history": history.copy(),
                "admission_crn_id": admission.crn_id,
                "admission_rollout_seed": admission.rollout_seed,
                "admission_perturbation_seed": admission.perturbation_seed,
                "admission_candidate_seed": int(admission_values["candidate_seed"]),
                "fall": admission.fall,
                "first_failure_step": admission.first_failure_step,
                "accepted": accepted,
                "accepted_group_index": accepted_group_index,
                "decision_reason": (
                    f"accepted_{config.admission_min_falls}_to_"
                    f"{config.admission_max_falls}_of_{config.admission_replicas}"
                    if accepted else
                    f"rejected_outside_{config.admission_min_falls}_to_"
                    f"{config.admission_max_falls}_of_{config.admission_replicas}"
                ),
            })
            admission_privileged_rows.append({
                "initial_tilt_rad": float(measurement.tilt_rad),
                "initial_height_m": float(measurement.height_m),
                "max_tilt_rad": admission.max_tilt_rad,
                "min_height_m": admission.min_height_m,
            })
            if accepted:
                candidates = recovery_program.preview_projected(history, nominal)
                label_values = branch_randomness(
                    role=role,
                    partition="label",
                    source_seed=int(config.source_seed),
                    proposal_index=proposal_index,
                    replicas=int(config.discovery_replicas),
                )
                evaluation = evaluate_same_state_group(
                    env,
                    snapshot,
                    candidates.requested,
                    _replica_bundle(label_values),
                    horizon_steps=config.horizon_steps,
                    continuation_policy=early_policy,
                    disturbance_program=None,
                    recovery_program=recovery_program,
                )
                for evaluation_name, candidate_name in (
                    ("candidate_requested", "requested"),
                    ("candidate_executed", "executed"),
                    ("candidate_q_target", "q_target"),
                ):
                    if not np.array_equal(
                        np.asarray(getattr(evaluation, evaluation_name)),
                        np.asarray(getattr(candidates, candidate_name)),
                    ):
                        raise RuntimeError(
                            "Stage-B execution disagrees with previewed "
                            f"{evaluation_name}"
                        )
                assembler.add(CollectedGroup(
                    identity=GroupIdentity(
                        group_id=proposal_id,
                        state_hash=state_hash,
                        trajectory_id=trajectory_id,
                        episode_id=episode_id,
                        episode_step=episode_step,
                        policy_training_seed=config.policy_training_seed,
                        source_seed=config.source_seed,
                        policy_source=preflight.source_fingerprint,
                        command_vx=float(env.cfg.move_speed),
                        acceptance_probability=1.0,
                        sampling_stratum="admission_positive_conditional",
                        trajectory_fingerprint_sha256=(
                            trajectory_fingerprint_sha256
                        ),
                    ),
                    observation_history=history,
                    candidate_kind=candidates.kind,
                    candidate_mask=candidates.mask,
                    evaluation=evaluation,
                    randomness=_group_randomness(label_values),
                    candidate_behavior_steps=candidates.behavior_steps,
                    privileged_features=privileged_features(env),
                ))
                if progress is not None:
                    progress({
                        "groups": assembler.group_count,
                        "target_groups": config.target_groups,
                        "proposals": len(admission_rows),
                        "source_steps": source_steps,
                        "trajectories": episode_number + 1,
                    })
                episode_number += 1
                if assembler.group_count < config.target_groups:
                    reset()
                continue
        step_result = env.step(source_action)
        source_steps += 1
        episode_step += 1
        if step_result.failure or episode_step >= config.max_episode_steps:
            episode_number += 1
            reset()

    admission, admission_privileged = _finalize_admission(
        rows=admission_rows,
        privileged_rows=admission_privileged_rows,
        config=config,
        generator_commit=preflight.generator_commit,
        protocol_sha256=preflight.parent_protocol_sha256,
        protocol_contract_sha256=preflight.parent_protocol_contract_sha256,
        fall_definition=_fall_definition(env),
        simulator_fingerprint=assembler.manifest["simulator_fingerprint"],
        source_policy=preflight.policy_set_manifest,
        action_application_contract=assembler.manifest[
            "action_application_contract"
        ],
    )
    admission.manifest.update({
        "stage_b_role": role,
        "policy_training_seed": int(config.policy_training_seed),
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256
        ),
        "seed_domain_hex": _domain(role, "admission").hex(),
    })
    admission_report = admission.validate(verify_hash=False)
    admission.manifest["content_sha256"] = admission_report["content_sha256"]
    admission_privileged.manifest.update({
        "stage_b_role": role,
        "execution_protocol_file_sha256": EXECUTION_PROTOCOL_FILE_SHA256,
        "execution_protocol_contract_sha256": (
            EXECUTION_PROTOCOL_CONTRACT_SHA256
        ),
        "deployable_content_sha256": admission_report["content_sha256"],
    })
    privileged_admission_report = admission_privileged.validate(
        admission, verify_hash=False
    )
    admission_privileged.manifest["content_sha256"] = (
        privileged_admission_report["content_sha256"]
    )
    labels, labels_privileged = assembler.finalize()
    if labels_privileged is None:
        raise AssertionError("Stage-B labels require a privileged sidecar")
    if labels.group_count != config.target_groups or labels.candidate_count != (
        CANDIDATES
    ) or labels.replica_count != config.discovery_replicas:
        raise StageBExecutionError("Stage-B label dimensions drifted")
    accepted = np.asarray(admission["accepted"], dtype=bool)
    if int(np.count_nonzero(accepted)) != labels.group_count:
        raise StageBExecutionError("admission and label group counts differ")
    if len(np.unique(np.asarray(admission["trajectory_id"])[accepted])) != (
        labels.group_count
    ):
        raise StageBExecutionError("Stage-B requires one state per trajectory")
    for admission_name, label_name in (
        ("proposal_id", "group_id"),
        ("state_hash", "state_hash"),
        ("trajectory_id", "trajectory_id"),
    ):
        if not np.array_equal(
            np.asarray(admission[admission_name])[accepted].astype(str),
            np.asarray(labels[label_name]).astype(str),
        ):
            raise StageBExecutionError(
                f"accepted admission {admission_name} differs from labels"
            )
    admission_seed_values = set(np.concatenate([
        np.asarray(admission[name], dtype=np.uint64).reshape(-1)
        for name in (
            "admission_crn_id",
            "admission_rollout_seed",
            "admission_perturbation_seed",
            "admission_candidate_seed",
        )
    ]).tolist())
    label_seed_values = set(np.concatenate([
        np.asarray(labels[name], dtype=np.uint64).reshape(-1)
        for name in ("crn_id", "rollout_seed", "perturbation_seed", "candidate_seed")
    ]).tolist())
    if admission_seed_values & label_seed_values:
        raise StageBExecutionError("Stage-B admission and label seeds overlap")
    return StageBRoleCollectionResult(
        role=role,
        admission=admission,
        admission_privileged=admission_privileged,
        labels=labels,
        labels_privileged=labels_privileged,
        source_steps=source_steps,
        trajectories=episode_number,
        proposals=len(admission_rows),
    )


__all__ = [
    "COLLECTION_PROTOCOL_VERSION",
    "DATASET_SPLIT_PREFIX",
    "StageBRoleCollectionPreflight",
    "StageBRoleCollectionResult",
    "collect_preflighted_stage_b_role",
    "preflight_stage_b_role_collection",
    "production_collection_config",
]
