"""One-shot V4 Stage-A state-dependent recovery confirmation.

The V4 primary rule is the uniform expectation over every exact per-state
discovery minimizer.  Discovery may be inspected once to create a selection
lock; audit paths are compared lexically with that lock, an irreversible
marker is published, and only then may an audit shard be touched.  A passing
report authorizes Stage B model work but never starts it or authorizes later
stages.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import threading
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml

from safety_data.closed_loop_recovery_collector import (
    AdmissionLedger,
    _derived_seed,
    canonical_protocol_sha256,
    role_randomness,
    seed_derivation_manifest,
)
import safety_data.closed_loop_recovery_triage as _v3
from safety_data.recovery_behaviors import RecoveryBehaviorConfig
from safety_data.schema import GroupedBranchDataset


PROTOCOL_NAME = "objective1_state_dependent_recovery_qsafe_v4"
PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "config" / "qsafe_state_dependent_recovery_v4.yaml"
)
PROTOCOL_CONTRACT_SHA256 = (
    "101484a5df78b22941a8988f9936c7fb40b4569ed5c555273843484275dcc977"
)
AUDIT_CONSUMED_SCHEMA_VERSION = (
    "qsafe.state_dependent_recovery_v4.audit_consumed.v1"
)
REPORT_SCHEMA_VERSION = "qsafe.state_dependent_recovery_v4.stage_a_report.v1"
SELECTION_SEMANTICS = {
    "schema_version": "qsafe.state_dependent_recovery_v4.selection.v1",
    "primary_selection": (
        "per_state_all_exact_discovery_minima_uniform_expectation"),
    "per_state_tie_rule": "uniform_expectation_all_exact_minima",
    "candidate_column_order_effect_on_ties": "forbidden",
    "selected_global_candidate_role": "diagnostic_only",
    "audit_runner_up_policy": "forbidden",
}

_CANDIDATE_PROTOCOL = RecoveryBehaviorConfig().manifest_protocol()
CANDIDATE_NAMES = tuple(_CANDIDATE_PROTOCOL["ordered_names"])
BEHAVIOR_STEPS = tuple(_CANDIDATE_PROTOCOL["behavior_override_steps"])
SOURCE_SEEDS = (8401, 8402, 8411, 8412, 8421, 8422)
AGE_STRATA = {
    25_438: (8401, 8402),
    50_030: (8411, 8412),
    100_359: (8421, 8422),
}
SEED_DOMAIN = b"qsafe_state_dependent_recovery_v4_seed\0"
SEED_ALGORITHM = (
    "high_bit_then_domain_low15_then_14_8_18_2_6_bitpack_v1")
SEED_DOMAIN_PREFIX_LOW15 = 18_561
SEED_ROLE_TAGS = (
    ("source_reset", 110),
    ("source_impulse", 111),
    ("source_action", 112),
    ("admission", 120),
    ("discovery", 130),
    ("audit", 140),
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DENIED_PREFIXES = ("formal", "sealed")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_V3_VALIDATOR_PATCH_LOCK = threading.RLock()


class StateDependentRecoveryV4Error(ValueError):
    """The immutable V4 protocol, firewall, or primary gate failed closed."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that fails closed instead of overwriting duplicate keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise StateDependentRecoveryV4Error(
                "canonical V4 YAML contains an unhashable mapping key") from exc
        if duplicate:
            raise StateDependentRecoveryV4Error(
                f"canonical V4 YAML contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StateDependentRecoveryV4Error(f"{name} must be a mapping")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise StateDependentRecoveryV4Error(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise StateDependentRecoveryV4Error(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise StateDependentRecoveryV4Error(f"{name} must be finite")
    return result


def _json_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(json.dumps(
            value, allow_nan=False, ensure_ascii=True, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise StateDependentRecoveryV4Error(
            f"{name} must be canonical JSON data") from exc


def _require_equal(actual: object, expected: object, name: str) -> None:
    if actual != expected:
        raise StateDependentRecoveryV4Error(f"{name} has drifted")


def _reject_protected_components(path: Path, name: str) -> None:
    for component in path.parts:
        folded = component.casefold()
        if any(folded.startswith(prefix) for prefix in _DENIED_PREFIXES):
            raise StateDependentRecoveryV4Error(
                f"{name} contains a denied path component")


def _absolute_repo_path(value: str | os.PathLike[str]) -> Path:
    raw = Path(value)
    anchored = raw if raw.is_absolute() else _REPOSITORY_ROOT / raw
    return Path(os.path.abspath(os.fspath(anchored)))


def _artifact_root(protocol: Mapping[str, Any]) -> Path:
    collection = _mapping(protocol.get("collection"), "collection")
    root = _absolute_repo_path(str(collection.get("artifact_root")))
    _reject_protected_components(root, "collection.artifact_root")
    return root


def _require_clean_head_protocol_binding() -> tuple[str, str]:
    """Return the current clean HEAD and canonical raw protocol digest."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(_REPOSITORY_ROOT),
             "status", "--porcelain=v1", "-z"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StateDependentRecoveryV4Error(
            "could not establish the current clean V4 generator commit") from exc
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) is None or (
            status.stdout):
        raise StateDependentRecoveryV4Error(
            "V4 evidence operations require the current clean git HEAD")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    try:
        descriptor = os.open(PROTOCOL_PATH, flags)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise StateDependentRecoveryV4Error(
                    "canonical V4 protocol must be a singly linked regular file")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except StateDependentRecoveryV4Error:
        raise
    except OSError as exc:
        raise StateDependentRecoveryV4Error(
            "canonical V4 protocol raw bytes are unreadable") from exc
    return commit, digest.hexdigest()


def _require_unchanged_clean_binding(
    expected_commit: str,
    expected_protocol_file_sha256: str,
    phase: str,
) -> None:
    observed = _require_clean_head_protocol_binding()
    if observed != (expected_commit, expected_protocol_file_sha256):
        raise StateDependentRecoveryV4Error(
            f"V4 clean HEAD/protocol binding changed {phase}")


def load_state_dependent_recovery_v4_protocol(
    path: str | os.PathLike[str] = PROTOCOL_PATH,
) -> dict[str, Any]:
    """Load and validate the one canonical V4 protocol file."""
    supplied = _absolute_repo_path(path)
    _reject_protected_components(supplied, "protocol path")
    if supplied != PROTOCOL_PATH:
        raise StateDependentRecoveryV4Error(
            "V4 requires the canonical protocol path")
    try:
        protocol = yaml.load(
            supplied.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except OSError as exc:
        raise StateDependentRecoveryV4Error(
            "canonical V4 protocol is unreadable") from exc
    validate_state_dependent_recovery_v4_protocol(protocol)
    return dict(protocol)


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    enforce_canonical_hash: bool = True,
) -> dict[str, Any]:
    protocol = _mapping(protocol, "protocol")
    _require_equal(protocol.get("protocol_schema_version"), 1,
                   "protocol_schema_version")
    _require_equal(protocol.get("protocol_name"), PROTOCOL_NAME,
                   "protocol_name")
    _require_equal(protocol.get("scope"),
                   "conditional_development_stage_a_confirmation", "scope")
    _require_equal(protocol.get("claim_eligible"), False, "claim_eligible")
    contract_sha256 = canonical_protocol_sha256(protocol)
    if enforce_canonical_hash and contract_sha256 != PROTOCOL_CONTRACT_SHA256:
        raise StateDependentRecoveryV4Error(
            "parsed protocol differs from the complete canonical V4 contract")

    parent = _mapping(protocol.get("parent_iterations"), "parent_iterations")
    _require_equal(parent.get("consumed_source_seeds"), [
        7601, 7602, 7603, 7801, 7802, 7811, 7812, 7821, 7822],
        "parent_iterations.consumed_source_seeds")
    _require_equal(parent.get("outcome_reuse"), "forbidden",
                   "parent_iterations.outcome_reuse")

    protection = _mapping(protocol.get("protection"), "protection")
    for name, expected in {
        "reject_old_protocols": True,
        "forbidden_legacy_machine_protocol": (
            "config/qsafe_evidence_protocol.yaml"),
        "assignment_before_candidate_outcomes": True,
        "optional_stopping": "forbidden",
        "sample_top_up": "forbidden",
        "failed_trials_remain_consumed": True,
        "audit_path_rule": (
            "lexical_commitment_check_then_marker_then_first_filesystem_probe"),
        "publication": "atomic_no_clobber_report_last",
    }.items():
        _require_equal(protection.get(name), expected, f"protection.{name}")
    _require_equal(protection.get("stage_A_denied_report_resume"), {
        "entrypoint": "resume-denied-report",
        "prerequisite": "existing_hash_bound_selection_lock_only",
        "required_selection_lock_values": {
            "audit_authorized": False,
            "data_gate": {
                "pass": False,
                "discovery_informativeness": {"pass": False},
            },
        },
        "selection_lock_mutation": "forbidden",
        "report_reconstruction": (
            "canonical_from_protocol_and_existing_selection_lock_only"),
        "repeated_call": (
            "accept_existing_report_only_if_byte_identical_to_reconstruction"),
        "report_publication": "atomic_no_clobber_report_last",
        "audit_path_parse_or_probe": "forbidden",
        "audit_outcome_hash_map_load_or_open": "forbidden",
    }, "protection.stage_A_denied_report_resume")
    denied = protection.get("denied_path_component_prefixes")
    if not isinstance(denied, list) or not set(_DENIED_PREFIXES).issubset(
            {str(value).casefold() for value in denied}):
        raise StateDependentRecoveryV4Error(
            "V4 must deny formal* and sealed* path components")

    seed_contract = _mapping(protocol.get("seed_derivation"), "seed_derivation")
    expected_seed_contract = {
        "algorithm": SEED_ALGORITHM,
        "domain_literal_ascii_escaped": (
            "qsafe_state_dependent_recovery_v4_seed\\0"),
        "domain_hex": SEED_DOMAIN.hex(),
        "domain_sha256_prefix_low15": SEED_DOMAIN_PREFIX_LOW15,
        "v4_tag_bit63": 1,
        "historical_v2_v3_tag_bit63": 0,
        "packed_field_order": [
            "v4_tag", "domain_prefix", "source_seed", "role_tag",
            "identity", "namespace", "index"],
        "packed_field_width_bits": [1, 15, 14, 8, 18, 2, 6],
        "production_caps_exclusive": {
            "source_seed": 1 << 14,
            "role_tag": 1 << 8,
            "identity": 1 << 18,
            "namespace": 1 << 2,
            "index": 1 << 6,
        },
        "stream_mapping": {
            "source_reset_rng": {
                "role": "source_reset", "identity": "episode_number",
                "namespace": 0, "index": 0},
            "episode_id": {
                "role": "source_reset", "identity": "episode_number",
                "namespace": 1, "index": 0},
            "source_impulse_rng": {
                "role": "source_impulse",
                "identity": (
                    "episode_number_times_max_episode_steps_plus_episode_step"),
                "namespace": 0, "index": 0},
            "source_action_rng": {
                "role": "source_action",
                "identity": (
                    "episode_number_times_max_episode_steps_plus_episode_step"),
                "namespace": 0, "index": 0},
            "branch_crn_id": {
                "roles": ["admission", "discovery", "audit"],
                "identity": "proposal_index", "namespace": 0,
                "index": "replica_index"},
            "branch_rollout_seed": {
                "roles": ["admission", "discovery", "audit"],
                "identity": "proposal_index", "namespace": 1,
                "index": "replica_index"},
            "branch_perturbation_seed": {
                "roles": ["admission", "discovery", "audit"],
                "identity": "proposal_index", "namespace": 2,
                "index": "replica_index"},
            "branch_candidate_seed": {
                "roles": ["admission", "discovery", "audit"],
                "identity": "proposal_index", "namespace": 3,
                "index": 0},
        },
        "injective_within_declared_caps": True,
        "truncation_after_field_packing": "forbidden",
        "role_tags": dict(SEED_ROLE_TAGS),
        "roles_exact": True,
        "disjoint_from_v2_v3_by_v4_tag_bit": True,
    }
    _require_equal(dict(seed_contract), expected_seed_contract,
                   "seed_derivation")

    target = _mapping(protocol.get("target"), "target")
    for name, expected in {
        "command_speed_mps": 0.30,
        "policy_hz": 50,
        "low_level_hz": 500,
        "horizon_policy_steps": 96,
        "cross_actor_seed_claim": "forbidden",
    }.items():
        _require_equal(target.get(name), expected, f"target.{name}")
    _require_equal(target.get("estimand"),
                   "equal_seed_per_state_discovery_locked_recovery_effect_on_"
                   "admission_positive_fixed_states", "target.estimand")
    action_contract = _mapping(
        target.get("action_application_contract"),
        "target.action_application_contract",
    )
    if set(action_contract) != {
            "q_target_semantic", "init_qpos", "action_offset", "joint_min",
            "joint_max", "projection", "max_joint_delta",
            "use_action_filter"}:
        raise StateDependentRecoveryV4Error(
            "target.action_application_contract must have the exact V4 keyset")
    _require_equal(
        action_contract.get("q_target_semantic"),
        "absolute_joint_position_sent",
        "target.action_application_contract.q_target_semantic",
    )
    _require_equal(
        action_contract.get("projection"),
        "clip_normalized_then_joint_bounds_then_slew_then_filter",
        "target.action_application_contract.projection",
    )
    _require_equal(
        action_contract.get("max_joint_delta"), None,
        "target.action_application_contract.max_joint_delta",
    )
    _require_equal(
        action_contract.get("use_action_filter"), False,
        "target.action_application_contract.use_action_filter",
    )
    for field in ("init_qpos", "action_offset", "joint_min", "joint_max"):
        values = action_contract.get(field)
        if not isinstance(values, list) or len(values) != 12 or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not np.isfinite(float(value)) for value in values):
            raise StateDependentRecoveryV4Error(
                f"target.action_application_contract.{field} must be a finite "
                "12-vector")

    policy_config = _mapping(protocol.get("policy_config"), "policy_config")
    _require_equal(policy_config.get("policy_training_seed"), 42,
                   "policy_config.policy_training_seed")
    early = protocol.get("early_task_policies")
    if not isinstance(early, list) or len(early) != 3:
        raise StateDependentRecoveryV4Error(
            "early_task_policies must contain the exact three seed-42 ages")
    expected_ages = (25_438, 50_030, 100_359)
    seed_age: dict[int, int] = {}
    for entry, age, seeds in zip(
            early, expected_ages, AGE_STRATA.values(), strict=True):
        item = _mapping(entry, "early_task_policy")
        _require_equal(item.get("training_step"), age,
                       "early_task_policy.training_step")
        _require_equal(item.get("source_seeds"), list(seeds),
                       "early_task_policy.source_seeds")
        for field in (
                "actor_sha256", "actor_state_dict_sha256",
                "policy_fingerprint_sha256", "checkpoint_fingerprint_sha256"):
            if _HEX64.fullmatch(str(item.get(field, ""))) is None:
                raise StateDependentRecoveryV4Error(
                    f"early_task_policy.{field} is not SHA-256")
        seed_age.update({int(seed): age for seed in seeds})
    mature = _mapping(protocol.get("mature_recovery_policy"),
                      "mature_recovery_policy")
    _require_equal(mature.get("training_step"), 500_000,
                   "mature_recovery_policy.training_step")
    _require_equal(
        mature.get("recovery_library_fingerprint_sha256"),
        "fcfb1fa541acf316f87dacf82b1fdeb9188d7a4b9df7f69544b567fb2c5d1045",
        "mature_recovery_policy.recovery_library_fingerprint_sha256",
    )

    collection = _mapping(protocol.get("collection"), "collection")
    expected_collection_scalars = {
        "artifact_root": "saved/qsafe_development/state_dependent_recovery_v4",
        "groups_per_source_seed": 64,
        "total_groups": 384,
        "max_groups_per_trajectory": 1,
        "settle_seconds": 0.04,
        "settle_policy_steps": 2,
        "total_candidate_replicas": 128,
        "total_candidate_branch_rollouts": 442_368,
        "audit_merge_before_selection": "forbidden",
    }
    for name, expected in expected_collection_scalars.items():
        _require_equal(collection.get(name), expected, f"collection.{name}")
    expected_templates = {
        "cohort_lock_filename": "cohort-lock.json",
        "attempt_shard_filename_template": "source-{source_seed}.attempt-started.json",
        "admission_shard_filename_template": "source-{source_seed}.admission.npz",
        "admission_privileged_shard_filename_template": (
            "source-{source_seed}.admission.privileged.npz"),
        "discovery_shard_filename_template": "source-{source_seed}.discovery.npz",
        "discovery_privileged_shard_filename_template": (
            "source-{source_seed}.discovery.privileged.npz"),
        "audit_shard_filename_template": "source-{source_seed}.audit.npz",
        "audit_privileged_shard_filename_template": (
            "source-{source_seed}.audit.privileged.npz"),
        "collection_report_shard_filename_template": (
            "source-{source_seed}.collection-report.json"),
        "admission_deployable_filename": "admission-ledger-deployable.npz",
        "admission_privileged_filename": "admission-ledger-privileged.npz",
        "admission_merge_report_filename": "admission-merge-report.json",
        "discovery_filename": "discovery-g384.npz",
        "discovery_privileged_filename": "discovery-g384-privileged.npz",
        "discovery_merge_report_filename": "discovery-merge-report.json",
        "selection_lock_filename": "selection-lock.json",
        "audit_consumed_filename": "audit-consumed.json",
        "triage_report_filename": "state-dependent-recovery-stage-a-report.json",
    }
    for name, expected in expected_templates.items():
        _require_equal(collection.get(name), expected, f"collection.{name}")
    _require_equal(collection.get("audit_analysis_input_order"),
                   list(SOURCE_SEEDS), "collection.audit_analysis_input_order")
    _require_equal(collection.get("candidates"),
                   RecoveryBehaviorConfig().manifest_protocol(),
                   "collection.candidates")
    admission = _mapping(collection.get("admission"), "collection.admission")
    for name, expected in {
        "replicas": 32,
        "horizon_policy_steps": 96,
        "accept_min_falls_inclusive": 6,
        "accept_max_falls_inclusive": 26,
        "all_proposals_recorded": True,
        "labels_used_in_effect_estimation": False,
    }.items():
        _require_equal(admission.get(name), expected,
                       f"collection.admission.{name}")
    partition = _mapping(collection.get("replica_partition"),
                         "collection.replica_partition")
    partition_expected = {
        "schema_version": "qsafe.physically_separate_replica_partition.v4_stage_a",
        "assignment_timing": "before_candidate_outcomes",
        "discovery_indices": {"start_inclusive": 0, "stop_exclusive": 64},
        "audit_indices": {"start_inclusive": 64, "stop_exclusive": 128},
        "discovery_replicas": 64,
        "audit_replicas": 64,
        "exhaustive": True,
        "disjoint_seed_domains": True,
        "physical_files": True,
    }
    _require_equal(dict(partition), partition_expected,
                   "collection.replica_partition")

    firewall = _mapping(protocol.get("firewall"), "firewall")
    for name, expected in {
        "selection_uses": ["admission_ledger", "discovery"],
        "selection_forbidden": [
            "audit", "consumed_v2_outcomes", "consumed_v3_outcomes"],
        "primary_selection": (
            "per_state_all_exact_discovery_minima_uniform_expectation"),
        "candidate_column_order_effect_on_ties": "forbidden",
        "selection_lock_no_clobber": True,
        "per_group_ties": "uniform_expectation_all_discovery_minima",
        "audit_consumed_marker_created_before_outcome_read": True,
        "audit_shards_opened_only_after_consumed_marker": True,
        "interrupted_audit_remains_consumed": True,
        "audit_runner_up_policy": "forbidden",
    }.items():
        _require_equal(firewall.get(name), expected, f"firewall.{name}")
    _require_equal(firewall.get("denied_report_crash_recovery"), {
        "entrypoint": "resume-denied-report",
        "reads": [
            "canonical_protocol", "existing_hash_bound_selection_lock"],
        "requires_false": [
            "audit_authorized", "data_gate_pass",
            "discovery_informativeness_pass"],
        "writes": "canonical_stage_A_failure_report_only",
        "repeated_call_requires_byte_identical_existing_report": True,
        "selection_lock_rewrite": "forbidden",
        "audit_path_parse_resolve_stat_hash_map_load_open": "forbidden",
    }, "firewall.denied_report_crash_recovery")

    statistics = _mapping(protocol.get("statistics"), "statistics")
    _require_equal(statistics.get("policy_age_strata"), {
        str(age): list(seeds) for age, seeds in AGE_STRATA.items()},
        "statistics.policy_age_strata")
    bootstrap = _mapping(statistics.get("bootstrap"), "statistics.bootstrap")
    bootstrap_expected = {
        "kind": "hierarchical_policy_age_seed_then_trajectory_group",
        "replicates": 50_000,
        "seed": 20_260_810,
        "rng_bit_generator": "numpy_PCG64",
        "chunk_size": 512,
        "draw_order": (
            "chunk_then_sorted_age_then_slot_seed_vector_C_then_group_matrix_C_order"),
        "quantile_method": "linear",
        "confidence": "one_sided_0.95",
        "resample_policy_age_strata": False,
        "resample_source_seeds_within_stratum": True,
        "resample_complete_groups_within_seed": True,
        "resample_candidates": False,
        "resample_replicas": False,
    }
    _require_equal(dict(bootstrap), bootstrap_expected, "statistics.bootstrap")

    gates = _mapping(protocol.get("triage_gates"), "triage_gates")
    data_gate = _mapping(gates.get("data"), "triage_gates.data")
    data_expected = {
        "independent_groups_exact": 384,
        "unique_source_trajectories_exact": 384,
        "groups_per_required_source_seed_exact": 64,
        "required_source_seeds": list(SOURCE_SEEDS),
        "candidates_per_group_exact": 9,
        "discovery_replicas_exact": 64,
        "audit_replicas_exact": 64,
        "horizon_policy_steps_exact": 96,
        "admission_replicas_exact": 32,
        "admission_falls_inclusive": [6, 26],
        "min_discovery_nominal_risk": 0.15,
        "max_discovery_nominal_risk": 0.75,
        "min_each_policy_age_discovery_nominal_risk": 0.10,
        "max_each_policy_age_discovery_nominal_risk": 0.90,
        "discovery_informativeness_checked_before_audit_open": True,
        "zero_duplicate_state_fingerprints": True,
        "zero_duplicate_trajectory_fingerprints": True,
    }
    _require_equal(dict(data_gate), data_expected, "triage_gates.data")
    primary = _mapping(gates.get("stage_A_primary"),
                       "triage_gates.stage_A_primary")
    primary_expected = {
        "rule": "per_state_all_exact_discovery_minima_uniform_expectation",
        "min_audit_absolute_reduction": 0.10,
        "min_one_sided_95_lcb_reduction": 0.07,
        "min_discovery_to_audit_pair_agreement": 0.70,
        "min_pair_agreement_one_sided_95_lcb": 0.65,
        "require_each_policy_age_positive": True,
        "min_positive_source_seeds": 6,
        "override_by_fixed_candidate": "forbidden",
        "override_by_oracle_or_reanalysis": "forbidden",
    }
    _require_equal(dict(primary), primary_expected,
                   "triage_gates.stage_A_primary")

    stage_a = _mapping(protocol.get("stage_A"), "stage_A")
    _require_equal(stage_a.get("actor_training_seed"), 42,
                   "stage_A.actor_training_seed")
    _require_equal(stage_a.get("source_seeds"), list(SOURCE_SEEDS),
                   "stage_A.source_seeds")
    _require_equal(stage_a.get("result_may_authorize"),
                   "stage_B_model_fit_only", "stage_A.result_may_authorize")
    stage_b = _mapping(protocol.get("stage_B"), "stage_B")
    _require_equal(stage_b.get("starts_only_if"), "stage_A_pass_true",
                   "stage_B.starts_only_if")
    _require_equal(stage_b.get("model_training_triggered_by_stage_A_tool"),
                   False, "stage_B.model_training_triggered_by_stage_A_tool")
    feature = _mapping(stage_b.get("feature_contract"),
                       "stage_B.feature_contract")
    for name, expected in {
        "schema_version": "qsafe.recovery_program_features.v1",
        "model_action_width": 82,
        "candidate_mask_contract": "all_K9_preview_candidates_valid",
        "recovery_library_fingerprint_sha256": (
            "fcfb1fa541acf316f87dacf82b1fdeb9188d7a4b9df7f69544b567fb2c5d1045"),
        "fallback_to_36d": "forbidden",
        "behavior_steps": list(BEHAVIOR_STEPS),
    }.items():
        _require_equal(feature.get(name), expected,
                       f"stage_B.feature_contract.{name}")
    stage_b_actor_training_seeds = {
        "fit": [43, 44, 45, 46],
        "probability_calibration": [47, 48],
        "uncertainty_calibration": [49, 50],
        "selector_calibration": [51, 52],
        "model_test": [53, 54, 55, 56],
    }
    _require_equal(stage_b.get("actor_training_seeds"),
                   stage_b_actor_training_seeds,
                   "stage_B.actor_training_seeds")
    stage_b_source_seeds = {
        "fit": [
            8501, 8502, 8503, 8504, 8511, 8512, 8513, 8514,
            8521, 8522, 8523, 8524,
        ],
        "probability_calibration": [8601, 8602, 8611, 8612, 8621, 8622],
        "uncertainty_calibration": [8631, 8632, 8641, 8642, 8651, 8652],
        "selector_calibration": [8661, 8662, 8671, 8672, 8681, 8682],
        "model_test": [
            8701, 8702, 8703, 8704, 8711, 8712, 8713, 8714,
            8721, 8722, 8723, 8724,
        ],
    }
    _require_equal(stage_b.get("source_seeds"), stage_b_source_seeds,
                   "stage_B.source_seeds")
    stage_b_actor_source_assignment = {
        "rule": (
            "nth_source_seed_in_each_age_block_belongs_to_nth_actor_training_seed"),
        "actor_checkpoint_steps": [25_000, 50_000, 100_000],
        "by_role": {
            "fit": {
                "43": [8501, 8511, 8521],
                "44": [8502, 8512, 8522],
                "45": [8503, 8513, 8523],
                "46": [8504, 8514, 8524],
            },
            "probability_calibration": {
                "47": [8601, 8611, 8621],
                "48": [8602, 8612, 8622],
            },
            "uncertainty_calibration": {
                "49": [8631, 8641, 8651],
                "50": [8632, 8642, 8652],
            },
            "selector_calibration": {
                "51": [8661, 8671, 8681],
                "52": [8662, 8672, 8682],
            },
            "model_test": {
                "53": [8701, 8711, 8721],
                "54": [8702, 8712, 8722],
                "55": [8703, 8713, 8723],
                "56": [8704, 8714, 8724],
            },
        },
    }
    _require_equal(stage_b.get("actor_source_assignment"),
                   stage_b_actor_source_assignment,
                   "stage_B.actor_source_assignment")
    stage_b_groups = {
        "fit": 1536,
        "probability_calibration": 384,
        "uncertainty_calibration": 384,
        "selector_calibration": 384,
        "model_test": 768,
    }
    _require_equal(stage_b.get("groups"), stage_b_groups, "stage_B.groups")
    stage_b_roles = list(stage_b_actor_training_seeds)
    role_replicas_32 = {role: 32 for role in stage_b_roles}
    label_replicas = dict(role_replicas_32)
    label_replicas["model_test"] = 64
    stage_b_replica_domains: dict[str, Any] = {}
    for role in stage_b_roles:
        token = role
        stage_b_replica_domains[role] = {
            "admission_literal_ascii_escaped": (
                "qsafe_state_dependent_recovery_v4_stage_b_"
                f"{token}_admission\\0"),
            "label_literal_ascii_escaped": (
                "qsafe_state_dependent_recovery_v4_stage_b_"
                f"{token}_label\\0"),
        }
    stage_b_replica_domains["all_ten_literals_pairwise_distinct"] = True
    _require_equal(stage_b.get("role_collection_contract"), {
        "role_order": stage_b_roles,
        "groups_per_source_seed": {
            "fit": 128,
            "probability_calibration": 64,
            "uncertainty_calibration": 64,
            "selector_calibration": 64,
            "model_test": 64,
        },
        "admission_replicas_exact": role_replicas_32,
        "label_replicas_exact": label_replicas,
        "admission_falls_inclusive": [6, 26],
        "assignment_timing": "before_any_candidate_outcome",
        "admission_and_label_replica_independence": {
            "physical_files_separate": True,
            "seed_domains_separate": True,
            "crn_ids_disjoint": True,
            "rollout_seeds_disjoint": True,
            "perturbation_seeds_disjoint": True,
            "candidate_seeds_disjoint": True,
            "admission_outcomes_used_as_labels": False,
        },
        "replica_seed_domains": stage_b_replica_domains,
        "five_role_pairwise_disjointness": {
            "dimensions": [
                "policy_training_seed", "actor_checkpoint_sha256",
                "actor_state_dict_sha256", "policy_fingerprint_sha256",
                "checkpoint_fingerprint_sha256", "state_fingerprint_sha256",
                "trajectory_fingerprint_sha256", "crn_id", "rollout_seed",
                "perturbation_seed", "candidate_seed",
            ],
            "all_ten_role_pairs_checked": True,
            "zero_collisions_required": True,
            "proof_report": "stage-b-split-disjointness-report.json",
        },
        "one_shot_report_last": True,
        "no_role_top_up": True,
    }, "stage_B.role_collection_contract")
    _require_equal(stage_b.get("normalization_contract"), {
        "source_role": "fit_only",
        "source_array": "stage-b/fit/labels-r32-deployable.npz",
        "fit_group_ids_only": True,
        "candidate_or_replica_outcome_weighting": "forbidden",
        "caller_supplied_statistics": "forbidden",
        "computed_before_model_test_consumption": True,
        "frozen_before_probability_calibration": True,
        "provenance_fields": [
            "schema_version", "source_role", "source_array_sha256",
            "source_group_ids_sha256", "observation_mean_f4_sha256",
            "observation_std_f4_sha256", "privileged_features_absent",
            "report_sha256",
        ],
        "privileged_features_absent": True,
    }, "stage_B.normalization_contract")

    stage_b_role_paths: dict[str, Any] = {
        "schema_version": "qsafe.stage_b_role_evidence_paths.v1",
        "required_per_role": [
            "attempt_marker", "source_admission_shards",
            "source_label_shards", "source_privileged_label_shards",
            "admission_array", "label_array", "privileged_label_array",
            "step_log", "collection_manifest", "completion_marker", "report",
        ],
    }
    for role in stage_b_roles:
        directory = role.replace("_", "-")
        label_r = label_replicas[role]
        base = f"stage-b/{directory}"
        stage_b_role_paths[role] = {
            "attempt_marker": f"{base}/attempt-started.json",
            "source_admission_shards": (
                f"{base}/source-{{source_seed}}.admission-r32.npz"),
            "source_label_shards": (
                f"{base}/source-{{source_seed}}.labels-r{label_r}.npz"),
            "source_privileged_label_shards": (
                f"{base}/source-{{source_seed}}.labels-r{label_r}.privileged.npz"),
            "admission_array": f"{base}/admission-r32.npz",
            "label_array": f"{base}/labels-r{label_r}-deployable.npz",
            "privileged_label_array": (
                f"{base}/labels-r{label_r}-privileged.npz"),
            "step_log": f"{base}/steps.jsonl",
            "collection_manifest": f"{base}/collection-manifest.json",
            "completion_marker": f"{base}/completed.json",
            "report": f"{base}/report.json",
        }
    _require_equal(stage_b.get("artifacts"), {
        "artifact_root_relative": "stage-b",
        "role_path_schema": stage_b_role_paths,
        "derived_paths": {
            "actor_bank_manifest": "stage-b/actor-bank-manifest.json",
            "split_disjointness_report": (
                "stage-b/stage-b-split-disjointness-report.json"),
            "normalization_report": (
                "stage-b/normalization-fit-only-report.json"),
            "trained_qsafe_artifact": "stage-b/qsafe-artifact",
            "probability_calibration_report": (
                "stage-b/probability-calibration-report.json"),
            "uncertainty_calibration_report": (
                "stage-b/uncertainty-calibration-report.json"),
            "selector_search_report": "stage-b/selector-search-report.json",
            "frozen_selector_bundle": (
                "stage-b/recovery-selector-bundle.json"),
            "matched_random_placebo_bundle": (
                "stage-b/matched-random-placebo-bundle.json"),
            "model_test_commitment_marker": (
                "stage-b/model-test-committed.json"),
            "model_test_consumed_marker": (
                "stage-b/model-test-consumed.json"),
            "stage_report": "state-dependent-recovery-stage-b-report.json",
        },
        "recovery_artifact_authorization": {
            "expected_manifest_sha256_source": (
                "accepted_stage_B_compiler_report"),
            "artifact_self_declared_hash_is_trust_root": False,
            "runtime_expected_manifest_sha256_required": True,
            "authorized_hash_bound_to_live_identity": True,
        },
        "every_file_bound_by_sha256_in_completion_marker": True,
        "reports_published_last_no_clobber": True,
    }, "stage_B.artifacts")
    _require_equal(stage_b.get("model"), {
        "ensemble_members": 5,
        "hidden_widths": [128, 128, 128],
        "epochs": 100,
        "batch_size": 64,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "weight_decay": 0.00001,
        "gradient_clip": 5.0,
        "seed": 20_260_810,
        "member_seed_formula": (
            "20260810_plus_1009_times_member_index_zero_based"),
        "device": "cpu",
        "torch_num_threads": 1,
        "torch_deterministic_algorithms": True,
        "optimizer_betas": [0.9, 0.999],
        "optimizer_epsilon": 0.00000001,
        "optimizer_amsgrad": False,
        "optimizer_foreach": False,
        "optimizer_fused": False,
    }, "stage_B.model")
    _require_equal(stage_b.get("loss"), {
        "absolute_risk_weight": 1.0,
        "state_risk_weight": 0.5,
        "relative_risk_weight": 1.0,
        "ranking_weight": 0.5,
        "ttf_weight": 0.1,
        "max_tilt_weight": 0.1,
        "min_height_weight": 0.1,
    }, "stage_B.loss")
    calibration = _mapping(stage_b.get("calibration"), "stage_B.calibration")
    for name, expected in {
        "probability_temperature_steps": 100,
        "probability_temperature_learning_rate": 0.05,
        "log_temperature_clamp": [-4.0, 4.0],
        "probability_temperature_optimizer": "Adam",
        "probability_temperature_optimizer_betas": [0.9, 0.999],
        "probability_temperature_optimizer_epsilon": 0.00000001,
        "probability_temperature_optimizer_foreach": False,
        "probability_temperature_optimizer_fused": False,
    }.items():
        _require_equal(calibration.get(name), expected,
                       f"stage_B.calibration.{name}")
    conformal = _mapping(
        calibration.get("split_conformal"),
        "stage_B.calibration.split_conformal",
    )
    _require_equal(conformal.get("uncertainty_source_seeds"),
                   [8631, 8632, 8641, 8642, 8651, 8652],
                   "stage_B split-conformal source seeds")
    for name, expected in {
        "risk_upper_familywise_alpha": 0.05,
        "benefit_lower_familywise_alpha": 0.05,
        "nonnominal_options": 8,
        "risk_upper_per_option_alpha": 0.00625,
        "benefit_lower_per_option_alpha": 0.00625,
        "cross_family_joint_error_correction": (
            "none_no_joint_coverage_claim"),
        "joint_sixteen_bound_coverage_claim": "forbidden",
        "selector_intersection_joint_coverage_claim": "forbidden",
    }.items():
        _require_equal(conformal.get(name), expected,
                       f"stage_B split-conformal {name}")
    _require_equal(conformal.get("rng"), "none_deterministic_stable_order",
                   "stage_B split-conformal RNG")
    _require_equal(
        conformal.get("nominal_trigger_score"),
        "predicted_nominal_risk_minus_empirical_nominal_risk",
        "stage_B split-conformal nominal trigger score",
    )
    _require_equal(
        conformal.get("nominal_trigger_rank_rule"),
        "one_based_min_n_ceil_n_plus_1_times_1_minus_alpha",
        "stage_B split-conformal nominal trigger rank rule",
    )
    selector = _mapping(
        calibration.get("selector_search"),
        "stage_B.calibration.selector_search",
    )
    for name, expected in {
        "selector_source_seeds": [8661, 8662, 8671, 8672, 8681, 8682],
        "grid_points_exact": 100,
        "bootstrap_replicates": 50_000,
        "bootstrap_seed": 20_260_811,
        "frozen_before_model_test": True,
        "task_q_gate": "forbidden",
        "ensemble_probability_std_ddof": 0,
        "candidate_choice_order": [
            "lowest_risk_ucb", "largest_benefit_lcb",
            "locked_candidate_index"],
        "empty_eligible_set": "deterministic_nominal_abstain",
        "candidate_mask_contract": "all_K9_preview_candidates_valid",
    }.items():
        _require_equal(selector.get(name), expected,
                       f"stage_B selector.{name}")
    _require_equal(selector.get("candidate_comparators"), {
        "nominal_risk_lcb_trigger": "greater_than_or_equal",
        "minimum_benefit_lcb": "strictly_greater_than",
        "maximum_candidate_risk_ucb": "less_than_or_equal",
        "maximum_ensemble_probability_std": "less_than_or_equal",
        "maximum_first_requested_action_rms_delta": "less_than_or_equal",
        "maximum_first_qtarget_rms_delta": "less_than_or_equal",
    }, "stage_B selector.candidate_comparators")
    _require_equal(calibration.get("frozen_selector_bundle"), {
        "schema_version": "qsafe.recovery_selector_bundle.v1",
        "includes": [
            "signed_conformal_offsets", "selector_config",
            "probability_calibration_report_sha256",
            "uncertainty_calibration_report_sha256",
            "selector_search_report_sha256", "candidate_choice_semantics",
            "ensemble_std_ddof"],
        "exact_top_level_fields": [
            "schema_version", "offsets", "selector_config",
            "calibration_and_search_report_sha256",
            "candidate_choice_semantics", "ensemble_std_ddof",
            "bundle_sha256"],
        "exact_offset_fields": [
            "nominal_lower", "risk_upper", "benefit_lower",
            "calibration_report_sha256"],
        "exact_report_hash_fields": [
            "probability_calibration", "uncertainty_calibration",
            "selector_search"],
        "canonical_sha256_required": True,
        "artifact_provenance_binding_required": True,
        "runtime_caller_override": "forbidden",
    }, "stage_B.calibration.frozen_selector_bundle")
    runtime = _mapping(
        protocol.get("persistent_option_runtime"),
        "persistent_option_runtime",
    )
    _require_equal(runtime.get("states"),
                   ["idle", "option", "spent_until_reset", "terminal"],
                   "persistent_option_runtime.states")
    _require_equal(runtime.get("max_option_starts_per_episode"), 1,
                   "persistent_option_runtime.max_option_starts_per_episode")
    _require_equal(runtime.get("reset_is_only_return_to_idle"), True,
                   "persistent_option_runtime.reset_is_only_return_to_idle")
    for name, expected in {
        "selection_input": "frozen_decision_proof_only",
        "naked_candidate_index_input": "forbidden",
        "first_step_application_must_equal_inference_preview": True,
        "action_projection_must_equal_recovery_library_manifest": True,
        "controller_issued_replay_proof_required": True,
        "controller_issued_runtime_step_proof_required": True,
        "replay_and_step_live_digest_required_before_insert": True,
        "step_replay_cross_field_validation": "exact",
        "actor_counter_snapshot_and_proposal_identity_in_transition_log": True,
        "stage_D_collector_adapter_required_before_first_online_outcome": True,
    }.items():
        _require_equal(runtime.get(name), expected,
                       f"persistent_option_runtime.{name}")
    _require_equal(runtime.get("counter_based_actor_shadow"), {
        "stateful_rng_provider": "forbidden",
        "algorithm": (
            "sha256_payload_digest_as_uint256_little_endian_to_numpy_PCG64"),
        "common_domain_literal_ascii_escaped": (
            "qsafe.recovery_actor_shadow.v1\\0"),
        "stage_C_tag_literal_ascii_escaped": "stage_c\\0",
        "stage_C_key_order": [
            "state_hash_sha256_raw32", "replica_u64le",
            "absolute_step_u64le", "stream_kind_u16le_length_then_ascii",
            "draw_index_u64le"],
        "stage_D_tag_literal_ascii_escaped": "stage_d\\0",
        "stage_D_key_order": [
            "training_seed_u64le", "absolute_exposure_step_u64le",
            "stream_kind_u16le_length_then_ascii", "draw_index_u64le"],
        "integer_range": "unsigned_64_bit",
        "stream_kind_regex": "^[a-z][a-z0-9_]{0,63}$",
        "stream_kind_exact": "nominal_actor",
        "draw_index_exact": 0,
        "pcg64_seed_integer_endianness": "little",
        "actor_noise_draw": (
            "numpy_Generator_standard_normal_shape12_dtype_float32"),
        "actor_provider_contract": (
            "deterministic_external_noise_only_equal_inputs_bit_identical"),
        "static_provider_manifest_and_fingerprint_revalidated_each_consume": True,
        "actor_snapshot_schema_version": "qsafe.actor_snapshot.v1",
        "actor_snapshot_exact_fields": [
            "schema_version", "actor_state_sha256", "actor_weight_version",
            "actor_update_hash_chain_sha256"],
        "actor_snapshot_revalidated_before_between_and_after_equal_input_calls": True,
        "stage_C_snapshot_constant_after_first_proposal": True,
        "stage_D_weight_version_monotonic": True,
        "stage_D_equal_version_requires_identical_snapshot": True,
        "stage_D_version_advance_requires_new_state_and_update_chain": True,
        "proposal_binds_snapshot_manifest_and_fingerprint": True,
        "proposal_audit_binds_counter_payload_seed_noise_action_provider_and_snapshot": True,
        "stage_C_golden_seed_sha256": (
            "125aa845f4dfcadb71bd0ebb16fc23a77a81909db3977268c1aef6c09dd87e34"),
        "stage_D_golden_seed_sha256": (
            "06b437c25e80d0193c61fff7c3c03b29b8886e563519ef414b1de89b948f5f26"),
        "stage_D_golden_noise_f4_sha256": (
            "5b1492577f16cf89821842b250d420101c30d21870daa376b90f93cb4cc55703"),
        "one_actor_draw_per_absolute_step": True,
    }, "persistent_option_runtime.counter_based_actor_shadow")
    placebo = _mapping(
        protocol.get("matched_random_placebo"), "matched_random_placebo")
    for name, expected in {
        "max_intervention_rate_mismatch": 0.02,
        "max_duration_total_variation": 0.05,
        "max_action_distance_ks": 0.10,
        "outcome_based_reweighting": "forbidden",
    }.items():
        _require_equal(placebo.get(name), expected,
                       f"matched_random_placebo.{name}")
    stage_c = _mapping(protocol.get("stage_C"), "stage_C")
    _require_equal(stage_c.get("starts_only_if"), "stage_B_pass_true",
                   "stage_C.starts_only_if")
    _require_equal(stage_c.get("actor_training_seeds"), [57, 58, 59, 60],
                   "stage_C.actor_training_seeds")
    _require_equal(stage_c.get("source_seeds"), [
        8801, 8802, 8803, 8804, 8811, 8812, 8813, 8814,
        8821, 8822, 8823, 8824,
    ], "stage_C.source_seeds")
    _require_equal(stage_c.get("groups_per_source_seed"), 100,
                   "stage_C.groups_per_source_seed")
    _require_equal(stage_c.get("total_groups"), 1200,
                   "stage_C.total_groups")
    _require_equal(stage_c.get("arms"), ["nominal", "qsafe", "placebo"],
                   "stage_C.arms")
    _require_equal(stage_c.get("arm_semantics"), {
        "nominal": "frozen_stage_C_sac_actor_only",
        "qsafe": "frozen_stage_C_sac_actor_plus_frozen_qsafe_selector",
        "placebo": (
            "frozen_stage_C_sac_actor_plus_frozen_matched_random_kernel"),
    }, "stage_C.arm_semantics")
    _require_equal(stage_c.get("actor_source_assignment"), {
        "rule": (
            "nth_source_seed_in_each_age_block_belongs_to_nth_actor_training_seed"),
        "actor_checkpoint_steps": [25_000, 50_000, 100_000],
        "mapping": {
            "57": [8801, 8811, 8821],
            "58": [8802, 8812, 8822],
            "59": [8803, 8813, 8823],
            "60": [8804, 8814, 8824],
        },
    }, "stage_C.actor_source_assignment")
    _require_equal(stage_c.get("actor_provider_identity"), {
        "mode": "fixed_checkpoint_per_actor_seed_and_age",
        "manifest_schema_version": "qsafe.stage_c_frozen_actor_bank.v1",
        "required_identity_fields": [
            "actor_training_seed", "checkpoint_step", "source_seed",
            "checkpoint_path", "actor_sha256", "actor_state_dict_sha256",
            "policy_fingerprint_sha256", "checkpoint_fingerprint_sha256",
            "policy_config_sha256", "generator_commit",
        ],
        "frozen_before_first_admission_outcome": True,
        "provider_mutation_after_freeze": "forbidden",
        "proposal_proof_binds_actor_fingerprint": True,
    }, "stage_C.actor_provider_identity")
    _require_equal(stage_c.get("replica_contract"), {
        "admission_replicas_exact": 32,
        "admission_falls_inclusive": [6, 26],
        "evaluation_replicas_per_arm_exact": 64,
        "evaluation_arms_exact": 3,
        "total_evaluation_branch_rollouts": 230_400,
        "admission_seed_domain_literal_ascii_escaped": (
            "qsafe_state_dependent_recovery_v4_stage_c_admission\\0"),
        "evaluation_seed_domain_literal_ascii_escaped": (
            "qsafe_state_dependent_recovery_v4_stage_c_evaluation\\0"),
        "admission_and_evaluation_seed_domains_disjoint": True,
        "admission_and_evaluation_physical_files_separate": True,
        "admission_crn_never_reused_for_evaluation": True,
        "evaluation_crn_shared_across_three_arms_within_state_replica": True,
        "evaluation_replica_indivisible_across_arms": True,
        "assignment_before_outcomes": True,
    }, "stage_C.replica_contract")
    _require_equal(stage_c.get("realized_placebo_balance"), {
        "recompute_from_raw_step_log": True,
        "population": "all_1200_complete_paired_state_groups",
        "weighting": (
            "equal_actor_seed_then_equal_age_source_then_equal_state_group"),
        "intervention_rate_denominator": "all_complete_state_groups",
        "duration_histogram_conditioning": "realized_interventions_only",
        "first_action_distance_ecdf_conditioning": (
            "realized_interventions_only"),
        "first_action_distance_metric": (
            "requested_action_rms_delta_from_nominal"),
        "zero_realized_interventions_in_qsafe_or_placebo": "fail",
        "max_absolute_intervention_rate_mismatch_inclusive": 0.02,
        "max_duration_histogram_total_variation_inclusive": 0.05,
        "max_first_action_distance_ks_inclusive": 0.10,
        "outcome_based_reweighting": "forbidden",
        "report_boolean_without_recomputation": "forbidden",
    }, "stage_C.realized_placebo_balance")
    _require_equal(stage_c.get("artifacts"), {
        "artifact_root_relative": "stage-c",
        "attempt_marker": "stage-c/attempt-started.json",
        "actor_bank_manifest": "stage-c/frozen-actor-bank-manifest.json",
        "source_admission_shards": (
            "stage-c/source-{source_seed}.admission-r32.npz"),
        "source_admission_privileged_shards": (
            "stage-c/source-{source_seed}.admission-r32.privileged.npz"),
        "admission_array": "stage-c/admission-r32.npz",
        "admission_privileged_array": (
            "stage-c/admission-r32-privileged.npz"),
        "admission_step_log": "stage-c/admission-steps.jsonl",
        "source_evaluation_shards": (
            "stage-c/source-{source_seed}.paired-r64.npz"),
        "source_evaluation_privileged_shards": (
            "stage-c/source-{source_seed}.paired-r64.privileged.npz"),
        "evaluation_array": "stage-c/paired-arms-r64.npz",
        "evaluation_privileged_array": (
            "stage-c/paired-arms-r64-privileged.npz"),
        "evaluation_array_shape": [1200, 3, 64],
        "evaluation_step_log": "stage-c/paired-steps.jsonl",
        "state_roster": "stage-c/state-roster.json",
        "qsafe_artifact_manifest": "stage-c/qsafe-artifact-manifest.json",
        "selector_bundle": "stage-c/recovery-selector-bundle.json",
        "placebo_bundle": "stage-c/matched-random-placebo-bundle.json",
        "placebo_balance_report": (
            "stage-c/realized-placebo-balance-report.json"),
        "completion_marker": "stage-c/completed.json",
        "stage_report": "state-dependent-recovery-stage-c-report.json",
        "every_file_bound_by_sha256_in_completion_marker": True,
        "report_published_last_no_clobber": True,
    }, "stage_C.artifacts")
    for name, expected in {
        "min_qsafe_reduction": 0.05,
        "min_qsafe_reduction_lcb": 0.03,
        "require_all_ages_positive": True,
        "require_all_actor_training_seeds_positive": True,
        "require_all_source_seeds_positive": True,
        "require_improved_pairs_gt_worsened_pairs": True,
        "require_qsafe_vs_placebo_lcb_gt_zero": True,
        "arm_order": ["nominal", "qsafe", "placebo"],
    }.items():
        _require_equal(stage_c.get(name), expected, f"stage_C.{name}")
    stage_c_bootstrap = _mapping(stage_c.get("bootstrap"),
                                 "stage_C.bootstrap")
    _require_equal(dict(stage_c_bootstrap), {
        "replicates": 50_000,
        "seed": 20_260_813,
        "rng_bit_generator": "numpy_PCG64",
        "quantile_method": "linear",
        "outer_unit": "actor_training_seed",
        "inner_unit": "complete_age_source_state_group",
        "same_draws_for_qsafe_vs_nominal_and_qsafe_vs_placebo": True,
    }, "stage_C.bootstrap")
    stage_d = _mapping(protocol.get("stage_D"), "stage_D")
    _require_equal(stage_d.get("starts_only_if"), "stage_C_pass_true",
                   "stage_D.starts_only_if")
    _require_equal(stage_d.get("speed_mps"), 0.30, "stage_D.speed_mps")
    _require_equal(stage_d.get("training_seeds"), list(range(201, 225)),
                   "stage_D.training_seeds")
    _require_equal(stage_d.get("fixed_policy_steps_per_seed_arm"), 500_000,
                   "stage_D.fixed_policy_steps_per_seed_arm")
    _require_equal(stage_d.get("arms"), [
        "pure_sac", "sac_frozen_qsafe", "sac_matched_random"],
        "stage_D.arms")
    _require_equal(stage_d.get("exposures_exact"), 72,
                   "stage_D.exposures_exact")
    _require_equal(stage_d.get("complete_paired_arms_per_training_seed"), 3,
                   "stage_D.complete_paired_arms_per_training_seed")
    _require_equal(stage_d.get("actor_provider_identity"), {
        "mode": "evolving_online_sac_actor",
        "manifest_schema_version": (
            "qsafe.stage_d_evolving_actor_provider.v1"),
        "initialization_identity_fields": [
            "training_seed", "initial_actor_sha256",
            "initial_reward_critic_sha256", "initial_target_critic_sha256",
            "initial_optimizer_sha256", "initial_reward_normalizer_sha256",
            "policy_config_sha256", "generator_commit",
        ],
        "all_three_arms_bit_identical_initialization_per_seed": True,
        "evolving_identity_fields_per_policy_step": [
            "training_seed", "arm", "absolute_exposure_step",
            "actor_weight_version", "actor_state_sha256",
            "actor_update_hash_chain_sha256", "nominal_proposal_sha256",
        ],
        "actor_update_hash_chain_covers_every_optimizer_update": True,
        "proposal_proof_binds_current_actor_fingerprint": True,
        "frozen_qsafe_selector_and_placebo_hashes_constant_across_exposure": (
            True),
    }, "stage_D.actor_provider_identity")
    lane_domain = (
        b"qsafe_state_dependent_recovery_v4_stage_d_lane_assignment\0")
    lane_master_seed = bytes.fromhex(
        "c35d46474a098e7f56179ac2cfca0c9072ebcb55da1756a3e3c6cbf6b0aa435e")
    lane_training_seeds = list(range(201, 225))
    lane_bits = [
        hashlib.sha256(
            lane_domain
            + lane_master_seed
            + seed.to_bytes(8, "little", signed=False)
            + (0).to_bytes(8, "little", signed=False)
        ).digest()[0] & 1
        for seed in lane_training_seeds
    ]
    lane_bits_sha256 = hashlib.sha256(bytes(lane_bits)).hexdigest()
    _require_equal(stage_d.get("randomized_execution_lane_assignment"), {
        "logical_pair": ["pure_sac", "sac_frozen_qsafe"],
        "randomized_physical_lanes": ["A", "B"],
        "placebo_physical_lane": "C",
        "assignment_timing": (
            "before_any_stage_D_simulator_step_or_outcome"),
        "master_seed_source": (
            "os_csprng_openssl_rand_256_before_any_stage_D_outcome"),
        "master_seed_hex": lane_master_seed.hex(),
        "master_seed_sha256": hashlib.sha256(lane_master_seed).hexdigest(),
        "domain_literal_ascii_escaped": (
            "qsafe_state_dependent_recovery_v4_stage_d_lane_assignment\\0"),
        "domain_hex": lane_domain.hex(),
        "bit_derivation": (
            "sha256_domain_then_master_seed_raw32_then_training_seed_u64le_"
            "then_draw_index_u64le_zero_take_digest_byte0_lsb"),
        "fair_bit_model": (
            "independent_bernoulli_half_from_csprng_seeded_domain_separated_hash"),
        "training_seed_order": lane_training_seeds,
        "expected_assignment_bits": lane_bits,
        "expected_assignment_bits_sha256": lane_bits_sha256,
        "bit_zero_mapping": {"A": "pure_sac", "B": "sac_frozen_qsafe"},
        "bit_one_mapping": {"A": "sac_frozen_qsafe", "B": "pure_sac"},
        "placebo_lane_assignment": (
            "fixed_preassigned_not_in_primary_pair_randomization"),
        "assignment_manifest": "stage-d/execution-lane-assignment.json",
        "assignment_manifest_required_fields": [
            "schema_version", "master_seed_sha256", "domain_hex",
            "training_seed_order", "assignment_bits",
            "assignment_bits_sha256", "per_seed_lane_mapping", "placebo_lane",
            "created_before_first_outcome", "manifest_sha256",
        ],
        "assignment_manifest_no_clobber_before_first_outcome": True,
        "outcome_dependent_reassignment": "forbidden",
    }, "stage_D.randomized_execution_lane_assignment")
    for name, expected in {
        "min_relative_fall_reduction": 0.20,
        "min_falls_per_1000_step_reduction": 0.40,
        "require_seed_cluster_lcb_gt_zero": True,
        "require_treatment_vs_placebo_lcb_gt_zero": True,
    }.items():
        _require_equal(stage_d.get(name), expected, f"stage_D.{name}")
    _require_equal(stage_d.get("paired_seed_bootstrap"), {
        "replicates": 50_000,
        "seed": 20_260_814,
        "rng_bit_generator": "numpy_PCG64",
        "quantile_method": "linear",
        "one_sided_lcb_quantile": 0.05,
        "resampling_unit": "complete_training_seed_three_arm_tuple",
        "resample_seed_indices_shape": [50_000, 24],
        "same_draws_for_all_online_metrics": True,
    }, "stage_D.paired_seed_bootstrap")
    _require_equal(stage_d.get("exact_sign_flip"), {
        "paired_units": "24_training_seeds",
        "assignments_exact": 16_777_216,
        "difference_per_seed": (
            "pure_sac_falls_per_1000_minus_sac_frozen_qsafe_falls_per_1000"),
        "observed_statistic": "arithmetic_mean_of_24_paired_differences",
        "favorable_direction": "larger_positive_reduction",
        "null_randomization": (
            "independently_swap_pure_sac_and_sac_frozen_qsafe_labels_within_each_seed"),
        "randomization_source": (
            "randomized_execution_lane_assignment_24_fair_bits"),
        "enumeration_order": (
            "unsigned_mask_0_through_2_pow_24_minus_1_seed201_is_lsb"),
        "zero_differences": "retained_and_both_sign_assignments_counted",
        "ties_to_observed_statistic": "included_in_tail",
        "p_value_formula": (
            "count_randomized_statistics_greater_than_or_equal_to_observed_"
            "divided_by_16777216"),
        "plus_one_correction": "false_full_enumeration",
        "required_assumption": (
            "within_seed_treatment_label_exchangeability_under_sharp_null"),
        "max_one_sided_p_inclusive": 0.05,
    }, "stage_D.exact_sign_flip")
    _require_equal(stage_d.get("relative_fall_gate"), {
        "baseline_metric": (
            "pooled_pure_sac_falls_per_1000_fixed_exposure_steps"),
        "treatment_metric": (
            "pooled_sac_frozen_qsafe_falls_per_1000_fixed_exposure_steps"),
        "formula": "baseline_minus_treatment_divided_by_baseline",
        "baseline_less_than_or_equal_to_zero": "gate_false",
    }, "stage_D.relative_fall_gate")
    _require_equal(stage_d.get("return_noninferiority_gate"), {
        "metric_per_seed_arm": (
            "1000_times_sum_transition_reward_divided_by_500000_policy_steps"),
        "aggregate": "arithmetic_mean_over_24_training_seeds",
        "margin_fraction_of_baseline_magnitude": 0.05,
        "baseline_magnitude_floor": 1.0,
        "signed_safe_formula": (
            "treatment_minus_baseline_greater_than_or_equal_to_minus_0.05_"
            "times_max_abs_baseline_1"),
        "ratio_division": "forbidden",
    }, "stage_D.return_noninferiority_gate")
    _require_equal(stage_d.get("velocity_gate"), {
        "per_step_error": (
            "absolute_body_frame_forward_velocity_minus_0.30_mps"),
        "metric_per_seed_arm": "arithmetic_mean_over_500000_policy_steps",
        "aggregate": "arithmetic_mean_over_24_training_seeds",
        "treatment_minus_baseline_max_inclusive_mps": 0.03,
        "reset_standup_and_settle_steps": "excluded",
    }, "stage_D.velocity_gate")
    _require_equal(stage_d.get("deadline_gate"), {
        "deadline_seconds": 0.02,
        "miss_comparator": (
            "inference_wall_time_strictly_greater_than_deadline"),
        "numerator": "sac_frozen_qsafe_policy_steps_with_deadline_miss",
        "denominator": "24_times_500000_sac_frozen_qsafe_policy_steps",
        "maximum_rate_strict": 0.001,
        "nonfinite_or_missing_timing": "gate_false",
    }, "stage_D.deadline_gate")
    _require_equal(stage_d.get("realized_placebo_balance"), {
        "recompute_from_raw_intervention_logs": True,
        "population": "all_24_complete_qsafe_and_placebo_exposures",
        "weighting": (
            "equal_training_seed_then_equal_policy_step_or_realized_option"),
        "intervention_rate_denominator": (
            "24_times_500000_policy_steps_per_arm"),
        "duration_histogram_conditioning": "realized_option_starts_only",
        "first_action_distance_ecdf_conditioning": (
            "realized_option_starts_only"),
        "first_action_distance_metric": (
            "requested_action_rms_delta_from_nominal"),
        "zero_realized_option_starts_in_qsafe_or_placebo": "fail",
        "max_absolute_intervention_rate_mismatch_inclusive": 0.02,
        "max_duration_histogram_total_variation_inclusive": 0.05,
        "max_first_action_distance_ks_inclusive": 0.10,
        "outcome_based_reweighting": "forbidden",
        "report_boolean_without_recomputation": "forbidden",
    }, "stage_D.realized_placebo_balance")
    _require_equal(stage_d.get("artifacts"), {
        "artifact_root_relative": "stage-d",
        "global_attempt_marker": "stage-d/attempt-started.json",
        "execution_lane_assignment_manifest": (
            "stage-d/execution-lane-assignment.json"),
        "seed_arm_root_template": (
            "stage-d/seed-{training_seed}/arm-{arm}"),
        "per_exposure_paths": {
            "attempt_marker": (
                "stage-d/seed-{training_seed}/arm-{arm}/attempt-started.json"),
            "initial_identity_manifest": (
                "stage-d/seed-{training_seed}/arm-{arm}/initial-identity.json"),
            "transition_log": (
                "stage-d/seed-{training_seed}/arm-{arm}/transitions.parquet"),
            "episode_log": (
                "stage-d/seed-{training_seed}/arm-{arm}/episodes.parquet"),
            "fall_and_exposure_array": (
                "stage-d/seed-{training_seed}/arm-{arm}/fall-exposure.npz"),
            "actor_update_hash_chain": (
                "stage-d/seed-{training_seed}/arm-{arm}/"
                "actor-update-hash-chain.jsonl"),
            "rng_stream_manifest": (
                "stage-d/seed-{training_seed}/arm-{arm}/rng-streams.json"),
            "intervention_step_log": (
                "stage-d/seed-{training_seed}/arm-{arm}/interventions.jsonl"),
            "timing_log": (
                "stage-d/seed-{training_seed}/arm-{arm}/timing.npz"),
            "completion_marker": (
                "stage-d/seed-{training_seed}/arm-{arm}/completed.json"),
        },
        "per_seed_paired_completion_marker": (
            "stage-d/seed-{training_seed}/paired-three-arm-completed.json"),
        "exposure_roster": "stage-d/exposure-roster-72.json",
        "paired_metric_array": "stage-d/paired-seed-metrics-24x3.npz",
        "sign_flip_report": "stage-d/exact-sign-flip-report.json",
        "bootstrap_report": "stage-d/paired-seed-bootstrap-report.json",
        "realized_placebo_balance_report": (
            "stage-d/realized-placebo-balance-report.json"),
        "completion_marker": "stage-d/completed-72-exposures.json",
        "stage_report": "state-dependent-recovery-stage-d-report.json",
        "every_exposure_file_bound_by_sha256": True,
        "every_exposure_completion_marker_no_clobber_report_last": True,
        "stage_report_published_last_no_clobber": True,
    }, "stage_D.artifacts")
    _require_equal(stage_d.get("objective1_pass_requires"),
                   "stage_A_and_stage_B_and_stage_C_and_stage_D",
                   "stage_D.objective1_pass_requires")
    fall_exposure = _mapping(
        protocol.get("fall_and_exposure_contract"),
        "fall_and_exposure_contract",
    )
    _require_equal(
        fall_exposure.get(
            "offline_branch_and_online_simulator_fall_predicate_identical"),
        True,
        "fall_and_exposure_contract simulator fall-predicate identity",
    )
    _require_equal(
        fall_exposure.get("hardware_supervisor_deployment_equivalence_claim"),
        "forbidden",
        "fall_and_exposure_contract hardware equivalence claim",
    )
    online_exposure = _mapping(
        fall_exposure.get("online_exposure"),
        "fall_and_exposure_contract.online_exposure",
    )
    _require_equal(online_exposure.get("policy_steps_per_seed_arm_exact"),
                   500_000,
                   "fall_and_exposure_contract.online_exposure steps")
    compiler = _mapping(
        protocol.get("authorization_compiler"), "authorization_compiler")
    for name, expected in {
        "schema_version": "qsafe.objective1_authorization_compiler.v4",
        "legacy_cli": "forbidden",
        "caller_supplied_gate_booleans": "forbidden",
        "naked_boolean_authorization_inputs": "forbidden",
        "canonical_paths_only": True,
        "recompute_metrics_from_immutable_arrays_and_step_logs": True,
        "verify_report_last_markers_hashes_rosters_and_exposure": True,
        "thresholds_and_bootstrap_from_this_protocol_only": True,
        "pass_expression": (
            "stage_A_and_stage_B_and_stage_C_and_fresh_030_online"),
        "stage_order": ["stage_A", "stage_B", "stage_C", "stage_D"],
        "phase1_and_phase2_bits_written_together": True,
        "malformed_or_missing_evidence": "raise_without_publication",
    }.items():
        _require_equal(compiler.get(name), expected,
                       f"authorization_compiler.{name}")
    _require_equal(compiler.get("canonical_report_filenames"), {
        "stage_A": "state-dependent-recovery-stage-a-report.json",
        "stage_B": "state-dependent-recovery-stage-b-report.json",
        "stage_C": "state-dependent-recovery-stage-c-report.json",
        "stage_D": "state-dependent-recovery-stage-d-report.json",
        "output": "objective1-authorization-report.json",
    }, "authorization_compiler.canonical_report_filenames")
    _require_equal(compiler.get("first_valid_failure_short_circuit"), {
        "enabled": True,
        "validate_current_stage_report_and_recompute_current_stage_gate": True,
        "require_every_later_stage_canonical_artifact_absent": True,
        "later_stage_absence_uses_protocol_inventory_only": True,
        "absence_definition": (
            "os_path_lexists_false_for_every_expanded_canonical_path_and_"
            "template_instance"),
        "later_stage_report_or_marker_present": (
            "malformed_raise_without_publication"),
        "valid_failure_output": {
            "objective1_pass": False,
            "phase1_pass": False,
            "phase2_authorized": False,
        },
        "do_not_require_later_stage_reports": True,
    }, "authorization_compiler.first_valid_failure_short_circuit")
    _require_equal(compiler.get("canonical_artifact_inventories"), {
        "stage_A": "collection_and_firewall_paths",
        "stage_B": "stage_B.artifacts",
        "stage_C": "stage_C.artifacts",
        "stage_D": "stage_D.artifacts",
    }, "authorization_compiler.canonical_artifact_inventories")

    authorization = _mapping(protocol.get("authorization"), "authorization")
    authorization_expected = {
        "stage_A_may_authorize_stage_B": True,
        "stage_B_authorized_before_audit": False,
        "stage_B_training_triggered": False,
        "paired_closed_loop_authorized": False,
        "online_training_authorized": False,
        "objective1_pass": False,
        "phase2_authorized": False,
    }
    _require_equal(dict(authorization), authorization_expected,
                   "authorization")

    protocol_copy = _json_copy(protocol, "protocol")
    return {
        "protocol": protocol_copy,
        "protocol_contract_sha256": contract_sha256,
        "collection": protocol_copy["collection"],
        "data_gate": protocol_copy["triage_gates"]["data"],
        "required_seeds": SOURCE_SEEDS,
        "seed_age": seed_age,
        "age_strata": AGE_STRATA,
        "groups": 384,
        "groups_per_seed": 64,
        "admission_replicas": 32,
        "discovery_replicas": 64,
        "audit_replicas": 64,
        "horizon": 96,
    }


def validate_state_dependent_recovery_v4_protocol(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact V4 A/B/C/D protocol without touching artifacts."""
    spec = _validate_protocol(protocol)
    return {
        "protocol_name": PROTOCOL_NAME,
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "actor_training_seed": 42,
        "claim_scope": "seed42_fixed_actor_conditional_mechanism_only",
        "required_source_seeds": list(SOURCE_SEEDS),
        "groups": 384,
        "candidates": 9,
        "admission_replicas": 32,
        "discovery_replicas": 64,
        "audit_replicas": 64,
        "horizon_policy_steps": 96,
        "bootstrap_replicates": 50_000,
        "bootstrap_seed": 20_260_810,
        "stage_B_training_triggered": False,
        "phase2_authorized": False,
    }


def validate_state_dependent_collection_readiness(
    *,
    protocol: Mapping[str, Any],
    collection_report_paths: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Validate report-last source commitments without touching audit NPZs."""
    spec = _validate_protocol(protocol)
    try:
        result = _v3._collection_readiness(
            collection_report_paths, protocol=protocol, spec=spec)
    except _v3.ClosedLoopRecoveryTriageError as exc:
        raise StateDependentRecoveryV4Error(str(exc)) from exc
    return _json_copy(result, "collection readiness")


def expected_v4_seed_manifest() -> dict[str, Any]:
    """Return a fresh copy of the exact four-field V4 RNG manifest."""
    return seed_derivation_manifest(
        seed_domain=SEED_DOMAIN,
        seed_role_tags=SEED_ROLE_TAGS,
        seed_algorithm=SEED_ALGORITHM,
    )


def _validate_admission_seed_leaf(
    ledger: AdmissionLedger,
    *,
    source_seed: int,
    file_sha256: str,
    content_sha256: str,
    protocol_file_sha256: str,
    protocol_contract_sha256: str,
) -> np.ndarray:
    validation = ledger.validate()
    if ledger.path is None or _v3._sha256_file(ledger.path) != file_sha256 or (
            validation["content_sha256"] != content_sha256):
        raise StateDependentRecoveryV4Error(
            "admission leaf differs from its report-last commitment")
    if ledger.manifest.get("protocol_sha256") != protocol_file_sha256 or (
            ledger.manifest.get("protocol_contract_sha256") !=
            protocol_contract_sha256) or ledger.manifest.get(
                "source_seed") != source_seed:
        raise StateDependentRecoveryV4Error(
            "admission leaf protocol/source identity drifted")
    proposals = int(validation["proposals"])
    proposal_index = np.asarray(ledger["proposal_index"])
    if proposal_index.shape != (proposals,) or not np.array_equal(
            proposal_index, np.arange(proposals, dtype=proposal_index.dtype)):
        raise StateDependentRecoveryV4Error(
            "admission proposal indices are not the original local order")
    for index in range(proposals):
        bundle, _ = role_randomness(
            source_seed=source_seed,
            proposal_index=index,
            replicas=32,
            role="admission",
            seed_domain=SEED_DOMAIN,
            role_tags=SEED_ROLE_TAGS,
            seed_algorithm=SEED_ALGORITHM,
        )
        for name, expected in (
            ("admission_crn_id", bundle.crn_id),
            ("admission_rollout_seed", bundle.rollout_seed),
            ("admission_perturbation_seed", bundle.perturbation_seed),
        ):
            if not np.array_equal(np.asarray(ledger[name])[index], expected):
                raise StateDependentRecoveryV4Error(
                    f"admission {name} does not use the V4 seed domain")
    accepted = np.asarray(ledger["accepted"], dtype=bool)
    positions = np.flatnonzero(accepted)
    if len(positions) != 64:
        raise StateDependentRecoveryV4Error(
            "admission leaf does not contain exactly 64 accepted positions")
    return positions


def _validate_discovery_seed_contract_before_lock(
    *,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    readiness: Mapping[str, Any],
    admission_path: Path,
    discovery_path: Path,
) -> None:
    # Merge completion JSON is safe to read before discovery.  It also binds
    # the canonical merged filenames to the report-last leaf commitments.
    try:
        _v3._validate_merge_completion_reports(
            protocol=protocol,
            spec=spec,
            readiness=readiness,
            admission_path=admission_path,
            discovery_path=discovery_path,
            admission=None,
            discovery=None,
        )
    except _v3.ClosedLoopRecoveryTriageError as exc:
        raise StateDependentRecoveryV4Error(str(exc)) from exc

    local_positions: dict[int, np.ndarray] = {}
    admission_commitments = readiness["role_commitments"]["admission"]
    for seed, commitment in zip(SOURCE_SEEDS, admission_commitments, strict=True):
        path = Path(str(commitment["path"]))
        _reject_protected_components(path, "admission leaf")
        ledger = AdmissionLedger.load(path)
        local_positions[seed] = _validate_admission_seed_leaf(
            ledger,
            source_seed=seed,
            file_sha256=str(commitment["file_sha256"]),
            content_sha256=str(commitment["content_sha256"]),
            protocol_file_sha256=str(readiness["protocol_file_sha256"]),
            protocol_contract_sha256=str(spec["protocol_contract_sha256"]),
        )

    discovery = GroupedBranchDataset.load(discovery_path)
    collection_protocol = _mapping(
        discovery.manifest.get("collection_protocol"),
        "discovery.collection_protocol",
    )
    _require_equal(collection_protocol.get("version"),
                   "qsafe.state_dependent_recovery.collection.v4_stage_a",
                   "discovery collection version")
    _require_equal(collection_protocol.get("seed_derivation"),
                   expected_v4_seed_manifest(),
                   "discovery seed-derivation manifest")
    _require_equal(discovery.manifest.get("split"),
                   "state_dependent_recovery_v4_stage_a_discovery",
                   "discovery split")
    recovery_binding = _mapping(
        discovery.manifest.get("recovery_program"),
        "discovery.recovery_program",
    )
    if set(recovery_binding) != {"manifest", "fingerprint_sha256"}:
        raise StateDependentRecoveryV4Error(
            "discovery recovery-program binding fields are not exact")
    recovery_manifest = _mapping(
        recovery_binding.get("manifest"),
        "discovery.recovery_program.manifest",
    )
    recovery_fingerprint = str(recovery_binding.get("fingerprint_sha256", ""))
    if recovery_fingerprint != canonical_protocol_sha256(recovery_manifest) or (
            recovery_manifest.get("candidate_protocol") != _CANDIDATE_PROTOCOL):
        raise StateDependentRecoveryV4Error(
            "discovery recovery-program manifest/fingerprint is invalid")
    _require_equal(
        recovery_fingerprint,
        spec["protocol"]["mature_recovery_policy"][
            "recovery_library_fingerprint_sha256"],
        "discovery recovery-program fingerprint",
    )
    source_seed = np.asarray(discovery["source_seed"], dtype=np.int64)
    if source_seed.shape != (384,):
        raise StateDependentRecoveryV4Error("discovery has the wrong group count")
    row = 0
    for seed in SOURCE_SEEDS:
        for proposal_index in local_positions[seed]:
            if int(source_seed[row]) != seed:
                raise StateDependentRecoveryV4Error(
                    "discovery source order differs from accepted admission")
            _, observed_discovery = role_randomness(
                source_seed=seed,
                proposal_index=int(proposal_index),
                replicas=64,
                role="discovery",
                seed_domain=SEED_DOMAIN,
                role_tags=SEED_ROLE_TAGS,
                seed_algorithm=SEED_ALGORITHM,
            )
            _, observed_audit = role_randomness(
                source_seed=seed,
                proposal_index=int(proposal_index),
                replicas=64,
                role="audit",
                seed_domain=SEED_DOMAIN,
                role_tags=SEED_ROLE_TAGS,
                seed_algorithm=SEED_ALGORITHM,
            )
            for name, expected in (
                ("crn_id", observed_discovery.crn_id),
                ("rollout_seed", observed_discovery.rollout_seed),
                ("perturbation_seed", observed_discovery.perturbation_seed),
                ("candidate_seed", observed_discovery.candidate_seed),
                ("preassigned_audit_crn_id", observed_audit.crn_id),
                ("preassigned_audit_rollout_seed", observed_audit.rollout_seed),
                ("preassigned_audit_perturbation_seed",
                 observed_audit.perturbation_seed),
                ("preassigned_audit_candidate_seed",
                 observed_audit.candidate_seed),
            ):
                value = np.asarray(discovery[name])[row]
                if not np.array_equal(value, expected):
                    raise StateDependentRecoveryV4Error(
                        f"discovery {name} does not use the exact V4 seed domain")
            row += 1
    if row != 384:
        raise StateDependentRecoveryV4Error(
            "accepted admission positions do not exhaust discovery rows")


@contextmanager
def _patched_v3_protocol_validator() -> Iterator[None]:
    """Reuse reviewed V3 artifact parsing with the exact V4 spec injected."""
    with _V3_VALIDATOR_PATCH_LOCK:
        original = _v3._validate_protocol
        _v3._validate_protocol = _validate_protocol
        try:
            yield
        finally:
            _v3._validate_protocol = original


def _publish_discovery_failure_report(
    *,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    lock: Mapping[str, Any],
    clean_commit: str,
    protocol_file_sha256: str,
) -> tuple[Path, str, dict[str, Any]]:
    """Publish or verify the audit-denied terminal report without audit I/O.

    The idempotent verification branch closes the only Stage-A crash window:
    the immutable selection lock may already exist while the report-last
    terminal record does not.  An existing report is accepted only when its
    bytes are exactly the canonical bytes that this lock would publish.
    """
    collection = spec["collection"]
    data_gate = _mapping(lock.get("data_gate"), "selection_lock.data_gate")
    informativeness = _mapping(
        data_gate.get("discovery_informativeness"),
        "selection_lock.discovery_informativeness",
    )
    if lock.get("audit_authorized") is not False or data_gate.get(
            "pass") is not False or informativeness.get("pass") is not False:
        raise StateDependentRecoveryV4Error(
            "failure report requires an audit-denied informativeness lock")
    if lock.get("generator_commit") != clean_commit or lock.get(
            "protocol_file_sha256") != protocol_file_sha256 or lock.get(
                "selection_semantics") != SELECTION_SEMANTICS:
        raise StateDependentRecoveryV4Error(
            "failure lock differs from clean protocol/selection semantics")
    report_path = _artifact_root(protocol) / str(
        collection["triage_report_filename"])
    try:
        report_file = _v3._artifact_path(
            report_path,
            protocol=protocol,
            expected_filename=str(collection["triage_report_filename"]),
            name="stage_A_failure_report_path",
        )
    except _v3.ClosedLoopRecoveryTriageError as exc:
        raise StateDependentRecoveryV4Error(str(exc)) from exc
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": protocol_file_sha256,
        "selection_lock_sha256": lock["selection_lock_sha256"],
        "audit_identifier": lock["audit_identifier"],
        "claim_scope": "seed42_fixed_actor_conditional_mechanism_only",
        "cross_actor_generalization_claim": False,
        "selection_semantics": _json_copy(
            SELECTION_SEMANTICS, "selection semantics"),
        "data_gate": _json_copy(data_gate, "data gate"),
        "stage_A_primary": {
            "tested": False,
            "pass": None,
            "reason": "discovery_informativeness_failed_before_audit_open",
        },
        "audit_opened_for_analysis": False,
        "audit_consumed": False,
        "decision": "no_model_training",
        "stage_B_authorized": False,
        "model_training_authorized": False,
        "model_training_triggered": False,
        "paired_closed_loop_authorized": False,
        "online_training_authorized": False,
        "objective1_pass": False,
        "phase2_authorized": False,
        "analysis_commit": clean_commit,
        "analysis_worktree_clean": True,
        "authorization_note": (
            "Discovery failed its preregistered informativeness gate; audit "
            "remained unopened and V4 stops before model training."),
    }
    expected_payload = _v3._canonical_json_bytes(report)
    try:
        report_sha256 = _v3._atomic_no_clobber_json(report_file, report)
    except _v3.ClosedLoopRecoveryTriageError as publish_exc:
        # A concurrent or resumed invocation is safe only if it observes the
        # exact canonical terminal report.  This reads the report path only;
        # no audit path is derived, parsed, or probed here.
        try:
            observed_payload = _v3._read_regular_bytes_once(
                report_file, "existing V4 Stage-A failure report")
        except _v3.ClosedLoopRecoveryTriageError:
            raise StateDependentRecoveryV4Error(str(publish_exc)) from publish_exc
        if observed_payload != expected_payload:
            raise StateDependentRecoveryV4Error(
                "existing V4 Stage-A failure report differs from the "
                "canonical audit-denied decision") from publish_exc
        report_sha256 = hashlib.sha256(observed_payload).hexdigest()
    return report_file, report_sha256, report


_AUDIT_DENIED_LOCK_FIELDS = frozenset({
    "schema_version",
    "protocol_name",
    "protocol_contract_sha256",
    "protocol_file_sha256",
    "generator_commit",
    "candidate_library_sha256",
    "policy_bundle_sha256",
    "created_at_utc",
    "input_artifacts",
    "candidate_order",
    "selected_global_candidate",
    "group_selection",
    "group_selection_sha256",
    "replica_partition",
    "replica_partition_sha256",
    "collection_readiness_sha256",
    "collection_readiness_manifest",
    "merge_readiness_sha256",
    "merge_readiness_manifest",
    "expected_audit_shards",
    "data_gate",
    "bootstrap",
    "triage_gates",
    "audit_identifier",
    "audit_authorized",
    "audit_runner_up_policy",
    "selection_semantics",
})
_READINESS_ROLES = (
    "admission",
    "admission_privileged",
    "discovery",
    "discovery_privileged",
    "audit",
    "audit_privileged",
)
_GROUP_SELECTION_FIELDS = frozenset({
    "group_index",
    "group_id",
    "state_hash",
    "trajectory_id",
    "source_seed",
    "policy_age",
    "admission_falls",
    "discovery_candidate_risk",
    "discovery_minimizer_indices",
    "discovery_minimizer_names",
    "uniform_weights",
})
_REPLICA_PARTITION_VECTOR_FIELDS = (
    "admission_crn_ids",
    "admission_rollout_seeds",
    "admission_perturbation_seeds",
    "discovery_crn_ids",
    "discovery_rollout_seeds",
    "discovery_perturbation_seeds",
    "audit_crn_ids",
    "audit_rollout_seeds",
    "audit_perturbation_seeds",
)
_REPLICA_PARTITION_FIELDS = frozenset({
    "group_index",
    "group_id",
    *_REPLICA_PARTITION_VECTOR_FIELDS,
    "discovery_candidate_seed",
    "audit_candidate_seed",
})


def _lock_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str] | set[str],
    name: str,
) -> None:
    if set(value) != set(expected):
        raise StateDependentRecoveryV4Error(
            f"audit-denied selection lock {name} has extra or missing fields")


def _lock_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise StateDependentRecoveryV4Error(
            f"audit-denied selection lock {name} must be lowercase SHA-256")
    return value


def _lock_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)):
        raise StateDependentRecoveryV4Error(
            f"audit-denied selection lock {name} must be an integer")
    result = int(value)
    if result < minimum:
        raise StateDependentRecoveryV4Error(
            f"audit-denied selection lock {name} must be at least {minimum}")
    return result


def _lock_v4_seed(value: object, name: str) -> int:
    result = _lock_integer(value, name, minimum=1 << 63)
    if result >= 1 << 64:
        raise StateDependentRecoveryV4Error(
            f"audit-denied selection lock {name} exceeds uint64")
    return result


def _lock_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StateDependentRecoveryV4Error(
            f"audit-denied selection lock {name} must be nonempty text")
    return value


def _validate_audit_denied_readiness_and_inputs(
    lock: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    """Validate self-contained pre-audit artifact identities without paths.

    Path strings carried by the lock are treated as opaque JSON values.  This
    function deliberately never constructs a ``Path``, resolves a target, or
    calls any filesystem primitive for a role artifact.
    """
    protocol = spec["protocol"]
    collection = spec["collection"]
    inputs = _mapping(lock.get("input_artifacts"),
                      "selection_lock.input_artifacts")
    _lock_exact_fields(inputs, {"admission", "discovery"}, "input_artifacts")
    admission_input = _mapping(
        inputs.get("admission"), "selection_lock.input_artifacts.admission")
    discovery_input = _mapping(
        inputs.get("discovery"), "selection_lock.input_artifacts.discovery")
    _lock_exact_fields(
        admission_input,
        {"filename", "file_sha256", "content_sha256", "proposal_count"},
        "input_artifacts.admission",
    )
    _lock_exact_fields(
        discovery_input,
        {"filename", "file_sha256", "content_sha256"},
        "input_artifacts.discovery",
    )
    if admission_input.get("filename") != collection[
            "admission_deployable_filename"] or discovery_input.get(
                "filename") != collection["discovery_filename"]:
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock input artifact filename mismatch")
    for role, artifact in (
        ("admission", admission_input),
        ("discovery", discovery_input),
    ):
        _lock_hash(artifact.get("file_sha256"),
                   f"input_artifacts.{role}.file_sha256")
        _lock_hash(artifact.get("content_sha256"),
                   f"input_artifacts.{role}.content_sha256")

    readiness = _mapping(
        lock.get("collection_readiness_manifest"),
        "selection_lock.collection_readiness_manifest",
    )
    _lock_exact_fields(readiness, {
        "schema_version",
        "protocol_name",
        "protocol_contract_sha256",
        "protocol_file_sha256",
        "generator_commit",
        "artifact_root",
        "required_source_seeds",
        "source_records",
        "role_commitments",
    }, "collection_readiness_manifest")
    if readiness.get("schema_version") != (
            _v3.COLLECTION_READINESS_SCHEMA_VERSION) or readiness.get(
                "protocol_name") != PROTOCOL_NAME or readiness.get(
                    "protocol_contract_sha256") != spec[
                        "protocol_contract_sha256"] or readiness.get(
                            "protocol_file_sha256") != lock[
                                "protocol_file_sha256"] or readiness.get(
                                    "generator_commit") != lock[
                                        "generator_commit"] or readiness.get(
                                            "required_source_seeds") != list(
                                                spec["required_seeds"]):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock readiness identity mismatch")
    if readiness.get("artifact_root") != str(_artifact_root(protocol)):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock readiness artifact root mismatch")
    readiness_sha256 = _lock_hash(
        lock.get("collection_readiness_sha256"),
        "collection_readiness_sha256",
    )
    if readiness_sha256 != _v3.canonical_sha256(readiness):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock collection readiness digest mismatch")

    roles = _mapping(
        readiness.get("role_commitments"),
        "selection_lock.collection_readiness_manifest.role_commitments",
    )
    _lock_exact_fields(roles, set(_READINESS_ROLES),
                       "collection readiness role_commitments")
    role_records: dict[str, list[Mapping[str, Any]]] = {}
    for role in _READINESS_ROLES:
        raw_records = roles.get(role)
        if not isinstance(raw_records, list) or len(raw_records) != len(
                spec["required_seeds"]):
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock readiness role count mismatch")
        checked: list[Mapping[str, Any]] = []
        for ordinal, (raw, seed) in enumerate(zip(
                raw_records, spec["required_seeds"], strict=True)):
            record = _mapping(
                raw, f"selection_lock.role_commitments.{role}[{ordinal}]")
            _lock_exact_fields(record, {
                "ordinal", "source_seed", "policy_training_step", "path",
                "file_sha256", "content_sha256",
            }, f"role_commitments.{role}[{ordinal}]")
            if _lock_integer(record.get("ordinal"), "role ordinal") != ordinal or (
                    _lock_integer(record.get("source_seed"), "role source_seed")
                    != seed) or _lock_integer(
                        record.get("policy_training_step"),
                        "role policy_training_step",
                        minimum=1,
                    ) != spec["seed_age"][seed]:
                raise StateDependentRecoveryV4Error(
                    "audit-denied selection lock readiness role order mismatch")
            # Deliberately opaque: no lexical parse, normalization, resolution,
            # stat, or open is permitted for any role path in resume mode.
            _lock_text(record.get("path"), f"role_commitments.{role}.path")
            _lock_hash(record.get("file_sha256"),
                       f"role_commitments.{role}.file_sha256")
            _lock_hash(record.get("content_sha256"),
                       f"role_commitments.{role}.content_sha256")
            checked.append(record)
        role_records[role] = checked

    source_records = readiness.get("source_records")
    if not isinstance(source_records, list) or len(source_records) != len(
            spec["required_seeds"]):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock readiness source record count mismatch")
    proposal_count = 0
    validation_fields = {
        "admission": {"proposals", "accepted", "content_sha256"},
        "admission_privileged": {"proposals", "content_sha256"},
        "discovery": {
            "groups", "max_candidates", "replicas", "horizon_steps",
            "content_sha256",
        },
        "discovery_privileged": {"groups", "content_sha256"},
        "audit": {
            "groups", "max_candidates", "replicas", "horizon_steps",
            "content_sha256",
        },
        "audit_privileged": {"groups", "content_sha256"},
    }
    for ordinal, (raw, seed) in enumerate(zip(
            source_records, spec["required_seeds"], strict=True)):
        source = _mapping(
            raw, f"selection_lock.readiness.source_records[{ordinal}]")
        _lock_exact_fields(source, {
            "ordinal", "source_seed", "policy_training_step",
            "protocol_file_sha256", "protocol_contract_sha256",
            "generator_commit", "collection_report_path",
            "collection_report_file_sha256", "cohort_lock",
            "attempt_marker", "outputs", "validations",
        }, f"readiness.source_records[{ordinal}]")
        if source.get("ordinal") != ordinal or source.get("source_seed") != seed or (
                source.get("policy_training_step") != spec["seed_age"][seed]) or (
                    source.get("protocol_file_sha256") != lock[
                        "protocol_file_sha256"]) or source.get(
                            "protocol_contract_sha256") != spec[
                                "protocol_contract_sha256"] or source.get(
                                    "generator_commit") != lock[
                                        "generator_commit"]:
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock readiness source identity mismatch")
        _lock_text(source.get("collection_report_path"),
                   "readiness.collection_report_path")
        _lock_hash(source.get("collection_report_file_sha256"),
                   "readiness.collection_report_file_sha256")
        for marker_name in ("cohort_lock", "attempt_marker"):
            marker = _mapping(
                source.get(marker_name),
                f"selection_lock.readiness.{marker_name}",
            )
            _lock_exact_fields(
                marker, {"path", "file_sha256", "contract_sha256"},
                f"readiness.{marker_name}",
            )
            _lock_text(marker.get("path"), f"readiness.{marker_name}.path")
            _lock_hash(marker.get("file_sha256"),
                       f"readiness.{marker_name}.file_sha256")
            _lock_hash(marker.get("contract_sha256"),
                       f"readiness.{marker_name}.contract_sha256")

        outputs = _mapping(
            source.get("outputs"), "selection_lock.readiness.outputs")
        validations = _mapping(
            source.get("validations"), "selection_lock.readiness.validations")
        _lock_exact_fields(outputs, set(_READINESS_ROLES),
                           "readiness.outputs")
        _lock_exact_fields(validations, set(_READINESS_ROLES),
                           "readiness.validations")
        for role in _READINESS_ROLES:
            output = _mapping(
                outputs.get(role), f"selection_lock.readiness.outputs.{role}")
            _lock_exact_fields(
                output, {"path", "file_sha256", "content_sha256"},
                f"readiness.outputs.{role}",
            )
            commitment = role_records[role][ordinal]
            if dict(output) != {
                "path": commitment["path"],
                "file_sha256": commitment["file_sha256"],
                "content_sha256": commitment["content_sha256"],
            }:
                raise StateDependentRecoveryV4Error(
                    "audit-denied selection lock readiness output commitment "
                    "mismatch")
            validation = _mapping(
                validations.get(role),
                f"selection_lock.readiness.validations.{role}",
            )
            _lock_exact_fields(
                validation, validation_fields[role],
                f"readiness.validations.{role}",
            )
            if validation.get("content_sha256") != output[
                    "content_sha256"]:
                raise StateDependentRecoveryV4Error(
                    "audit-denied selection lock validation/content mismatch")
            if role == "admission":
                proposals = _lock_integer(
                    validation.get("proposals"), "admission proposals",
                    minimum=1)
                accepted = _lock_integer(
                    validation.get("accepted"), "admission accepted")
                if accepted != spec["groups_per_seed"] or proposals < accepted or (
                        proposals > int(collection[
                            "max_proposals_per_source_seed"])):
                    raise StateDependentRecoveryV4Error(
                        "audit-denied selection lock admission validation mismatch")
                proposal_count += proposals
            elif role == "admission_privileged":
                if _lock_integer(
                        validation.get("proposals"),
                        "admission privileged proposals",
                        minimum=1,
                ) != _lock_integer(
                    validations["admission"]["proposals"],
                    "admission proposals",
                    minimum=1,
                ):
                    raise StateDependentRecoveryV4Error(
                        "audit-denied selection lock privileged proposal mismatch")
            elif role in ("discovery", "audit"):
                expected_replicas = (
                    spec["discovery_replicas"] if role == "discovery"
                    else spec["audit_replicas"])
                expected_values = {
                    "groups": spec["groups_per_seed"],
                    "max_candidates": len(CANDIDATE_NAMES),
                    "replicas": expected_replicas,
                    "horizon_steps": spec["horizon"],
                }
                if any(validation.get(name) != expected
                       for name, expected in expected_values.items()):
                    raise StateDependentRecoveryV4Error(
                        "audit-denied selection lock role validation mismatch")
            elif validation.get("groups") != spec["groups_per_seed"]:
                raise StateDependentRecoveryV4Error(
                    "audit-denied selection lock privileged group mismatch")

    if admission_input.get("proposal_count") != proposal_count:
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock admission proposal identity mismatch")
    expected_audit = lock.get("expected_audit_shards")
    if not isinstance(expected_audit, list) or expected_audit != role_records[
            "audit"]:
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock audit commitment mismatch")

    merge = _mapping(
        lock.get("merge_readiness_manifest"),
        "selection_lock.merge_readiness_manifest",
    )
    _lock_exact_fields(merge, {
        "schema_version", "protocol_contract_sha256",
        "collection_readiness_sha256", "admission_merge_report",
        "discovery_merge_report",
    }, "merge_readiness_manifest")
    if merge.get("schema_version") != (
            "qsafe.closed_loop_recovery_triage.merge_readiness.v1") or (
                merge.get("protocol_contract_sha256") != spec[
                    "protocol_contract_sha256"]) or merge.get(
                        "collection_readiness_sha256") != readiness_sha256:
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock merge readiness identity mismatch")
    merge_sha256 = _lock_hash(
        lock.get("merge_readiness_sha256"), "merge_readiness_sha256")
    if merge_sha256 != _v3.canonical_sha256(merge):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock merge readiness digest mismatch")
    for role, field in (
        ("admission", "admission_merge_report"),
        ("discovery", "discovery_merge_report"),
    ):
        record = _mapping(
            merge.get(field), f"selection_lock.merge_readiness.{field}")
        _lock_exact_fields(record, {
            "path", "file_sha256", "output_file_sha256",
            "output_content_sha256",
        }, f"merge_readiness.{field}")
        _lock_text(record.get("path"), f"merge_readiness.{field}.path")
        _lock_hash(record.get("file_sha256"),
                   f"merge_readiness.{field}.file_sha256")
        if record.get("output_file_sha256") != inputs[role][
                "file_sha256"] or record.get(
                    "output_content_sha256") != inputs[role][
                        "content_sha256"]:
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock merge/input artifact mismatch")


def _validate_audit_denied_selection_records(
    lock: Mapping[str, Any],
    spec: Mapping[str, Any],
    data_gate: Mapping[str, Any],
) -> None:
    groups = lock.get("group_selection")
    partitions = lock.get("replica_partition")
    if not isinstance(groups, list) or len(groups) != spec["groups"] or not (
            isinstance(partitions, list) and len(partitions) == spec["groups"]):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock group/replica length mismatch")
    if lock.get("group_selection_sha256") != _v3.canonical_sha256(groups) or (
            lock.get("replica_partition_sha256") !=
            _v3.canonical_sha256(partitions)):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock group/replica commitment mismatch")

    risks = np.empty(
        (spec["groups"], len(CANDIDATE_NAMES)), dtype=np.float64)
    source_seed = np.empty(spec["groups"], dtype=np.int64)
    group_ids: list[str] = []
    state_hashes: list[str] = []
    trajectory_ids: list[str] = []
    seed_domains: dict[str, list[int]] = {
        name: [] for name in _REPLICA_PARTITION_VECTOR_FIELDS
    }
    seed_domains.update({
        "discovery_candidate_seed": [],
        "audit_candidate_seed": [],
    })
    vector_lengths = {
        "admission_crn_ids": spec["admission_replicas"],
        "admission_rollout_seeds": spec["admission_replicas"],
        "admission_perturbation_seeds": spec["admission_replicas"],
        "discovery_crn_ids": spec["discovery_replicas"],
        "discovery_rollout_seeds": spec["discovery_replicas"],
        "discovery_perturbation_seeds": spec["discovery_replicas"],
        "audit_crn_ids": spec["audit_replicas"],
        "audit_rollout_seeds": spec["audit_replicas"],
        "audit_perturbation_seeds": spec["audit_replicas"],
    }
    admission_lower, admission_upper = map(
        int, spec["data_gate"]["admission_falls_inclusive"])
    for index, (raw_group, raw_partition) in enumerate(zip(
            groups, partitions, strict=True)):
        group = _mapping(
            raw_group, f"selection_lock.group_selection[{index}]")
        partition = _mapping(
            raw_partition, f"selection_lock.replica_partition[{index}]")
        _lock_exact_fields(group, _GROUP_SELECTION_FIELDS,
                           f"group_selection[{index}]")
        _lock_exact_fields(partition, _REPLICA_PARTITION_FIELDS,
                           f"replica_partition[{index}]")
        if group.get("group_index") != index or partition.get(
                "group_index") != index:
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock group order mismatch")
        group_id = _lock_text(group.get("group_id"),
                              f"group_selection[{index}].group_id")
        if partition.get("group_id") != group_id:
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock group/replica identity mismatch")
        state_hash = _lock_hash(
            group.get("state_hash"), f"group_selection[{index}].state_hash")
        trajectory_id = _lock_text(
            group.get("trajectory_id"),
            f"group_selection[{index}].trajectory_id",
        )
        seed = _lock_integer(
            group.get("source_seed"),
            f"group_selection[{index}].source_seed",
        )
        if seed not in spec["required_seeds"] or group.get(
                "policy_age") != spec["seed_age"][seed]:
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock source seed/policy age mismatch")
        admission_falls = _lock_integer(
            group.get("admission_falls"),
            f"group_selection[{index}].admission_falls",
        )
        if not admission_lower <= admission_falls <= admission_upper:
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock admission falls out of bounds")
        try:
            row = np.asarray(
                group.get("discovery_candidate_risk"), dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock discovery risks are invalid") from exc
        if row.shape != (len(CANDIDATE_NAMES),) or not np.all(
                np.isfinite(row)) or np.any(row < 0.0) or np.any(row > 1.0) or (
                    not np.all(
                        row * spec["discovery_replicas"] == np.rint(
                            row * spec["discovery_replicas"]))):
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock discovery risks are invalid")
        raw_winners = group.get("discovery_minimizer_indices")
        if not isinstance(raw_winners, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_winners):
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock minimizer indices are invalid")
        winners = np.asarray(raw_winners, dtype=np.int64)
        expected_winners = np.flatnonzero(row == np.min(row))
        if not np.array_equal(winners, expected_winners) or group.get(
                "discovery_minimizer_names") != [
                    CANDIDATE_NAMES[value] for value in expected_winners]:
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock per-state minimizers mismatch")
        expected_weights = [1.0 / len(expected_winners)] * len(
            expected_winners)
        if group.get("uniform_weights") != expected_weights:
            raise StateDependentRecoveryV4Error(
                "audit-denied selection lock per-state weights mismatch")

        for field, count in vector_lengths.items():
            raw_values = partition.get(field)
            if not isinstance(raw_values, list) or len(raw_values) != count or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    or value < 1 << 63 or value >= 1 << 64
                    for value in raw_values) or len(
                        set(raw_values)) != count:
                raise StateDependentRecoveryV4Error(
                    f"audit-denied selection lock {field} is invalid")
            seed_domains[field].extend(raw_values)
        for field in ("discovery_candidate_seed", "audit_candidate_seed"):
            seed_domains[field].append(_lock_v4_seed(
                partition.get(field), f"replica_partition.{field}"))

        risks[index] = row
        source_seed[index] = seed
        group_ids.append(group_id)
        state_hashes.append(state_hash)
        trajectory_ids.append(trajectory_id)

    if any(len(set(values)) != spec["groups"] for values in (
            group_ids, state_hashes, trajectory_ids)):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock group identities are not unique")
    if set(map(int, source_seed)) != set(spec["required_seeds"]) or any(
            np.count_nonzero(source_seed == seed) != spec["groups_per_seed"]
            for seed in spec["required_seeds"]):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock source-seed composition mismatch")
    for field, values in seed_domains.items():
        if len(set(values)) != len(values):
            raise StateDependentRecoveryV4Error(
                f"audit-denied selection lock {field} is not globally unique")
    domains = [set(values) for values in seed_domains.values()]
    if any(domains[left].intersection(domains[right])
           for left in range(len(domains))
           for right in range(left + 1, len(domains))):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock seed domains overlap")

    global_scores = []
    for candidate in range(1, len(CANDIDATE_NAMES)):
        per_seed = [
            float(np.mean(
                risks[source_seed == seed, candidate], dtype=np.float64))
            for seed in spec["required_seeds"]
        ]
        global_scores.append(float(np.mean(per_seed, dtype=np.float64)))
    expected_global_index = 1 + int(np.argmin(
        np.asarray(global_scores, dtype=np.float64)))
    expected_global_table = [{
        "candidate_index": candidate,
        "candidate_name": CANDIDATE_NAMES[candidate],
        "equal_seed_discovery_risk": global_scores[candidate - 1],
    } for candidate in range(1, len(CANDIDATE_NAMES))]
    global_choice = _mapping(
        lock.get("selected_global_candidate"),
        "selection_lock.selected_global_candidate",
    )
    _lock_exact_fields(global_choice, {
        "candidate_index", "candidate_name", "selection_scope",
        "exact_tie_break", "discovery_candidate_table",
    }, "selected_global_candidate")
    if global_choice.get("candidate_index") != expected_global_index or (
            global_choice.get("candidate_name") != CANDIDATE_NAMES[
                expected_global_index]) or global_choice.get(
                    "selection_scope") != "eight_nonnominal_candidates" or (
                        global_choice.get("exact_tie_break") !=
                        "locked_candidate_order") or global_choice.get(
                            "discovery_candidate_table") != expected_global_table:
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock global selection mismatch")

    informativeness = _v3._discovery_informativeness(
        risks[:, 0], source_seed, spec)
    if informativeness.get("pass") is not False or data_gate.get(
            "discovery_informativeness") != informativeness:
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock informativeness record mismatch")


def _read_audit_denied_selection_lock(
    path: Path,
    *,
    expected_sha256: str,
    spec: Mapping[str, Any],
    clean_commit: str,
    protocol_file_sha256: str,
) -> dict[str, Any]:
    """Read one failure lock without using the audit-authorized V3 reader."""
    if _HEX64.fullmatch(expected_sha256) is None:
        raise StateDependentRecoveryV4Error(
            "expected_selection_lock_sha256 must be lowercase SHA-256")
    try:
        payload = _v3._read_regular_bytes_once(
            path, "V4 audit-denied selection lock")
    except _v3.ClosedLoopRecoveryTriageError as exc:
        raise StateDependentRecoveryV4Error(str(exc)) from exc
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise StateDependentRecoveryV4Error(
            "selection-lock file hash differs from the required hash")
    try:
        lock = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StateDependentRecoveryV4Error(
            "could not parse audit-denied selection lock") from exc
    if not isinstance(lock, dict) or payload != _v3._canonical_json_bytes(lock):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock is not canonical JSON")
    _lock_exact_fields(lock, _AUDIT_DENIED_LOCK_FIELDS, "top-level mapping")

    protocol = spec["protocol"]
    collection = spec["collection"]
    expected_identity = {
        "schema_version": _v3.SELECTION_LOCK_SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": protocol_file_sha256,
        "generator_commit": clean_commit,
        "candidate_library_sha256": _v3.canonical_sha256(
            collection["candidates"]),
        "policy_bundle_sha256": _v3.canonical_sha256({
            "policy_config": protocol["policy_config"],
            "early_task_policies": protocol["early_task_policies"],
            "mature_recovery_policy": protocol["mature_recovery_policy"],
        }),
        "candidate_order": list(CANDIDATE_NAMES),
        "bootstrap": protocol["statistics"]["bootstrap"],
        "triage_gates": protocol["triage_gates"],
        "selection_semantics": SELECTION_SEMANTICS,
        "audit_authorized": False,
        "audit_runner_up_policy": "forbidden",
    }
    for name, expected in expected_identity.items():
        if lock.get(name) != expected:
            raise StateDependentRecoveryV4Error(
                f"audit-denied selection lock {name} mismatch")
    if _HEX64.fullmatch(str(lock.get("audit_identifier", ""))) is None:
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock has an invalid audit identifier")
    created_at = lock.get("created_at_utc")
    try:
        created = datetime.fromisoformat(created_at) if isinstance(
            created_at, str) else None
    except ValueError as exc:
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock created_at_utc is invalid") from exc
    if created is None or created.utcoffset() != timezone.utc.utcoffset(None):
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock created_at_utc is not UTC")

    data_gate = _mapping(lock.get("data_gate"), "selection_lock.data_gate")
    _lock_exact_fields(data_gate, {
        "structural_contract_pass",
        "independent_groups",
        "unique_state_fingerprints",
        "unique_trajectory_fingerprints",
        "groups_per_source_seed",
        "required_source_seeds",
        "candidates",
        "admission_replicas",
        "discovery_replicas",
        "audit_replicas_preassigned",
        "horizon_policy_steps",
        "discovery_informativeness",
        "pass",
    }, "data_gate")
    expected_gate_values = {
        "structural_contract_pass": True,
        "independent_groups": spec["groups"],
        "unique_state_fingerprints": spec["groups"],
        "unique_trajectory_fingerprints": spec["groups"],
        "groups_per_source_seed": spec["groups_per_seed"],
        "required_source_seeds": list(spec["required_seeds"]),
        "candidates": len(CANDIDATE_NAMES),
        "admission_replicas": spec["admission_replicas"],
        "discovery_replicas": spec["discovery_replicas"],
        "audit_replicas_preassigned": spec["audit_replicas"],
        "horizon_policy_steps": spec["horizon"],
        "pass": False,
    }
    for name, expected in expected_gate_values.items():
        if data_gate.get(name) != expected:
            raise StateDependentRecoveryV4Error(
                f"audit-denied selection lock data_gate.{name} mismatch")
    informativeness = _mapping(
        data_gate.get("discovery_informativeness"),
        "selection_lock.discovery_informativeness",
    )
    if informativeness.get("pass") is not False:
        raise StateDependentRecoveryV4Error(
            "audit-denied selection lock is not an informativeness failure")
    _validate_audit_denied_readiness_and_inputs(lock, spec)
    _validate_audit_denied_selection_records(lock, spec, data_gate)

    result = _json_copy(lock, "audit-denied selection lock")
    result["selection_lock_sha256"] = expected_sha256
    return result


def resume_state_dependent_discovery_failure_report(
    *,
    protocol: Mapping[str, Any],
    selection_lock_path: str | os.PathLike[str],
    expected_selection_lock_sha256: str,
) -> dict[str, Any]:
    """Recover report-last publication after an audit-denied lock crash.

    This operation has no audit-path argument and never derives an audit path.
    It can only reproduce the terminal ``no_model_training`` report already
    determined by a hash-bound, structurally valid, uninformative lock.
    """
    spec = _validate_protocol(protocol)
    clean_commit, protocol_file_sha256 = (
        _require_clean_head_protocol_binding())
    collection = spec["collection"]
    try:
        lock_file = _v3._artifact_path(
            selection_lock_path,
            protocol=protocol,
            expected_filename=str(collection["selection_lock_filename"]),
            name="selection_lock_path",
        )
    except _v3.ClosedLoopRecoveryTriageError as exc:
        raise StateDependentRecoveryV4Error(str(exc)) from exc
    lock = _read_audit_denied_selection_lock(
        lock_file,
        expected_sha256=expected_selection_lock_sha256,
        spec=spec,
        clean_commit=clean_commit,
        protocol_file_sha256=protocol_file_sha256,
    )
    _require_unchanged_clean_binding(
        clean_commit, protocol_file_sha256,
        "before resumed failure-report publication",
    )
    report_file, report_sha256, report = _publish_discovery_failure_report(
        protocol=protocol,
        spec=spec,
        lock=lock,
        clean_commit=clean_commit,
        protocol_file_sha256=protocol_file_sha256,
    )
    _require_unchanged_clean_binding(
        clean_commit, protocol_file_sha256,
        "during resumed failure-report publication",
    )
    return {
        "selection_lock": str(lock_file),
        "selection_lock_sha256": expected_selection_lock_sha256,
        "stage_A_failure_report": str(report_file),
        "stage_A_failure_report_sha256": report_sha256,
        "audit_authorized": False,
        "audit_opened_for_analysis": False,
        "audit_consumed": False,
        "decision": report["decision"],
        "stage_B_authorized": False,
        "model_training_authorized": False,
        "objective1_pass": False,
        "phase2_authorized": False,
    }


def create_state_dependent_selection_lock(
    *,
    protocol: Mapping[str, Any],
    admission_path: str | os.PathLike[str],
    discovery_path: str | os.PathLike[str],
    collection_report_paths: Sequence[str | os.PathLike[str]],
    selection_lock_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create the no-clobber discovery lock without touching audit shards."""
    spec = _validate_protocol(protocol)
    clean_commit, protocol_file_sha256 = (
        _require_clean_head_protocol_binding())
    collection = spec["collection"]
    try:
        admission_file = _v3._artifact_path(
            admission_path,
            protocol=protocol,
            expected_filename=str(collection["admission_deployable_filename"]),
            name="admission_path",
        )
        discovery_file = _v3._artifact_path(
            discovery_path,
            protocol=protocol,
            expected_filename=str(collection["discovery_filename"]),
            name="discovery_path",
        )
        readiness = _v3._collection_readiness(
            collection_report_paths, protocol=protocol, spec=spec)
        if readiness.get("generator_commit") != clean_commit:
            raise StateDependentRecoveryV4Error(
                "collection readiness differs from the current clean HEAD")
        if readiness.get("protocol_file_sha256") != protocol_file_sha256:
            raise StateDependentRecoveryV4Error(
                "collection readiness differs from the canonical protocol file")
        _validate_discovery_seed_contract_before_lock(
            protocol=protocol,
            spec=spec,
            readiness=readiness,
            admission_path=admission_file,
            discovery_path=discovery_file,
        )
        with _patched_v3_protocol_validator():
            result = _v3.create_selection_lock(
                protocol=protocol,
                admission_path=admission_file,
                discovery_path=discovery_file,
                collection_report_paths=collection_report_paths,
                selection_lock_path=selection_lock_path,
                selection_semantics=SELECTION_SEMANTICS,
            )
    except (_v3.ClosedLoopRecoveryTriageError, OSError, ValueError) as exc:
        if isinstance(exc, StateDependentRecoveryV4Error):
            raise
        raise StateDependentRecoveryV4Error(str(exc)) from exc
    _require_unchanged_clean_binding(
        clean_commit, protocol_file_sha256, "during selection-lock creation")
    if result.get("selection_semantics") != SELECTION_SEMANTICS:
        raise StateDependentRecoveryV4Error(
            "persisted selection lock lacks the exact V4 primary semantics")
    result["primary_selection"] = SELECTION_SEMANTICS["primary_selection"]
    result["selected_global_candidate_is_diagnostic_only"] = True
    audit_authorized = result.get("audit_authorized")
    if not isinstance(audit_authorized, bool):
        raise StateDependentRecoveryV4Error(
            "persisted selection lock lacks a boolean audit authorization")
    if not audit_authorized:
        report_file, report_sha256, _ = _publish_discovery_failure_report(
            protocol=protocol,
            spec=spec,
            lock=result,
            clean_commit=clean_commit,
            protocol_file_sha256=protocol_file_sha256,
        )
        result["stage_A_failure_report"] = str(report_file)
        result["stage_A_failure_report_sha256"] = report_sha256
        result["decision"] = "no_model_training"
    return result


def _locked_audit_paths_before_consumption(
    values: Sequence[str | os.PathLike[str]],
    *,
    protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> list[Path]:
    """Validate audit spellings and commitments without any filesystem call."""
    if isinstance(values, (str, bytes, os.PathLike)):
        raise StateDependentRecoveryV4Error(
            "audit_paths must be the ordered six-path sequence")
    supplied = list(values)
    if len(supplied) != 6:
        raise StateDependentRecoveryV4Error(
            "audit_paths must contain exactly six physical shards")
    root = _artifact_root(protocol)
    expected = [
        root / str(spec["collection"]["audit_shard_filename_template"]).format(
            source_seed=seed)
        for seed in SOURCE_SEEDS
    ]
    normalized = []
    for ordinal, (raw, canonical) in enumerate(zip(
            supplied, expected, strict=True)):
        path = _absolute_repo_path(raw)
        _reject_protected_components(path, f"audit_paths[{ordinal}]")
        if path != canonical:
            raise StateDependentRecoveryV4Error(
                "audit shard paths must follow the canonical source order")
        normalized.append(path)
    commitments = lock.get("expected_audit_shards")
    if not isinstance(commitments, list) or len(commitments) != 6:
        raise StateDependentRecoveryV4Error(
            "selection lock lacks exhaustive audit commitments")
    for ordinal, (path, seed, raw) in enumerate(zip(
            normalized, SOURCE_SEEDS, commitments, strict=True)):
        record = _mapping(raw, f"expected_audit_shards[{ordinal}]")
        if record.get("ordinal") != ordinal or record.get(
                "source_seed") != seed or _absolute_repo_path(str(
                    record.get("path", ""))) != path or any(
                        _HEX64.fullmatch(str(record.get(field, ""))) is None
                        for field in ("file_sha256", "content_sha256")):
            raise StateDependentRecoveryV4Error(
                "audit path/order/hash commitment differs from selection lock")
    return normalized


def _evaluate_stage_a_risks(
    *,
    discovery_risk: np.ndarray,
    audit_risk: np.ndarray,
    source_seed: np.ndarray,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    discovery = np.asarray(discovery_risk, dtype=np.float64)
    audit = np.asarray(audit_risk, dtype=np.float64)
    seeds = np.asarray(source_seed, dtype=np.int64)
    if discovery.shape != (384, 9) or audit.shape != (384, 9) or (
            seeds.shape != (384,)) or not np.all(np.isfinite(discovery)) or (
                not np.all(np.isfinite(audit))) or np.any(discovery < 0.0) or (
                    np.any(discovery > 1.0)) or np.any(audit < 0.0) or np.any(
                        audit > 1.0):
        raise StateDependentRecoveryV4Error(
            "Stage-A risks must be finite G384/K9 probabilities")
    if any(np.count_nonzero(seeds == seed) != 64 for seed in SOURCE_SEEDS):
        raise StateDependentRecoveryV4Error(
            "Stage-A risks require 64 groups for each locked source seed")
    winners = [
        np.flatnonzero(row == np.min(row)) for row in discovery
    ]
    if any(len(indices) == 0 for indices in winners):
        raise StateDependentRecoveryV4Error("discovery minimizer set is empty")
    conditional_risk = np.asarray([
        float(np.mean(audit[group, indices], dtype=np.float64))
        for group, indices in enumerate(winners)
    ], dtype=np.float64)
    nominal = audit[:, 0]
    effect = nominal - conditional_risk
    pair_group, comparisons, tie_comparisons = _v3._pair_agreement_groups(
        discovery, audit)
    metrics = np.column_stack((nominal, effect, pair_group))
    bootstrap = spec["protocol"]["statistics"]["bootstrap"]
    draws = _v3._hierarchical_bootstrap(
        metrics,
        seeds,
        spec["age_strata"],
        replicates=int(bootstrap["replicates"]),
        seed=int(bootstrap["seed"]),
        chunk_size=int(bootstrap["chunk_size"]),
    )
    estimates = np.asarray([
        _v3._equal_seed_mean(metrics[:, column], seeds, SOURCE_SEEDS)
        for column in range(metrics.shape[1])
    ], dtype=np.float64)
    lower = np.quantile(
        draws, 0.05, axis=0, method=str(bootstrap["quantile_method"]))
    seed_effects = _v3._seed_effects(effect, seeds, SOURCE_SEEDS)
    age_effects = _v3._age_effects(effect, seeds, AGE_STRATA)
    gate = spec["protocol"]["triage_gates"]["stage_A_primary"]
    checks = {
        "audit_absolute_reduction": bool(
            estimates[1] >= float(gate["min_audit_absolute_reduction"])),
        "one_sided_95_lcb_reduction": bool(
            lower[1] >= float(gate["min_one_sided_95_lcb_reduction"])),
        "discovery_to_audit_pair_agreement": bool(
            estimates[2] >= float(
                gate["min_discovery_to_audit_pair_agreement"])),
        "pair_agreement_one_sided_95_lcb": bool(
            lower[2] >= float(
                gate["min_pair_agreement_one_sided_95_lcb"])),
        "each_policy_age_positive": bool(
            all(value > 0.0 for value in age_effects.values())),
        "all_six_source_seeds_positive": bool(
            sum(value > 0.0 for value in seed_effects.values()) >= int(
                gate["min_positive_source_seeds"])),
    }
    passed = bool(all(checks.values()))
    winner_count = np.asarray([len(value) for value in winners], dtype=np.float64)
    return {
        "primary_rule": gate["rule"],
        "audit_absolute_reduction": float(estimates[1]),
        "one_sided_95_lcb": float(lower[1]),
        "pair_agreement": float(estimates[2]),
        "pair_agreement_one_sided_95_lcb": float(lower[2]),
        "source_seed_effects": seed_effects,
        "policy_age_effects": age_effects,
        "checks": checks,
        "pass": passed,
        "discovery_minimizer_tie_fraction": _v3._equal_seed_mean(
            (winner_count > 1.0).astype(np.float64), seeds, SOURCE_SEEDS),
        "mean_discovery_minimizer_count": _v3._equal_seed_mean(
            winner_count, seeds, SOURCE_SEEDS),
        "pair_comparisons": int(comparisons),
        "pair_tie_comparisons": int(tie_comparisons),
        "audit_nominal_risk": {
            "equal_seed_estimate": float(estimates[0]),
            "one_sided_95_lcb": float(lower[0]),
            "source_seed_risks": _v3._seed_effects(
                nominal, seeds, SOURCE_SEEDS),
            "policy_age_risks": _v3._age_effects(
                nominal, seeds, AGE_STRATA),
        },
        "bootstrap_metric_columns": [
            "audit_nominal_risk", "per_state_primary_effect", "pair_agreement"],
    }


def evaluate_state_dependent_stage_a(
    *,
    protocol: Mapping[str, Any],
    discovery_fall: np.ndarray,
    audit_fall: np.ndarray,
    source_seed: np.ndarray,
) -> dict[str, Any]:
    """Pure array evaluator used by the one-shot audit and synthetic tests."""
    spec = _validate_protocol(protocol)
    discovery = np.asarray(discovery_fall)
    audit = np.asarray(audit_fall)
    if discovery.shape != (384, 9, 64) or audit.shape != (384, 9, 64):
        raise StateDependentRecoveryV4Error(
            "Stage-A fall arrays must have exact shape [384,9,64]")
    if discovery.dtype.kind not in "biu" or audit.dtype.kind not in "biu" or (
            not np.all(np.isin(discovery, (0, 1, False, True)))) or not np.all(
                np.isin(audit, (0, 1, False, True))):
        raise StateDependentRecoveryV4Error(
            "Stage-A fall arrays must contain only binary outcomes")
    return _evaluate_stage_a_risks(
        discovery_risk=np.mean(discovery, axis=2, dtype=np.float64),
        audit_risk=np.mean(audit, axis=2, dtype=np.float64),
        source_seed=source_seed,
        spec=spec,
    )


def consume_and_evaluate_state_dependent_audit(
    *,
    protocol: Mapping[str, Any],
    selection_lock_path: str | os.PathLike[str],
    expected_selection_lock_sha256: str,
    audit_paths: Sequence[str | os.PathLike[str]],
    audit_consumed_path: str | os.PathLike[str],
    expected_generator_commit: str,
    expected_protocol_file_sha256: str,
) -> dict[str, Any]:
    """Publish the marker, open audit once, and compute the V4 primary gate."""
    spec = _validate_protocol(protocol)
    clean_commit, protocol_file_sha256 = (
        _require_clean_head_protocol_binding())
    if expected_generator_commit != clean_commit:
        raise StateDependentRecoveryV4Error(
            "expected generator commit differs from the current clean HEAD")
    if expected_protocol_file_sha256 != protocol_file_sha256:
        raise StateDependentRecoveryV4Error(
            "expected protocol hash differs from the canonical raw V4 protocol")
    collection = spec["collection"]
    try:
        lock_file = _v3._artifact_path(
            selection_lock_path,
            protocol=protocol,
            expected_filename=str(collection["selection_lock_filename"]),
            name="selection_lock_path",
        )
        lock = _v3._read_selection_lock(
            lock_file, expected_selection_lock_sha256, spec)
    except _v3.ClosedLoopRecoveryTriageError as exc:
        raise StateDependentRecoveryV4Error(str(exc)) from exc
    if lock.get("generator_commit") != expected_generator_commit:
        raise StateDependentRecoveryV4Error(
            "selection lock differs from the clean generator commit")
    if lock.get("protocol_file_sha256") != expected_protocol_file_sha256:
        raise StateDependentRecoveryV4Error(
            "selection lock differs from the canonical protocol file")
    if lock.get("selection_semantics") != SELECTION_SEMANTICS:
        raise StateDependentRecoveryV4Error(
            "selection lock does not bind the exact V4 primary semantics")

    # Pure lexical/JSON checks only.  No audit path is probed above or inside
    # this helper.  The next filesystem operation on an audit shard is below
    # the marker publication.
    audit_files = _locked_audit_paths_before_consumption(
        audit_paths, protocol=protocol, spec=spec, lock=lock)
    try:
        consumed_file = _v3._artifact_path(
            audit_consumed_path,
            protocol=protocol,
            expected_filename=str(collection["audit_consumed_filename"]),
            name="audit_consumed_path",
        )
    except _v3.ClosedLoopRecoveryTriageError as exc:
        raise StateDependentRecoveryV4Error(str(exc)) from exc
    if os.path.lexists(os.fspath(consumed_file)):
        raise StateDependentRecoveryV4Error(
            "V4 audit has already been consumed or reserved")
    _require_unchanged_clean_binding(
        clean_commit, protocol_file_sha256, "before irreversible audit consumption")
    marker = {
        "schema_version": AUDIT_CONSUMED_SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": lock["protocol_file_sha256"],
        "selection_lock_sha256": expected_selection_lock_sha256,
        "audit_identifier": lock["audit_identifier"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "irreversibly_consumed_before_outcome_read",
    }
    try:
        marker_sha256 = _v3._atomic_no_clobber_json(consumed_file, marker)
        audit = _v3._load_audit_shards_after_consumption(audit_files, spec)
    except _v3.ClosedLoopRecoveryTriageError as exc:
        raise StateDependentRecoveryV4Error(str(exc)) from exc

    validated = lock["_validated"]
    if audit["file_sha256"] != [
            record["file_sha256"] for record in lock["expected_audit_shards"]
    ] or audit["content_sha256"] != [
            record["content_sha256"] for record in lock["expected_audit_shards"]
    ]:
        raise StateDependentRecoveryV4Error(
            "audit shards differ from report-last selection commitments")
    if audit["generator_commit"] != lock["generator_commit"] or audit[
            "protocol_file_sha256"] != lock["protocol_file_sha256"]:
        raise StateDependentRecoveryV4Error(
            "audit generator/protocol identity differs from selection lock")
    if not _v3._same_identities(validated, audit):
        raise StateDependentRecoveryV4Error(
            "audit group identities/order differ from selection lock")
    for observed, expected in (
        ("crn_id", "audit_crn_id"),
        ("rollout_seed", "audit_rollout_seed"),
        ("perturbation_seed", "audit_perturbation_seed"),
        ("candidate_seed", "audit_candidate_seed"),
    ):
        if not np.array_equal(audit[observed], validated[expected]):
            raise StateDependentRecoveryV4Error(
                f"audit {observed} differs from preassigned V4 seeds")

    primary = _evaluate_stage_a_risks(
        discovery_risk=validated["discovery_risk"],
        audit_risk=np.mean(audit["fall"], axis=2, dtype=np.float64),
        source_seed=audit["source_seed"],
        spec=spec,
    )
    passed = bool(primary["pass"])
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "protocol_contract_sha256": spec["protocol_contract_sha256"],
        "protocol_file_sha256": lock["protocol_file_sha256"],
        "selection_lock_sha256": expected_selection_lock_sha256,
        "audit_identifier": lock["audit_identifier"],
        "audit_consumed_marker_sha256": marker_sha256,
        "audit_shard_file_sha256": audit["file_sha256"],
        "audit_shard_content_sha256": audit["content_sha256"],
        "claim_scope": "seed42_fixed_actor_conditional_mechanism_only",
        "cross_actor_generalization_claim": False,
        "estimand": (
            "equal_groups_within_source_seed_then_equal_six_source_seeds; "
            "uniform audit risk over exact per-state discovery minima; "
            "conditional on admission-positive fixed states"),
        "data_gate": {
            "pass": True,
            "groups": 384,
            "groups_per_source_seed": 64,
            "required_source_seeds": list(SOURCE_SEEDS),
            "candidates": 9,
            "discovery_replicas": 64,
            "audit_replicas": 64,
            "horizon_policy_steps": 96,
            "discovery_informativeness": lock["data_gate"][
                "discovery_informativeness"],
            "v4_seed_domain_bound_before_lock": True,
        },
        "bootstrap": {
            **_json_copy(
                spec["protocol"]["statistics"]["bootstrap"], "bootstrap"),
            "replicates_used": 50_000,
            "seed_used": 20_260_810,
            "override_used": False,
            "metric_columns": primary.pop("bootstrap_metric_columns"),
        },
        "stage_A_primary": primary,
        "decision": "authorize_stage_B_only" if passed else "no_model_training",
        "stage_B_authorized": passed,
        "model_training_authorized": passed,
        "model_training_triggered": False,
        "paired_closed_loop_authorized": False,
        "online_training_authorized": False,
        "objective1_pass": False,
        "phase2_authorized": False,
        "authorization_note": (
            "A pass authorizes only the separately invoked Stage-B workflow; "
            "this audit never fits a model and cannot authorize Stage C, D, "
            "Objective 1, speed expansion, or a cross-actor claim."),
    }
    json.dumps(report, allow_nan=False, sort_keys=True)
    return report


def v4_seed(
    source_seed: int,
    identity: int,
    role: str,
    namespace: int,
    index: int = 0,
) -> int:
    """Expose the immutable V4 derivation for collectors and tests."""
    return _derived_seed(
        source_seed,
        identity,
        role,
        namespace,
        index,
        seed_domain=SEED_DOMAIN,
        role_tags=SEED_ROLE_TAGS,
        seed_algorithm=SEED_ALGORITHM,
    )


__all__ = [
    "AGE_STRATA",
    "AUDIT_CONSUMED_SCHEMA_VERSION",
    "BEHAVIOR_STEPS",
    "CANDIDATE_NAMES",
    "PROTOCOL_CONTRACT_SHA256",
    "PROTOCOL_NAME",
    "PROTOCOL_PATH",
    "REPORT_SCHEMA_VERSION",
    "SELECTION_SEMANTICS",
    "SEED_DOMAIN",
    "SEED_ALGORITHM",
    "SEED_DOMAIN_PREFIX_LOW15",
    "SEED_ROLE_TAGS",
    "SOURCE_SEEDS",
    "StateDependentRecoveryV4Error",
    "consume_and_evaluate_state_dependent_audit",
    "create_state_dependent_selection_lock",
    "evaluate_state_dependent_stage_a",
    "expected_v4_seed_manifest",
    "load_state_dependent_recovery_v4_protocol",
    "resume_state_dependent_discovery_failure_report",
    "v4_seed",
    "validate_state_dependent_collection_readiness",
    "validate_state_dependent_recovery_v4_protocol",
]
