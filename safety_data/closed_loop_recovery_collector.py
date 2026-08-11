"""Outcome-firewalled collection for closed-loop recovery triage v3.

This module is intentionally separate from the v1/v2 native collector.  Its
admission rollouts run only the nominal early policy, while accepted states are
evaluated into physically separate discovery and audit grouped datasets.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from safety_data.collector import (
    CollectedGroup,
    GroupIdentity,
    GroupRandomness,
    GroupedBranchAssembler,
    PRIVILEGED_FEATURE_NAMES,
    _action_application_contract,
    _policy_identity,
    privileged_features,
)
from safety_data.native import (
    NativeGroupEvaluation,
    ReplicaSeedBundle,
    evaluate_same_state_group,
)
from safety_data.paths import (
    ProtectedEvidencePathError,
    assert_development_path,
    assert_safe_evidence_output,
    require_workflow_authorized_or_safe_input,
)
from safety_data.schema import GroupedBranchDataset, PrivilegedBranchView


ADMISSION_SCHEMA_VERSION = "qsafe.closed_loop_admission.v3"
ADMISSION_PRIVILEGED_SCHEMA_VERSION = (
    "qsafe.closed_loop_admission_privileged.v3")
COLLECTION_PROTOCOL_VERSION = "qsafe.closed_loop_recovery_collection.v3"
FALL_HEIGHT_REFERENCE = "base_link_body_origin_world_z"
FALL_SAMPLING_CADENCE = (
    "first_failing_50Hz_post_action_boundary_after_10_low_level_substeps")
FIRST_FAILURE_STEP_SEMANTICS = (
    "first_failing_post_action_policy_boundary_1_through_H")

_ROLE_TAGS = {
    "source_reset": 10,
    "source_impulse": 11,
    "source_action": 12,
    "admission": 20,
    "discovery": 30,
    "audit": 40,
}
_V3_SEED_DOMAIN = b"qsafe_closed_loop_v3_seed\0"
_V3_ROLE_TAG_ITEMS = tuple(_ROLE_TAGS.items())
_SHA256_LOW63_SEED_ALGORITHM = "sha256_low63_v1"
_INJECTIVE_V4_SEED_ALGORITHM = (
    "high_bit_then_domain_low15_then_14_8_18_2_6_bitpack_v1")
_INJECTIVE_V4_STREAM_MAPPING = {
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
        "identity": "proposal_index", "namespace": 3, "index": 0},
}


def seed_derivation_manifest(
    *,
    seed_domain: bytes,
    seed_role_tags: Mapping[str, int] | Iterable[tuple[str, int]],
    seed_algorithm: str,
) -> dict[str, Any]:
    """Build a fresh canonical manifest for a non-default seed contract."""
    items = (
        seed_role_tags.items()
        if isinstance(seed_role_tags, Mapping)
        else seed_role_tags
    )
    result = {
        "domain_hex": seed_domain.hex(),
        "role_tags": {name: int(value) for name, value in items},
        "algorithm": seed_algorithm,
    }
    if seed_algorithm == _INJECTIVE_V4_SEED_ALGORITHM:
        result["stream_mapping"] = copy.deepcopy(
            _INJECTIVE_V4_STREAM_MAPPING)
    return result


def _canonical_hash(manifest: Mapping[str, Any], arrays: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(b"qsafe_closed_loop_v3_content\0")
    clean_manifest = copy.deepcopy(dict(manifest))
    clean_manifest.pop("content_sha256", None)
    digest.update(json.dumps(
        clean_manifest, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8"))
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        if value.dtype.kind in "US":
            rendered = json.dumps(
                value.astype(str).tolist(), ensure_ascii=True,
                separators=(",", ":")).encode("utf-8")
            digest.update(rendered)
        else:
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_protocol_sha256(protocol: Mapping[str, Any]) -> str:
    """Hash parsed protocol semantics independently from YAML formatting."""
    try:
        payload = json.dumps(
            dict(protocol), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("protocol must be canonical JSON-compatible data") from exc
    return hashlib.sha256(payload).hexdigest()


def _fall_definition(env: Any) -> dict[str, Any]:
    """Return the exact v3 label contract carried by every outcome artifact."""
    return {
        "max_abs_roll_pitch_rad": float(env.cfg.fallen_orientation_rad),
        "min_base_height_m": 0.18,
        "tilt_comparator": "greater_than_or_equal",
        "height_comparator": "strict_less_than",
        "height_reference": FALL_HEIGHT_REFERENCE,
        "sampling_cadence": FALL_SAMPLING_CADENCE,
        "within_policy_hold_crossings": "not_observed",
        "first_failure_step_semantics": FIRST_FAILURE_STEP_SEMANTICS,
    }


def _derived_seed(
    source_seed: int,
    identity: int,
    role: str,
    namespace: int,
    index: int = 0,
    *,
    seed_domain: bytes = _V3_SEED_DOMAIN,
    role_tags: tuple[tuple[str, int], ...] = _V3_ROLE_TAG_ITEMS,
    seed_algorithm: str = _SHA256_LOW63_SEED_ALGORITHM,
) -> int:
    try:
        tags = dict(role_tags)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed role tags must be unique (name, integer) pairs") from exc
    if len(tags) != len(role_tags) or set(tags) != set(_ROLE_TAGS):
        raise ValueError("seed role tags must bind the exact six collector roles")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in tags.values()) or len(set(tags.values())) != len(tags):
        raise ValueError("seed role tags must be unique nonnegative integers")
    if not isinstance(seed_domain, bytes) or not seed_domain or not (
            seed_domain.endswith(b"\0")):
        raise ValueError("seed domain must be nonempty bytes ending in NUL")
    if role not in tags:
        raise ValueError(f"unknown v3 RNG role {role!r}")
    values = (source_seed, identity, namespace, index)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)) or int(value) < 0 for value in values):
        raise ValueError("v3 seed inputs must be nonnegative integers")
    if seed_algorithm == _INJECTIVE_V4_SEED_ALGORITHM:
        source_seed_value, identity_value, namespace_value, index_value = (
            int(value) for value in values)
        role_tag = int(tags[role])
        widths_and_values = (
            (14, source_seed_value, "source_seed"),
            (8, role_tag, "role_tag"),
            (18, identity_value, "identity"),
            (2, namespace_value, "namespace"),
            (6, index_value, "index"),
        )
        for width, value, name in widths_and_values:
            if value >= 1 << width:
                raise ValueError(
                    f"injective V4 seed {name} exceeds its {width}-bit cap")
        # Bit 63 is reserved for V4, while every historical SHA-low63 seed has
        # it cleared.  The low 15 bits of the full-domain digest and every
        # remaining field are then packed without truncation.  The complete
        # 64-bit mapping is injective within the declared production caps and
        # disjoint from V2/V3 by construction.
        packed = int.from_bytes(
            hashlib.sha256(seed_domain).digest()[:2], "little") & 0x7FFF
        for width, value, _ in widths_and_values:
            packed = (packed << width) | value
        if packed >= 1 << 63:
            raise AssertionError("injective V4 seed exceeded 63 bits")
        return (1 << 63) | packed
    if seed_algorithm != _SHA256_LOW63_SEED_ALGORITHM:
        raise ValueError(f"unknown seed_algorithm={seed_algorithm!r}")
    digest = hashlib.sha256(seed_domain)
    digest.update(int(tags[role]).to_bytes(4, "little"))
    for value in values:
        digest.update(int(value).to_bytes(16, "little", signed=False))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def _step_rng(seed: int, step: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([
        int(seed), 0, int(step)]))


def role_randomness(
    *,
    source_seed: int,
    proposal_index: int,
    replicas: int,
    role: str,
    seed_domain: bytes = _V3_SEED_DOMAIN,
    role_tags: tuple[tuple[str, int], ...] = _V3_ROLE_TAG_ITEMS,
    seed_algorithm: str = _SHA256_LOW63_SEED_ALGORITHM,
) -> tuple[ReplicaSeedBundle, GroupRandomness]:
    """Return globally disjoint admission/discovery/audit seed namespaces."""
    if role not in ("admission", "discovery", "audit"):
        raise ValueError("branch role must be admission, discovery, or audit")
    if isinstance(replicas, bool) or not isinstance(
            replicas, (int, np.integer)) or int(replicas) <= 0:
        raise ValueError("replicas must be a positive integer")
    count = int(replicas)
    crn_id = np.asarray([
        _derived_seed(
            source_seed, proposal_index, role, 0, replica,
            seed_domain=seed_domain, role_tags=role_tags,
            seed_algorithm=seed_algorithm)
        for replica in range(count)
    ], dtype=np.uint64)
    rollout_seed = np.asarray([
        _derived_seed(
            source_seed, proposal_index, role, 1, replica,
            seed_domain=seed_domain, role_tags=role_tags,
            seed_algorithm=seed_algorithm)
        for replica in range(count)
    ], dtype=np.uint64)
    perturbation_seed = np.asarray([
        _derived_seed(
            source_seed, proposal_index, role, 2, replica,
            seed_domain=seed_domain, role_tags=role_tags,
            seed_algorithm=seed_algorithm)
        for replica in range(count)
    ], dtype=np.uint64)
    candidate_seed = _derived_seed(
        source_seed, proposal_index, role, 3, 0,
        seed_domain=seed_domain, role_tags=role_tags,
        seed_algorithm=seed_algorithm)
    bundle = ReplicaSeedBundle(
        crn_id=crn_id,
        rollout_seed=rollout_seed,
        perturbation_seed=perturbation_seed,
    )
    return bundle, GroupRandomness(
        crn_id=crn_id,
        rollout_seed=rollout_seed,
        perturbation_seed=perturbation_seed,
        candidate_seed=candidate_seed,
    )


@dataclass(frozen=True)
class NominalReplicaEvaluation:
    fall: np.ndarray
    first_failure_step: np.ndarray
    max_tilt_rad: np.ndarray
    min_height_m: np.ndarray
    crn_id: np.ndarray
    rollout_seed: np.ndarray
    perturbation_seed: np.ndarray


def _capture_component(component: Any, name: str) -> Any:
    capture = getattr(component, "capture_branch_state", None)
    restore = getattr(component, "restore_branch_state", None)
    if callable(capture) != callable(restore):
        raise TypeError(
            f"{name} must implement both capture_branch_state and "
            "restore_branch_state, or neither")
    return copy.deepcopy(capture()) if callable(capture) else None


def _restore_component(component: Any, state: Any) -> None:
    restore = getattr(component, "restore_branch_state", None)
    if callable(restore):
        restore(copy.deepcopy(state))


def evaluate_nominal_admission(
    *,
    env: Any,
    snapshot: Any,
    nominal_first_action: np.ndarray,
    seeds: ReplicaSeedBundle,
    horizon_steps: int,
    continuation_policy: Any,
) -> NominalReplicaEvaluation:
    """Evaluate admission without constructing or touching recovery actions."""
    first = np.asarray(nominal_first_action, dtype=np.float32).reshape(-1)
    if first.shape != (env.cfg.num_joints,) or not np.all(np.isfinite(first)):
        raise ValueError("nominal_first_action must be one finite joint action")
    if np.any(first < -1.0 - 1e-6) or np.any(first > 1.0 + 1e-6):
        raise ValueError("nominal_first_action must lie in normalized [-1,1]")
    if isinstance(horizon_steps, bool) or not isinstance(
            horizon_steps, (int, np.integer)) or int(horizon_steps) <= 0:
        raise ValueError("horizon_steps must be a positive integer")
    horizon = int(horizon_steps)
    replicas = seeds.replica_count
    fall = np.zeros(replicas, dtype=bool)
    first_failure = np.full(replicas, horizon + 1, dtype=np.int16)
    max_tilt = np.zeros(replicas, dtype=np.float32)
    min_height = np.full(replicas, np.inf, dtype=np.float32)
    policy_state = _capture_component(continuation_policy, "continuation_policy")
    try:
        for replica, rollout_seed in enumerate(seeds.rollout_seed):
            env.restore(snapshot)
            _restore_component(continuation_policy, policy_state)
            if env.measurement().failure:
                raise ValueError("admission snapshot is already a failure")
            for step in range(horizon):
                if step == 0:
                    action = first
                else:
                    action = continuation_policy(
                        env.record_observation(), step,
                        _step_rng(int(rollout_seed), step))
                result = env.step(action)
                max_tilt[replica] = max(max_tilt[replica], result.tilt_rad)
                min_height[replica] = min(min_height[replica], result.height_m)
                if result.failure:
                    fall[replica] = True
                    first_failure[replica] = step + 1
                    break
    finally:
        env.restore(snapshot)
        _restore_component(continuation_policy, policy_state)
    return NominalReplicaEvaluation(
        fall=fall,
        first_failure_step=first_failure,
        max_tilt_rad=max_tilt,
        min_height_m=min_height,
        crn_id=seeds.crn_id.copy(),
        rollout_seed=seeds.rollout_seed.copy(),
        perturbation_seed=seeds.perturbation_seed.copy(),
    )


@dataclass
class AdmissionLedger:
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    path: Path | None = None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def validate(self, *, verify_hash: bool = True) -> dict[str, Any]:
        required = {
            "proposal_id", "proposal_index", "state_hash", "trajectory_id",
            "episode_id", "episode_step", "source_seed",
            "policy_training_step", "policy_source", "obs_history",
            "admission_crn_id", "admission_rollout_seed",
            "admission_perturbation_seed", "fall", "first_failure_step",
            "accepted", "accepted_group_index", "decision_reason",
        }
        optional = {"admission_candidate_seed"}
        if not required.issubset(self.arrays) or set(self.arrays) - required - optional:
            raise ValueError(
                f"admission ledger fields differ: missing={required-set(self.arrays)}, "
                f"extra={set(self.arrays)-required}")
        if self.manifest.get("schema_version") != ADMISSION_SCHEMA_VERSION or (
                self.manifest.get("feature_view") != "deployable_admission"):
            raise ValueError("invalid admission ledger manifest")
        fall_definition = self.manifest.get("fall_definition")
        if not isinstance(fall_definition, Mapping) or set(fall_definition) != {
                "max_abs_roll_pitch_rad", "min_base_height_m",
                "tilt_comparator", "height_comparator",
                "height_reference", "sampling_cadence",
                "within_policy_hold_crossings",
                "first_failure_step_semantics",
        }:
            raise ValueError("admission ledger requires the exact fall definition")
        try:
            tilt_threshold = float(fall_definition["max_abs_roll_pitch_rad"])
            height_threshold = float(fall_definition["min_base_height_m"])
        except (TypeError, ValueError) as exc:
            raise ValueError("admission fall thresholds must be numeric") from exc
        if not np.isfinite(tilt_threshold) or tilt_threshold <= 0.0 or not (
                np.isfinite(height_threshold) and height_threshold > 0.0):
            raise ValueError("admission fall thresholds must be finite and positive")
        if fall_definition["height_reference"] != FALL_HEIGHT_REFERENCE or (
                fall_definition["tilt_comparator"] !=
                "greater_than_or_equal") or fall_definition[
                    "height_comparator"] != "strict_less_than" or (
                fall_definition["sampling_cadence"] != FALL_SAMPLING_CADENCE) or (
                    fall_definition["within_policy_hold_crossings"] !=
                    "not_observed") or fall_definition[
                        "first_failure_step_semantics"] != (
                            FIRST_FAILURE_STEP_SEMANTICS):
            raise ValueError("admission fall sampling/reference contract drifted")
        proposals = len(np.asarray(self.arrays["proposal_id"]))
        if proposals == 0:
            raise ValueError("admission ledger cannot be empty")
        for name, value in self.arrays.items():
            if np.asarray(value).ndim == 0 or len(np.asarray(value)) != proposals:
                raise ValueError(f"admission field {name} is not proposal-first")
        if not np.array_equal(
                np.asarray(self.arrays["proposal_index"]),
                np.arange(proposals, dtype=np.int64)):
            raise ValueError("proposal_index must preserve exhaustive ledger order")
        for name in ("proposal_id", "state_hash"):
            values = np.asarray(self.arrays[name]).astype(str)
            if np.any(values == "") or len(np.unique(values)) != proposals:
                raise ValueError(f"{name} must be nonempty and unique")
        for name in ("trajectory_id", "policy_source", "decision_reason"):
            values = np.asarray(self.arrays[name])
            if values.dtype.kind not in "US" or np.any(values.astype(str) == ""):
                raise ValueError(f"{name} must contain nonempty text")
        for name in (
            "proposal_index", "episode_id", "episode_step", "source_seed",
            "policy_training_step", "accepted_group_index",
        ):
            value = np.asarray(self.arrays[name])
            if value.dtype.kind not in "iu":
                raise ValueError(f"{name} must use an integer dtype")
            if name != "accepted_group_index" and np.any(value < 0):
                raise ValueError(f"{name} must contain nonnegative integers")
        shard_contract = self.manifest.get("shards")
        if shard_contract is None:
            source_seed = int(self.manifest.get("source_seed", -1))
            policy_step = int(self.manifest.get("policy_training_step", -1))
            if source_seed < 0 or not np.all(
                    np.asarray(self.arrays["source_seed"]) == source_seed):
                raise ValueError("admission rows must match manifest source_seed")
            if policy_step < 0 or not np.all(
                    np.asarray(self.arrays["policy_training_step"]) == policy_step):
                raise ValueError(
                    "admission rows must match manifest policy_training_step")
            if len(np.unique(np.asarray(self.arrays["policy_source"]))) != 1:
                raise ValueError(
                    "leaf admission rows must use one frozen policy identity")
        else:
            if not isinstance(shard_contract, list) or len(shard_contract) < 2:
                raise ValueError("merged admission shards must be a nontrivial list")
            source_seeds = self.manifest.get("source_seeds")
            policy_steps = self.manifest.get("policy_training_steps")
            if not isinstance(source_seeds, list) or not isinstance(
                    policy_steps, list) or len(source_seeds) != len(
                        shard_contract) or len(policy_steps) != len(shard_contract):
                raise ValueError("merged admission source metadata is malformed")
            offset = 0
            for ordinal, (shard, source_seed, policy_step) in enumerate(zip(
                    shard_contract, source_seeds, policy_steps, strict=True)):
                if not isinstance(shard, Mapping) or shard.get(
                        "ordinal") != ordinal:
                    raise ValueError("merged admission shard order is malformed")
                count = int(shard.get("proposals", -1))
                stop = offset + count
                if count <= 0 or stop > proposals or int(shard.get(
                        "source_seed", -1)) != int(source_seed) or int(
                            shard.get("policy_training_step", -1)) != int(
                                policy_step):
                    raise ValueError("merged admission shard provenance is malformed")
                selection = slice(offset, stop)
                if not np.all(np.asarray(
                        self.arrays["source_seed"])[selection] == int(source_seed)):
                    raise ValueError("merged admission source-seed rows are misaligned")
                if not np.all(np.asarray(self.arrays[
                        "policy_training_step"])[selection] == int(policy_step)):
                    raise ValueError("merged admission policy-age rows are misaligned")
                if len(np.unique(np.asarray(
                        self.arrays["policy_source"])[selection])) != 1:
                    raise ValueError(
                        "each admission shard must use one frozen policy identity")
                offset = stop
            if offset != proposals:
                raise ValueError("merged admission shard sizes do not exhaust rows")
        history = np.asarray(self.arrays["obs_history"])
        if history.shape != (proposals, 5, 46) or not np.all(np.isfinite(history)):
            raise ValueError("admission obs_history must have shape [P,5,46]")
        fall = np.asarray(self.arrays["fall"])
        admission_replicas = int(self.manifest.get("admission_replicas", -1))
        if fall.shape != (proposals, admission_replicas):
            raise ValueError("admission fall must have shape [P,R_admission]")
        if not np.all(np.isin(fall, (0, 1, False, True))):
            raise ValueError("admission fall labels must be binary")
        for name in (
            "admission_crn_id", "admission_rollout_seed",
            "admission_perturbation_seed",
        ):
            value = np.asarray(self.arrays[name])
            if value.shape != fall.shape or value.dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer [P,R] array")
            if len(np.unique(value)) != value.size:
                raise ValueError(f"{name} seeds must be globally unique")
        combined_seeds = np.concatenate([
            np.asarray(self.arrays[name], dtype=np.uint64).reshape(-1)
            for name in (
                "admission_crn_id", "admission_rollout_seed",
                "admission_perturbation_seed")
        ])
        if len(np.unique(combined_seeds)) != combined_seeds.size:
            raise ValueError("admission RNG namespaces must be disjoint")
        if "admission_candidate_seed" in self.arrays:
            candidate_seed = np.asarray(self.arrays["admission_candidate_seed"])
            if candidate_seed.shape != (proposals,) or candidate_seed.dtype.kind not in "iu" or np.any(candidate_seed < 0):
                raise ValueError(
                    "admission_candidate_seed must be a nonnegative integer [P] vector")
            if len(np.unique(candidate_seed)) != proposals:
                raise ValueError("admission candidate seeds must be globally unique")
            combined_with_candidate = np.concatenate([
                combined_seeds, candidate_seed.astype(np.uint64, copy=False)
            ])
            if len(np.unique(combined_with_candidate)) != len(combined_with_candidate):
                raise ValueError(
                    "admission candidate seeds overlap another admission namespace")
        accepted = np.asarray(self.arrays["accepted"])
        if accepted.dtype.kind not in "biu" or not np.all(
                np.isin(accepted, (0, 1, False, True))):
            raise ValueError("accepted must be binary")
        accepted_bool = accepted.astype(bool)
        counts = fall.astype(bool).sum(axis=1)
        lower = int(self.manifest["accept_min_falls_inclusive"])
        upper = int(self.manifest["accept_max_falls_inclusive"])
        expected = (counts >= lower) & (counts <= upper)
        if not np.array_equal(accepted_bool, expected):
            raise ValueError("admission decision disagrees with locked fall bounds")
        group_index = np.asarray(self.arrays["accepted_group_index"])
        if group_index.shape != (proposals,) or group_index.dtype.kind not in "iu":
            raise ValueError("accepted_group_index must be integer [P]")
        if np.any(group_index[~accepted_bool] != -1) or not np.array_equal(
                group_index[accepted_bool],
                np.arange(np.count_nonzero(accepted_bool), dtype=group_index.dtype)):
            raise ValueError("accepted_group_index must enumerate accepted proposals")
        reasons = np.asarray(self.arrays["decision_reason"]).astype(str)
        accepted_reason = f"accepted_{lower}_to_{upper}_of_{admission_replicas}"
        rejected_reason = (
            f"rejected_outside_{lower}_to_{upper}_of_{admission_replicas}")
        expected_reasons = np.where(
            accepted_bool, accepted_reason, rejected_reason)
        if not np.array_equal(reasons, expected_reasons):
            raise ValueError("admission decision_reason disagrees with decision")
        failure_step = np.asarray(self.arrays["first_failure_step"])
        horizon = int(self.manifest["horizon_steps"])
        if failure_step.shape != fall.shape or failure_step.dtype.kind not in "iu":
            raise ValueError("admission first_failure_step must be integer [P,R]")
        if np.any(failure_step[fall.astype(bool)] < 1) or np.any(
                failure_step[fall.astype(bool)] > horizon) or np.any(
                    failure_step[~fall.astype(bool)] != horizon + 1):
            raise ValueError("admission first-failure labels are invalid")
        content_hash = _canonical_hash(self.manifest, self.arrays)
        recorded = self.manifest.get("content_sha256")
        if verify_hash and recorded is not None and recorded != content_hash:
            raise ValueError("admission ledger content hash mismatch")
        return {
            "proposals": proposals,
            "accepted": int(np.count_nonzero(accepted_bool)),
            "content_sha256": content_hash,
        }

    def save(self, path: str | Path) -> Path:
        output = assert_development_path(assert_safe_evidence_output(path))
        if output.suffix != ".npz":
            raise ValueError("admission ledger must use .npz")
        report = self.validate(verify_hash=False)
        manifest = copy.deepcopy(self.manifest)
        manifest["content_sha256"] = report["content_sha256"]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            manifest_json=np.asarray(json.dumps(
                manifest, sort_keys=True, separators=(",", ":"))),
            **self.arrays,
        )
        self.manifest = manifest
        self.path = output
        return output

    @classmethod
    def load(cls, path: str | Path) -> "AdmissionLedger":
        try:
            source = assert_development_path(
                require_workflow_authorized_or_safe_input(
                    path, allowed_roles=(
                        "admission",
                        "stage_b_fit_admission",
                        "stage_b_probability_calibration_admission",
                        "stage_b_uncertainty_calibration_admission",
                        "stage_b_selector_calibration_admission",
                        "stage_b_model_test_admission",
                        "stage_b_model_test_producer_admission",
                    )))
        except ProtectedEvidencePathError as exc:
            raise ValueError(str(exc)) from exc
        with np.load(source, allow_pickle=False) as payload:
            manifest = json.loads(str(payload["manifest_json"].item()))
            arrays = {
                name: payload[name].copy()
                for name in payload.files if name != "manifest_json"
            }
        value = cls(manifest, arrays, source)
        value.validate()
        return value


def load_admission_ledger_blind(path: str | Path) -> AdmissionLedger:
    """Load an admission shard without inspecting outcome semantics.

    Model-Test production is allowed to carry the complete immutable shard,
    but must not derive accepted counts, fall rates, or any other outcome
    statistic while assembling the aggregate.  This loader therefore checks
    only the manifest/content hash and leaves semantic validation to the
    later, authorized evaluator.
    """
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        if "manifest_json" not in payload.files:
            raise ValueError("blind admission shard has no manifest_json")
        manifest = json.loads(str(payload["manifest_json"].item()))
        arrays = {
            name: payload[name].copy()
            for name in payload.files if name != "manifest_json"
        }
    recorded = manifest.get("content_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("blind admission shard lacks content_sha256")
    actual = _canonical_hash(manifest, arrays)
    if actual != recorded:
        raise ValueError("blind admission shard content hash mismatch")
    return AdmissionLedger(manifest, arrays, source)


def merge_admission_ledgers_blind(
    ledgers: list[AdmissionLedger] | tuple[AdmissionLedger, ...],
) -> AdmissionLedger:
    """Mechanically merge admission shards without reading fall/accepted data."""
    items = list(ledgers)
    if len(items) < 2:
        raise ValueError("at least two admission ledgers are required")
    reference = items[0]
    locked_fields = (
        "schema_version", "feature_view", "generator_commit",
        "protocol_sha256", "protocol_contract_sha256", "fall_definition",
        "simulator_fingerprint", "source_policy", "continuation_policy",
        "action_application_contract", "admission_replicas",
        "horizon_steps", "accept_min_falls_inclusive",
        "accept_max_falls_inclusive", "all_proposals_recorded",
        "candidate_outcomes_used_for_admission",
    )
    keys = set(reference.arrays)
    hashes: list[str] = []
    source_seeds: list[int] = []
    policy_steps: list[int] = []
    for index, item in enumerate(items):
        if set(item.arrays) != keys:
            raise ValueError(f"blind admission shard {index} array fields differ")
        for name in locked_fields:
            if item.manifest.get(name) != reference.manifest.get(name):
                raise ValueError(
                    f"blind admission shard {index} changes locked field {name}")
        actual = _canonical_hash(item.manifest, item.arrays)
        if item.manifest.get("content_sha256") != actual:
            raise ValueError(f"blind admission shard {index} content hash mismatch")
        hashes.append(actual)
        source_seeds.append(int(item.manifest.get("source_seed", -1)))
        policy_steps.append(int(item.manifest.get("policy_training_step", -1)))
        for name in keys:
            value = np.asarray(item.arrays[name])
            ref_value = np.asarray(reference.arrays[name])
            if value.ndim == 0 or value.shape[0] != len(
                    np.asarray(item.arrays["proposal_id"])):
                raise ValueError(
                    f"blind admission shard {index} field {name} is not proposal-first")
            if value.shape[1:] != ref_value.shape[1:] or (
                    value.dtype.kind not in "US" and value.dtype != ref_value.dtype):
                raise ValueError(
                    f"blind admission shard {index} field {name} shape/dtype drifted")
    if len(set(source_seeds)) != len(items) or any(seed < 0 for seed in source_seeds):
        raise ValueError("blind admission source seeds must be unique/nonnegative")
    for name in (
        "proposal_id", "state_hash", "admission_crn_id",
        "admission_rollout_seed", "admission_perturbation_seed",
    ):
        combined = np.concatenate([
            np.asarray(item.arrays[name]).reshape(-1) for item in items
        ])
        if len(np.unique(combined)) != combined.size:
            raise ValueError(f"blind admission shards overlap on {name}")
    if "admission_candidate_seed" in keys:
        combined = np.concatenate([
            np.asarray(item.arrays["admission_candidate_seed"]).reshape(-1)
            for item in items
        ])
        if len(np.unique(combined)) != combined.size:
            raise ValueError(
                "blind admission shards overlap on admission_candidate_seed")
    arrays = {
        name: np.concatenate([np.asarray(item.arrays[name]) for item in items], axis=0)
        for name in sorted(keys)
    }
    arrays["proposal_index"] = np.arange(
        len(np.asarray(arrays["proposal_id"])), dtype=np.int64)
    manifest = {name: copy.deepcopy(reference.manifest[name]) for name in locked_fields}
    manifest["source_seeds"] = source_seeds
    manifest["policy_training_steps"] = policy_steps
    manifest["shards"] = [
        {
            "ordinal": index,
            "source_seed": source_seeds[index],
            "policy_training_step": policy_steps[index],
            "proposals": int(len(np.asarray(item.arrays["proposal_id"]))),
            "content_sha256": hashes[index],
        }
        for index, item in enumerate(items)
    ]
    manifest["content_sha256"] = _canonical_hash(manifest, arrays)
    return AdmissionLedger(manifest, arrays)


def save_admission_ledger_blind(ledger: AdmissionLedger, path: str | Path) -> Path:
    """Write an admission shard without semantic outcome validation."""
    output = assert_development_path(assert_safe_evidence_output(path))
    if output.suffix != ".npz":
        raise ValueError("admission ledger must use .npz")
    manifest = copy.deepcopy(ledger.manifest)
    manifest["content_sha256"] = _canonical_hash(manifest, ledger.arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        manifest_json=np.asarray(json.dumps(
            manifest, sort_keys=True, separators=(",", ":"))),
        **ledger.arrays,
    )
    ledger.manifest = manifest
    ledger.path = output
    return output


@dataclass
class AdmissionPrivilegedView:
    manifest: dict[str, Any]
    proposal_id: np.ndarray
    state_hash: np.ndarray
    initial_tilt_rad: np.ndarray
    initial_height_m: np.ndarray
    max_tilt_rad: np.ndarray
    min_height_m: np.ndarray
    path: Path | None = None

    def validate(
        self,
        ledger: AdmissionLedger,
        *,
        verify_hash: bool = True,
    ) -> dict[str, Any]:
        ledger_report = ledger.validate()
        if self.manifest.get(
                "schema_version") != ADMISSION_PRIVILEGED_SCHEMA_VERSION or (
                    self.manifest.get("feature_view")
                    != "privileged_admission_diagnostic_only"):
            raise ValueError("invalid privileged admission manifest")
        if self.manifest.get("deployable_content_sha256") != ledger_report[
                "content_sha256"]:
            raise ValueError("privileged admission view links wrong ledger")
        proposals = ledger_report["proposals"]
        if not np.array_equal(
                np.asarray(self.proposal_id).astype(str),
                np.asarray(ledger.arrays["proposal_id"]).astype(str)) or not (
                    np.array_equal(
                        np.asarray(self.state_hash).astype(str),
                        np.asarray(ledger.arrays["state_hash"]).astype(str))):
            raise ValueError("privileged admission identities do not align")
        replicas = int(ledger.manifest["admission_replicas"])
        for name in ("initial_tilt_rad", "initial_height_m"):
            value = np.asarray(getattr(self, name))
            if value.shape != (proposals,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite [P]")
        for name in ("max_tilt_rad", "min_height_m"):
            value = np.asarray(getattr(self, name))
            if value.shape != (proposals, replicas) or not np.all(
                    np.isfinite(value)):
                raise ValueError(f"{name} must be finite [P,R]")
        arrays = {
            "proposal_id": np.asarray(self.proposal_id),
            "state_hash": np.asarray(self.state_hash),
            "initial_tilt_rad": np.asarray(self.initial_tilt_rad),
            "initial_height_m": np.asarray(self.initial_height_m),
            "max_tilt_rad": np.asarray(self.max_tilt_rad),
            "min_height_m": np.asarray(self.min_height_m),
        }
        content_hash = _canonical_hash(self.manifest, arrays)
        recorded = self.manifest.get("content_sha256")
        if verify_hash and recorded is not None and recorded != content_hash:
            raise ValueError("privileged admission content hash mismatch")
        return {"proposals": proposals, "content_sha256": content_hash}

    def save(self, path: str | Path, ledger: AdmissionLedger) -> Path:
        output = assert_development_path(assert_safe_evidence_output(path))
        if output.suffix != ".npz":
            raise ValueError("privileged admission view must use .npz")
        report = self.validate(ledger, verify_hash=False)
        manifest = copy.deepcopy(self.manifest)
        manifest["content_sha256"] = report["content_sha256"]
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            manifest_json=np.asarray(json.dumps(
                manifest, sort_keys=True, separators=(",", ":"))),
            proposal_id=self.proposal_id,
            state_hash=self.state_hash,
            initial_tilt_rad=self.initial_tilt_rad,
            initial_height_m=self.initial_height_m,
            max_tilt_rad=self.max_tilt_rad,
            min_height_m=self.min_height_m,
        )
        self.manifest = manifest
        self.path = output
        return output

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        ledger: AdmissionLedger,
    ) -> "AdmissionPrivilegedView":
        try:
            source = assert_development_path(
                require_workflow_authorized_or_safe_input(
                    path, allowed_roles=("admission_privileged",)))
        except ProtectedEvidencePathError as exc:
            raise ValueError(str(exc)) from exc
        with np.load(source, allow_pickle=False) as payload:
            manifest = json.loads(str(payload["manifest_json"].item()))
            value = cls(
                manifest=manifest,
                proposal_id=payload["proposal_id"].copy(),
                state_hash=payload["state_hash"].copy(),
                initial_tilt_rad=payload["initial_tilt_rad"].copy(),
                initial_height_m=payload["initial_height_m"].copy(),
                max_tilt_rad=payload["max_tilt_rad"].copy(),
                min_height_m=payload["min_height_m"].copy(),
                path=source,
            )
        value.validate(ledger)
        return value


@dataclass(frozen=True)
class ClosedLoopRecoveryCollectionConfig:
    source_seed: int
    policy_training_step: int
    policy_training_seed: int = 42
    target_groups: int = 64
    horizon_steps: int = 96
    admission_replicas: int = 32
    admission_min_falls: int = 6
    admission_max_falls: int = 26
    discovery_replicas: int = 64
    audit_replicas: int = 64
    max_episode_steps: int = 100
    max_proposals: int = 4096
    max_trajectories: int = 2048
    proposal_cooldown_steps: int = 5
    settle_seconds: float = 0.04
    source_impulse_interval_steps: int = 10
    source_linear_std_mps: float = 1.0
    source_angular_std_radps: float = 4.0
    proposal_min_tilt_rad: float = 0.10
    proposal_max_height_m: float = 0.32
    seed_domain: bytes = _V3_SEED_DOMAIN
    seed_role_tags: tuple[tuple[str, int], ...] = _V3_ROLE_TAG_ITEMS
    seed_algorithm: str = _SHA256_LOW63_SEED_ALGORITHM
    dataset_split_prefix: str = "closed_loop_recovery_v3"
    collection_protocol_version: str = COLLECTION_PROTOCOL_VERSION
    trajectory_id_prefix: str = "closed-loop-v3"
    explicit_filter_settings_in_action_contract: bool = False

    def __post_init__(self) -> None:
        positive = (
            "target_groups", "horizon_steps", "admission_replicas",
            "discovery_replicas", "audit_replicas", "max_episode_steps",
            "max_proposals", "max_trajectories",
            "source_impulse_interval_steps",
        )
        nonnegative = (
            "source_seed", "policy_training_step", "policy_training_seed",
            "proposal_cooldown_steps",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                    value, (int, np.integer)) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(
                    value, (int, np.integer)) or int(value) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 0 <= self.admission_min_falls <= self.admission_max_falls <= (
                self.admission_replicas):
            raise ValueError("admission bounds must lie within replica count")
        if self.target_groups > self.max_proposals:
            raise ValueError("target_groups cannot exceed max_proposals")
        if self.target_groups > self.max_trajectories:
            raise ValueError("target_groups cannot exceed max_trajectories")
        if self.horizon_steps > np.iinfo(np.int16).max - 1:
            raise ValueError(
                "horizon_steps must be at most int16.max - 1 so H+1 is "
                "representable")
        for name in (
            "settle_seconds", "source_linear_std_mps",
            "source_angular_std_radps", "proposal_min_tilt_rad",
            "proposal_max_height_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        # Exercise the same validation used by every generated stream while
        # keeping the historical defaults bit-identical.
        _derived_seed(
            0, 0, "source_reset", 0,
            seed_domain=self.seed_domain,
            role_tags=self.seed_role_tags,
            seed_algorithm=self.seed_algorithm,
        )
        if self.seed_algorithm == _INJECTIVE_V4_SEED_ALGORITHM:
            if int(self.source_seed) >= 1 << 14:
                raise ValueError(
                    "injective V4 source_seed exceeds its 14-bit cap")
            if any(int(tag) >= 1 << 8 for _, tag in self.seed_role_tags):
                raise ValueError(
                    "injective V4 role_tag exceeds its 8-bit cap")
            if int(self.max_proposals) > 1 << 18:
                raise ValueError(
                    "injective V4 max_proposals exceeds proposal identity cap")
            if int(self.max_trajectories) > 1 << 18 or (
                    int(self.max_trajectories)
                    * int(self.max_episode_steps) > 1 << 18):
                raise ValueError(
                    "injective V4 trajectory/step identity exceeds its "
                    "18-bit cap")
            if any(int(value) > 1 << 6 for value in (
                    self.admission_replicas,
                    self.discovery_replicas,
                    self.audit_replicas)):
                raise ValueError(
                    "injective V4 replica count exceeds its 6-bit index cap")
        for name in (
                "dataset_split_prefix", "collection_protocol_version",
                "trajectory_id_prefix"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be nonempty text")
        if not isinstance(
                self.explicit_filter_settings_in_action_contract,
                (bool, np.bool_),
        ):
            raise ValueError(
                "explicit_filter_settings_in_action_contract must be boolean")


@dataclass(frozen=True)
class ClosedLoopRecoveryCollectionResult:
    admission: AdmissionLedger
    admission_privileged: AdmissionPrivilegedView
    discovery: GroupedBranchDataset
    discovery_privileged: PrivilegedBranchView
    audit: GroupedBranchDataset
    audit_privileged: PrivilegedBranchView
    source_steps: int
    trajectories: int
    proposals: int


@dataclass(frozen=True)
class ClosedLoopRecoveryCollectionPreflight:
    """Outcome-free objects whose construction can fail deterministically."""

    env: Any
    early_policy: Any
    recovery_program: Any
    config: ClosedLoopRecoveryCollectionConfig
    generator_commit: str
    protocol_sha256: str
    protocol_contract_sha256: str
    policy_set_manifest: dict[str, Any]
    source_fingerprint: str
    candidate_protocol: dict[str, Any]
    recovery_program_binding: dict[str, Any]
    discovery_assembler: GroupedBranchAssembler
    audit_assembler: GroupedBranchAssembler


def merge_admission_ledgers(
    ledgers: list[AdmissionLedger] | tuple[AdmissionLedger, ...],
) -> AdmissionLedger:
    """Merge leaf proposal ledgers in the same locked source-shard order."""
    items = list(ledgers)
    if len(items) < 2:
        raise ValueError("at least two admission ledgers are required")
    reports = [item.validate() for item in items]
    reference = items[0].manifest
    locked_fields = (
        "schema_version", "feature_view", "generator_commit",
        "protocol_sha256", "protocol_contract_sha256",
        "fall_definition", "simulator_fingerprint", "source_policy",
        "continuation_policy", "action_application_contract",
        "admission_replicas", "horizon_steps",
        "accept_min_falls_inclusive", "accept_max_falls_inclusive",
        "all_proposals_recorded", "candidate_outcomes_used_for_admission",
    )
    for index, item in enumerate(items):
        for name in locked_fields:
            if item.manifest.get(name) != reference.get(name):
                raise ValueError(
                    f"admission shard {index} changes locked field {name}")
        if set(item.arrays) != set(items[0].arrays):
            raise ValueError(f"admission shard {index} array fields differ")
    for name in (
        "proposal_id", "state_hash", "admission_crn_id",
        "admission_rollout_seed", "admission_perturbation_seed",
    ):
        flattened = [np.asarray(item[name]).reshape(-1) for item in items]
        combined = np.concatenate(flattened)
        if len(np.unique(combined)) != combined.size:
            raise ValueError(f"admission shards overlap on {name}")
    if all("admission_candidate_seed" in item.arrays for item in items):
        combined = np.concatenate([
            np.asarray(item["admission_candidate_seed"]).reshape(-1)
            for item in items
        ])
        if len(np.unique(combined)) != combined.size:
            raise ValueError("admission shards overlap on admission_candidate_seed")
    arrays = {
        name: np.concatenate([np.asarray(item[name]) for item in items], axis=0)
        for name in sorted(items[0].arrays)
    }
    arrays["proposal_index"] = np.arange(
        len(arrays["proposal_id"]), dtype=np.int64)
    accepted = np.asarray(arrays["accepted"], dtype=bool)
    arrays["accepted_group_index"] = np.full(
        len(accepted), -1, dtype=np.int64)
    arrays["accepted_group_index"][accepted] = np.arange(
        np.count_nonzero(accepted), dtype=np.int64)
    manifest = {
        name: copy.deepcopy(reference[name]) for name in locked_fields
    }
    manifest["source_seeds"] = [
        int(item.manifest["source_seed"]) for item in items]
    manifest["policy_training_steps"] = [
        int(item.manifest["policy_training_step"]) for item in items]
    if len(set(manifest["source_seeds"])) != len(items):
        raise ValueError("admission source seed shards must be unique")
    manifest["shards"] = [
        {
            "ordinal": index,
            "source_seed": int(item.manifest["source_seed"]),
            "policy_training_step": int(item.manifest["policy_training_step"]),
            "proposals": int(report["proposals"]),
            "accepted": int(report["accepted"]),
            "content_sha256": str(report["content_sha256"]),
        }
        for index, (item, report) in enumerate(zip(items, reports, strict=True))
    ]
    merged = AdmissionLedger(manifest, arrays)
    report = merged.validate(verify_hash=False)
    merged.manifest["content_sha256"] = report["content_sha256"]
    return merged


def merge_admission_privileged_views(
    views: list[AdmissionPrivilegedView] | tuple[AdmissionPrivilegedView, ...],
    ledgers: list[AdmissionLedger] | tuple[AdmissionLedger, ...],
    merged_ledger: AdmissionLedger,
) -> AdmissionPrivilegedView:
    """Merge physically separate admission diagnostics in ledger order."""
    view_items = list(views)
    ledger_items = list(ledgers)
    if len(view_items) != len(ledger_items) or len(view_items) < 2:
        raise ValueError("privileged and admission shard counts must align")
    reports = [
        view.validate(ledger)
        for view, ledger in zip(view_items, ledger_items, strict=True)
    ]
    generator_commit = view_items[0].manifest.get("generator_commit")
    protocol_sha256 = view_items[0].manifest.get("protocol_sha256")
    protocol_contract_sha256 = view_items[0].manifest.get(
        "protocol_contract_sha256")
    for index, view in enumerate(view_items):
        if view.manifest.get("generator_commit") != generator_commit or (
                view.manifest.get("protocol_sha256") != protocol_sha256) or (
                    view.manifest.get("protocol_contract_sha256")
                    != protocol_contract_sha256):
            raise ValueError(
                f"privileged admission shard {index} changes provenance")
    merged_report = merged_ledger.validate()
    merged = AdmissionPrivilegedView(
        manifest={
            "schema_version": ADMISSION_PRIVILEGED_SCHEMA_VERSION,
            "feature_view": "privileged_admission_diagnostic_only",
            "generator_commit": generator_commit,
            "protocol_sha256": protocol_sha256,
            "protocol_contract_sha256": protocol_contract_sha256,
            "deployable_content_sha256": merged_report["content_sha256"],
            "shards": [
                {
                    "ordinal": index,
                    "content_sha256": report["content_sha256"],
                    "proposals": report["proposals"],
                }
                for index, report in enumerate(reports)
            ],
        },
        proposal_id=np.concatenate([view.proposal_id for view in view_items]),
        state_hash=np.concatenate([view.state_hash for view in view_items]),
        initial_tilt_rad=np.concatenate([
            view.initial_tilt_rad for view in view_items]),
        initial_height_m=np.concatenate([
            view.initial_height_m for view in view_items]),
        max_tilt_rad=np.concatenate([
            view.max_tilt_rad for view in view_items], axis=0),
        min_height_m=np.concatenate([
            view.min_height_m for view in view_items], axis=0),
    )
    report = merged.validate(merged_ledger, verify_hash=False)
    merged.manifest["content_sha256"] = report["content_sha256"]
    return merged


def _common_collection_manifest(
    *,
    role: str,
    config: ClosedLoopRecoveryCollectionConfig,
    protocol_sha256: str,
    protocol_contract_sha256: str,
) -> dict[str, Any]:
    result = {
        "version": config.collection_protocol_version,
        "role": role,
        "scope": "conditional_development_mechanism_triage_only",
        "protocol_sha256": protocol_sha256,
        "protocol_contract_sha256": protocol_contract_sha256,
        "selection_timing": "admission_before_candidate_outcomes",
        "admission": {
            "replicas": int(config.admission_replicas),
            "horizon_steps": int(config.horizon_steps),
            "accept_min_falls_inclusive": int(config.admission_min_falls),
            "accept_max_falls_inclusive": int(config.admission_max_falls),
            "candidate_outcomes_used": False,
        },
        "sampling_strata": {
            "admission_positive_conditional": {
                "predicate": "locked_nominal_admission_positive",
                "acceptance_probability": 1.0,
            },
        },
        "acceptance_probability_field_semantics": (
            "unit_analysis_weight_within_conditional_cohort_not_source_stream_"
            "inclusion_probability"),
        "natural_incidence_claim": False,
        "max_groups_per_trajectory": 1,
        "replica_seed_contract": "explicit_three_stream_v1",
        "physical_replica_role_files": True,
        "branch_disturbance": "zero",
    }
    if config.seed_domain != _V3_SEED_DOMAIN or config.seed_role_tags != (
            _V3_ROLE_TAG_ITEMS) or config.seed_algorithm != (
                _SHA256_LOW63_SEED_ALGORITHM):
        result["seed_derivation"] = seed_derivation_manifest(
            seed_domain=config.seed_domain,
            seed_role_tags=config.seed_role_tags,
            seed_algorithm=config.seed_algorithm,
        )
    return result


def _assembler(
    *,
    role: str,
    env: Any,
    config: ClosedLoopRecoveryCollectionConfig,
    generator_commit: str,
    protocol_sha256: str,
    protocol_contract_sha256: str,
    policy_set_manifest: Mapping[str, Any],
    candidate_protocol: Mapping[str, Any],
    recovery_program_binding: Mapping[str, Any],
) -> GroupedBranchAssembler:
    return GroupedBranchAssembler(
        split=f"{config.dataset_split_prefix}_{role}",
        horizon_steps=config.horizon_steps,
        generator_commit=generator_commit,
        simulator_fingerprint=env.simulator_fingerprint(),
        source_policy=policy_set_manifest,
        continuation_policy=policy_set_manifest,
        candidate_protocol=candidate_protocol,
        fall_definition=_fall_definition(env),
        action_application_contract=_action_application_contract(
            env,
            include_filter_settings=(
                config.explicit_filter_settings_in_action_contract),
        ),
        collection_protocol=_common_collection_manifest(
            role=role, config=config, protocol_sha256=protocol_sha256,
            protocol_contract_sha256=protocol_contract_sha256),
        recovery_program=recovery_program_binding,
        privileged_feature_names=PRIVILEGED_FEATURE_NAMES,
    )


def preflight_closed_loop_recovery_collection(
    *,
    env: Any,
    early_policy: Any,
    recovery_program: Any,
    policy_set_manifest: Mapping[str, Any],
    config: ClosedLoopRecoveryCollectionConfig,
    generator_commit: str,
    protocol_sha256: str,
    protocol_contract_sha256: str,
) -> ClosedLoopRecoveryCollectionPreflight:
    """Build all deterministic collection bindings before an attempt marker.

    The caller may safely run this before any simulator rollout.  The returned
    empty assemblers are then passed into the one-shot collector so policy,
    candidate, simulator-manifest, and schema construction cannot first fail
    after the attempt has already been consumed.
    """
    source_manifest, source_fingerprint = _policy_identity(
        early_policy, "early source")
    if int(source_manifest.get("training_step", -1)) != int(
            config.policy_training_step):
        raise ValueError("loaded early policy training step differs from config")
    for method_name in ("sample_action", "deterministic_action"):
        if not callable(getattr(early_policy, method_name, None)):
            raise TypeError(
                f"early source policy must expose {method_name}()")
    for method_name in (
        "reset_standing", "apply_base_velocity_impulse", "record_observation",
        "measurement", "capture", "restore", "step", "simulator_fingerprint",
    ):
        if not callable(getattr(env, method_name, None)):
            raise TypeError(f"collection environment must expose {method_name}()")
    manifest_method = getattr(recovery_program, "manifest_protocol", None)
    full_manifest_method = getattr(recovery_program, "manifest", None)
    fingerprint_method = getattr(recovery_program, "fingerprint", None)
    preview_method = getattr(recovery_program, "preview_projected", None)
    if not callable(manifest_method):
        raise TypeError("recovery_program must expose manifest_protocol()")
    if not callable(preview_method):
        raise TypeError("recovery_program must expose preview_projected()")
    if not callable(full_manifest_method) or not callable(fingerprint_method):
        raise TypeError(
            "recovery_program must expose manifest() and fingerprint()")
    candidate_protocol = copy.deepcopy(dict(manifest_method()))
    program_manifest = copy.deepcopy(dict(full_manifest_method()))
    program_fingerprint = fingerprint_method()
    if program_manifest.get("candidate_protocol") != candidate_protocol or not (
            isinstance(program_fingerprint, str)) or program_fingerprint != (
                canonical_protocol_sha256(program_manifest)):
        raise ValueError(
            "recovery program manifest/fingerprint/candidate binding drifted")
    recovery_program_binding = {
        "manifest": program_manifest,
        "fingerprint_sha256": program_fingerprint,
    }
    discovery_assembler = _assembler(
        role="discovery", env=env, config=config,
        generator_commit=generator_commit, protocol_sha256=protocol_sha256,
        protocol_contract_sha256=protocol_contract_sha256,
        policy_set_manifest=policy_set_manifest,
        candidate_protocol=candidate_protocol,
        recovery_program_binding=recovery_program_binding,
    )
    audit_assembler = _assembler(
        role="audit", env=env, config=config,
        generator_commit=generator_commit, protocol_sha256=protocol_sha256,
        protocol_contract_sha256=protocol_contract_sha256,
        policy_set_manifest=policy_set_manifest,
        candidate_protocol=candidate_protocol,
        recovery_program_binding=recovery_program_binding,
    )
    return ClosedLoopRecoveryCollectionPreflight(
        env=env,
        early_policy=early_policy,
        recovery_program=recovery_program,
        config=config,
        generator_commit=generator_commit,
        protocol_sha256=protocol_sha256,
        protocol_contract_sha256=protocol_contract_sha256,
        policy_set_manifest=copy.deepcopy(dict(policy_set_manifest)),
        source_fingerprint=source_fingerprint,
        candidate_protocol=candidate_protocol,
        recovery_program_binding=recovery_program_binding,
        discovery_assembler=discovery_assembler,
        audit_assembler=audit_assembler,
    )


def _finalize_admission(
    *,
    rows: list[dict[str, Any]],
    privileged_rows: list[dict[str, Any]],
    config: ClosedLoopRecoveryCollectionConfig,
    generator_commit: str,
    protocol_sha256: str,
    protocol_contract_sha256: str,
    fall_definition: Mapping[str, Any],
    simulator_fingerprint: Mapping[str, Any],
    source_policy: Mapping[str, Any],
    action_application_contract: Mapping[str, Any],
) -> tuple[AdmissionLedger, AdmissionPrivilegedView]:
    if not rows or len(rows) != len(privileged_rows):
        raise ValueError("admission rows must be nonempty and aligned")
    arrays = {
        name: np.stack([np.asarray(row[name]) for row in rows])
        if np.asarray(rows[0][name]).ndim > 0
        else np.asarray([row[name] for row in rows])
        for name in rows[0]
    }
    manifest = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "feature_view": "deployable_admission",
        "generator_commit": generator_commit,
        "protocol_sha256": protocol_sha256,
        "protocol_contract_sha256": protocol_contract_sha256,
        "fall_definition": copy.deepcopy(dict(fall_definition)),
        "simulator_fingerprint": copy.deepcopy(dict(simulator_fingerprint)),
        "source_policy": copy.deepcopy(dict(source_policy)),
        "continuation_policy": copy.deepcopy(dict(source_policy)),
        "action_application_contract": copy.deepcopy(
            dict(action_application_contract)),
        "source_seed": int(config.source_seed),
        "policy_training_step": int(config.policy_training_step),
        "admission_replicas": int(config.admission_replicas),
        "horizon_steps": int(config.horizon_steps),
        "accept_min_falls_inclusive": int(config.admission_min_falls),
        "accept_max_falls_inclusive": int(config.admission_max_falls),
        "all_proposals_recorded": True,
        "candidate_outcomes_used_for_admission": False,
    }
    ledger = AdmissionLedger(manifest, arrays)
    ledger_report = ledger.validate(verify_hash=False)
    ledger.manifest["content_sha256"] = ledger_report["content_sha256"]
    privileged = AdmissionPrivilegedView(
        manifest={
            "schema_version": ADMISSION_PRIVILEGED_SCHEMA_VERSION,
            "feature_view": "privileged_admission_diagnostic_only",
            "generator_commit": generator_commit,
            "protocol_sha256": protocol_sha256,
            "protocol_contract_sha256": protocol_contract_sha256,
            "deployable_content_sha256": ledger_report["content_sha256"],
        },
        proposal_id=np.asarray([row["proposal_id"] for row in rows]),
        state_hash=np.asarray([row["state_hash"] for row in rows]),
        initial_tilt_rad=np.asarray([
            row["initial_tilt_rad"] for row in privileged_rows],
            dtype=np.float32),
        initial_height_m=np.asarray([
            row["initial_height_m"] for row in privileged_rows],
            dtype=np.float32),
        max_tilt_rad=np.stack([
            row["max_tilt_rad"] for row in privileged_rows]).astype(np.float32),
        min_height_m=np.stack([
            row["min_height_m"] for row in privileged_rows]).astype(np.float32),
    )
    privileged_report = privileged.validate(ledger, verify_hash=False)
    privileged.manifest["content_sha256"] = privileged_report["content_sha256"]
    return ledger, privileged


def collect_preflighted_closed_loop_recovery_triage(
    *,
    preflight: ClosedLoopRecoveryCollectionPreflight,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> ClosedLoopRecoveryCollectionResult:
    """Collect from one immutable, outcome-free preflight binding."""
    if not isinstance(preflight, ClosedLoopRecoveryCollectionPreflight):
        raise TypeError("preflight must come from the v3 preflight builder")
    prepared = preflight
    env = prepared.env
    early_policy = prepared.early_policy
    recovery_program = prepared.recovery_program
    config = prepared.config
    generator_commit = prepared.generator_commit
    protocol_sha256 = prepared.protocol_sha256
    protocol_contract_sha256 = prepared.protocol_contract_sha256
    source_fingerprint = prepared.source_fingerprint
    discovery_assembler = prepared.discovery_assembler
    audit_assembler = prepared.audit_assembler

    if discovery_assembler.group_count != 0 or audit_assembler.group_count != 0:
        raise ValueError("preflight assemblers must be unused")

    admission_rows: list[dict[str, Any]] = []
    admission_privileged_rows: list[dict[str, Any]] = []
    preassigned_audit_randomness: list[GroupRandomness] = []
    source_steps = 0
    episode_number = 0
    episode_step = 0
    last_proposal_step = -config.proposal_cooldown_steps

    def reset() -> None:
        nonlocal episode_step, last_proposal_step
        if episode_number >= config.max_trajectories:
            raise RuntimeError(
                "collector exhausted max_trajectories before target groups")
        env.reset_standing(
            settle_seconds=config.settle_seconds,
            rng=np.random.default_rng(_derived_seed(
                config.source_seed, episode_number, "source_reset", 0,
                seed_domain=config.seed_domain,
                role_tags=config.seed_role_tags,
                seed_algorithm=config.seed_algorithm)),
        )
        episode_step = 0
        last_proposal_step = -config.proposal_cooldown_steps

    reset()
    while discovery_assembler.group_count < config.target_groups:
        if len(admission_rows) >= config.max_proposals:
            raise RuntimeError(
                "collector exhausted max_proposals before target groups")
        if episode_step > 0 and episode_step % (
                config.source_impulse_interval_steps) == 0:
            impulse_rng = np.random.default_rng(_derived_seed(
                config.source_seed,
                episode_number * config.max_episode_steps + episode_step,
                "source_impulse", 0,
                seed_domain=config.seed_domain,
                role_tags=config.seed_role_tags,
                seed_algorithm=config.seed_algorithm))
            env.apply_base_velocity_impulse(
                linear_velocity_delta=impulse_rng.normal(
                    0.0, config.source_linear_std_mps, size=3),
                angular_velocity_delta=impulse_rng.normal(
                    0.0, config.source_angular_std_radps, size=3),
            )
        history = env.record_observation()
        observation = history[-1]
        source_action = early_policy.sample_action(
            observation,
            np.random.default_rng(_derived_seed(
                config.source_seed,
                episode_number * config.max_episode_steps + episode_step,
                "source_action", 0,
                seed_domain=config.seed_domain,
                role_tags=config.seed_role_tags,
                seed_algorithm=config.seed_algorithm)),
        )
        measurement = env.measurement()
        if measurement.failure:
            episode_number += 1
            reset()
            continue
        cooldown_ready = (
            episode_step - last_proposal_step >= config.proposal_cooldown_steps)
        pre_screen = (
            float(measurement.tilt_rad) >= config.proposal_min_tilt_rad
            or float(measurement.height_m) <= config.proposal_max_height_m)
        accepted = False
        if cooldown_ready and pre_screen:
            proposal_index = len(admission_rows)
            last_proposal_step = episode_step
            snapshot = env.capture()
            if env.measurement().failure:
                raise RuntimeError("proposal snapshot is already a failure")
            state_hash = snapshot.compound_sha256()
            trajectory_id = (
                f"{config.trajectory_id_prefix}:source-{config.source_seed}:"
                f"trajectory-{episode_number}")
            proposal_id = f"{trajectory_id}:step-{episode_step}"
            admission_bundle, admission_randomness = role_randomness(
                source_seed=config.source_seed,
                proposal_index=proposal_index,
                replicas=config.admission_replicas,
                role="admission",
                seed_domain=config.seed_domain,
                role_tags=config.seed_role_tags,
                seed_algorithm=config.seed_algorithm,
            )
            nominal = early_policy.deterministic_action(observation)
            admission = evaluate_nominal_admission(
                env=env,
                snapshot=snapshot,
                nominal_first_action=nominal,
                seeds=admission_bundle,
                horizon_steps=config.horizon_steps,
                continuation_policy=early_policy,
            )
            fall_count = int(np.count_nonzero(admission.fall))
            accepted = bool(
                config.admission_min_falls <= fall_count
                <= config.admission_max_falls)
            accepted_group_index = (
                discovery_assembler.group_count if accepted else -1)
            admission_rows.append({
                "proposal_id": proposal_id,
                "proposal_index": proposal_index,
                "state_hash": state_hash,
                "trajectory_id": trajectory_id,
                "episode_id": _derived_seed(
                    config.source_seed, episode_number, "source_reset", 1,
                    seed_domain=config.seed_domain,
                    role_tags=config.seed_role_tags,
                    seed_algorithm=config.seed_algorithm),
                "episode_step": episode_step,
                "source_seed": config.source_seed,
                "policy_training_step": config.policy_training_step,
                "policy_source": source_fingerprint,
                "obs_history": history.copy(),
                "admission_crn_id": admission.crn_id,
                "admission_rollout_seed": admission.rollout_seed,
                "admission_perturbation_seed": admission.perturbation_seed,
                "admission_candidate_seed": admission_randomness.candidate_seed,
                "fall": admission.fall,
                "first_failure_step": admission.first_failure_step,
                "accepted": accepted,
                "accepted_group_index": accepted_group_index,
                "decision_reason": (
                    f"accepted_{config.admission_min_falls}_to_"
                    f"{config.admission_max_falls}_of_"
                    f"{config.admission_replicas}" if accepted else
                    f"rejected_outside_{config.admission_min_falls}_to_"
                    f"{config.admission_max_falls}_of_"
                    f"{config.admission_replicas}"),
            })
            admission_privileged_rows.append({
                "initial_tilt_rad": float(measurement.tilt_rad),
                "initial_height_m": float(measurement.height_m),
                "max_tilt_rad": admission.max_tilt_rad,
                "min_height_m": admission.min_height_m,
            })
            if accepted:
                candidates = recovery_program.preview_projected(history, nominal)
                discovery_bundle, discovery_randomness = role_randomness(
                    source_seed=config.source_seed,
                    proposal_index=proposal_index,
                    replicas=config.discovery_replicas,
                    role="discovery",
                    seed_domain=config.seed_domain,
                    role_tags=config.seed_role_tags,
                    seed_algorithm=config.seed_algorithm,
                )
                audit_bundle, audit_randomness = role_randomness(
                    source_seed=config.source_seed,
                    proposal_index=proposal_index,
                    replicas=config.audit_replicas,
                    role="audit",
                    seed_domain=config.seed_domain,
                    role_tags=config.seed_role_tags,
                    seed_algorithm=config.seed_algorithm,
                )
                preassigned_audit_randomness.append(audit_randomness)
                discovery = evaluate_same_state_group(
                    env,
                    snapshot,
                    candidates.requested,
                    discovery_bundle,
                    horizon_steps=config.horizon_steps,
                    continuation_policy=early_policy,
                    disturbance_program=None,
                    recovery_program=recovery_program,
                )
                audit = evaluate_same_state_group(
                    env,
                    snapshot,
                    candidates.requested,
                    audit_bundle,
                    horizon_steps=config.horizon_steps,
                    continuation_policy=early_policy,
                    disturbance_program=None,
                    recovery_program=recovery_program,
                )
                for evaluation in (discovery, audit):
                    for evaluation_name, candidate_name in (
                        ("candidate_requested", "requested"),
                        ("candidate_executed", "executed"),
                        ("candidate_q_target", "q_target"),
                    ):
                        if not np.array_equal(
                                np.asarray(getattr(evaluation, evaluation_name)),
                                np.asarray(getattr(candidates, candidate_name))):
                            raise RuntimeError(
                                "closed-loop execution disagrees with previewed "
                                f"{evaluation_name}")
                identity = GroupIdentity(
                    group_id=proposal_id,
                    state_hash=state_hash,
                    trajectory_id=trajectory_id,
                    episode_id=_derived_seed(
                        config.source_seed, episode_number, "source_reset", 1,
                        seed_domain=config.seed_domain,
                        role_tags=config.seed_role_tags,
                        seed_algorithm=config.seed_algorithm),
                    episode_step=episode_step,
                    policy_training_seed=config.policy_training_seed,
                    source_seed=config.source_seed,
                    policy_source=source_fingerprint,
                    command_vx=float(env.cfg.move_speed),
                    acceptance_probability=1.0,
                    sampling_stratum="admission_positive_conditional",
                )
                privileged = privileged_features(env)
                for assembler, evaluation, randomness in (
                    (discovery_assembler, discovery, discovery_randomness),
                    (audit_assembler, audit, audit_randomness),
                ):
                    assembler.add(CollectedGroup(
                        identity=identity,
                        observation_history=history,
                        candidate_kind=candidates.kind,
                        candidate_mask=candidates.mask,
                        evaluation=evaluation,
                        randomness=randomness,
                        candidate_behavior_steps=candidates.behavior_steps,
                        privileged_features=privileged,
                    ))
                if progress is not None:
                    progress({
                        "groups": discovery_assembler.group_count,
                        "target_groups": config.target_groups,
                        "proposals": len(admission_rows),
                        "source_steps": source_steps,
                        "trajectories": episode_number + 1,
                    })
                episode_number += 1
                if discovery_assembler.group_count < config.target_groups:
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
        generator_commit=generator_commit,
        protocol_sha256=protocol_sha256,
        protocol_contract_sha256=protocol_contract_sha256,
        fall_definition=_fall_definition(env),
        simulator_fingerprint=discovery_assembler.manifest[
            "simulator_fingerprint"],
        source_policy=prepared.policy_set_manifest,
        action_application_contract=discovery_assembler.manifest[
            "action_application_contract"],
    )
    discovery, discovery_privileged = discovery_assembler.finalize()
    audit, audit_privileged = audit_assembler.finalize()
    assert discovery_privileged is not None and audit_privileged is not None
    if len(preassigned_audit_randomness) != discovery.group_count:
        raise RuntimeError("preassigned audit seed rows do not align with groups")
    discovery.arrays["preassigned_audit_crn_id"] = np.stack([
        value.crn_id for value in preassigned_audit_randomness])
    discovery.arrays["preassigned_audit_rollout_seed"] = np.stack([
        value.rollout_seed for value in preassigned_audit_randomness])
    discovery.arrays["preassigned_audit_perturbation_seed"] = np.stack([
        value.perturbation_seed for value in preassigned_audit_randomness])
    discovery.arrays["preassigned_audit_candidate_seed"] = np.asarray([
        value.candidate_seed for value in preassigned_audit_randomness],
        dtype=np.uint64)
    discovery_report = discovery.validate(verify_hash=False)
    discovery_privileged.manifest["deployable_content_sha256"] = (
        discovery_report["content_sha256"])
    if not np.array_equal(discovery["group_id"], audit["group_id"]) or not (
            np.array_equal(discovery["state_hash"], audit["state_hash"])):
        raise RuntimeError("discovery and audit identities do not align")
    accepted = np.asarray(admission["accepted"], dtype=bool)
    for identity_name in ("proposal_id", "state_hash", "trajectory_id"):
        admission_name = (
            "group_id" if identity_name == "proposal_id" else identity_name)
        if not np.array_equal(
                np.asarray(admission[identity_name])[accepted].astype(str),
                np.asarray(discovery[admission_name]).astype(str)):
            raise RuntimeError(
                f"accepted admission {identity_name} does not align with "
                "discovery groups")
    if len(np.unique(np.asarray(admission["trajectory_id"])[accepted])) != (
            discovery.group_count):
        raise RuntimeError("accepted groups must use one source trajectory each")
    for preassigned_name, audit_name in (
        ("preassigned_audit_crn_id", "crn_id"),
        ("preassigned_audit_rollout_seed", "rollout_seed"),
        ("preassigned_audit_perturbation_seed", "perturbation_seed"),
        ("preassigned_audit_candidate_seed", "candidate_seed"),
    ):
        if not np.array_equal(
                np.asarray(discovery[preassigned_name]),
                np.asarray(audit[audit_name])):
            raise RuntimeError(
                f"{preassigned_name} does not match the physical audit shard")
    role_seed_sets = {
        "admission": set(np.concatenate([
            np.asarray(admission[name], dtype=np.uint64).reshape(-1)
            for name in (
                "admission_crn_id", "admission_rollout_seed",
                "admission_perturbation_seed")
        ]).tolist()),
        "discovery": set(np.concatenate([
            np.asarray(discovery[name], dtype=np.uint64).reshape(-1)
            for name in (
                "crn_id", "rollout_seed", "perturbation_seed",
                "candidate_seed")
        ]).tolist()),
        "audit": set(np.concatenate([
            np.asarray(audit[name], dtype=np.uint64).reshape(-1)
            for name in (
                "crn_id", "rollout_seed", "perturbation_seed",
                "candidate_seed")
        ]).tolist()),
    }
    for left, right in (
        ("admission", "discovery"),
        ("admission", "audit"),
        ("discovery", "audit"),
    ):
        if role_seed_sets[left] & role_seed_sets[right]:
            raise RuntimeError(f"{left}/{right} RNG seed domains overlap")
    return ClosedLoopRecoveryCollectionResult(
        admission=admission,
        admission_privileged=admission_privileged,
        discovery=discovery,
        discovery_privileged=discovery_privileged,
        audit=audit,
        audit_privileged=audit_privileged,
        source_steps=source_steps,
        trajectories=episode_number,
        proposals=len(admission_rows),
    )


def collect_closed_loop_recovery_triage(
    *,
    env: Any,
    early_policy: Any,
    recovery_program: Any,
    policy_set_manifest: Mapping[str, Any],
    config: ClosedLoopRecoveryCollectionConfig,
    generator_commit: str,
    protocol_sha256: str,
    protocol_contract_sha256: str,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> ClosedLoopRecoveryCollectionResult:
    """Convenience API that preflights and then immediately collects."""
    prepared = preflight_closed_loop_recovery_collection(
        env=env,
        early_policy=early_policy,
        recovery_program=recovery_program,
        policy_set_manifest=policy_set_manifest,
        config=config,
        generator_commit=generator_commit,
        protocol_sha256=protocol_sha256,
        protocol_contract_sha256=protocol_contract_sha256,
    )
    return collect_preflighted_closed_loop_recovery_triage(
        preflight=prepared,
        progress=progress,
    )


__all__ = [
    "ADMISSION_PRIVILEGED_SCHEMA_VERSION",
    "ADMISSION_SCHEMA_VERSION",
    "AdmissionLedger",
    "AdmissionPrivilegedView",
    "ClosedLoopRecoveryCollectionConfig",
    "ClosedLoopRecoveryCollectionPreflight",
    "ClosedLoopRecoveryCollectionResult",
    "NominalReplicaEvaluation",
    "collect_closed_loop_recovery_triage",
    "collect_preflighted_closed_loop_recovery_triage",
    "canonical_protocol_sha256",
    "evaluate_nominal_admission",
    "merge_admission_ledgers",
    "merge_admission_privileged_views",
    "preflight_closed_loop_recovery_collection",
    "role_randomness",
    "seed_derivation_manifest",
]
