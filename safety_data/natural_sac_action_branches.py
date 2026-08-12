"""Natural-SAC same-state action branches for action-conditioned Q_safe.

The natural PPO archive supervises state danger.  This module supplies the
missing counterfactual supervision needed to decide which SAC action is safer:
states are selected without reading candidate outcomes, then every candidate
is continued from the exact same native snapshot with paired RNG streams.
No force, impulse, recovery sequence, or settle transition is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from safety_data.candidates import (
    ACTOR_SAMPLE_COUNT,
    EvidenceCandidateConfig,
    InsufficientCandidateSupportError,
    build_evidence_candidates,
)
from safety_data.action_oracle_candidates import build_action_oracle_candidates
from safety_data.collector import (
    CollectedGroup,
    GroupIdentity,
    GroupRandomness,
    GroupedBranchAssembler,
    _action_application_contract,
)
from safety_data.native import ReplicaSeedBundle, evaluate_same_state_group
from safety_data.natural_sac_recovery import _snapshot_from_row


RISK_BAND_EDGES = (0.0, 0.25, 0.50, 0.75, 1.0)
EARLY_NEAR_FALL_RISK_MIN = 0.08
EARLY_NEAR_FALL_RISK_MAX = 0.60
EARLY_NEAR_FALL_UNCERTAINTY_MAX = 0.20


def _u64(domain: bytes, identity: bytes, *parts: int) -> int:
    digest = hashlib.sha256(domain + b"\0" + identity)
    for part in parts:
        digest.update(int(part).to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest()[:8], "little")


@dataclass(frozen=True)
class NaturalActionBranchPlan:
    row_index: np.ndarray
    identity: np.ndarray
    state_risk: np.ndarray
    state_uncertainty: np.ndarray
    risk_band: np.ndarray
    acceptance_probability: np.ndarray

    def __post_init__(self) -> None:
        count = len(np.asarray(self.row_index))
        values = {
            "row_index": np.asarray(self.row_index, dtype=np.int64),
            "identity": np.asarray(self.identity, dtype="S64"),
            "state_risk": np.asarray(self.state_risk, dtype=np.float32),
            "state_uncertainty": np.asarray(
                self.state_uncertainty, dtype=np.float32),
            "risk_band": np.asarray(self.risk_band, dtype=np.int8),
            "acceptance_probability": np.asarray(
                self.acceptance_probability, dtype=np.float64),
        }
        if any(value.shape != (count,) for value in values.values()):
            raise ValueError("natural action branch plan arrays must be one-dimensional")
        if count == 0 or len(np.unique(values["row_index"])) != count:
            raise ValueError("natural action branch plan must contain unique rows")
        if len(set(map(bytes, values["identity"]))) != count:
            raise ValueError("natural action branch plan identities must be unique")
        if not np.all(np.isfinite(values["state_risk"])) or np.any(
                values["state_risk"] < 0.0) or np.any(values["state_risk"] > 1.0):
            raise ValueError("state risk must be finite probabilities")
        if not np.all(np.isfinite(values["state_uncertainty"])) or np.any(
                values["state_uncertainty"] < 0.0):
            raise ValueError("state uncertainty must be finite and nonnegative")
        if np.any(values["risk_band"] < 0) or np.any(values["risk_band"] > 3):
            raise ValueError("risk band must lie in [0,3]")
        if not np.all(np.isfinite(values["acceptance_probability"])) or np.any(
                values["acceptance_probability"] <= 0.0) or np.any(
                    values["acceptance_probability"] > 1.0):
            raise ValueError("acceptance probability must lie in (0,1]")
        for name, value in values.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def build_risk_stratified_plan(
    *, identities: np.ndarray, state_risk: np.ndarray,
    state_uncertainty: np.ndarray | None = None, groups: int,
) -> NaturalActionBranchPlan:
    """Select deterministic early near-fall states before any branching.

    The absolute risk window rejects both ordinary low-risk walking and the
    most advanced, likely already unrecoverable states.  This admission rule
    reads neither the natural future label nor candidate branch outcomes.
    """
    identities = np.asarray(identities, dtype="S64")
    risk = np.asarray(state_risk, dtype=np.float64)
    uncertainty = (
        np.zeros_like(risk)
        if state_uncertainty is None
        else np.asarray(state_uncertainty, dtype=np.float64))
    if identities.ndim != 1 or risk.shape != identities.shape:
        raise ValueError("identities and state_risk must be aligned vectors")
    if uncertainty.shape != risk.shape or not np.all(np.isfinite(uncertainty)) or (
            np.any(uncertainty < 0.0)):
        raise ValueError("state_uncertainty must align and be nonnegative")
    if isinstance(groups, bool) or not isinstance(groups, (int, np.integer)) or (
            not 1 <= int(groups) <= len(risk)):
        raise ValueError("groups must be a positive count no larger than the source")
    if len(set(map(bytes, identities))) != len(identities):
        raise ValueError("source identities must be unique")
    if not np.all(np.isfinite(risk)) or np.any(risk < 0.0) or np.any(risk > 1.0):
        raise ValueError("state_risk must contain finite probabilities")

    eligible = (
        (risk >= EARLY_NEAR_FALL_RISK_MIN)
        & (risk <= EARLY_NEAR_FALL_RISK_MAX)
        & (uncertainty <= EARLY_NEAR_FALL_UNCERTAINTY_MAX)
    )
    eligible_rows = np.flatnonzero(eligible)
    if len(eligible_rows) < int(groups):
        raise ValueError(
            "source does not contain enough pre-outcome early near-fall states; "
            f"eligible={len(eligible_rows)}, required={int(groups)}")
    quantiles = np.quantile(risk[eligible_rows], RISK_BAND_EDGES)
    band = np.full(len(risk), -1, dtype=np.int8)
    band[eligible_rows] = np.searchsorted(
        quantiles[1:-1], risk[eligible_rows], side="right").astype(np.int8)
    quota = np.full(4, int(groups) // 4, dtype=np.int64)
    quota[:int(groups) % 4] += 1
    available = np.bincount(band[eligible_rows], minlength=4)
    # Empty/tiny bands can occur when calibrated scores tie.  Reassign their
    # unused quota deterministically to bands with remaining capacity.
    selected: list[int] = []
    selected_band: list[int] = []
    remaining = int(groups)
    used = np.zeros(4, dtype=np.int64)
    while remaining:
        made_progress = False
        for band_index in range(4):
            target = min(int(quota[band_index]), int(available[band_index]))
            if used[band_index] >= target:
                continue
            rows = np.flatnonzero(band == band_index)
            ordered = sorted(
                map(int, rows),
                key=lambda row: hashlib.sha256(
                    b"qsafe.natural_action_plan.v1\0" + bytes(identities[row])
                ).digest(),
            )
            row = ordered[int(used[band_index])]
            selected.append(row)
            selected_band.append(band_index)
            used[band_index] += 1
            remaining -= 1
            made_progress = True
            if remaining == 0:
                break
        if made_progress:
            continue
        capacities = available - used
        if not np.any(capacities > 0):
            raise RuntimeError("risk plan exhausted source rows")
        quota[int(np.argmax(capacities))] += remaining

    rows = np.asarray(selected, dtype=np.int64)
    bands = np.asarray(selected_band, dtype=np.int8)
    probabilities = np.asarray([
        used[int(value)] / available[int(value)] for value in bands
    ], dtype=np.float64)
    return NaturalActionBranchPlan(
        row_index=rows,
        identity=identities[rows],
        state_risk=risk[rows].astype(np.float32),
        state_uncertainty=uncertainty[rows].astype(np.float32),
        risk_band=bands,
        acceptance_probability=probabilities,
    )


class _SessionContinuation:
    def __init__(self, sample: Callable[[np.ndarray, np.random.Generator], np.ndarray]):
        self._sample = sample

    def __call__(self, history: np.ndarray, step: int,
                 rng: np.random.Generator) -> np.ndarray:
        del step
        return self._sample(np.asarray(history, dtype=np.float32)[-1], rng)


def collect_natural_action_groups(
    *, arrays: Mapping[str, np.ndarray], source_manifest: Mapping[str, Any],
    plan: NaturalActionBranchPlan, env: Any, policy: Any,
    sample_action: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    generator_commit: str, replicas: int = 4, horizon_steps: int = 96,
    candidate_config: EvidenceCandidateConfig | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> Any:
    """Branch one preselected natural-SAC source into a grouped dataset."""
    if isinstance(replicas, bool) or not isinstance(replicas, (int, np.integer)) or (
            int(replicas) <= 0):
        raise ValueError("replicas must be a positive integer")
    if int(horizon_steps) != 96:
        raise ValueError("Objective 1 natural action labels require H96")
    if source_manifest.get("external_force") != "verified_zero" or (
            source_manifest.get("recovery_executed") is not False):
        raise ValueError("source must be unforced natural SAC without recovery")
    if int(source_manifest.get("horizon_policy_steps", -1)) != 96:
        raise ValueError("source natural SAC horizon differs from H96")
    config = candidate_config or EvidenceCandidateConfig()
    candidate_protocol = config.manifest_protocol()
    candidate_protocol = {
        "version": "qsafe.action_oracle_candidates.v1",
        "count": 24,
        "nominal_index": 0,
        "base_local_protocol": candidate_protocol,
        "state_dependent_slots": list(range(16, 24)),
        "protocol_resolved_per_state_by": "build_action_oracle_candidates",
    }
    policy_manifest = policy.manifest()
    policy_fingerprint = policy.fingerprint()
    assembler = GroupedBranchAssembler(
        # This first product exists to validate candidate-space headroom.  It
        # must not silently become model-fit data before the oracle gate passes.
        split="natural_sac_action_oracle_development",
        horizon_steps=96,
        generator_commit=generator_commit,
        simulator_fingerprint=env.simulator_fingerprint(),
        source_policy=policy_manifest,
        continuation_policy=policy_manifest,
        candidate_protocol=candidate_protocol,
        fall_definition={
            "max_abs_roll_pitch_rad": float(env.cfg.fallen_orientation_rad),
            "min_base_height_m": 0.18,
        },
        action_application_contract=_action_application_contract(env),
        collection_protocol={
            "version": "qsafe.natural_sac_action_branches.v1",
            "selection_timing": "before_candidate_outcomes",
            "selection_inputs": ["snapshot_identity", "calibrated_state_risk"],
            "state_uncertainty_input": "calibrated_ensemble_uncertainty",
            "early_near_fall_risk_interval_inclusive": [
                EARLY_NEAR_FALL_RISK_MIN, EARLY_NEAR_FALL_RISK_MAX],
            "early_near_fall_uncertainty_max_inclusive": (
                EARLY_NEAR_FALL_UNCERTAINTY_MAX),
            "purpose": "candidate_space_oracle_gate_before_model_training",
            "model_training_authorized": False,
            "branch_outcome_used_for_selection": False,
            "natural_fall_label_used_for_selection": False,
            "external_force": "verified_zero",
            "impulse": "forbidden",
            "settle_after_restore": False,
            "same_state_common_random_numbers": True,
            "risk_band_edges": list(RISK_BAND_EDGES),
            "replicas": int(replicas),
        },
    )
    continuation = _SessionContinuation(sample_action)
    skipped = 0
    for plan_index, row in enumerate(plan.row_index):
        row = int(row)
        if bytes(arrays["identity"][row]) != bytes(plan.identity[plan_index]):
            raise RuntimeError("branch plan identity differs from source row")
        snapshot = _snapshot_from_row(arrays, row)
        env.restore(snapshot)
        if np.any(env.data.xfrc_applied != 0.0):
            raise RuntimeError("natural action snapshot contains external force")
        history = env.observation_history()
        identity = bytes(plan.identity[plan_index])
        nominal = np.asarray(arrays["action_requested"][row], dtype=np.float32)
        deterministic = policy.deterministic_action(history[-1])
        actor_samples = np.stack([
            sample_action(
                history[-1],
                np.random.default_rng(_u64(
                    b"qsafe.natural_action_candidate_actor.v1", identity, index)),
            )
            for index in range(ACTOR_SAMPLE_COUNT)
        ])
        candidate_seed = _u64(b"qsafe.natural_action_candidates.v1", identity)
        try:
            local_candidates = build_evidence_candidates(
                nominal=nominal,
                deterministic_mean=deterministic,
                previous_requested=env.previous_action_requested,
                actor_samples=actor_samples,
                action_applier=env.action_applier,
                current_qpos=np.asarray(
                    env.data.qpos[env.qpos_addresses], dtype=np.float32),
                candidate_seed=candidate_seed,
                config=config,
            )
            candidates = build_action_oracle_candidates(
                local_candidates, observation_history=history,
                action_applier=env.action_applier)
        except InsufficientCandidateSupportError:
            skipped += 1
            continue
        crn = np.asarray([
            _u64(b"qsafe.natural_action_crn.v1", identity, replica)
            for replica in range(int(replicas))
        ], dtype=np.uint64)
        rollout = np.asarray([
            _u64(b"qsafe.natural_action_rollout.v1", identity, replica)
            for replica in range(int(replicas))
        ], dtype=np.uint64)
        perturbation = np.asarray([
            _u64(b"qsafe.natural_action_no_perturb.v1", identity, replica)
            for replica in range(int(replicas))
        ], dtype=np.uint64)
        randomness = GroupRandomness(
            crn_id=crn, rollout_seed=rollout,
            perturbation_seed=perturbation, candidate_seed=candidate_seed)
        evaluation = evaluate_same_state_group(
            env, snapshot, candidates.requested,
            ReplicaSeedBundle(crn, rollout, perturbation),
            horizon_steps=96, continuation_policy=continuation,
            disturbance_program=None,
        )
        if not np.array_equal(evaluation.candidate_q_target, candidates.q_target):
            raise RuntimeError("branch first actions differ from candidate preview")
        episode_id = int(arrays["episode_id"][row])
        episode_step = int(arrays["episode_step"][row])
        source_seed = int(source_manifest["source_seed"])
        trajectory = f"natural-sac:{source_seed}:episode-{episode_id}"
        assembler.add(CollectedGroup(
            identity=GroupIdentity(
                group_id=f"{trajectory}:step-{episode_step}",
                state_hash=snapshot.compound_sha256(),
                trajectory_id=trajectory,
                episode_id=episode_id,
                episode_step=episode_step,
                policy_training_seed=int(source_manifest["actor_seed"]),
                source_seed=source_seed,
                policy_source=policy_fingerprint,
                command_vx=0.30,
                acceptance_probability=float(
                    plan.acceptance_probability[plan_index]),
                sampling_stratum=f"calibrated_risk_band_{int(plan.risk_band[plan_index])}",
            ),
            observation_history=np.asarray(history, dtype=np.float32),
            candidate_kind=candidates.kind,
            candidate_mask=candidates.mask,
            evaluation=evaluation,
            randomness=randomness,
        ))
        if progress is not None:
            progress({
                "groups": assembler.group_count,
                "planned_groups": len(plan.row_index),
                "skipped_candidate_support": skipped,
                "valid_candidates": candidates.valid_count,
                "fall_fraction": float(np.mean(
                    evaluation.fall[candidates.mask])),
            })
    dataset, privileged = assembler.finalize()
    if privileged is not None:
        raise RuntimeError("natural action dataset unexpectedly produced privileged data")
    return dataset


def validate_natural_source_manifest(
    manifest_path: str | Path, source_path: str | Path,
    *, sha256: Callable[[Path], str],
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    source_path = Path(source_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "qsafe.natural_sac_states.v1":
        raise ValueError("source manifest schema is not natural SAC v1")
    if manifest.get("output_sha256") != sha256(source_path):
        raise ValueError("source manifest hash differs from source data")
    if manifest.get("external_force") != "verified_zero" or (
            manifest.get("recovery_executed") is not False):
        raise ValueError("source was forced or executed recovery")
    if int(manifest.get("horizon_policy_steps", -1)) != 96:
        raise ValueError("source labels are not H96")
    return manifest
