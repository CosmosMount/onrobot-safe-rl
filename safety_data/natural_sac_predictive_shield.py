"""One-step successor-risk Q_safe shield for native SAC snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from safety_data.candidates import (
    ACTOR_SAMPLE_COUNT,
    EvidenceCandidateConfig,
    build_evidence_candidates,
)
from safety_data.natural_sac_calibration import CalibratedStateRiskPredictor
from safety_data.policies import FrozenDroQPolicy
from safety_data.natural_sac_recovery import _snapshot_from_row
from train.config import load_app_config
from train.mujoco_snapshot_env import BranchSnapshot, MujocoSnapshotEnv


def _rng(identity: bytes, arm: int, step: int, substream: int = 0) -> np.random.Generator:
    digest = hashlib.sha256(b"qsafe.predictive_shield.v1\0" + identity)
    for value in (arm, step, substream):
        digest.update(int(value).to_bytes(8, "little"))
    return np.random.default_rng(int.from_bytes(digest.digest()[:8], "little"))


@dataclass(frozen=True)
class PredictiveRollout:
    fall: bool
    first_failure_step: int
    interventions: int
    eligible_steps: int
    mean_selected_successor_risk: float


class _SessionActor:
    def __init__(self, policy: FrozenDroQPolicy, sample_action: Any) -> None:
        self.policy = policy
        self.sample = sample_action

    def sample_action(self, observation: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
        return self.sample(observation, rng)

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        return self.policy.deterministic_action(observation)


def _candidate_actions(
    env: MujocoSnapshotEnv, actor: FrozenDroQPolicy,
    observation_history: np.ndarray, nominal: np.ndarray,
    *, identity: bytes, step: int,
) -> Any:
    observation = observation_history[-1]
    deterministic = actor.deterministic_action(observation)
    samples = np.stack([
        actor.sample_action(observation, _rng(identity, 2, step, index))
        for index in range(ACTOR_SAMPLE_COUNT)
    ])
    seed = int(_rng(identity, 3, step).integers(0, np.iinfo(np.int64).max))
    return build_evidence_candidates(
        nominal=nominal, deterministic_mean=deterministic,
        previous_requested=env.previous_action_requested,
        actor_samples=samples, action_applier=env.action_applier,
        current_qpos=np.asarray(env.data.qpos[env.qpos_addresses], dtype=np.float32),
        candidate_seed=seed, config=EvidenceCandidateConfig())


def rollout_predictive_shield(
    env: MujocoSnapshotEnv, snapshot: BranchSnapshot, *,
    actor: FrozenDroQPolicy, predictor: CalibratedStateRiskPredictor,
    identity: bytes, horizon_steps: int = 96,
    risk_threshold: float = 0.16658837339093935,
    uncertainty_max: float = 0.20,
    arm: str = "shield",
) -> PredictiveRollout:
    """Run nominal, Q_safe successor selection, or matched-random selection."""
    if arm not in {"nominal", "shield", "placebo"}:
        raise ValueError("arm must be nominal, shield, or placebo")
    env.restore(snapshot)
    interventions = 0
    eligible_steps = 0
    selected_risks = []
    first_failure = horizon_steps + 1
    for step in range(horizon_steps):
        history = env.observation_history() if step == 0 else env.record_observation()
        nominal = actor.sample_action(history[-1], _rng(identity, 1, step))
        action = nominal
        if arm != "nominal":
            current_risk, current_std = predictor(history)
            if current_risk[0] >= risk_threshold and current_std[0] <= uncertainty_max:
                eligible_steps += 1
                candidates = _candidate_actions(
                    env, actor, history, nominal, identity=identity, step=step)
                branch = env.capture()
                valid_indices = np.flatnonzero(candidates.mask)
                successor_histories = []
                successor_fall = []
                for index in valid_indices:
                    env.restore(branch)
                    result = env.step(candidates.requested[index])
                    successor_fall.append(result.failure)
                    successor_histories.append(
                        history if result.failure else env.record_observation())
                successor_risk, successor_std = predictor(
                    np.asarray(successor_histories, dtype=np.float32))
                score = np.where(
                    np.asarray(successor_fall), 1.0,
                    np.clip(successor_risk + successor_std, 0.0, 1.0))
                if arm == "shield":
                    selected_local = int(np.argmin(score))
                else:
                    selected_local = int(_rng(identity, 4, step).integers(
                        0, len(valid_indices)))
                selected_index = int(valid_indices[selected_local])
                selected_risks.append(float(score[selected_local]))
                env.restore(branch)
                action = candidates.requested[selected_index]
                interventions += int(selected_index != 0)
        result = env.step(action)
        if result.failure:
            first_failure = step + 1
            break
    return PredictiveRollout(
        fall=first_failure <= horizon_steps,
        first_failure_step=first_failure,
        interventions=interventions,
        eligible_steps=eligible_steps,
        mean_selected_successor_risk=(
            float(np.mean(selected_risks)) if selected_risks else 0.0),
    )


def evaluate_predictive_plan_source(
    *, source_data: str | Path, source_manifest: str | Path,
    plan_path: str | Path, model_path: str | Path, output: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Evaluate three paired arms for one protected/development source."""
    import json
    import os

    source_data = Path(source_data).resolve()
    source_manifest = Path(source_manifest).resolve()
    plan_path = Path(plan_path).resolve()
    output = Path(output).resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError("predictive shield output already exists")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    with np.load(plan_path, allow_pickle=False) as loaded:
        mask = loaded["source_seed"] == int(manifest["source_seed"])
        plan = {name: loaded[name][mask].copy() for name in loaded.files}
    if limit is not None:
        for name in plan:
            plan[name] = plan[name][:int(limit)]
    if len(plan["identity"]) == 0:
        raise ValueError("plan has no rows for source")
    robot, train, _ = load_app_config(manifest["config_path"])
    policy = FrozenDroQPolicy  # retain explicit type identity near loader call
    from safety_data.policies import load_frozen_droq_policy
    actor = load_frozen_droq_policy(
        manifest["actor_manifest"]["actor_path"], manifest["config_path"],
        observation_dim=robot.obs_dim, action_dim=robot.num_joints,
        training_step=int(manifest["actor_training_step"]), device="cpu")
    del policy
    env = MujocoSnapshotEnv(
        manifest["model_path"], robot, policy_frequency=train.control_frequency,
        max_joint_delta=train.max_joint_delta, use_action_filter=False)
    predictor = CalibratedStateRiskPredictor(model_path, device="cpu")
    arms = ("nominal", "shield", "placebo")
    fall = np.empty((len(plan["identity"]), 3), dtype=bool)
    first_failure = np.empty((len(plan["identity"]), 3), dtype=np.int16)
    interventions = np.empty((len(plan["identity"]), 3), dtype=np.int16)
    eligible = np.empty((len(plan["identity"]), 3), dtype=np.int16)
    with np.load(source_data, allow_pickle=False) as arrays, actor.inference_session() as sample:
        session_actor = _SessionActor(actor, sample)
        for index, row in enumerate(plan["row_index"]):
            row = int(row)
            if bytes(arrays["identity"][row]) != bytes(plan["identity"][index]):
                raise RuntimeError("predictive plan/source identity mismatch")
            snapshot = _snapshot_from_row(arrays, row)
            for arm_index, arm in enumerate(arms):
                result = rollout_predictive_shield(
                    env, snapshot, actor=session_actor, predictor=predictor,
                    identity=bytes(plan["identity"][index]), arm=arm)
                fall[index, arm_index] = result.fall
                first_failure[index, arm_index] = result.first_failure_step
                interventions[index, arm_index] = result.interventions
                eligible[index, arm_index] = result.eligible_steps
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **plan, fall=fall,
                        first_failure_step=first_failure,
                        interventions=interventions, eligible_steps=eligible)
    os.link(temporary, output); temporary.unlink()
    report = {
        "schema_version": "qsafe.natural_sac_predictive_shield.v1",
        "source_seed": int(manifest["source_seed"]),
        "actor_seed": int(manifest["actor_seed"]),
        "actor_training_step": int(manifest["actor_training_step"]),
        "states": len(fall), "arms": list(arms),
        "fall_counts": fall.sum(axis=0).tolist(),
        "interventions": interventions.sum(axis=0).tolist(),
        "external_force": "verified_zero", "phase2_authorized": False,
    }
    report_path = output.with_suffix(".manifest.json")
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report
