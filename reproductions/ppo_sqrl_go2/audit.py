"""Fresh-state K16/R8/H96 audit of formal-v2 SQRL rejection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reproductions.sqrl_go2.algo.networks import TanhGaussianActor
from reproductions.sqrl_go2.algo.safety_critic import (
    SafetyCriticConfig as FormalSafetyConfig,
    SafetyCriticLearner as FormalSafetyLearner,
)
from safety_data.native import ReplicaSeedBundle, evaluate_same_state_group
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv

from .checkpoint import module_sha256
from .io import write_json_no_clobber
from .protocol import Protocol, reserve_output_directory
from .statistics import (
    matched_state_only_rejection, risk_enrichment,
    state_cluster_bootstrap_difference,
)


def _u64(domain: bytes, seed: int, *parts: int) -> int:
    digest = hashlib.sha256(domain + b"\0" + int(seed).to_bytes(8, "little"))
    for part in parts:
        digest.update(int(part).to_bytes(8, "little"))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


class FrozenFormalPolicy:
    def __init__(self, actor: TanhGaussianActor, safety: torch.nn.Module,
                 env: MujocoSnapshotEnv, *, epsilon: float, candidates: int):
        self.actor = actor.eval()
        self.safety = safety.eval()
        self.env = env
        self.epsilon = float(epsilon)
        self.candidates = int(candidates)

    def sample(self, history: np.ndarray, count: int,
               rng: np.random.Generator) -> np.ndarray:
        observation = torch.from_numpy(np.asarray(history, np.float32).reshape(1, -1))
        with torch.no_grad():
            mean, log_std = self.actor(observation)
        noise = torch.from_numpy(rng.standard_normal((count, 12)).astype(np.float32))
        return torch.tanh(mean.expand(count, -1) + log_std.exp().expand(
            count, -1) * noise).numpy().astype(np.float32)

    def projected_and_risk(self, history: np.ndarray,
                           candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current_q = self.env.data.qpos[self.env.qpos_addresses]
        projected = self.env.action_applier.preview_many(candidates, current_q)
        executed = np.stack([item.action_executed for item in projected])
        observation = torch.from_numpy(np.asarray(history, np.float32).reshape(1, -1)).expand(
            len(candidates), -1)
        with torch.no_grad():
            risk = self.safety(observation, torch.from_numpy(executed)).numpy()
        return executed.astype(np.float32), risk.astype(np.float32)

    def __call__(self, history: np.ndarray, step: int,
                 rng: np.random.Generator) -> np.ndarray:
        candidates = self.sample(history, self.candidates, rng)
        _, risk = self.projected_and_risk(history, candidates)
        safe = np.flatnonzero(risk <= self.epsilon)
        return candidates[int(safe[0]) if len(safe) else int(np.argmin(risk))]


def load_formal_seed(seed: int, formal_root: str | Path,
                     device: str = "cpu"):
    root = Path(formal_root) / f"seed_{seed}" / "pretrain_030"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    actor_cfg = payload["sac"]["config"]
    actor = TanhGaussianActor(
        actor_cfg["observation_dim"], actor_cfg["action_dim"],
        tuple(actor_cfg["hidden_dims"]))
    actor.load_state_dict(payload["sac"]["actor"])
    safety_cfg = FormalSafetyConfig(**payload["safety"]["config"])
    safety = FormalSafetyLearner(safety_cfg, device=device)
    safety.load_checkpoint(payload["safety"], load_optimizer=False)
    if module_sha256(actor) != manifest["actor_sha256"] or module_sha256(
            safety.critic) != manifest["safety_sha256"]:
        raise ValueError(f"formal-v2 lineage hash failed for seed {seed}")
    if int(manifest["seed"]) != seed or manifest["phase"] != "pretrain":
        raise ValueError("formal-v2 seed/phase lineage differs")
    return actor, safety.critic, manifest


def _fresh_state_hashes(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.glob("**/*.npz"):
        try:
            with np.load(path, allow_pickle=False) as loaded:
                for field in ("snapshot_sha256", "state_hash", "identity",
                              "episode_identity", "rng_identity"):
                    if field in loaded:
                        result.update(str(value) for value in loaded[field].reshape(-1))
        except (OSError, ValueError):
            continue
    return result


def collect_audit_seed(
    *, seed: int, formal_root: str | Path, output: str | Path,
    model_path: str | Path = "/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml",
    config_path: str | Path = "reproductions/sqrl_go2/config/target_040.yaml",
    historical_root: str | Path = "saved", protocol: Protocol | None = None,
) -> dict[str, Any]:
    cfg = protocol or Protocol()
    output = reserve_output_directory(output)
    actor, safety, lineage = load_formal_seed(seed, formal_root)
    robot, train, _ = load_app_config(config_path)
    if robot.move_speed != cfg.target_command or train.use_action_filter or (
            train.max_joint_delta is not None):
        raise ValueError("audit runtime differs from target protocol")
    env = MujocoSnapshotEnv(
        model_path, robot, policy_frequency=train.control_frequency,
        max_joint_delta=None, use_action_filter=False)
    policy = FrozenFormalPolicy(
        actor, safety, env, epsilon=cfg.epsilon_safe,
        candidates=cfg.mask_candidates)
    historical = _fresh_state_hashes(Path(historical_root))
    snapshots = []
    identities = []
    episode_identities = []
    rng_identities = []
    candidates = []
    executed = []
    risks = []
    episode = 0
    while len(snapshots) < cfg.audit_states_per_seed:
        episode_rng = np.random.default_rng(_u64(b"ppo_sqrl.audit.episode", seed, episode))
        env.reset_standing(rng=episode_rng)
        selected_step = 32 + _u64(
            b"ppo_sqrl.audit.state_step", seed, episode) % (500 - 32)
        for step in range(500):
            history = env.record_observation()
            if step == selected_step:
                snapshot = env.capture()
                identity = snapshot.compound_sha256()
                if identity in historical or identity in identities:
                    raise RuntimeError("audit state overlaps historical or current corpus")
                candidate_rng = np.random.default_rng(
                    _u64(b"ppo_sqrl.audit.candidate", seed, episode))
                raw = policy.sample(history, cfg.audit_candidates, candidate_rng)
                projected, risk = policy.projected_and_risk(history, raw)
                snapshots.append(snapshot)
                identities.append(identity)
                episode_identities.append(hashlib.sha256(
                    f"ppo_sqrl.audit.episode/{seed}/{episode}".encode()).hexdigest())
                rng_identities.append(hashlib.sha256(
                    f"ppo_sqrl.audit.rng/{seed}/{episode}/{selected_step}".encode()).hexdigest())
                if episode_identities[-1] in historical or rng_identities[-1] in historical:
                    raise RuntimeError("audit episode or RNG identity overlaps history")
                candidates.append(raw)
                executed.append(projected)
                risks.append(risk)
            action = policy.sample(history, 1, episode_rng)[0]
            measurement = env.step(action)
            if measurement.failure:
                break
        episode += 1
    outcomes = np.zeros((cfg.audit_states_per_seed, cfg.audit_candidates,
                         cfg.audit_replicas), dtype=bool)
    first_failure = np.full_like(outcomes, cfg.audit_horizon + 1, dtype=np.int16)
    for state, (snapshot, raw) in enumerate(zip(snapshots, candidates, strict=True)):
        base = _u64(b"ppo_sqrl.audit.branch", seed, state)
        bundle = ReplicaSeedBundle(
            crn_id=np.arange(cfg.audit_replicas, dtype=np.uint64) + base,
            rollout_seed=np.arange(cfg.audit_replicas, dtype=np.uint64) + base + 10_000,
            perturbation_seed=np.arange(cfg.audit_replicas, dtype=np.uint64) + base + 20_000)
        result = evaluate_same_state_group(
            env, snapshot, raw, bundle, horizon_steps=cfg.audit_horizon,
            continuation_policy=policy)
        outcomes[state] = result.fall
        first_failure[state] = result.first_failure_step
    risk_values = np.stack(risks)
    action_rejected = risk_values > cfg.epsilon_safe
    state_scores = risk_values.mean(axis=1)
    state_rejected, state_threshold = matched_state_only_rejection(
        state_scores, action_rejected)
    action_summary = risk_enrichment(
        outcomes, np.broadcast_to(action_rejected[:, :, None], outcomes.shape))
    state_summary = risk_enrichment(
        outcomes, np.broadcast_to(state_rejected[:, :, None], outcomes.shape))
    np.savez_compressed(
        output / "audit_arrays.npz",
        identity=np.asarray(identities, dtype="S64"),
        episode_identity=np.asarray(episode_identities, dtype="S64"),
        rng_identity=np.asarray(rng_identities, dtype="S64"),
        candidate_requested=np.stack(candidates),
        candidate_executed=np.stack(executed),
        qsafe=risk_values, action_rejected=action_rejected,
        state_score=state_scores, state_rejected=state_rejected,
        fall=outcomes, first_failure_step=first_failure)
    report = {
        "schema_version": "ppo_sqrl_go2.mechanism_seed.v1",
        "seed": seed, "states": cfg.audit_states_per_seed,
        "candidates": cfg.audit_candidates, "replicas": cfg.audit_replicas,
        "horizon": cfg.audit_horizon,
        "lineage": {
            "checkpoint": lineage["checkpoint"],
            "actor_sha256": lineage["actor_sha256"],
            "safety_sha256": lineage["safety_sha256"],
        },
        "state_only_threshold": state_threshold,
        "action_conditioned": action_summary,
        "state_only": state_summary,
        "action_difference_ci95": state_cluster_bootstrap_difference(
            outcomes, np.broadcast_to(action_rejected[:, :, None], outcomes.shape),
            seed=cfg.bootstrap_seed + seed,
            replicates=cfg.bootstrap_replicates),
        "state_selection_reads_branch_outcomes": False,
        "historical_overlap_count": 0,
        "unique_episode_identity_count": len(set(episode_identities)),
        "unique_rng_identity_count": len(set(rng_identities)),
    }
    write_json_no_clobber(output / "manifest.json", report)
    return report
