#!/usr/bin/env python3
"""Run an isolated Q_safe update-intensity diagnostic on a saved SQRL replay."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.agents import create_agent
from train.config import load_app_config
from train.loop import _agent_hashes


def _spaces(robot_cfg):
    observation_space = gym.spaces.Box(
        -100.0, 100.0, (robot_cfg.obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(
        -1.0, 1.0, (robot_cfg.num_joints,), dtype=np.float32)
    return observation_space, action_space


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def _flatten(replay):
    items = []
    near32 = []
    for trajectory in replay._trajectories:
        unsafe_indices = [
            index for index, item in enumerate(trajectory)
            if float(item["unsafe"]) >= 0.5
        ]
        labels = np.zeros(len(trajectory), dtype=np.bool_)
        for failure_index in unsafe_indices:
            labels[max(0, failure_index - 31):failure_index + 1] = True
        items.extend(trajectory)
        near32.extend(labels.tolist())
    return items, np.asarray(near32, dtype=np.bool_)


@torch.no_grad()
def _candidate_metrics(agent, observations: torch.Tensor, *,
                       seed: int) -> dict[str, float | int]:
    count = int(agent._cfg.safety_num_candidates)
    rng_state = torch.random.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state(agent._device)
        if agent._device.type == "cuda" else None)
    torch.manual_seed(seed)
    repeated = observations.to(agent._device).repeat_interleave(count, dim=0)
    candidate_actions, _ = agent._actor(
        observations=agent._actor_observations(repeated),
        training=False, sample=True)
    candidate_risks = agent._risk(repeated, candidate_actions).reshape(
        len(observations), count)
    minima = candidate_risks.min(dim=1).values.cpu().numpy()
    torch.random.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state, agent._device)
    epsilon = float(agent._cfg.safety_epsilon)
    return {
        "states": len(observations),
        "candidate_count": count,
        "candidate_mean_q": float(candidate_risks.mean().item()),
        "candidate_min_q_median": float(np.median(minima)),
        "candidate_min_q_q05": float(np.quantile(minima, 0.05)),
        "safe_set_coverage": float(np.mean(minima <= epsilon)),
    }


def _evaluate(agent, *, state_indices: np.ndarray,
              task_state_sets: dict[str, torch.Tensor]) -> dict[str, object]:
    items, near32 = _flatten(agent._safety_replay)
    observations = torch.as_tensor(
        np.stack([item["observation"] for item in items]),
        dtype=torch.float32, device=agent._device)
    actions = torch.as_tensor(
        np.stack([item["action"] for item in items]),
        dtype=torch.float32, device=agent._device)
    risks = agent._risk(observations, actions).cpu().numpy().reshape(-1)
    unsafe = np.asarray(
        [float(item["unsafe"]) >= 0.5 for item in items], dtype=np.bool_)

    safety_candidates = _candidate_metrics(
        agent, observations[state_indices], seed=87001)
    epsilon = float(agent._cfg.safety_epsilon)
    normal = ~unsafe
    return {
        "transitions": len(items),
        "trajectories": agent._safety_replay.trajectory_count,
        "unsafe_samples": int(unsafe.sum()),
        "normal_samples": int(normal.sum()),
        "mean_q": float(risks.mean()),
        "unsafe_mean_q": float(risks[unsafe].mean()) if unsafe.any() else None,
        "normal_mean_q": float(risks[normal].mean()) if normal.any() else None,
        "unsafe_auroc": _auc(unsafe, risks),
        "near_failure_h32_samples": int(near32.sum()),
        "near_failure_h32_auroc": _auc(near32, risks),
        "candidate_states": safety_candidates["states"],
        "candidate_count": safety_candidates["candidate_count"],
        "candidate_min_q_median": safety_candidates[
            "candidate_min_q_median"],
        "candidate_min_q_q05": safety_candidates["candidate_min_q_q05"],
        "safe_set_coverage": safety_candidates["safe_set_coverage"],
        "epsilon": epsilon,
        "task_replay": {
            name: _candidate_metrics(
                agent, states, seed=88001 + index)
            for index, (name, states) in enumerate(task_state_sets.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/go2_50hz_sqrl_paper_pretrain.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument(
        "--milestones", default="0,250,500,1000",
        help="Comma-separated cumulative Q_safe update counts to evaluate.")
    parser.add_argument("--evaluation-states", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.updates <= 0:
        raise ValueError("--updates must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if not (source / "agent/safety_critic.pt").exists():
        raise FileNotFoundError(f"invalid source checkpoint: {source}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)

    robot_cfg, _, agent_cfg = load_app_config(args.config, agent="paper_sqrl")
    agent_cfg.device_type = args.device
    agent_cfg.buffer_device_type = args.device
    observation_space, action_space = _spaces(robot_cfg)
    torch.manual_seed(args.seed)
    agent = create_agent(observation_space, action_space, {}, agent_cfg)
    agent.load(str(source / "agent"))
    agent.load_replay_buffer(str(source / "replay"))
    if not agent._safety_replay.can_sample():
        raise RuntimeError("source safety replay cannot be sampled")
    if agent._safety_replay.failure_count == 0:
        raise RuntimeError("source safety replay has no failure samples")

    before_hashes = _agent_hashes(agent)
    items, _ = _flatten(agent._safety_replay)
    rng = np.random.default_rng(args.seed)
    state_indices = rng.choice(
        len(items), min(args.evaluation_states, len(items)), replace=False)
    task_payload = torch.load(
        source / "replay/replay_buffer.pt", map_location="cpu",
        weights_only=False)
    task_count = int(task_payload["num_in_buffer"])
    task_observations = task_payload["observation"][:task_count]
    evaluation_count = min(args.evaluation_states, task_count)
    recent_pool = np.arange(max(0, task_count - 5000), task_count)
    task_state_sets = {
        "uniform": task_observations[rng.choice(
            task_count, evaluation_count, replace=False)],
        "recent_5000": task_observations[rng.choice(
            recent_pool, min(evaluation_count, len(recent_pool)),
            replace=False)],
    }
    milestones = sorted({
        int(value) for value in args.milestones.split(",") if value.strip()
    } | {0, args.updates})
    if milestones[0] < 0 or milestones[-1] > args.updates:
        raise ValueError("milestones must be between zero and --updates")

    evaluations = {"0": _evaluate(
        agent, state_indices=state_indices, task_state_sets=task_state_sets)}
    last_loss = None
    for update_index in range(1, args.updates + 1):
        info = agent._update_safety()
        if not info:
            raise RuntimeError("Q_safe update unexpectedly produced no metrics")
        last_loss = float(info["safety/loss"])
        if update_index in milestones:
            evaluations[str(update_index)] = _evaluate(
                agent, state_indices=state_indices,
                task_state_sets=task_state_sets)

    after_hashes = _agent_hashes(agent)
    if before_hashes["actor_hash"] != after_hashes["actor_hash"]:
        raise RuntimeError("actor changed during Q_safe-only diagnostic")
    if before_hashes["reward_critic_hash"] != after_hashes["reward_critic_hash"]:
        raise RuntimeError("reward critic changed during Q_safe-only diagnostic")
    if before_hashes["safety_critic_hash"] == after_hashes["safety_critic_hash"]:
        raise RuntimeError("safety critic did not change")

    shutil.copytree(source, output)
    agent.save(str(output / "agent"))
    agent.save_replay_buffer(str(output / "replay"))
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {
        "kind": "paper_sqrl_qsafe_update_intensity_diagnostic",
        "source_checkpoint": str(source),
        "output_checkpoint": str(output),
        "config": str(Path(args.config).resolve()),
        "seed": args.seed,
        "additional_safety_updates": args.updates,
        "last_loss": last_loss,
        "before_hashes": before_hashes,
        "after_hashes": after_hashes,
        "evaluations": evaluations,
        "git_commit": commit,
        "git_dirty": dirty,
        "strict_reproduction_result_unchanged": True,
    }
    (output / "recalibration_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
