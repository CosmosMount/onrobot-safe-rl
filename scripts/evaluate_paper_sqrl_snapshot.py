#!/usr/bin/env python3
"""Exact-snapshot causal evaluation of the paper SQRL action selector."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.agents import create_agent
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


def _spaces(robot_cfg):
    return (
        gym.spaces.Box(
            -100.0, 100.0, (robot_cfg.obs_dim,), dtype=np.float32),
        gym.spaces.Box(
            -1.0, 1.0, (robot_cfg.num_joints,), dtype=np.float32),
    )


@torch.no_grad()
def _nominal(agent, observation: np.ndarray, *, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    obs = torch.as_tensor(
        observation[None, :], dtype=torch.float32, device=agent._device)
    action, _ = agent._actor(
        observations=agent._actor_observations(obs),
        training=False, sample=True)
    return action[0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def _candidate_set(agent, observation: np.ndarray, *, seed: int) -> dict:
    """Use Eq. 3 on one fixed candidate set; candidate zero is nominal."""
    count = int(agent._cfg.safety_num_candidates)
    torch.manual_seed(seed)
    obs = torch.as_tensor(
        np.repeat(observation[None, :], count, axis=0),
        dtype=torch.float32, device=agent._device)
    actions, info = agent._actor(
        observations=agent._actor_observations(obs),
        training=False, sample=True)
    risks = agent._risk(obs, actions).reshape(-1)
    safe_indices = torch.nonzero(
        risks <= float(agent._cfg.safety_epsilon), as_tuple=False).reshape(-1)
    no_safe = safe_indices.numel() == 0
    if no_safe:
        selected_index = int(torch.argmin(risks).item())
    else:
        log_probs = info["log_prob"].reshape(-1)
        weights = torch.softmax(log_probs[safe_indices], dim=0)
        torch.manual_seed(seed + 7919)
        offset = int(torch.multinomial(weights, 1).item())
        selected_index = int(safe_indices[offset].item())
    values = actions.cpu().numpy().astype(np.float32)
    risk_values = risks.cpu().numpy()
    return {
        "nominal": values[0],
        "selected": values[selected_index],
        "nominal_risk": float(risk_values[0]),
        "selected_risk": float(risk_values[selected_index]),
        "selected_index": selected_index,
        "replaced": selected_index != 0,
        "no_safe": no_safe,
        "safe_rate": float(len(safe_indices) / count),
    }


def _rollout(env, agent, snapshot, previous_action, horizon: int, *,
             mode: str, first: dict, seed: int) -> dict:
    env.restore(snapshot)
    previous = np.asarray(previous_action, dtype=np.float32).copy()
    failure_step = None
    near_failure_step = None
    max_tilt = 0.0
    min_height = np.inf
    replacement_steps = []
    no_safe_steps = []
    for index in range(horizon):
        observation = env.observation(previous)
        step_seed = seed + index * 104729
        if index == 0:
            selection = first
            action = (
                selection["selected"] if mode in {"one_step", "closed_loop"}
                else selection["nominal"])
            if mode in {"one_step", "closed_loop"} and selection["replaced"]:
                replacement_steps.append(index + 1)
            if mode in {"one_step", "closed_loop"} and selection["no_safe"]:
                no_safe_steps.append(index + 1)
        elif mode == "closed_loop":
            selection = _candidate_set(agent, observation, seed=step_seed)
            action = selection["selected"]
            if selection["replaced"]:
                replacement_steps.append(index + 1)
            if selection["no_safe"]:
                no_safe_steps.append(index + 1)
        else:
            action = _nominal(agent, observation, seed=step_seed)
        measurement = env.step(action)
        previous = action
        max_tilt = max(max_tilt, measurement.tilt_rad)
        min_height = min(min_height, measurement.height_m)
        if near_failure_step is None and measurement.near_failure:
            near_failure_step = index + 1
        if measurement.failure:
            failure_step = index + 1
            break
    return {
        "failure_step": failure_step,
        "near_failure_step": near_failure_step,
        "max_tilt_rad": float(max_tilt),
        "min_height_m": float(min_height),
        "replacement_steps": replacement_steps,
        "no_safe_steps": no_safe_steps,
    }


def _occurred_by(record: dict, horizon: int, key: str) -> bool:
    step = record[key]
    return step is not None and int(step) <= horizon


def _paired_pvalue(improved: int, worsened: int) -> float | None:
    discordant = improved + worsened
    if discordant == 0:
        return None
    lower = min(improved, worsened)
    probability = sum(
        math.comb(discordant, index) for index in range(lower + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def _summarize(records: list[dict], mode: str, horizon: int) -> dict:
    nominal = [
        _occurred_by(record["nominal"], horizon, "failure_step")
        for record in records
    ]
    treatment = [
        _occurred_by(record[mode], horizon, "failure_step")
        for record in records
    ]
    improved = sum(n and not t for n, t in zip(nominal, treatment))
    worsened = sum(not n and t for n, t in zip(nominal, treatment))
    nominal_failures = sum(nominal)
    treatment_failures = sum(treatment)
    return {
        "pairs": len(records),
        "nominal_failures": nominal_failures,
        "selected_failures": treatment_failures,
        "absolute_failure_delta": treatment_failures - nominal_failures,
        "failure_rate_delta": (
            (treatment_failures - nominal_failures) / len(records)),
        "improved": improved,
        "worsened": worsened,
        "both_fail": sum(n and t for n, t in zip(nominal, treatment)),
        "neither_fail": sum(
            not n and not t for n, t in zip(nominal, treatment)),
        "pairwise_improvement_accuracy": (
            improved / (improved + worsened)
            if improved + worsened else None),
        "exact_paired_pvalue": _paired_pvalue(improved, worsened),
        "total_replacements": sum(
            sum(step <= horizon for step in record[mode]["replacement_steps"])
            for record in records),
        "total_no_safe": sum(
            sum(step <= horizon for step in record[mode]["no_safe_steps"])
            for record in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/go2_50hz_sqrl_paper_finetune.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model", default=(
            "/home/xyz/code/unitree_mujoco/"
            "unitree_robots/go2/scene_empty.xml"))
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=642)
    parser.add_argument("--disturbance-interval", type=int, default=10)
    parser.add_argument("--linear-impulse-std", type=float, default=1.0)
    parser.add_argument("--angular-impulse-std", type=float, default=4.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    robot_cfg, train_cfg, agent_cfg = load_app_config(
        args.config, agent="paper_sqrl")
    agent_cfg.device_type = args.device
    agent_cfg.buffer_device_type = args.device
    observation_space, action_space = _spaces(robot_cfg)
    agent = create_agent(observation_space, action_space, {}, agent_cfg)
    checkpoint = Path(args.checkpoint).resolve()
    agent.load(str(checkpoint / "agent"))
    env = MujocoSnapshotEnv(
        args.model, robot_cfg,
        policy_frequency=train_cfg.control_frequency)
    rng = np.random.default_rng(args.seed)
    env.reset_standing(rng=rng)
    previous = np.zeros(robot_cfg.num_joints, dtype=np.float32)
    episode_step = 0
    records = []

    verification = env.capture()
    before = env.observation(previous)
    env.step(np.zeros(robot_cfg.num_joints, dtype=np.float32))
    env.restore(verification)
    restore_error = float(np.max(np.abs(
        before - env.observation(previous))))
    if restore_error > 1e-5:
        raise RuntimeError(f"snapshot restore error {restore_error}")

    for pair_index in range(args.pairs):
        if (args.disturbance_interval > 0 and episode_step > 0
                and episode_step % args.disturbance_interval == 0):
            env.apply_base_velocity_impulse(
                linear_velocity_delta=rng.normal(
                    0.0, args.linear_impulse_std, size=3),
                angular_velocity_delta=rng.normal(
                    0.0, args.angular_impulse_std, size=3))
        observation = env.observation(previous)
        seed = args.seed * 10_000_000 + pair_index * 1000
        first = _candidate_set(agent, observation, seed=seed)
        snapshot = env.capture()
        record = {
            "pair_index": pair_index,
            "episode_step": episode_step,
            "nominal_risk": first["nominal_risk"],
            "selected_risk": first["selected_risk"],
            "safe_rate": first["safe_rate"],
            "initial_replaced": first["replaced"],
            "initial_no_safe": first["no_safe"],
        }
        for mode in ("nominal", "one_step", "closed_loop"):
            record[mode] = _rollout(
                env, agent, snapshot, previous, args.horizon,
                mode=mode, first=first, seed=seed)
        records.append(record)
        env.restore(snapshot)
        measurement = env.step(first["nominal"])
        previous = first["nominal"]
        episode_step += 1
        if measurement.failure or episode_step >= args.episode_steps:
            env.reset_standing(rng=rng)
            previous = np.zeros(robot_cfg.num_joints, dtype=np.float32)
            episode_step = 0

    summary = {
        "kind": "paper_sqrl_exact_snapshot_causal_evaluation",
        "checkpoint": str(checkpoint),
        "config": str(Path(args.config).resolve()),
        "seed": args.seed,
        "pairs": len(records),
        "candidate_count": int(agent._cfg.safety_num_candidates),
        "epsilon": float(agent._cfg.safety_epsilon),
        "initial_replacement_rate": float(np.mean([
            record["initial_replaced"] for record in records])),
        "initial_no_safe_rate": float(np.mean([
            record["initial_no_safe"] for record in records])),
        "initial_safe_candidate_rate": float(np.mean([
            record["safe_rate"] for record in records])),
        "mean_predicted_risk_improvement": float(np.mean([
            record["nominal_risk"] - record["selected_risk"]
            for record in records])),
        "snapshot_restore_max_observation_error": restore_error,
        "disturbance": {
            "interval": args.disturbance_interval,
            "linear_std": args.linear_impulse_std,
            "angular_std": args.angular_impulse_std,
        },
        "horizons": {
            str(horizon): {
                "one_step": _summarize(records, "one_step", horizon),
                "closed_loop": _summarize(records, "closed_loop", horizon),
            }
            for horizon in (8, 16, 32) if horizon <= args.horizon
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"summary": summary, "records": records}, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
