#!/usr/bin/env python3
"""Exact-snapshot comparison of nominal and repeatedly shielded policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl.agents import create_agent
from scripts.evaluate_p16_snapshot_replacements import (
    _actor_actions,
    _risks,
    _select,
)
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


def _rollout(
    env, agent, snapshot, previous_action, horizon, *,
    shield: bool, seed: int,
):
    env.restore(snapshot)
    previous = np.asarray(previous_action, dtype=np.float32).copy()
    replacements = 0
    no_safe = 0
    failure_step = None
    for step in range(horizon):
        observation = env.observation(previous)
        nominal = _actor_actions(
            agent, observation, 1, sample=False, seed=0)[0]
        action = nominal
        if shield:
            selection = _select(
                agent, observation, nominal,
                seed=seed * 1000 + step)
            action = selection["action"]
            replacements += int(selection["replaced"])
            no_safe += int(selection["no_safe"])
        measurement = env.step(action)
        previous = action
        if measurement.failure:
            failure_step = step + 1
            break
    return {
        "failure": failure_step is not None,
        "failure_step": failure_step,
        "replacements": replacements,
        "no_safe": no_safe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/go2_50hz_safe_adaptive_gated_v3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=642)
    parser.add_argument("--disturbance-interval", type=int, default=10)
    parser.add_argument("--linear-impulse-std", type=float, default=1.0)
    parser.add_argument("--angular-impulse-std", type=float, default=4.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        default=("/home/xyz/code/unitree_mujoco/"
                 "unitree_robots/go2/scene_empty.xml"))
    args = parser.parse_args()

    robot_cfg, train_cfg, agent_cfg = load_app_config(
        args.config, agent="safe_droq")
    agent_cfg.device_type = "cuda" if torch.cuda.is_available() else "cpu"
    agent_cfg.buffer_device_type = agent_cfg.device_type
    agent = create_agent(
        gym.spaces.Box(
            -100.0, 100.0, (robot_cfg.obs_dim,), dtype=np.float32),
        gym.spaces.Box(
            -1.0, 1.0, (robot_cfg.num_joints,), dtype=np.float32),
        {}, agent_cfg)
    checkpoint = Path(args.checkpoint).resolve()
    agent.load(str(checkpoint))
    agent._cfg.safety_pretrained_path = str(
        checkpoint / "safety_critic.pt")
    env = MujocoSnapshotEnv(
        args.model, robot_cfg,
        policy_frequency=train_cfg.control_frequency)
    rng = np.random.default_rng(args.seed)
    env.reset_standing(rng=rng)
    previous = np.zeros(robot_cfg.num_joints, dtype=np.float32)
    episode_step = 0
    records = []
    natural_step = 0
    while len(records) < args.pairs:
        if episode_step > 0 and episode_step % args.disturbance_interval == 0:
            env.apply_base_velocity_impulse(
                linear_velocity_delta=rng.normal(
                    0.0, args.linear_impulse_std, size=3),
                angular_velocity_delta=rng.normal(
                    0.0, args.angular_impulse_std, size=3))
        observation = env.observation(previous)
        nominal = _actor_actions(
            agent, observation, 1, sample=False, seed=0)[0]
        risk = float(_risks(agent, observation, nominal[None, :])[0])
        snapshot = env.capture()
        if risk > float(agent.cfg.safety_epsilon):
            nominal_result = _rollout(
                env, agent, snapshot, previous, args.horizon,
                shield=False, seed=args.seed + len(records))
            shield_result = _rollout(
                env, agent, snapshot, previous, args.horizon,
                shield=True, seed=args.seed + len(records))
            records.append({
                "natural_step": natural_step,
                "initial_risk": risk,
                "nominal": nominal_result,
                "shield": shield_result,
            })
            env.restore(snapshot)
        measurement = env.step(nominal)
        previous = nominal
        episode_step += 1
        natural_step += 1
        if measurement.failure or episode_step >= 400:
            env.reset_standing(rng=rng)
            previous = np.zeros(
                robot_cfg.num_joints, dtype=np.float32)
            episode_step = 0

    nominal_failures = sum(r["nominal"]["failure"] for r in records)
    shield_failures = sum(r["shield"]["failure"] for r in records)
    improved = sum(
        r["nominal"]["failure"] and not r["shield"]["failure"]
        for r in records)
    worsened = sum(
        not r["nominal"]["failure"] and r["shield"]["failure"]
        for r in records)
    summary = {
        "pairs": len(records),
        "horizon": args.horizon,
        "nominal_failures": nominal_failures,
        "shield_failures": shield_failures,
        "absolute_failure_delta": shield_failures - nominal_failures,
        "improved": improved,
        "worsened": worsened,
        "total_replacements": sum(
            r["shield"]["replacements"] for r in records),
        "total_no_safe": sum(
            r["shield"]["no_safe"] for r in records),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
