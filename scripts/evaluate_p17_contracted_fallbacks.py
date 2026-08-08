#!/usr/bin/env python3
"""Exact-snapshot evaluation of previous-action contraction fallbacks."""

from __future__ import annotations

import argparse
import json

import gymnasium as gym
import numpy as np
import torch

from rl.agents import create_agent
from safety_data.paths import assert_development_path
from scripts.evaluate_p16_snapshot_replacements import (
    _actor_actions,
    _branch,
    _risks,
)
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/go2_50hz_safe_adaptive_gated_v3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model",
        default=("/home/xyz/code/unitree_mujoco/"
                 "unitree_robots/go2/scene_empty.xml"))
    parser.add_argument("--natural-steps", type=int, default=30000)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--episode-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disturbance-interval", type=int, default=10)
    parser.add_argument("--linear-impulse-std", type=float, default=1.0)
    parser.add_argument("--angular-impulse-std", type=float, default=4.0)
    parser.add_argument(
        "--alphas", default="0.0,0.25,0.5,0.75",
        help="selected = alpha * nominal + (1-alpha) * previous")
    parser.add_argument(
        "--output",
        default="saved/experiments/p17_contracted_fallbacks.json")
    args = parser.parse_args()
    assert_development_path(args.config)
    checkpoint_path = assert_development_path(args.checkpoint)
    model_path = assert_development_path(args.model)
    output_path = assert_development_path(args.output)
    alphas = [float(value) for value in args.alphas.split(",")]

    robot_cfg, train_cfg, agent_cfg = load_app_config(
        args.config, agent="safe_droq")
    agent_cfg.device_type = "cuda" if torch.cuda.is_available() else "cpu"
    agent_cfg.buffer_device_type = agent_cfg.device_type
    observation_space = gym.spaces.Box(
        -100.0, 100.0, (robot_cfg.obs_dim,), dtype=np.float32)
    action_space = gym.spaces.Box(
        -1.0, 1.0, (robot_cfg.num_joints,), dtype=np.float32)
    agent = create_agent(
        observation_space, action_space, {}, agent_cfg)
    agent.load(str(checkpoint_path))

    env = MujocoSnapshotEnv(
        model_path, robot_cfg,
        policy_frequency=train_cfg.control_frequency,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter)
    rng = np.random.default_rng(args.seed)
    env.reset_standing(rng=rng)
    episode_step = 0
    records = []

    for natural_step in range(args.natural_steps):
        if (
            args.disturbance_interval > 0
            and episode_step > 0
            and episode_step % args.disturbance_interval == 0
        ):
            env.apply_base_velocity_impulse(
                linear_velocity_delta=rng.normal(
                    0.0, args.linear_impulse_std, size=3),
                angular_velocity_delta=rng.normal(
                    0.0, args.angular_impulse_std, size=3))
        observation = env.record_observation()[-1]
        previous = env.previous_action_requested
        nominal = _actor_actions(
            agent, observation, 1, sample=True,
            seed=args.seed * 1_000_000 + natural_step)[0]
        nominal_risk = float(_risks(
            agent, observation, nominal[None, :])[0])
        snapshot = env.capture()
        if nominal_risk > float(agent.cfg.safety_epsilon):
            actions = {
                str(alpha): np.clip(
                    alpha * nominal + (1.0 - alpha) * previous,
                    -1.0, 1.0).astype(np.float32)
                for alpha in alphas
            }
            record = {
                "natural_step": natural_step,
                "nominal_risk": nominal_risk,
                "horizons": {},
            }
            for horizon in (8, 16, 32):
                nominal_result = _branch(
                    env, agent, snapshot, nominal, horizon)
                record["horizons"][str(horizon)] = {
                    "nominal": nominal_result,
                    "contracted": {
                        alpha: _branch(
                            env, agent, snapshot, action, horizon)
                        for alpha, action in actions.items()
                    },
                }
            records.append(record)
            env.restore(snapshot)
            if len(records) >= args.max_pairs:
                break

        measurement = env.step(nominal)
        episode_step += 1
        if measurement.failure or episode_step >= args.episode_steps:
            env.reset_standing(rng=rng)
            episode_step = 0

    summary = {
        "checkpoint": str(checkpoint_path),
        "pairs": len(records),
        "epsilon": float(agent.cfg.safety_epsilon),
        "alphas": {},
    }
    for alpha in map(str, alphas):
        summary["alphas"][alpha] = {}
        for horizon in (8, 16, 32):
            results = [r["horizons"][str(horizon)] for r in records]
            nominal_failures = sum(
                item["nominal"]["failure"] for item in results)
            selected_failures = sum(
                item["contracted"][alpha]["failure"] for item in results)
            improved = sum(
                item["nominal"]["failure"]
                and not item["contracted"][alpha]["failure"]
                for item in results)
            worsened = sum(
                not item["nominal"]["failure"]
                and item["contracted"][alpha]["failure"]
                for item in results)
            summary["alphas"][alpha][str(horizon)] = {
                "nominal_failures": nominal_failures,
                "selected_failures": selected_failures,
                "improved": improved,
                "worsened": worsened,
                "absolute_failure_delta": (
                    selected_failures - nominal_failures),
            }

    output = output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
