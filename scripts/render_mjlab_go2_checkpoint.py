#!/usr/bin/env python3
"""Render a finite, no-push Go2 PPO checkpoint rollout to MP4."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio.v3 as iio
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.steps <= 0:
        raise ValueError("checkpoint must exist and steps must be positive")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg = load_env_cfg("Unitree-Go2-Flat")
    agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
    cfg.scene.num_envs = 1
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.viewer.width = args.width
    cfg.viewer.height = args.height
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (0.40, 0.40)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)

    environment = ManagerBasedRlEnv(
        cfg=cfg, device="cuda:0", render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(environment, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device="cuda:0")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    observation = wrapped.get_observations()
    frames = []
    with torch.inference_mode():
        for _ in range(args.steps):
            action = policy(observation)
            observation, _, _, _ = wrapped.step(action)
            frame = environment.render()
            if frame is None:
                raise RuntimeError("MjLab renderer returned no frame")
            frames.append(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.output, frames, fps=50, codec="libx264",
                pixelformat="yuv420p")
    environment.close()
    print(args.output.resolve())


if __name__ == "__main__":
    main()
