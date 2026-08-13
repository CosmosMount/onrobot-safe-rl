#!/usr/bin/env python3
"""Run the walk_in_the_park-compatible Go2 learner against the controller."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train.go2_sync_env import Go2SyncEnv


def load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if float(config.get("control_frequency", 20.0)) != 20.0:
        raise ValueError("walk_in_the_park requires control_frequency=20 Hz")
    if int(config.get("utd_ratio", 1)) < 1:
        raise ValueError("walk_in_the_park requires utd_ratio >= 1")
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "config/go2_walk_in_the_park.yaml"))
    parser.add_argument("--jaxsac", default=str(REPO_ROOT.parent / "jaxsac"))
    parser.add_argument("--policy-socket")
    parser.add_argument("--state-socket")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--start-training", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--benchmark-only", action="store_true",
                        help="run actor+UTD=20 20Hz gate without sending robot actions")
    parser.add_argument("--wandb", action="store_true",
                        help="enable Weights & Biases logging")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--progress-interval", type=int, default=None,
                        help="print rollout progress every N actions")
    args = parser.parse_args()
    config = load_config(args.config)
    sys.path.insert(0, args.jaxsac)
    from train.walk_in_the_park import run

    benchmark_only = args.benchmark_only or bool(config.get("benchmark_only", False))
    wandb_run = None
    wandb_enabled = args.wandb or bool(config.get("wandb", False))
    if wandb_enabled:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B logging requested but wandb is not installed; "
                "install it with `pip install wandb`") from exc
        wandb_run = wandb.init(
            project=args.wandb_project or config.get("wandb_project", "onrobot-safe-rl"),
            name=args.wandb_name or config.get("wandb_name"),
            mode=args.wandb_mode or config.get("wandb_mode", "online"),
            config={**config, "jax_platforms": os.environ.get("JAX_PLATFORMS")},
        )
    if benchmark_only:
        # No socket, controller, reset, or action is needed for the compute
        # preflight. This makes the gate safe to run before the robot starts.
        env = SimpleNamespace(observation_space_shape=(46,),
                              action_space_shape=(12,))
    else:
        env = Go2SyncEnv(
            policy_socket=args.policy_socket or config.get("policy_socket", "/tmp/go2_policy.sock"),
            state_socket=args.state_socket or config.get("state_socket", "/tmp/go2_policy.sock.state"),
            max_episode_steps=int(config.get("max_episode_steps", 400)),
            sport_velocity_world_frame=bool(config.get("sport_velocity_world_frame", True)),
            init_qpos=config.get("init_qpos"),
            action_offset=config.get("action_offset"))
    try:
        _, metrics = run(
            env,
            max_steps=args.max_steps if args.max_steps is not None else int(config.get("max_steps", 1_000_000)),
            start_training=(args.start_training if args.start_training is not None
                            else int(config.get("start_training", 10_000))),
            batch_size=args.batch_size if args.batch_size is not None else int(config.get("batch_size", 256)),
            utd_ratio=int(config.get("utd_ratio", 1)),
            replay_capacity=int(config.get("replay_capacity", 1_000_000)),
            seed=args.seed if args.seed is not None else int(config.get("seed", 42)),
            enforce_20hz=bool(config.get("enforce_20hz", True)),
            benchmark_only=benchmark_only,
            wandb_run=wandb_run,
            wandb_log_interval=int(config.get("wandb_log_interval", 100)),
            progress_interval=(args.progress_interval
                               if args.progress_interval is not None
                               else int(config.get("progress_interval", 100))))
        print(metrics, flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
