#!/usr/bin/env python3
"""Run the Go2 learner against the controller."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train.env import Go2SyncEnv


def load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if float(config.get("control_frequency", 20.0)) != 20.0:
        raise ValueError("training requires control_frequency=20 Hz")
    if int(config.get("utd_ratio", 1)) < 1:
        raise ValueError("training requires utd_ratio >= 1")
    batch_size = int(config.get("batch_size", 256))
    # UTD sampling is with replacement; only one mini-batch of distinct
    # transitions is required before training can start.
    required_replay = batch_size
    if int(config.get("start_training", 10_000)) < required_replay:
        raise ValueError(
            f"start_training must be at least batch_size "
            f"({required_replay}); got {config.get('start_training')}")
    return config


def build_algorithm_config(config: dict):
    """Build the complete DroQ config; do not silently use library defaults."""
    from jaxsac.config import AlgorithmConfig

    fields = set(AlgorithmConfig.__dataclass_fields__)
    algorithm_values = {name: config[name] for name in fields if name in config}
    if "hidden_dims" in algorithm_values:
        algorithm_values["hidden_dims"] = tuple(algorithm_values["hidden_dims"])
    return AlgorithmConfig(**algorithm_values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "config/go2.yaml"))
    parser.add_argument("--jaxsac", default=str(REPO_ROOT.parent / "jaxsac"))
    parser.add_argument("--policy-socket")
    parser.add_argument("--state-socket")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--start-training", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
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
    from train.train import run
    algorithm_config = build_algorithm_config(config)

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
    env = Go2SyncEnv(
        policy_socket=args.policy_socket or config.get("policy_socket", "/tmp/go2_policy.v3.sock"),
        state_socket=args.state_socket or config.get("state_socket", "/tmp/go2_policy.v3.sock.state"),
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
            config=algorithm_config,
            enforce_20hz=bool(config.get("enforce_20hz", True)),
            wandb_run=wandb_run,
            log_interval=int(config.get("wandb_log_interval", 100)),
            progress_interval=(args.progress_interval
                               if args.progress_interval is not None
                               else int(config.get("progress_interval", 100))),
            save_dir=config.get("save_dir"),
            checkpoint_interval=int(config.get("checkpoint_interval", 1000)),
            resume=bool(config.get("resume", False)))
        print(metrics, flush=True)
    except KeyboardInterrupt:
        # Ctrl-C is a normal user shutdown, including when reset() is
        # blocked waiting for the next controller state packet.  Keep the
        # finally block below responsible for STOP/pose recovery, but do not
        # print a traceback for an intentional interruption.
        print("\n[train] interrupted by user; shutting down safely.",
              flush=True)
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except KeyboardInterrupt:
                print("[train] shutdown interrupted; local sockets will be "
                      "closed.", flush=True)
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
