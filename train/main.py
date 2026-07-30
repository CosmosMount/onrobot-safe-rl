"""CLI for Go2 FlashSAC training."""

from __future__ import annotations

import argparse
import os

from rl.agents import create_agent
from rl.agents.base.agent import BaseAgent
from runtime.inference.dds import DdsConfig
from train.config import load_app_config
from train.env import Go2Env
from train.loop import run_play, run_training
from train.replay import install_flashsac_numpy_replay


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Go2 online training client")
    parser.add_argument(
        "--mode",
        choices=("train", "play"),
        default="train",
        help="train connects to the standalone runtime; play runs deterministic policy rollouts.",
    )
    parser.add_argument(
        "--config-profile",
        choices=("go2", "simulation", "real_robot"),
        default="go2",
        help="Configuration profile.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML overlay path; overrides --config-profile.",
    )
    parser.add_argument(
        "--agent",
        choices=("flashsac", "droq", "safe_droq"),
        default=None,
        help="Override train.agent from the selected configuration profile.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Snapshot directory for --mode play. Defaults to latest in save_dir.",
    )
    parser.add_argument(
        "--safety-mode",
        choices=("logging", "masking"),
        default=None,
        help="Override safe_droq safety mode.",
    )
    parser.add_argument(
        "--safety-pretrained-path",
        default=None,
        help="Load only Q_safe weights from a safety_critic.pt file.",
    )
    parser.add_argument(
        "--freeze-safety-critic",
        action="store_true",
        help="Keep a transferred Q_safe fixed during target SAC training.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Override train.save_dir for isolated comparisons.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the W&B run name.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Override the experiment/group name.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override train.max_steps (useful for smoke tests and matched runs).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging for this run.",
    )
    parser.add_argument(
        "--play-episodes",
        type=int,
        default=1,
        help="Number of deterministic episodes to run in --mode play.",
    )
    return parser.parse_args(argv)


def _build_env(robot_cfg, train_cfg) -> Go2Env:
    return Go2Env(
        dds_config=DdsConfig(
            domain_id=robot_cfg.domain_id,
            interface=robot_cfg.interface,
        ),
        go2_config=robot_cfg,
        control_frequency=train_cfg.control_frequency,
        max_episode_steps=train_cfg.max_episode_steps,
        ipc_socket=robot_cfg.ipc_socket,
        max_joint_delta=train_cfg.max_joint_delta,
        use_action_filter=train_cfg.use_action_filter,
        reset_hold_steps=train_cfg.reset_hold_steps,
        reset_joint_tolerance=train_cfg.reset_joint_tolerance,
        recovery_stable_steps=train_cfg.recovery_stable_steps,
        standup_timeout_steps=train_cfg.standup_timeout_steps,
        abort_on_unstable_reset=train_cfg.abort_on_unstable_reset,
        seed=train_cfg.seed,
    )


def _build_agent(env: Go2Env, agent_cfg) -> BaseAgent:
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info={},
        cfg=agent_cfg,
    )
    replay_adapter = (
        "FlashSACNumpyReplay"
        if install_flashsac_numpy_replay(agent, env, agent_cfg)
        else "agent-default"
    )
    print(f"[train] replay adapter={replay_adapter}", flush=True)
    return agent


def main(argv=None) -> int:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    args = _parse_args(argv)
    robot_cfg, train_cfg, agent_cfg = load_app_config(
        path=args.config,
        profile=args.config_profile,
        agent=args.agent,
    )
    if args.save_dir is not None:
        train_cfg.save_dir = args.save_dir
    if args.run_name is not None:
        train_cfg.wandb_run_name = args.run_name
    if args.experiment_name is not None:
        train_cfg.experiment_name = args.experiment_name
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise SystemExit("--max-steps must be positive")
        train_cfg.max_steps = args.max_steps
    if args.no_wandb:
        train_cfg.wandb = False
    if args.safety_mode is not None:
        if str(agent_cfg.agent_type) != "safe_droq":
            raise SystemExit("--safety-mode requires --agent safe_droq")
        agent_cfg.safety_mode = args.safety_mode
    if args.safety_pretrained_path is not None:
        if str(agent_cfg.agent_type) != "safe_droq":
            raise SystemExit(
                "--safety-pretrained-path requires --agent safe_droq")
        agent_cfg.safety_pretrained_path = args.safety_pretrained_path
    if args.freeze_safety_critic:
        if str(agent_cfg.agent_type) != "safe_droq":
            raise SystemExit(
                "--freeze-safety-critic requires --agent safe_droq")
        agent_cfg.freeze_safety_critic = True
    env = _build_env(robot_cfg, train_cfg)
    agent = _build_agent(env, agent_cfg)

    print(
        f"[train] mode={args.mode} profile={args.config_profile} "
        f"agent={train_cfg.agent} experiment={train_cfg.experiment_name} "
        f"dds={robot_cfg.domain_id}/{robot_cfg.interface} "
        f"device={agent_cfg.device_type} replay=numpy "
        f"start_training={train_cfg.start_training} max_steps={train_cfg.max_steps}",
        flush=True,
    )

    if args.mode == "play":
        return run_play(
            agent,
            env,
            train_cfg,
            checkpoint=args.checkpoint,
            episodes=args.play_episodes,
        )
    run_training(agent, env, train_cfg)
    return 0
