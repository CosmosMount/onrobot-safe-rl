"""Compatibility CLI for Go2 walk training."""

from __future__ import annotations

import argparse
import os

from train.learner import run_in_process, run_play
from train.config import load_app_config


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Go2 online training')
    parser.add_argument(
        '--mode',
        choices=('in_process', 'play'),
        default='in_process',
        help=('Runtime layout. in_process runs online training; play loads a '
              'saved policy and runs deterministic rollouts.'),
    )
    parser.add_argument(
        '--config',
        default='config/go2.yaml',
        help='Single YAML config shared by Python train and C++ controller.',
    )
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Snapshot path for --mode play. Defaults to latest in save_dir.',
    )
    parser.add_argument(
        '--play-episodes',
        type=int,
        default=1,
        help='Number of deterministic episodes to run in --mode play.',
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

    args = _parse_args(argv)
    robot_cfg, train_cfg, agent_cfgs = load_app_config(args.config)

    print(f'[train] mode={args.mode} '
          f'config={args.config} '
          f'agent={train_cfg.agent} '
          f'experiment={train_cfg.experiment_name} '
          f'dds={robot_cfg.domain_id}/{robot_cfg.interface} '
          f'init_qpos={robot_cfg.init_qpos[:3]}... '
          f'standup=controller '
          f'explore_scale={train_cfg.explore_action_scale} '
          f'max_steps={train_cfg.max_steps} '
          f'reset_hold={train_cfg.reset_hold_steps} '
          f'recovery_stable={train_cfg.recovery_stable_steps}',
          flush=True)

    if args.mode == 'play':
        return run_play(
            robot_cfg,
            train_cfg,
            agent_cfgs,
            checkpoint=args.checkpoint,
            episodes=args.play_episodes,
        )
    return run_in_process(robot_cfg, train_cfg, agent_cfgs)
