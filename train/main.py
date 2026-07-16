"""Compatibility CLI for Go2 walk training."""

from __future__ import annotations

import argparse
import os

from learner.learner import run_in_process, run_split
from train.config import load_app_config


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Go2 online training')
    parser.add_argument(
        '--mode',
        choices=('in_process', 'split'),
        default='in_process',
        help=('Runtime layout. in_process keeps collector and learner in one '
              'process; split reserves the P2 split execution entrypoint.'),
    )
    parser.add_argument(
        '--config-profile',
        choices=('go2', 'simulation', 'real_robot'),
        default='go2',
        help=('Configuration profile. go2 keeps the compatibility file; '
              'simulation/real_robot use config/common.yaml overlays.'),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

    args = _parse_args(argv)
    robot_cfg, train_cfg, droq_cfg = load_app_config(
        profile=args.config_profile)

    print(f'[train] mode={args.mode} '
          f'profile={args.config_profile} '
          f'config={robot_cfg.domain_id}/{robot_cfg.interface} '
          f'init_qpos={robot_cfg.init_qpos[:3]}... '
          f'standup=controller '
          f'explore_scale={train_cfg.explore_action_scale} '
          f'reset_hold={train_cfg.reset_hold_steps} '
          f'recovery_stable={train_cfg.recovery_stable_steps}',
          flush=True)

    if args.mode == 'split':
        return run_split(robot_cfg, train_cfg, droq_cfg)
    return run_in_process(robot_cfg, train_cfg, droq_cfg)
