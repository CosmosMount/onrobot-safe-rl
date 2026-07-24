"""Compatibility CLI for Go2 walk training."""

from __future__ import annotations

import argparse
import os

from learner.learner import (run_in_process, run_play,
                             run_safety_collection, run_safety_eval, run_split)
from train.config import load_app_config


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Go2 online training')
    parser.add_argument(
        '--mode',
        choices=('in_process', 'split', 'play', 'safety_eval',
                 'safety_collect'),
        default='in_process',
        help=('Runtime layout. in_process keeps collector and learner in one '
              'process; split runs collector and learner on separate threads; '
              'play loads a saved policy and runs deterministic rollouts.'),
    )
    parser.add_argument(
        '--config-profile',
        choices=('go2', 'simulation', 'real_robot'),
        default='go2',
        help=('Configuration profile. go2 keeps the compatibility file; '
              'simulation/real_robot use config/common.yaml overlays.'),
    )
    parser.add_argument(
        '--config',
        default=None,
        help=('Optional YAML path. Preserves the compatibility command '
              '--config config/go2.yaml and overrides --config-profile.'),
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
    parser.add_argument(
        '--action-noise-std',
        type=float,
        default=None,
        help=('Gaussian action perturbation for safety_collect/safety_eval. '
              'Q_safe remains read-only during evaluation.'),
    )
    parser.add_argument(
        '--rollout-seed',
        type=int,
        default=None,
        help='Explicit held-out noise seed for safety_collect/safety_eval.',
    )
    parser.add_argument(
        '--safety-mask',
        action='store_true',
        help='Enable SQRL candidate masking in safety_eval/safety_collect.',
    )
    parser.add_argument(
        '--safety-mask-epsilon',
        type=float,
        default=None,
        help='Override Q_safe threshold for --safety-mask evaluation.',
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

    args = _parse_args(argv)
    robot_cfg, train_cfg, droq_cfg = load_app_config(
        path=args.config, profile=args.config_profile)

    print(f'[train] mode={args.mode} '
          f'profile={args.config_profile} '
          f'experiment={train_cfg.experiment_name} '
          f'config={robot_cfg.domain_id}/{robot_cfg.interface} '
          f'init_qpos={robot_cfg.init_qpos[:3]}... '
          f'standup=controller '
          f'explore_scale={train_cfg.explore_action_scale} '
          f'max_steps={train_cfg.max_steps} '
          f'reset_hold={train_cfg.reset_hold_steps} '
          f'recovery_stable={train_cfg.recovery_stable_steps}',
          flush=True)

    if args.mode == 'split':
        return run_split(robot_cfg, train_cfg, droq_cfg)
    if args.mode == 'play':
        return run_play(
            robot_cfg,
            train_cfg,
            droq_cfg,
            checkpoint=args.checkpoint,
            episodes=args.play_episodes,
        )
    if args.mode == 'safety_eval':
        return run_safety_eval(
            robot_cfg, train_cfg, droq_cfg,
            checkpoint=args.checkpoint, episodes=args.play_episodes,
            action_noise_std=args.action_noise_std or 0.0,
            rollout_seed=args.rollout_seed,
            safety_mask=args.safety_mask,
            safety_mask_epsilon=args.safety_mask_epsilon)
    if args.mode == 'safety_collect':
        return run_safety_collection(
            robot_cfg, train_cfg, droq_cfg,
            checkpoint=args.checkpoint, episodes=args.play_episodes,
            action_noise_std=args.action_noise_std,
            rollout_seed=args.rollout_seed,
            safety_mask=args.safety_mask,
            safety_mask_epsilon=args.safety_mask_epsilon)
    return run_in_process(robot_cfg, train_cfg, droq_cfg)
