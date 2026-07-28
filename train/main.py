"""Compatibility CLI for Go2 walk training."""

from __future__ import annotations

import argparse
import dataclasses
import os

from learner.learner import (run_in_process, run_play,
                             run_safety_collection, run_safety_eval, run_split)
from learner.safety_retrain import run_safety_retrain
from train.config import Go2Config, TrainConfig, load_app_config


def _parse_held_out_seeds(raw: str | None) -> set[int] | None:
    if raw is None or raw.strip() == '':
        return None
    return {int(part.strip()) for part in raw.split(',') if part.strip()}


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Go2 online training')
    parser.add_argument(
        '--mode',
        choices=('in_process', 'split', 'play', 'safety_eval',
                 'safety_collect', 'safety_retrain',
                 'sqrl_pretrain', 'sqrl_finetune'),
        default='in_process',
        help=('Runtime layout. in_process/split train SAC; play rolls out a '
              'checkpoint; safety_* are Q_safe tools; sqrl_pretrain/finetune '
              'run SQRL Route A (constrained sampling; finetune adds nu).'),
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
        help=argparse.SUPPRESS,  # Removed: pipeline is Q_safe-only.
    )
    parser.add_argument(
        '--safety-mask-epsilon',
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--allow-min-risk-fallback',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--update-q-safe',
        action='store_true',
        help=('safety_collect only: update Q_safe online while collecting. '
              'Default freezes Q_safe and writes episode artifacts instead.'),
    )
    parser.add_argument(
        '--dataset-dir',
        default=None,
        help=('safety_collect only: directory for per-episode safety '
              'artifacts. Defaults to saved/safety_datasets/collect_*.'),
    )
    parser.add_argument(
        '--update-checkpoint',
        action='store_true',
        help=('safety_collect only: also write training_snapshot under the '
              'source checkpoint directory. Off by default when Q_safe is '
              'frozen so reference critics are not overwritten.'),
    )
    parser.add_argument(
        '--retrain-steps',
        type=int,
        default=5000,
        help='safety_retrain only: number of Q_safe gradient steps.',
    )
    parser.add_argument(
        '--held-out-seeds',
        default=None,
        help=('safety_retrain only: comma-separated rollout seeds reserved '
              'for validation. Default holds out ~20%% of unique seeds.'),
    )
    parser.add_argument(
        '--include-checkpoint-replay',
        action='store_true',
        help=('safety_retrain only: mix the source checkpoint safety replay '
              'into the train set in addition to episode artifacts.'),
    )
    parser.add_argument(
        '--save-dir',
        default=None,
        help=('Output directory for safety_retrain / sqrl_* snapshots.'),
    )
    parser.add_argument(
        '--from-scratch',
        action='store_true',
        help=('sqrl_pretrain only: do not warm-start from SAC 12584; '
              'jointly train pi and Q_safe from step 0.'),
    )
    parser.add_argument(
        '--move-speed',
        type=float,
        default=None,
        help='Override config move_speed (m/s) used by the walk reward.',
    )
    parser.add_argument(
        '--safety-conservative-weight',
        type=float,
        default=None,
        help=('Override CSC/CQL-style Q_safe conservative weight. '
              'Zero preserves the existing safety critic loss.'),
    )
    return parser.parse_args(argv)


_DEFAULT_SQRL_WARM_START = (
    'saved/checkpoints_58d/training_snapshot_000000012584.pkl')
_DEFAULT_SQRL_SAVE_DIR = 'saved/checkpoints_sqrl'


def apply_move_speed(robot_cfg: Go2Config, move_speed: float) -> Go2Config:
    """Return a copy of robot_cfg with a new target walk speed."""
    if move_speed <= 0.0:
        raise ValueError(f'move_speed must be positive, got {move_speed}')
    return dataclasses.replace(robot_cfg, move_speed=float(move_speed))


def _configure_sqrl_mode(args, train_cfg: TrainConfig, droq_cfg):
    """Apply SQRL Route A flags for sqrl_pretrain / sqrl_finetune."""
    phase = 'pretrain' if args.mode == 'sqrl_pretrain' else 'finetune'
    from_scratch = bool(getattr(args, 'from_scratch', False))
    if from_scratch and phase != 'pretrain':
        raise SystemExit('--from-scratch is only valid with --mode sqrl_pretrain')
    train_cfg.sqrl_enabled = True
    train_cfg.sqrl_phase = phase
    train_cfg.safety_critic_enabled = True
    train_cfg.safety_replay_enabled = True
    train_cfg.resume_checkpoint = not from_scratch
    if args.save_dir:
        train_cfg.save_dir = args.save_dir
    elif train_cfg.save_dir in ('saved/checkpoints', 'saved/checkpoints_58d',
                                'saved/checkpoints_step5'):
        train_cfg.save_dir = _DEFAULT_SQRL_SAVE_DIR
    if from_scratch:
        train_cfg.warm_start_checkpoint = None
        train_cfg.resume_checkpoint = False
    elif args.checkpoint:
        train_cfg.warm_start_checkpoint = args.checkpoint
    elif not train_cfg.warm_start_checkpoint:
        if phase == 'pretrain':
            train_cfg.warm_start_checkpoint = _DEFAULT_SQRL_WARM_START
        else:
            raise SystemExit(
                'sqrl_finetune requires --checkpoint pointing to a '
                'sqrl_pretrain snapshot with safety_critic_state.')
    train_cfg.experiment_name = f'sqrl_{phase}'
    droq_cfg['safety_lagrange_lr'] = float(train_cfg.sqrl_lagrange_lr)
    return train_cfg, droq_cfg


def main(argv=None) -> int:
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

    args = _parse_args(argv)
    if args.safety_mask or args.allow_min_risk_fallback:
        raise SystemExit(
            'Legacy heuristic --safety-mask is withdrawn. Use '
            '--mode sqrl_pretrain / sqrl_finetune for SQRL Route A.')
    robot_cfg, train_cfg, droq_cfg = load_app_config(
        path=args.config, profile=args.config_profile)
    if args.move_speed is not None:
        robot_cfg = apply_move_speed(robot_cfg, args.move_speed)
    if args.safety_conservative_weight is not None:
        if args.safety_conservative_weight < 0.0:
            raise SystemExit('--safety-conservative-weight must be >= 0')
        train_cfg.safety_conservative_weight = float(
            args.safety_conservative_weight)

    if args.mode in ('sqrl_pretrain', 'sqrl_finetune'):
        train_cfg, droq_cfg = _configure_sqrl_mode(args, train_cfg, droq_cfg)

    print(f'[train] mode={args.mode} '
          f'profile={args.config_profile} '
          f'experiment={train_cfg.experiment_name} '
          f'config={robot_cfg.domain_id}/{robot_cfg.interface} '
          f'move_speed={robot_cfg.move_speed} '
          f'init_qpos={robot_cfg.init_qpos[:3]}... '
          f'standup=controller '
          f'explore_scale={train_cfg.explore_action_scale} '
          f'max_steps={train_cfg.max_steps} '
          f'reset_hold={train_cfg.reset_hold_steps} '
          f'recovery_stable={train_cfg.recovery_stable_steps}',
          flush=True)
    if train_cfg.safety_conservative_weight > 0.0:
        print(
            '[train] conservative Q_safe '
            f'alpha={train_cfg.safety_conservative_weight} '
            f'actions={train_cfg.safety_conservative_num_actions}',
            flush=True)
    if train_cfg.sqrl_enabled:
        print(f'[train] SQRL phase={train_cfg.sqrl_phase} '
              f'eps={train_cfg.sqrl_epsilon} K={train_cfg.sqrl_num_candidates} '
              f'from_scratch={bool(getattr(args, "from_scratch", False))} '
              f'warm_start={train_cfg.warm_start_checkpoint} '
              f'save_dir={train_cfg.save_dir}',
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
            rollout_seed=args.rollout_seed)
    if args.mode == 'safety_collect':
        return run_safety_collection(
            robot_cfg, train_cfg, droq_cfg,
            checkpoint=args.checkpoint, episodes=args.play_episodes,
            action_noise_std=args.action_noise_std,
            rollout_seed=args.rollout_seed,
            freeze_q_safe=not args.update_q_safe,
            dataset_dir=args.dataset_dir,
            update_checkpoint=args.update_checkpoint)
    if args.mode == 'safety_retrain':
        if not args.checkpoint:
            raise SystemExit(
                'safety_retrain requires --checkpoint (reference Q_safe).')
        if not args.dataset_dir:
            raise SystemExit(
                'safety_retrain requires --dataset-dir with episode artifacts.')
        return run_safety_retrain(
            train_cfg, droq_cfg,
            checkpoint=args.checkpoint,
            dataset_dir=args.dataset_dir,
            retrain_steps=args.retrain_steps,
            held_out_seeds=_parse_held_out_seeds(args.held_out_seeds),
            include_checkpoint_replay=args.include_checkpoint_replay,
            save_dir=args.save_dir)
    # in_process, sqrl_pretrain, sqrl_finetune
    return run_in_process(robot_cfg, train_cfg, droq_cfg)
