#!/usr/bin/env python3
"""Run the P15 common-actor multi-speed SAC versus SQRL protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from learner.checkpoint import (
    latest_snapshot,
    load_training_snapshot_metadata,
    snapshot_agent_hash,
)
from learner.learner import run_in_process
from scripts.run_sqrl_ft_speed_sweep import (
    _run_sac_ft,
    _run_sqrl_ft,
    _speed_tag,
)
from scripts.run_sqrl_transfer_scratch import (
    _bounce_controller,
    _fresh_dir,
    _stabilize,
    _tee_stdout,
)
from train.config import load_app_config
from train.main import apply_move_speed


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(
        value, indent=2, allow_nan=True), encoding='utf-8')
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _validate_common_checkpoint(
        checkpoint: Path,
        *,
        min_transitions: int,
        min_episodes: int,
) -> dict:
    metadata = load_training_snapshot_metadata(checkpoint)
    coverage = metadata.get('speed_coverage')
    if not isinstance(coverage, dict):
        raise RuntimeError('common checkpoint has no speed coverage manifest')
    if not bool(coverage.get('frontier_complete')):
        raise RuntimeError('1.00 m/s frontier did not pass its promotion gate')
    if not bool(coverage.get('coverage_complete')):
        raise RuntimeError('balanced speed coverage is incomplete')
    transitions = coverage.get('balanced_transitions') or {}
    episodes = coverage.get('balanced_episodes') or {}
    failed = [
        speed for speed in coverage.get('speed_bins', [])
        if int(transitions.get(f'{float(speed):.2f}', 0)) < min_transitions
        or int(episodes.get(f'{float(speed):.2f}', 0)) < min_episodes
    ]
    if failed:
        raise RuntimeError(f'common checkpoint under-covered speeds: {failed}')
    return metadata


def _run_common_sac(args, seed: int, seed_root: Path) -> Path:
    common_dir = seed_root / 'common_sac'
    existing = latest_snapshot(common_dir)
    if args.reuse_complete and existing is not None:
        _validate_common_checkpoint(
            Path(existing),
            min_transitions=args.balance_min_transitions,
            min_episodes=args.balance_min_episodes)
        return Path(existing)
    _fresh_dir(common_dir)
    robot_cfg, cfg, droq_cfg = load_app_config(args.config)
    robot_cfg = apply_move_speed(robot_cfg, args.cmd_min)
    cfg.seed = seed
    cfg.experiment_name = f'p15_common_sac_seed{seed}'
    cfg.save_dir = str(common_dir)
    cfg.resume_checkpoint = False
    cfg.warm_start_checkpoint = None
    cfg.max_steps = (
        args.curriculum_max_steps + args.balance_max_steps)
    cfg.checkpoint_interval = min(10_000, cfg.max_steps)
    cfg.cmd_speed_curriculum = True
    cfg.cmd_speed_curriculum_mode = 'performance_then_balanced'
    cfg.cmd_speed_min = args.cmd_min
    cfg.cmd_speed_max = args.cmd_max
    cfg.cmd_speed_increment = args.cmd_increment
    cfg.cmd_speed_frontier_probability = 0.75
    cfg.cmd_speed_promotion_window = args.promotion_window
    cfg.cmd_speed_min_episode_length = args.min_episode_length
    cfg.cmd_speed_min_velocity_ratio = args.min_velocity_ratio
    cfg.cmd_speed_max_fall_rate = args.max_fall_rate
    cfg.cmd_speed_curriculum_max_steps = args.curriculum_max_steps
    cfg.cmd_speed_balance_min_transitions = (
        args.balance_min_transitions)
    cfg.cmd_speed_balance_min_episodes = args.balance_min_episodes
    cfg.cmd_speed_balance_max_steps = args.balance_max_steps
    cfg.sqrl_enabled = False
    cfg.sqrl_actor_lagrange_enabled = False
    # Keep a fresh Q_safe state and safety replay in the common checkpoint,
    # but never update or consult Q_safe during common actor training.
    cfg.safety_replay_enabled = True
    cfg.safety_critic_enabled = True
    cfg.safety_critic_update_interval = 0
    cfg.max_episode_steps = args.episode_steps
    cfg.wandb = bool(args.wandb)
    cfg.wandb_run_name = cfg.experiment_name
    if args.bounce:
        _bounce_controller(f'before {cfg.experiment_name}')
    _stabilize(robot_cfg, cfg)
    log_path = seed_root / 'logs' / 'common_sac.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _tee_stdout(log_path):
        run_in_process(robot_cfg, cfg, dict(droq_cfg))
    checkpoint = latest_snapshot(common_dir)
    if checkpoint is None:
        raise RuntimeError('common SAC run produced no checkpoint')
    _validate_common_checkpoint(
        Path(checkpoint),
        min_transitions=args.balance_min_transitions,
        min_episodes=args.balance_min_episodes)
    return Path(checkpoint)


def _run_command(command: list[str]) -> None:
    print('[P15] ' + ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def _bootstrap_ci(values, seed=0):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [None, None]
    if len(array) == 1:
        return [float(array[0]), float(array[0])]
    rng = np.random.default_rng(seed)
    means = np.mean(
        rng.choice(array, size=(10_000, len(array)), replace=True),
        axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _aggregate(rows, finetune_steps):
    metric_names = (
        'run/falls_per_1000_steps',
        'run/fall_rate_per_episode',
        'run/average_return',
        'run/average_episode_length',
        'rolling/forward_velocity_mean',
        'rolling/tracking_error_mean',
        'rolling/near_failure_rate',
        'rolling/intervention_rate',
        'rolling/safety_cost_mean',
        'rolling/critic_loss_mean',
        'wall_time_sec',
        'run/policy_step_throughput',
    )
    sqrl_only_names = (
        'sqrl/active_steps',
        'sqrl/action_change_steps',
        'sqrl/replaced_steps',
        'sqrl/no_safe_steps',
        'sqrl/emergency_steps',
        'sqrl/replacement_rate',
        'sqrl/replacement_failure_rate_h8',
        'sqrl/replacement_failure_rate_h16',
        'sqrl/replacement_failure_rate_h32',
        'sqrl/false_negative_falls_h8',
        'sqrl/false_negative_falls_h16',
        'sqrl/false_negative_falls_h32',
    )
    paired = []
    keys = sorted({
        (int(row['seed']), float(row['ft_speed'])) for row in rows})
    for seed, speed in keys:
        sac = next((
            row for row in rows
            if row['seed'] == seed and row['algo'] == 'sac'
            and abs(row['ft_speed'] - speed) < 1e-9), None)
        sqrl = next((
            row for row in rows
            if row['seed'] == seed and row['algo'] == 'sqrl'
            and abs(row['ft_speed'] - speed) < 1e-9), None)
        if sac is None or sqrl is None:
            continue
        sac_falls = sac.get('run/falls_per_1000_steps')
        sqrl_falls = sqrl.get('run/falls_per_1000_steps')
        if sac_falls is None:
            sac_falls = (
                1000.0 * float(sac.get('falls_total_end') or 0)
                / finetune_steps)
        if sqrl_falls is None:
            sqrl_falls = (
                1000.0 * float(sqrl.get('falls_total_end') or 0)
                / finetune_steps)
        metric_pairs = {}
        for name in metric_names:
            sac_value = sac.get(name)
            sqrl_value = sqrl.get(name)
            if sac_value is None or sqrl_value is None:
                continue
            try:
                sac_numeric = float(sac_value)
                sqrl_numeric = float(sqrl_value)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(sac_numeric)
                    and np.isfinite(sqrl_numeric)):
                continue
            metric_pairs[name] = {
                'sac': sac_numeric,
                'sqrl': sqrl_numeric,
                'sqrl_minus_sac': sqrl_numeric - sac_numeric,
            }
        paired.append({
            'seed': seed,
            'speed': speed,
            'sac_falls_per_1000': sac_falls,
            'sqrl_falls_per_1000': sqrl_falls,
            'sqrl_minus_sac_falls_per_1000':
                sqrl_falls - sac_falls,
            'fall_percentage_reduction': (
                None if sac_falls == 0.0
                else (sac_falls - sqrl_falls) / sac_falls),
            'metric_pairs': metric_pairs,
            'sqrl_control_metrics': {
                name: sqrl[name]
                for name in sqrl_only_names if name in sqrl
            },
        })
    summary = {}
    for speed in sorted({row['speed'] for row in paired}):
        fall_values = [
            row['sqrl_minus_sac_falls_per_1000']
            for row in paired if row['speed'] == speed]
        metric_summary = {}
        for name in metric_names:
            cells = [
                row['metric_pairs'][name]
                for row in paired
                if row['speed'] == speed
                and name in row['metric_pairs']
            ]
            if not cells:
                continue
            differences = [
                cell['sqrl_minus_sac'] for cell in cells]
            metric_summary[name] = {
                'mean_sac': float(np.mean([
                    cell['sac'] for cell in cells])),
                'mean_sqrl': float(np.mean([
                    cell['sqrl'] for cell in cells])),
                'mean_paired_difference': float(
                    np.mean(differences)),
                'paired_difference_bootstrap_95_ci': _bootstrap_ci(
                    differences,
                    seed=int(round(speed * 1000))
                    + len(metric_summary)),
            }
        summary[f'{speed:.2f}'] = {
            'seeds': len(fall_values),
            'mean_sqrl_minus_sac_falls_per_1000': (
                float(np.mean(fall_values)) if fall_values else None),
            'bootstrap_95_ci': _bootstrap_ci(
                fall_values, seed=int(round(speed * 1000))),
            'metrics': metric_summary,
        }
    return {'paired_rows': paired, 'by_speed': summary}


def _attach_summary(row: dict) -> dict:
    path = Path(row['save_dir']) / 'training_summary.json'
    if path.is_file():
        row.update(_read_json(path))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--targets', default='0.40,0.50,0.80,1.00')
    parser.add_argument('--cmd-min', type=float, default=0.30)
    parser.add_argument('--cmd-max', type=float, default=1.00)
    parser.add_argument('--cmd-increment', type=float, default=0.05)
    parser.add_argument('--curriculum-max-steps', type=int, default=100000)
    parser.add_argument('--balance-max-steps', type=int, default=30000)
    parser.add_argument('--balance-min-transitions', type=int, default=1600)
    parser.add_argument('--balance-min-episodes', type=int, default=4)
    parser.add_argument('--promotion-window', type=int, default=8)
    parser.add_argument('--min-episode-length', type=float, default=300.0)
    parser.add_argument('--min-velocity-ratio', type=float, default=0.75)
    parser.add_argument('--max-fall-rate', type=float, default=0.125)
    parser.add_argument('--episode-steps', type=int, default=400)
    parser.add_argument(
        '--natural-transitions-per-speed', type=int, default=1600)
    parser.add_argument('--branch-snapshots-per-speed', type=int, default=40)
    parser.add_argument('--q-safe-steps', type=int, default=3000)
    parser.add_argument('--finetune-steps', type=int, default=4000)
    parser.add_argument('--out-root', type=Path, default=Path('saved/p15'))
    parser.add_argument('--model', default=(
        '/home/xyz/code/unitree_mujoco/unitree_robots/go2/'
        'scene_empty.xml'))
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--bounce', action='store_true')
    parser.add_argument('--reuse-complete', action='store_true')
    parser.add_argument(
        '--phase', choices=('common', 'all'), default='all',
        help='Use common for a controller-connected curriculum smoke test.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Write the resolved protocol without starting the controller.')
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    targets = [
        float(value) for value in args.targets.split(',') if value.strip()]
    args.out_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        'protocol': 'P15',
        'seeds': seeds,
        'targets': targets,
        'speed_range': [
            args.cmd_min, args.cmd_max, args.cmd_increment],
        'curriculum_max_steps': args.curriculum_max_steps,
        'balance_max_steps': args.balance_max_steps,
        'balance_min_transitions': args.balance_min_transitions,
        'balance_min_episodes': args.balance_min_episodes,
        'promotion_window': args.promotion_window,
        'min_episode_length': args.min_episode_length,
        'min_velocity_ratio': args.min_velocity_ratio,
        'max_fall_rate': args.max_fall_rate,
        'episode_steps': args.episode_steps,
        'natural_transitions_per_speed':
            args.natural_transitions_per_speed,
        'branch_snapshots_per_speed':
            args.branch_snapshots_per_speed,
        'q_safe_steps': args.q_safe_steps,
        'finetune_steps': args.finetune_steps,
        'primary_sqrl': {
            'action_masking_only': True,
            'actor_lagrange': False,
            'K': 32,
            'epsilon': 0.20,
            'support_gate': True,
            'q_safe_frozen': True,
        },
    }
    _write_json(args.out_root / 'resolved_protocol.json', resolved)
    if args.dry_run:
        print(json.dumps(resolved, indent=2))
        return 0

    rows = []
    seed_manifests = {}
    for seed_index, seed in enumerate(seeds):
        seed_root = args.out_root / f'seed_{seed}'
        seed_root.mkdir(parents=True, exist_ok=True)
        common = _run_common_sac(args, seed, seed_root)
        common_metadata = _validate_common_checkpoint(
            common,
            min_transitions=args.balance_min_transitions,
            min_episodes=args.balance_min_episodes)
        common_hash = snapshot_agent_hash(common)
        if args.phase == 'common':
            seed_manifests[str(seed)] = {
                'common_checkpoint': str(common.resolve()),
                'common_actor_hash': common_hash,
                'speed_coverage': common_metadata['speed_coverage'],
            }
            _write_json(
                seed_root / 'seed_manifest.json',
                seed_manifests[str(seed)])
            continue

        branches = seed_root / 'branches' / 'p15_multispeed.pkl'
        collect_command = [
            sys.executable,
            '-m', 'scripts.collect_p15_multispeed_branches',
            '--checkpoint', str(common),
            '--config', args.config,
            '--model', args.model,
            '--output', str(branches),
            '--seed', str(seed),
            '--cmd-min', str(args.cmd_min),
            '--cmd-max', str(args.cmd_max),
            '--cmd-increment', str(args.cmd_increment),
            '--natural-steps-per-speed',
            str(args.natural_transitions_per_speed),
            '--snapshots-per-speed',
            str(args.branch_snapshots_per_speed),
            '--min-episodes-per-speed',
            str(args.balance_min_episodes),
        ]
        if args.reuse_complete:
            collect_command.append('--reuse-complete')
        if not (args.reuse_complete and branches.is_file()):
            _run_command(collect_command)

        qsafe_root = seed_root / 'q_safe'
        qsafe_report = qsafe_root / 'p15_qsafe_report.json'
        train_command = [
            sys.executable,
            '-m', 'scripts.train_p15_qsafe',
            '--common-checkpoint', str(common),
            '--branches', str(branches),
            '--config', args.config,
            '--output-root', str(qsafe_root),
            '--seed', str(seed),
            '--cmd-min', str(args.cmd_min),
            '--cmd-max', str(args.cmd_max),
            '--cmd-increment', str(args.cmd_increment),
            '--gate-speeds', ','.join(map(str, targets)),
            '--min-natural-transitions',
            str(args.natural_transitions_per_speed),
            '--min-branch-snapshots',
            str(args.branch_snapshots_per_speed),
            '--steps', str(args.q_safe_steps),
        ]
        if not (args.reuse_complete and qsafe_report.is_file()):
            _run_command(train_command)
        report = _read_json(qsafe_report)
        qsafe_checkpoint = Path(report['q_safe_checkpoint'])
        if snapshot_agent_hash(qsafe_checkpoint) != common_hash:
            raise RuntimeError('Q_safe checkpoint no longer shares common actor')
        failed_gates = [
            speed for speed, gate in report['gates'].items()
            if not gate['p15_gate_passed']]
        paired_evaluation = {
            speed: {
                'p15_gate_passed': gate['p15_gate_passed'],
                'nominal_failure_rate': gate['control_metrics'][
                    'control_nominal_failure_rate'],
                'selected_failure_rate': gate['control_metrics'][
                    'control_selected_failure_rate'],
                'actual_failure_reduction': gate['control_metrics'][
                    'control_nominal_relative_failure_reduction'],
                'replacement_contribution': gate['control_metrics'][
                    'control_replacement_failure_contribution'],
                'fallback_contribution': gate['control_metrics'][
                    'control_fallback_failure_contribution'],
            }
            for speed, gate in report['gates'].items()
        }
        _write_json(
            seed_root / 'frozen_paired_evaluation.json',
            paired_evaluation)
        seed_manifests[str(seed)] = {
            'common_checkpoint': str(common.resolve()),
            'common_actor_hash': common_hash,
            'speed_coverage': common_metadata['speed_coverage'],
            'q_safe_checkpoint': str(qsafe_checkpoint.resolve()),
            'q_safe_actor_hash': snapshot_agent_hash(qsafe_checkpoint),
            'failed_gates': failed_gates,
            'frozen_paired_evaluation': paired_evaluation,
        }
        _write_json(seed_root / 'seed_manifest.json',
                    seed_manifests[str(seed)])
        if failed_gates:
            _write_json(args.out_root / 'p15_results.json', {
                'status': 'stopped_on_gate_failure',
                'failed_seed': seed,
                'failed_speeds': failed_gates,
                'seed_manifests': seed_manifests,
                'rows': rows,
            })
            raise RuntimeError(
                f'P15 seed {seed} gate failed at {failed_gates}; '
                'masking finetune was not started')

        source_step = int(common_metadata.get(
            'speed_coverage', {}).get('step', 0) or
            int(common.stem.rsplit('_', 1)[-1]))
        for speed in targets:
            tag = _speed_tag(speed)
            gate_path = Path(report['gates'][f'{speed:.2f}']['path'])
            sac_name = f'p15_seed{seed}_sac_{tag}'
            sac_row = _run_sac_ft(
                config=args.config,
                pre_ckpt=common,
                ft_dir=seed_root / 'finetune' / sac_name,
                log_path=seed_root / 'logs' / f'{sac_name}.log',
                n_pre=source_step,
                n_ft=args.finetune_steps,
                ft_speed=speed,
                wandb=args.wandb,
                run_name=sac_name,
                bounce=args.bounce,
                seed=seed)
            sac_row['seed'] = seed
            rows.append(_attach_summary(sac_row))

            sqrl_name = f'p15_seed{seed}_sqrl_{tag}'
            sqrl_row = _run_sqrl_ft(
                config=args.config,
                pre_ckpt=qsafe_checkpoint,
                ft_dir=seed_root / 'finetune' / sqrl_name,
                log_path=seed_root / 'logs' / f'{sqrl_name}.log',
                n_pre=source_step,
                n_ft=args.finetune_steps,
                ft_speed=speed,
                wandb=args.wandb,
                run_name=sqrl_name,
                bounce=args.bounce,
                control_metrics_path=gate_path,
                support_gate_enabled=True,
                actor_lagrange_enabled=False,
                freeze_safety_critic=True,
                prevalidated_control_gate=True,
                num_candidates=32,
                epsilon=0.20,
                seed=seed)
            sqrl_row['seed'] = seed
            rows.append(_attach_summary(sqrl_row))
        # Seed 42 is the mandatory pilot. Only reaching here authorizes the
        # remaining seeds listed by the caller.
        if seed_index == 0 and seed != 42 and len(seeds) > 1:
            raise RuntimeError('the first multi-seed P15 run must be seed 42')

    result = {
        'status': (
            'common_phase_complete'
            if args.phase == 'common' else 'complete'),
        'protocol': resolved,
        'seed_manifests': seed_manifests,
        'rows': rows,
        'paired': _aggregate(rows, args.finetune_steps),
    }
    _write_json(args.out_root / 'p15_results.json', result)
    print(json.dumps(result['paired'], indent=2, allow_nan=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
