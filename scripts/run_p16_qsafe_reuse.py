#!/usr/bin/env python3
"""P16: reuse a 0.30 m/s Q_safe while training a fresh 0.30 m/s SAC."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from flax import serialization

from jaxrl.agents import DroQLearner
from jaxrl.data.replay_buffer import ReplayBuffer
from jaxrl.data.safety_replay import SafetyReplayManager
from jaxrl.env.specs import BoxSpec
from learner.checkpoint import (
    agent_state_hash,
    latest_snapshot,
    save_training_snapshot,
    snapshot_agent_hash,
)
from learner.learner import run_in_process
from scripts.compose_qsafe_transfer_checkpoint import (
    compose_qsafe_transfer_checkpoint,
)
from scripts.run_sqrl_ft_speed_sweep import _run_sac_ft, _run_sqrl_ft
from scripts.run_sqrl_transfer_scratch import (
    _fresh_dir,
    _scale_schedule_for_short_run,
    _stabilize,
    _tee_stdout,
)
from train.config import load_app_config
from train.main import apply_move_speed


_FALL_EVENT = re.compile(r'\[step (?P<step>\d+)\] episode done \(fallen\)')
_ROLL = re.compile(
    r'\[step (?P<step>\d+)\] rolling n=\d+ .*?falls=(?P<falls>\d+)')


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_predecessor(pid: int, result_path: Path | None) -> dict:
    started = time.time()
    last_notice = 0.0
    while _pid_alive(pid):
        now = time.time()
        if now - last_notice >= 300.0 or last_notice == 0.0:
            print(
                f'[P16] waiting for P15 pid={pid}; '
                f'elapsed={now - started:.0f}s', flush=True)
            last_notice = now
        time.sleep(30.0)
    predecessor: dict[str, object] = {
        'pid': int(pid),
        'waited_sec': float(time.time() - started),
    }
    if result_path is not None:
        if not result_path.is_file():
            raise RuntimeError(
                f'P15 process ended without terminal result: {result_path}')
        predecessor['result_path'] = str(result_path.resolve())
        predecessor['result'] = json.loads(
            result_path.read_text(encoding='utf-8'))
    print('[P16] P15 terminal result observed; starting P16', flush=True)
    return predecessor


def create_target_step0_checkpoint(
        config: str,
        output_dir: Path,
        *,
        seed: int,
        speed: float,
) -> tuple[Path, dict[str, object]]:
    robot_cfg, cfg, droq_cfg = load_app_config(config)
    robot_cfg = apply_move_speed(robot_cfg, speed)
    cfg.seed = int(seed)
    observation_spec = BoxSpec(
        shape=(robot_cfg.obs_dim,), dtype=np.float32,
        low=-np.inf, high=np.inf)
    action_spec = BoxSpec(
        shape=(robot_cfg.num_joints,), dtype=np.float32,
        low=-np.ones(robot_cfg.num_joints, dtype=np.float32),
        high=np.ones(robot_cfg.num_joints, dtype=np.float32))
    agent = DroQLearner.create(
        cfg.seed, observation_spec, action_spec, **dict(droq_cfg))
    reward_replay = ReplayBuffer(
        observation_spec, action_spec, cfg.buffer_size)
    reward_replay.seed(cfg.seed)
    safety_replay = SafetyReplayManager(
        recent_capacity=cfg.safety_recent_capacity,
        failure_capacity=cfg.safety_failure_capacity,
        boundary_capacity=cfg.safety_boundary_capacity,
        recovery_capacity=cfg.safety_recovery_capacity,
        all_capacity=cfg.buffer_size,
        failure_history=cfg.safety_failure_history,
        n_step=cfg.safety_critic_n_step,
        failure_horizons=cfg.safety_failure_horizons,
        seed=cfg.seed)
    _fresh_dir(output_dir)
    state_hash = agent_state_hash(serialization.to_state_dict(agent))
    path = save_training_snapshot(
        output_dir, agent=agent, replay_buffer=reward_replay,
        safety_replay=safety_replay, step=0,
        metadata={
            'protocol': 'P16',
            'experiment_name': 'p16_target_step0',
            'seed': int(seed),
            'move_speed': float(speed),
            'obs_dim': int(robot_cfg.obs_dim),
            'action_dim': int(robot_cfg.num_joints),
            'agent_state_hash': state_hash,
            'reward_replay_size': 0,
        })
    return path, {
        'checkpoint': str(path.resolve()),
        'agent_hash': state_hash,
        'reward_replay_size': 0,
        'seed': int(seed),
    }


def _run_source(
        config: str,
        output_dir: Path,
        log_path: Path,
        *,
        seed: int,
        speed: float,
        steps: int,
        wandb: bool,
) -> tuple[Path, dict[str, object]]:
    robot_cfg, cfg, droq_cfg = load_app_config(config)
    robot_cfg = apply_move_speed(robot_cfg, speed)
    cfg.seed = int(seed)
    _fresh_dir(output_dir)
    _stabilize(robot_cfg, cfg)
    cfg.experiment_name = 'p16_qsafe_source'
    cfg.save_dir = str(output_dir)
    cfg.max_steps = int(steps)
    cfg.checkpoint_interval = min(1000, max(steps // 4, 1))
    cfg.resume_checkpoint = False
    cfg.warm_start_checkpoint = None
    cfg.cmd_speed_curriculum = False
    cfg.sqrl_enabled = False
    cfg.safety_replay_enabled = True
    cfg.safety_critic_enabled = True
    cfg.safety_critic_update_interval = 1
    cfg.wandb = bool(wandb)
    cfg.wandb_run_name = 'p16_qsafe_source'
    _scale_schedule_for_short_run(cfg, steps)
    with _tee_stdout(log_path):
        run_in_process(robot_cfg, cfg, dict(droq_cfg))
    checkpoint = latest_snapshot(output_dir)
    if checkpoint is None:
        raise RuntimeError('P16 source produced no checkpoint')
    with checkpoint.open('rb') as stream:
        payload = pickle.load(stream)
    if 'safety_critic_state' not in payload:
        raise RuntimeError('P16 source checkpoint has no Q_safe')
    return checkpoint, _read_training_result(output_dir, log_path)


def _read_training_result(output_dir: Path, log_path: Path) -> dict[str, object]:
    summary_path = output_dir / 'training_summary.json'
    summary = (
        json.loads(summary_path.read_text(encoding='utf-8'))
        if summary_path.is_file() else {})
    text = (
        log_path.read_text(encoding='utf-8', errors='replace')
        if log_path.is_file() else '')
    fall_steps = [int(match.group('step')) for match in _FALL_EVENT.finditer(text)]
    curve = [
        {'step': int(match.group('step')), 'falls': int(match.group('falls'))}
        for match in _ROLL.finditer(text)
    ]
    return {
        **summary,
        'fall_event_steps': fall_steps,
        'first_fall_step': fall_steps[0] if fall_steps else None,
        'cumulative_fall_curve': curve,
        'checkpoint': str(latest_snapshot(output_dir) or ''),
        'log_path': str(log_path.resolve()),
    }


def _run_module(args: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('[P16] ' + ' '.join(args), flush=True)
    with log_path.open('w', encoding='utf-8') as stream:
        completed = subprocess.run(
            [sys.executable, '-u', '-m', *args],
            stdout=stream, stderr=subprocess.STDOUT,
            cwd=Path(__file__).resolve().parents[1],
            check=False)
    if completed.returncode:
        raise RuntimeError(
            f'module {args[0]} failed ({completed.returncode}); see {log_path}')


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=True), encoding='utf-8')
    temporary.replace(path)


def run_target(
        *,
        config: str,
        root: Path,
        source_checkpoint: Path,
        seed: int,
        speed: float,
        steps: int,
        branch_natural_steps: int,
        branch_snapshots: int,
        wandb: bool,
) -> dict[str, object]:
    target_root = root / f'target_seed_{seed}'
    logs = target_root / 'logs'
    base_checkpoint, base = create_target_step0_checkpoint(
        config, target_root / 'step0_base', seed=seed, speed=speed)
    transfer_checkpoint = target_root / 'step0_qsafe_transfer.pkl'
    transfer_meta = compose_qsafe_transfer_checkpoint(
        base_checkpoint, source_checkpoint, transfer_checkpoint)
    if snapshot_agent_hash(base_checkpoint) != snapshot_agent_hash(
            transfer_checkpoint):
        raise AssertionError('P16 base and Q_safe target actor hashes differ')

    branches = target_root / 'target_actor_branches.pkl'
    interval = max(branch_natural_steps // max(branch_snapshots, 1), 1)
    _run_module([
        'scripts.collect_mujoco_branches',
        '--config', config,
        '--checkpoint', str(transfer_checkpoint),
        '--move-speed', str(speed),
        '--output', str(branches),
        '--seed', str(seed + 50_000),
        '--natural-steps', str(branch_natural_steps),
        '--natural-action-noise-std', '0.05',
        '--natural-episode-max-steps', '100',
        '--max-episodes', '100',
        '--snapshot-interval', str(interval),
        '--perturbation-count', '8',
        '--policy-sample-count', '8',
        '--horizons', '8,16,32',
    ], logs / 'target_branch_collection.log')
    gate_path = target_root / 'p16_transfer_gate.json'
    _run_module([
        'scripts.evaluate_p16_transfer',
        '--config', config,
        '--checkpoint', str(transfer_checkpoint),
        '--branches', str(branches),
        '--output', str(gate_path),
        '--seed', str(seed),
        '--horizon', '32',
        '--epsilon', '0.20',
    ], logs / 'target_transfer_gate.log')
    gate = json.loads(gate_path.read_text(encoding='utf-8'))

    common = dict(
        config=config, n_pre=0, n_ft=steps,
        ft_speed=speed, wandb=wandb, bounce=False, seed=seed)
    sac_dir = target_root / 'sac'
    _run_sac_ft(
        **common, pre_ckpt=base_checkpoint,
        ft_dir=sac_dir, log_path=logs / 'sac.log',
        run_name=f'p16_sac_seed{seed}')
    sac_result = _read_training_result(sac_dir, logs / 'sac.log')

    logging_dir = target_root / 'qsafe_logging'
    _run_sqrl_ft(
        **common, pre_ckpt=transfer_checkpoint,
        ft_dir=logging_dir, log_path=logs / 'qsafe_logging.log',
        run_name=f'p16_qsafe_logging_seed{seed}',
        freeze_safety_critic=True, logging_only=True,
        actor_lagrange_enabled=False, num_candidates=32, epsilon=0.20,
        support_gate_enabled=True)
    logging_result = _read_training_result(
        logging_dir, logs / 'qsafe_logging.log')

    masking_result = None
    status = 'transfer_gate_failed'
    if bool(gate.get('p16_gate_passed', False)):
        masking_dir = target_root / 'qsafe_masking'
        _run_sqrl_ft(
            **common, pre_ckpt=transfer_checkpoint,
            ft_dir=masking_dir, log_path=logs / 'qsafe_masking.log',
            run_name=f'p16_qsafe_masking_seed{seed}',
            control_metrics_path=gate_path,
            support_gate_enabled=True,
            actor_lagrange_enabled=False,
            freeze_safety_critic=True,
            prevalidated_control_gate=True,
            num_candidates=32, epsilon=0.20)
        masking_result = _read_training_result(
            masking_dir, logs / 'qsafe_masking.log')
        status = 'complete'
    else:
        print(
            '[P16] transfer gate failed; masking is intentionally skipped: '
            f'{gate.get("p16_gate_failed_checks")}', flush=True)

    result = {
        'protocol': 'P16',
        'status': status,
        'seed': int(seed),
        'speed': float(speed),
        'target_step0': base,
        'transfer': transfer_meta,
        'gate': gate,
        'groups': {
            'A_pure_sac': sac_result,
            'C_qsafe_logging_only': logging_result,
            'B_qsafe_masking': masking_result,
        },
    }
    _write_json(target_root / 'p16_target_result.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--out-root', type=Path,
                        default=Path('saved/p16_qsafe_reuse'))
    parser.add_argument('--source-seed', type=int, default=142)
    parser.add_argument('--target-seeds', default='42')
    parser.add_argument('--speed', type=float, default=0.30)
    parser.add_argument('--source-steps', type=int, default=15000)
    parser.add_argument('--target-steps', type=int, default=15000)
    parser.add_argument('--branch-natural-steps', type=int, default=1200)
    parser.add_argument('--branch-snapshots', type=int, default=120)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wait-pid', type=int)
    parser.add_argument('--p15-result', type=Path)
    args = parser.parse_args()

    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    predecessor = None
    if args.wait_pid is not None:
        predecessor = wait_for_predecessor(
            args.wait_pid,
            args.p15_result.resolve() if args.p15_result else None)

    result: dict[str, object] = {
        'protocol': 'P16',
        'status': 'running',
        'design': {
            'source_speed': args.speed,
            'source_seed': args.source_seed,
            'target_seeds': [
                int(value) for value in args.target_seeds.split(',')],
            'target_speed': args.speed,
            'source_actor_transferred': False,
            'source_reward_critic_transferred': False,
            'source_reward_replay_transferred': False,
            'source_qsafe_transferred': True,
            'target_groups_share_step0_agent_and_empty_reward_replay': True,
        },
        'predecessor': predecessor,
        'targets': [],
    }
    result_path = root / 'p16_results.json'
    _write_json(result_path, result)
    try:
        source_dir = root / f'source_seed_{args.source_seed}'
        source_checkpoint, source_result = _run_source(
            args.config, source_dir, source_dir / 'source.log',
            seed=args.source_seed, speed=args.speed,
            steps=args.source_steps, wandb=args.wandb)
        result['source'] = {
            **source_result,
            'seed': args.source_seed,
            'speed': args.speed,
            'checkpoint': str(source_checkpoint.resolve()),
            'agent_hash': snapshot_agent_hash(source_checkpoint),
            'role': 'Q_safe producer only; not a target comparison group',
        }
        _write_json(result_path, result)
        for seed in result['design']['target_seeds']:
            target = run_target(
                config=args.config, root=root,
                source_checkpoint=source_checkpoint,
                seed=int(seed), speed=args.speed, steps=args.target_steps,
                branch_natural_steps=args.branch_natural_steps,
                branch_snapshots=args.branch_snapshots,
                wandb=args.wandb)
            result['targets'].append(target)
            _write_json(result_path, result)
        result['status'] = 'complete'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = f'{type(exc).__name__}: {exc}'
        _write_json(result_path, result)
        raise
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
