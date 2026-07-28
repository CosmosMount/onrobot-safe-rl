#!/usr/bin/env python3
"""Reuse slow pretrain; sweep faster finetune speeds for SQRL vs SAC.

Writes stage logs, optional wandb runs, and a falls/convergence summary table.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from learner.checkpoint import latest_snapshot
from learner.learner import run_in_process
from scripts.parse_ft_train_log import parse_ft_log
from scripts.run_sqrl_transfer_scratch import (
    _bounce_controller, _fresh_dir, _scale_schedule_for_short_run,
    _stabilize, _tee_stdout)
from train.config import load_app_config
from train.main import _configure_sqrl_mode, apply_move_speed


def _speed_tag(speed: float) -> str:
    return f'v{int(round(speed * 100)):03d}'


def _read_wandb_meta(save_dir: str | Path) -> dict:
    path = Path(save_dir) / 'wandb_run.json'
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _run_sqrl_ft(*, config: str, pre_ckpt: Path, ft_dir: Path, log_path: Path,
                 n_pre: int, n_ft: int, ft_speed: float, wandb: bool,
                 run_name: str, bounce: bool) -> dict:
    robot_cfg, train_cfg, droq_cfg = load_app_config(path=config)
    robot_cfg = apply_move_speed(robot_cfg, ft_speed)
    droq_cfg = dict(droq_cfg)
    _fresh_dir(ft_dir)
    if bounce:
        _bounce_controller(f'before {run_name}')
    _stabilize(robot_cfg, train_cfg)
    ns = argparse.Namespace(
        mode='sqrl_finetune', checkpoint=str(pre_ckpt),
        save_dir=str(ft_dir), from_scratch=False)
    ft_cfg, ft_droq = _configure_sqrl_mode(ns, train_cfg, droq_cfg)
    ft_cfg.resume_checkpoint = False
    ft_cfg.warm_start_checkpoint = str(pre_ckpt)
    ft_cfg.max_steps = n_pre + n_ft
    ft_cfg.checkpoint_interval = min(1000, max(n_ft // 4, 1))
    ft_cfg.sqrl_activation_steps = min(int(ft_cfg.sqrl_activation_steps), 200)
    ft_cfg.sqrl_epsilon_anneal_steps = min(
        int(ft_cfg.sqrl_epsilon_anneal_steps), 500)
    ft_cfg.warmup = True
    ft_cfg.wandb = bool(wandb)
    ft_cfg.wandb_run_name = run_name
    ft_cfg.experiment_name = run_name
    ft_cfg.cmd_speed_curriculum = False
    _scale_schedule_for_short_run(ft_cfg, n_ft)
    t0 = time.time()
    with _tee_stdout(log_path):
        run_in_process(robot_cfg, ft_cfg, ft_droq)
    wall = time.time() - t0
    ckpt = latest_snapshot(ft_dir)
    text = log_path.read_text(encoding='utf-8', errors='replace')
    metrics = parse_ft_log(text, start_step=n_pre, ft_speed=ft_speed)
    metrics.update({
        'algo': 'sqrl',
        'ft_speed': ft_speed,
        'save_dir': str(ft_dir),
        'checkpoint': str(ckpt) if ckpt else None,
        'log_path': str(log_path),
        'wall_time_sec': wall,
        'wandb': _read_wandb_meta(ft_dir),
    })
    return metrics


def _run_sac_ft(*, config: str, pre_ckpt: Path, ft_dir: Path, log_path: Path,
                n_pre: int, n_ft: int, ft_speed: float, wandb: bool,
                run_name: str, bounce: bool) -> dict:
    robot_cfg, train_cfg, droq_cfg = load_app_config(path=config)
    robot_cfg = apply_move_speed(robot_cfg, ft_speed)
    droq_cfg = dict(droq_cfg)
    _fresh_dir(ft_dir)
    if bounce:
        _bounce_controller(f'before {run_name}')
    _stabilize(robot_cfg, train_cfg)
    sac_ft = train_cfg
    sac_ft.sqrl_enabled = False
    sac_ft.safety_critic_enabled = False
    sac_ft.experiment_name = run_name
    sac_ft.save_dir = str(ft_dir)
    sac_ft.max_steps = n_pre + n_ft
    sac_ft.checkpoint_interval = min(1000, max(n_ft // 4, 1))
    sac_ft.warmup = True
    sac_ft.wandb = bool(wandb)
    sac_ft.wandb_run_name = run_name
    sac_ft.resume_checkpoint = False
    sac_ft.warm_start_checkpoint = str(pre_ckpt)
    sac_ft.cmd_speed_curriculum = False
    _scale_schedule_for_short_run(sac_ft, n_ft)
    t0 = time.time()
    with _tee_stdout(log_path):
        run_in_process(robot_cfg, sac_ft, droq_cfg)
    wall = time.time() - t0
    ckpt = latest_snapshot(ft_dir)
    text = log_path.read_text(encoding='utf-8', errors='replace')
    metrics = parse_ft_log(text, start_step=n_pre, ft_speed=ft_speed)
    metrics.update({
        'algo': 'sac',
        'ft_speed': ft_speed,
        'save_dir': str(ft_dir),
        'checkpoint': str(ckpt) if ckpt else None,
        'log_path': str(log_path),
        'wall_time_sec': wall,
        'wandb': _read_wandb_meta(ft_dir),
    })
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--pretrain-steps', type=int, default=12000)
    parser.add_argument('--finetune-steps', type=int, default=4000)
    parser.add_argument('--ft-speeds', default='0.40,0.45,0.50')
    parser.add_argument(
        '--sqrl-pre-dir', default='saved/checkpoints_sqrl_xfer_v2_pre')
    parser.add_argument(
        '--sac-pre-dir', default='saved/checkpoints_sac_xfer_v2_pre')
    parser.add_argument(
        '--out-root', default='saved/checkpoints_ft_speed_sweep')
    parser.add_argument(
        '--stage-log-dir',
        default='saved/safety_evaluation/ft_speed_sweep_logs')
    parser.add_argument(
        '--output',
        default='saved/safety_evaluation/ft_speed_sweep_summary.json')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no-bounce', action='store_true')
    parser.add_argument(
        '--algos', default='sqrl,sac',
        help='Comma list: sqrl,sac')
    args = parser.parse_args()

    speeds = [float(x) for x in args.ft_speeds.split(',') if x.strip()]
    algos = [x.strip() for x in args.algos.split(',') if x.strip()]
    n_pre = int(args.pretrain_steps)
    n_ft = int(args.finetune_steps)
    bounce = not args.no_bounce
    stage_log_dir = Path(args.stage_log_dir)
    stage_log_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    sqrl_pre = latest_snapshot(args.sqrl_pre_dir)
    sac_pre = latest_snapshot(args.sac_pre_dir)
    if 'sqrl' in algos and sqrl_pre is None:
        raise SystemExit(f'No SQRL pre checkpoint in {args.sqrl_pre_dir}')
    if 'sac' in algos and sac_pre is None:
        raise SystemExit(f'No SAC pre checkpoint in {args.sac_pre_dir}')

    rows: list[dict] = []
    for speed in speeds:
        tag = _speed_tag(speed)
        if 'sqrl' in algos:
            run_name = f'sqrl_ft_{tag}'
            print(f'[sweep] === {run_name} speed={speed} ===', flush=True)
            row = _run_sqrl_ft(
                config=args.config, pre_ckpt=Path(sqrl_pre),
                ft_dir=out_root / run_name,
                log_path=stage_log_dir / f'{run_name}.log',
                n_pre=n_pre, n_ft=n_ft, ft_speed=speed,
                wandb=args.wandb, run_name=run_name, bounce=bounce)
            rows.append(row)
            print(f'[sweep] {run_name} falls={row["falls_total_end"]} '
                  f'vel={row["final_forward_vel"]} '
                  f'converged={row["converged_step"]} '
                  f'wall_s={row["wall_time_sec"]:.1f} '
                  f'wandb={row["wandb"].get("url") or row["wandb"].get("name")}',
                  flush=True)
        if 'sac' in algos:
            run_name = f'sac_ft_{tag}'
            print(f'[sweep] === {run_name} speed={speed} ===', flush=True)
            row = _run_sac_ft(
                config=args.config, pre_ckpt=Path(sac_pre),
                ft_dir=out_root / run_name,
                log_path=stage_log_dir / f'{run_name}.log',
                n_pre=n_pre, n_ft=n_ft, ft_speed=speed,
                wandb=args.wandb, run_name=run_name, bounce=bounce)
            rows.append(row)
            print(f'[sweep] {run_name} falls={row["falls_total_end"]} '
                  f'vel={row["final_forward_vel"]} '
                  f'converged={row["converged_step"]} '
                  f'wall_s={row["wall_time_sec"]:.1f} '
                  f'wandb={row["wandb"].get("url") or row["wandb"].get("name")}',
                  flush=True)

    # Pairwise comparison table
    table = []
    for speed in speeds:
        tag = _speed_tag(speed)
        sq = next((r for r in rows if r['algo'] == 'sqrl' and abs(r['ft_speed'] - speed) < 1e-9), None)
        sa = next((r for r in rows if r['algo'] == 'sac' and abs(r['ft_speed'] - speed) < 1e-9), None)
        table.append({
            'ft_speed': speed,
            'sqrl_falls': None if sq is None else sq['falls_total_end'],
            'sac_falls': None if sa is None else sa['falls_total_end'],
            'sqrl_converged_step': None if sq is None else sq['converged_step'],
            'sac_converged_step': None if sa is None else sa['converged_step'],
            'sqrl_final_vel': None if sq is None else sq['final_forward_vel'],
            'sac_final_vel': None if sa is None else sa['final_forward_vel'],
            'sqrl_wall_s': None if sq is None else sq['wall_time_sec'],
            'sac_wall_s': None if sa is None else sa['wall_time_sec'],
            'sqrl_wandb': None if sq is None else sq.get('wandb'),
            'sac_wandb': None if sa is None else sa.get('wandb'),
            'sqrl_fewer_falls': (
                None if sq is None or sa is None
                or sq['falls_total_end'] is None
                or sa['falls_total_end'] is None
                else sq['falls_total_end'] < sa['falls_total_end']),
        })

    summary = {
        'protocol': 'ft_speed_sweep_same_pre',
        'pretrain_steps': n_pre,
        'finetune_steps': n_ft,
        'sqrl_pre': str(sqrl_pre) if sqrl_pre else None,
        'sac_pre': str(sac_pre) if sac_pre else None,
        'ft_speeds': speeds,
        'wandb_requested': bool(args.wandb),
        'runs': rows,
        'table': table,
        'offline_v2_baseline': {
            'path': 'saved/safety_evaluation/sqrl_xfer_v2_ft_falls_offline.json',
            'note': 'Prior 0.40 FT: SQRL falls=5 vs SAC falls=8',
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'[sweep] summary={out}', flush=True)
    print('[sweep] TABLE', json.dumps(table, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
