#!/usr/bin/env python3
"""Multi-speed cmd-obs pretrain → held-out probes → fixed-c finetune.

Protocol (plan Multi-speed SQRL vs SAC):
  - Obs includes normalized command speed.
  - SQRL + SAC pretrain with curriculum c ∈ [0.30, 1.0].
  - After pre: short held-out at fixed speeds (no updates).
  - Then finetune at each fixed speed; record training falls.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from learner.checkpoint import latest_snapshot
from learner.learner import run_in_process
from scripts.parse_ft_train_log import parse_ft_log
from scripts.run_sqrl_ft_speed_sweep import (
    _read_wandb_meta, _run_sac_ft, _run_sqrl_ft, _speed_tag)
from scripts.run_sqrl_transfer_scratch import (
    _bounce_controller, _fresh_dir, _heldout_fixed_speed,
    _scale_schedule_for_short_run, _stabilize, _tee_stdout)
from train.config import load_app_config
from train.main import _configure_sqrl_mode, apply_move_speed


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def _load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, list):
        raise ValueError(f'Expected a JSON list in {path}')
    return value


def _completed_ft_row(*, ft_dir: Path, log_path: Path, algo: str,
                      speed: float, start_step: int,
                      target_step: int) -> dict | None:
    """Rehydrate a completed FT run without touching its directory."""
    checkpoint = latest_snapshot(ft_dir)
    if checkpoint is None or not log_path.is_file():
        return None
    from learner.checkpoint import load_training_snapshot_metadata
    # Filename is authoritative even when an older snapshot lacks useful
    # metadata. Keep metadata read here to surface corrupt files early.
    load_training_snapshot_metadata(checkpoint)
    try:
        checkpoint_step = int(Path(checkpoint).stem.rsplit('_', 1)[-1])
    except ValueError:
        return None
    if checkpoint_step < target_step:
        return None
    text = log_path.read_text(encoding='utf-8', errors='replace')
    metrics = parse_ft_log(text, start_step=start_step, ft_speed=speed)
    metrics.update({
        'algo': algo,
        'ft_speed': speed,
        'save_dir': str(ft_dir),
        'checkpoint': str(checkpoint),
        'log_path': str(log_path),
        'wall_time_sec': None,
        'wandb': _read_wandb_meta(ft_dir),
        'reused_completed_run': True,
    })
    return metrics


def _enable_curriculum(train_cfg, *, n_pre: int, c_min: float, c_max: float):
    train_cfg.cmd_speed_curriculum = True
    train_cfg.cmd_speed_min = float(c_min)
    train_cfg.cmd_speed_max = float(c_max)
    train_cfg.cmd_speed_curriculum_steps = int(n_pre)
    return train_cfg


def _disable_curriculum(train_cfg):
    train_cfg.cmd_speed_curriculum = False
    return train_cfg


def _run_sqrl_multispeed_pre(*, config: str, pre_dir: Path, log_path: Path,
                            n_pre: int, c_min: float, c_max: float,
                            wandb: bool, run_name: str, bounce: bool) -> Path:
    robot_cfg, train_cfg, droq_cfg = load_app_config(path=config)
    # Base cfg speed unused while curriculum is on; keep min as default.
    robot_cfg = apply_move_speed(robot_cfg, c_min)
    droq_cfg = dict(droq_cfg)
    _fresh_dir(pre_dir)
    if bounce:
        _bounce_controller(f'before {run_name}')
    _stabilize(robot_cfg, train_cfg)
    ns = argparse.Namespace(
        mode='sqrl_pretrain', checkpoint=None, save_dir=str(pre_dir),
        from_scratch=True)
    pre_cfg, pre_droq = _configure_sqrl_mode(ns, train_cfg, droq_cfg)
    pre_cfg = _enable_curriculum(
        pre_cfg, n_pre=n_pre, c_min=c_min, c_max=c_max)
    pre_cfg.max_steps = n_pre
    pre_cfg.checkpoint_interval = min(2000, max(n_pre // 4, 1))
    pre_cfg.warmup = True
    pre_cfg.wandb = bool(wandb)
    pre_cfg.wandb_run_name = run_name
    pre_cfg.experiment_name = run_name
    _scale_schedule_for_short_run(pre_cfg, n_pre)
    with _tee_stdout(log_path):
        run_in_process(robot_cfg, pre_cfg, pre_droq)
    ckpt = latest_snapshot(pre_dir)
    if ckpt is None:
        raise RuntimeError(f'SQRL multi-speed pre produced no snapshot in {pre_dir}')
    return Path(ckpt)


def _run_sac_multispeed_pre(*, config: str, pre_dir: Path, log_path: Path,
                            n_pre: int, c_min: float, c_max: float,
                            wandb: bool, run_name: str, bounce: bool) -> Path:
    robot_cfg, train_cfg, droq_cfg = load_app_config(path=config)
    robot_cfg = apply_move_speed(robot_cfg, c_min)
    droq_cfg = dict(droq_cfg)
    _fresh_dir(pre_dir)
    if bounce:
        _bounce_controller(f'before {run_name}')
    _stabilize(robot_cfg, train_cfg)
    sac = _enable_curriculum(train_cfg, n_pre=n_pre, c_min=c_min, c_max=c_max)
    sac.sqrl_enabled = False
    sac.safety_critic_enabled = False
    sac.experiment_name = run_name
    sac.save_dir = str(pre_dir)
    sac.max_steps = n_pre
    sac.checkpoint_interval = min(2000, max(n_pre // 4, 1))
    sac.warmup = True
    sac.wandb = bool(wandb)
    sac.wandb_run_name = run_name
    sac.resume_checkpoint = False
    sac.warm_start_checkpoint = None
    _scale_schedule_for_short_run(sac, n_pre)
    with _tee_stdout(log_path):
        run_in_process(robot_cfg, sac, droq_cfg)
    ckpt = latest_snapshot(pre_dir)
    if ckpt is None:
        raise RuntimeError(f'SAC multi-speed pre produced no snapshot in {pre_dir}')
    return Path(ckpt)


def _run_heldout(*, config: str, checkpoint: Path, speed: float,
                 max_steps: int, use_sqrl: bool, bounce: bool,
                 seed: int) -> dict:
    robot_cfg, train_cfg, droq_cfg = load_app_config(path=config)
    robot_cfg = apply_move_speed(robot_cfg, speed)
    train_cfg = _disable_curriculum(train_cfg)
    droq_cfg = dict(droq_cfg)
    if bounce:
        _bounce_controller(f'before heldout {"sqrl" if use_sqrl else "sac"} '
                           f'v={speed}')
    _stabilize(robot_cfg, train_cfg)
    return _heldout_fixed_speed(
        robot_cfg=robot_cfg, train_cfg=train_cfg, droq_cfg=droq_cfg,
        checkpoint=str(checkpoint), max_steps=max_steps,
        action_noise_std=0.0, rollout_seed=seed, use_sqrl=use_sqrl,
        epsilon=float(train_cfg.sqrl_epsilon),
        num_candidates=int(train_cfg.sqrl_num_candidates),
        noise_mode='candidate')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--pretrain-steps', type=int, default=12000)
    parser.add_argument('--finetune-steps', type=int, default=4000)
    parser.add_argument('--heldout-steps', type=int, default=1000)
    parser.add_argument('--cmd-min', type=float, default=0.30)
    parser.add_argument('--cmd-max', type=float, default=1.0)
    parser.add_argument(
        '--probe-speeds', default='0.40,0.50,0.60,0.80,1.00')
    parser.add_argument(
        '--out-root', default='saved/checkpoints_multispeed')
    parser.add_argument(
        '--stage-log-dir',
        default='saved/safety_evaluation/multispeed_logs')
    parser.add_argument(
        '--output',
        default='saved/safety_evaluation/multispeed_sqrl_sac_summary.json')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--no-bounce', action='store_true')
    parser.add_argument(
        '--skip-pre', action='store_true',
        help='Reuse existing multispeed pre dirs under out-root')
    parser.add_argument(
        '--skip-heldout', action='store_true')
    parser.add_argument(
        '--skip-ft', action='store_true')
    parser.add_argument(
        '--skip-ft-done', action='store_true',
        help=('Reuse an FT run when its latest checkpoint reaches '
              'pretrain_steps + finetune_steps; incomplete runs are rerun.'))
    parser.add_argument(
        '--heldout-cache', default=None,
        help=('JSON cache for held-out rows. Defaults to '
              '<stage-log-dir>/heldout_rows.json.'))
    parser.add_argument(
        '--finetune-cache', default=None,
        help=('JSON cache updated after every FT run. Defaults to '
              '<stage-log-dir>/finetune_rows.json.'))
    parser.add_argument(
        '--algos', default='sqrl,sac')
    parser.add_argument('--heldout-seed', type=int, default=9100)
    args = parser.parse_args()

    speeds = [float(x) for x in args.probe_speeds.split(',') if x.strip()]
    algos = [x.strip() for x in args.algos.split(',') if x.strip()]
    n_pre = int(args.pretrain_steps)
    n_ft = int(args.finetune_steps)
    n_hold = int(args.heldout_steps)
    bounce = not args.no_bounce
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stage_log_dir = Path(args.stage_log_dir)
    stage_log_dir.mkdir(parents=True, exist_ok=True)
    heldout_cache = Path(
        args.heldout_cache or stage_log_dir / 'heldout_rows.json')
    finetune_cache = Path(
        args.finetune_cache or stage_log_dir / 'finetune_rows.json')

    sqrl_pre_dir = out_root / 'sqrl_pre'
    sac_pre_dir = out_root / 'sac_pre'
    sqrl_pre = None
    sac_pre = None

    if not args.skip_pre:
        if 'sqrl' in algos:
            print('[ms] === SQRL multi-speed pretrain ===', flush=True)
            sqrl_pre = _run_sqrl_multispeed_pre(
                config=args.config, pre_dir=sqrl_pre_dir,
                log_path=stage_log_dir / 'sqrl_pre.log',
                n_pre=n_pre, c_min=args.cmd_min, c_max=args.cmd_max,
                wandb=args.wandb, run_name='sqrl_multispeed_pre',
                bounce=bounce)
            print(f'[ms] SQRL pre ckpt={sqrl_pre}', flush=True)
        if 'sac' in algos:
            print('[ms] === SAC multi-speed pretrain ===', flush=True)
            sac_pre = _run_sac_multispeed_pre(
                config=args.config, pre_dir=sac_pre_dir,
                log_path=stage_log_dir / 'sac_pre.log',
                n_pre=n_pre, c_min=args.cmd_min, c_max=args.cmd_max,
                wandb=args.wandb, run_name='sac_multispeed_pre',
                bounce=bounce)
            print(f'[ms] SAC pre ckpt={sac_pre}', flush=True)
    else:
        sqrl_pre = latest_snapshot(sqrl_pre_dir)
        sac_pre = latest_snapshot(sac_pre_dir)
        if 'sqrl' in algos and sqrl_pre is None:
            raise SystemExit(f'No SQRL pre in {sqrl_pre_dir}')
        if 'sac' in algos and sac_pre is None:
            raise SystemExit(f'No SAC pre in {sac_pre_dir}')
        if sqrl_pre is not None:
            sqrl_pre = Path(sqrl_pre)
        if sac_pre is not None:
            sac_pre = Path(sac_pre)

    heldout_rows: list[dict] = []
    if args.skip_heldout:
        heldout_rows = _load_rows(heldout_cache)
        if not heldout_rows:
            raise SystemExit(
                f'--skip-heldout requires cached rows in {heldout_cache}')
        print(f'[ms] reused {len(heldout_rows)} held-out rows from '
              f'{heldout_cache}', flush=True)
    else:
        for speed in speeds:
            tag = _speed_tag(speed)
            if 'sqrl' in algos and sqrl_pre is not None:
                print(f'[ms] === heldout SQRL {tag} ===', flush=True)
                row = _run_heldout(
                    config=args.config, checkpoint=sqrl_pre, speed=speed,
                    max_steps=n_hold, use_sqrl=True, bounce=bounce,
                    seed=args.heldout_seed)
                row['algo'] = 'sqrl'
                heldout_rows.append(row)
                _write_json(heldout_cache, heldout_rows)
                print(f'[ms] heldout sqrl {tag} falls={row["falls"]} '
                      f'vel={row["mean_forward_vel"]}', flush=True)
            if 'sac' in algos and sac_pre is not None:
                print(f'[ms] === heldout SAC {tag} ===', flush=True)
                row = _run_heldout(
                    config=args.config, checkpoint=sac_pre, speed=speed,
                    max_steps=n_hold, use_sqrl=False, bounce=bounce,
                    seed=args.heldout_seed + 1)
                row['algo'] = 'sac'
                heldout_rows.append(row)
                _write_json(heldout_cache, heldout_rows)
                print(f'[ms] heldout sac {tag} falls={row["falls"]} '
                      f'vel={row["mean_forward_vel"]}', flush=True)

    ft_rows: list[dict] = _load_rows(finetune_cache)
    if args.skip_ft_done:
        # Populate the cache from any complete on-disk runs before deciding
        # whether new simulator work is needed. This also lets --skip-ft emit
        # a useful partial summary without starting the environment.
        ft_root = out_root / 'ft'
        for speed in speeds:
            tag = _speed_tag(speed)
            for algo in algos:
                run_name = f'{algo}_ft_{tag}'
                row = _completed_ft_row(
                    ft_dir=ft_root / run_name,
                    log_path=stage_log_dir / f'{run_name}.log',
                    algo=algo, speed=speed, start_step=n_pre,
                    target_step=n_pre + n_ft)
                if row is None:
                    continue
                ft_rows = [
                    old for old in ft_rows
                    if not (old.get('algo') == algo
                            and abs(float(old.get('ft_speed', -1)) - speed)
                            < 1e-9)]
                ft_rows.append(row)
        _write_json(finetune_cache, ft_rows)
    if not args.skip_ft:
        ft_root = out_root / 'ft'
        ft_root.mkdir(parents=True, exist_ok=True)
        for speed in speeds:
            tag = _speed_tag(speed)
            if 'sqrl' in algos and sqrl_pre is not None:
                run_name = f'sqrl_ft_{tag}'
                print(f'[ms] === {run_name} ===', flush=True)
                ft_dir = ft_root / run_name
                log_path = stage_log_dir / f'{run_name}.log'
                row = None
                if args.skip_ft_done:
                    row = _completed_ft_row(
                        ft_dir=ft_dir, log_path=log_path, algo='sqrl',
                        speed=speed, start_step=n_pre,
                        target_step=n_pre + n_ft)
                if row is not None:
                    print(f'[ms] reuse completed {run_name} '
                          f'checkpoint={row["checkpoint"]}', flush=True)
                else:
                # Finetune must NOT use curriculum (fixed target speed).
                    row = _run_sqrl_ft(
                        config=args.config, pre_ckpt=sqrl_pre,
                        ft_dir=ft_dir, log_path=log_path,
                        n_pre=n_pre, n_ft=n_ft, ft_speed=speed,
                        wandb=args.wandb, run_name=run_name, bounce=bounce)
                # Ensure curriculum off is reflected if train mutated shared cfg
                # inside helper — helpers load fresh configs each call.
                ft_rows = [
                    old for old in ft_rows
                    if not (old.get('algo') == 'sqrl'
                            and abs(float(old.get('ft_speed', -1)) - speed)
                            < 1e-9)]
                ft_rows.append(row)
                _write_json(finetune_cache, ft_rows)
                print(f'[ms] {run_name} falls={row["falls_total_end"]} '
                      f'vel={row["final_forward_vel"]}', flush=True)
            if 'sac' in algos and sac_pre is not None:
                run_name = f'sac_ft_{tag}'
                print(f'[ms] === {run_name} ===', flush=True)
                ft_dir = ft_root / run_name
                log_path = stage_log_dir / f'{run_name}.log'
                row = None
                if args.skip_ft_done:
                    row = _completed_ft_row(
                        ft_dir=ft_dir, log_path=log_path, algo='sac',
                        speed=speed, start_step=n_pre,
                        target_step=n_pre + n_ft)
                if row is not None:
                    print(f'[ms] reuse completed {run_name} '
                          f'checkpoint={row["checkpoint"]}', flush=True)
                else:
                    row = _run_sac_ft(
                        config=args.config, pre_ckpt=sac_pre,
                        ft_dir=ft_dir, log_path=log_path,
                        n_pre=n_pre, n_ft=n_ft, ft_speed=speed,
                        wandb=args.wandb, run_name=run_name, bounce=bounce)
                ft_rows = [
                    old for old in ft_rows
                    if not (old.get('algo') == 'sac'
                            and abs(float(old.get('ft_speed', -1)) - speed)
                            < 1e-9)]
                ft_rows.append(row)
                _write_json(finetune_cache, ft_rows)
                print(f'[ms] {run_name} falls={row["falls_total_end"]} '
                      f'vel={row["final_forward_vel"]}', flush=True)

    # Patch ft helpers: they don't disable curriculum. Fix by wrapping
    # apply — already load_app_config defaults curriculum=False, OK.

    table = []
    for speed in speeds:
        tag = _speed_tag(speed)
        h_sq = next((r for r in heldout_rows
                     if r.get('algo') == 'sqrl'
                     and abs(r['move_speed'] - speed) < 1e-9), None)
        h_sa = next((r for r in heldout_rows
                     if r.get('algo') == 'sac'
                     and abs(r['move_speed'] - speed) < 1e-9), None)
        f_sq = next((r for r in ft_rows
                     if r.get('algo') == 'sqrl'
                     and abs(r['ft_speed'] - speed) < 1e-9), None)
        f_sa = next((r for r in ft_rows
                     if r.get('algo') == 'sac'
                     and abs(r['ft_speed'] - speed) < 1e-9), None)
        table.append({
            'speed': speed,
            'pre_heldout_sqrl_falls': None if h_sq is None else h_sq['falls'],
            'pre_heldout_sac_falls': None if h_sa is None else h_sa['falls'],
            'pre_heldout_sqrl_vel': (
                None if h_sq is None else h_sq['mean_forward_vel']),
            'pre_heldout_sac_vel': (
                None if h_sa is None else h_sa['mean_forward_vel']),
            'ft_sqrl_falls': (
                None if f_sq is None else f_sq['falls_total_end']),
            'ft_sac_falls': (
                None if f_sa is None else f_sa['falls_total_end']),
            'ft_sqrl_vel': (
                None if f_sq is None else f_sq['final_forward_vel']),
            'ft_sac_vel': (
                None if f_sa is None else f_sa['final_forward_vel']),
            'ft_sqrl_converged': (
                None if f_sq is None else f_sq['converged_step']),
            'ft_sac_converged': (
                None if f_sa is None else f_sa['converged_step']),
        })

    summary = {
        'protocol': 'multispeed_cmd_obs_sqrl_vs_sac',
        'cmd_speed_min': float(args.cmd_min),
        'cmd_speed_max': float(args.cmd_max),
        'pretrain_steps': n_pre,
        'finetune_steps': n_ft,
        'heldout_steps': n_hold,
        'probe_speeds': speeds,
        'sqrl_pre': str(sqrl_pre) if sqrl_pre else None,
        'sac_pre': str(sac_pre) if sac_pre else None,
        'heldout': heldout_rows,
        'finetune': ft_rows,
        'table': table,
        'wandb_requested': bool(args.wandb),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, summary)
    print(f'[ms] summary={out}', flush=True)
    print('[ms] TABLE', json.dumps(table, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
