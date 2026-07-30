#!/usr/bin/env python3
"""Run SAC/SQRL stress cells with a fresh simulator and controller each time."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from scripts.run_sqrl_transfer_scratch import (
    _heldout_fixed_speed, _stabilize)
from train.config import load_app_config
from train.main import apply_move_speed


REPO = Path(__file__).resolve().parents[1]
SIM_DIR = Path('/home/xyz/code/unitree_mujoco/simulate/build')


def _stop_exact(name: str) -> None:
    subprocess.run(
        ['pkill', '-TERM', '-x', name],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        result = subprocess.run(
            ['pgrep', '-x', name], check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode != 0:
            return
        time.sleep(0.1)
    subprocess.run(
        ['pkill', '-KILL', '-x', name],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def restart_stack(log_dir: Path) -> tuple[subprocess.Popen, subprocess.Popen]:
    _stop_exact('go2_control')
    _stop_exact('unitree_mujoco')
    socket = Path('/tmp/go2_policy.sock')
    socket.unlink(missing_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sim_log = (log_dir / 'simulator.log').open('a', encoding='utf-8')
    controller_log = (
        log_dir / 'controller.log').open('a', encoding='utf-8')
    sim_environment = dict(os.environ)
    if not sim_environment.get('DISPLAY'):
        sim_environment['DISPLAY'] = ':1'
    sim = subprocess.Popen(
        ['./unitree_mujoco', '-r', 'go2', '-s', 'scene_empty.xml',
         '-i', '1', '-n', 'lo'],
        cwd=SIM_DIR, stdout=sim_log, stderr=subprocess.STDOUT,
        start_new_session=True, env=sim_environment)
    time.sleep(5.0)
    controller = subprocess.Popen(
        [str(REPO / 'controller/build/go2_control'),
         str(REPO / 'config/go2.yaml')],
        cwd=REPO, stdout=controller_log, stderr=subprocess.STDOUT,
        start_new_session=True)
    for _ in range(100):
        if sim.poll() is not None:
            raise RuntimeError('unitree_mujoco exited during startup')
        if controller.poll() is not None:
            raise RuntimeError('go2_control exited during startup')
        if socket.exists():
            time.sleep(2.0)
            return sim, controller
        time.sleep(0.1)
    raise RuntimeError('controller policy socket did not appear')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--speed', type=float, default=0.50)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--noise', type=float, default=0.20)
    parser.add_argument('--epsilon', type=float, default=0.20)
    parser.add_argument('--seeds', default='9300,9301,9302')
    parser.add_argument('--ensemble-size', type=int, default=3)
    parser.add_argument(
        '--algos', default='sac,sqrl',
        help='Comma-separated subset of sac,sqrl.')
    parser.add_argument(
        '--sample-policy', action='store_true',
        help=('Use the SAC policy distribution directly instead of its mode '
              'plus external Gaussian noise. SQRL already samples candidates.'))
    parser.add_argument(
        '--log-qsafe', action='store_true',
        help='Load and log Q_safe during SAC cells (checkpoint must contain it).')
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    algos = {value.strip() for value in args.algos.split(',') if value.strip()}
    unknown = algos.difference({'sac', 'sqrl'})
    if unknown:
        raise SystemExit(f'unknown --algos values: {sorted(unknown)}')
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for use_sqrl in (False, True):
            if ('sqrl' if use_sqrl else 'sac') not in algos:
                continue
            cell = f'{"sqrl" if use_sqrl else "sac"}_seed{seed}'
            print(f'[isolated] restarting stack for {cell}', flush=True)
            restart_stack(args.output.parent / 'stack_logs')
            robot, cfg, droq = load_app_config(path='config/go2.yaml')
            robot = apply_move_speed(robot, args.speed)
            cfg.safety_critic_ensemble_size = args.ensemble_size
            _stabilize(robot, cfg)
            row = _heldout_fixed_speed(
                robot_cfg=robot, train_cfg=cfg, droq_cfg=dict(droq),
                checkpoint=args.checkpoint, max_steps=args.steps,
                action_noise_std=args.noise, rollout_seed=seed,
                use_sqrl=use_sqrl, epsilon=args.epsilon,
                num_candidates=cfg.sqrl_num_candidates,
                noise_mode='candidate' if use_sqrl else 'post',
                log_qsafe=bool(args.log_qsafe and not use_sqrl),
                sample_policy=bool(args.sample_policy and not use_sqrl))
            row['algo'] = 'sqrl_structured' if use_sqrl else 'sac'
            row['isolated_stack'] = True
            rows.append(row)
            args.output.write_text(
                json.dumps(rows, indent=2), encoding='utf-8')
            print(f'[isolated] completed {cell}: '
                  f'falls={row["falls"]} '
                  f'vel={row["mean_forward_vel"]:.3f} '
                  f'no_safe={row["no_safe_rate"]:.3f}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
