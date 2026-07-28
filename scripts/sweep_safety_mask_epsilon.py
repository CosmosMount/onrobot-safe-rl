#!/usr/bin/env python3
"""Legacy epsilon sweep for Stage-2 shielding (withdrawn).

Exits immediately: active pipeline is Q_safe-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from collector.legacy_env import build_legacy_env
from jaxrl.env.env import prepare_env
from learner.learner import run_safety_eval
from train.config import load_app_config
from train.env import UnstableResetError


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        default='saved/checkpoints_step5/training_snapshot_000000015327.pkl')
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--rollout-seed', type=int, default=9010)
    parser.add_argument('--action-noise-std', type=float, default=0.50)
    parser.add_argument('--play-episodes', type=int, default=2)
    parser.add_argument(
        '--epsilons', default='0.10,0.15,0.20,0.25,0.30,0.40')
    parser.add_argument(
        '--output-dir',
        default='saved/safety_evaluation/epsilon_sweep_ref15327')
    parser.add_argument(
        '--include-unmasked', action='store_true', default=True)
    parser.add_argument(
        '--allow-min-risk-fallback', action='store_true',
        help='Use legacy argmin(Q_safe) fallback during the sweep.')
    return parser.parse_args()


def _stabilize(robot_cfg, train_cfg, attempts: int = 6) -> None:
    env = prepare_env(
        build_legacy_env(robot_cfg, train_cfg, train_cfg.seed),
        rescale_actions=False, seed=train_cfg.seed)
    try:
        for attempt in range(attempts):
            try:
                env.reset(standup=True, with_recovery=True, grace_period=True)
                print(f'[sweep] stabilized attempt={attempt}', flush=True)
                return
            except UnstableResetError:
                time.sleep(2.0)
        raise RuntimeError('Could not stabilize robot before epsilon sweep')
    finally:
        env.close()


def _summarize(report: dict) -> dict:
    keys = (
        'failures', 'num_episodes', 'average_episode_length', 'average_return',
        'q_safe_auroc', 'pre_failure_vs_normal_delta', 'mask_rate',
        'no_safe_candidate_rate', 'fallback_previous_rate',
        'fallback_contracted_rate', 'fallback_policy_mean_rate',
        'fallback_hold_previous_rate', 'fallback_min_risk_rate',
        'selected_q_safe_mean')
    return {key: report.get(key) for key in keys}


def main() -> int:
    # Stage-2 shield withdrawn; keep file for historical reference only.
    print(
        'Action masking/shielding is withdrawn. Use Q_safe-only '
        'safety_eval (no --safety-mask).',
        file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
