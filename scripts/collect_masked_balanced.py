#!/usr/bin/env python3
"""Legacy masked balanced collection for Stage-2 shielding (withdrawn).

Exits immediately: active pipeline is Q_safe-only ``safety_collect``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from collector.legacy_env import build_legacy_env
from jaxrl.env.env import prepare_env
from learner.learner import run_safety_collection
from learner.safety_dataset import load_manifest
from train.config import load_app_config
from train.env import UnstableResetError


def _stabilize_robot(robot_cfg, train_cfg, *, attempts: int = 6) -> bool:
    """Run standup/recovery until the env accepts a stable policy start."""
    env = prepare_env(
        build_legacy_env(robot_cfg, train_cfg, train_cfg.seed),
        rescale_actions=False, seed=train_cfg.seed)
    try:
        for attempt in range(attempts):
            try:
                env.reset(standup=True, with_recovery=True, grace_period=True)
                print(f'[collect] stabilized attempt={attempt}', flush=True)
                return True
            except UnstableResetError:
                print(f'[collect] stabilize retry={attempt}', flush=True)
                time.sleep(2.0)
        return False
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--checkpoint',
        default='saved/checkpoints_step5/training_snapshot_000000015327.pkl')
    parser.add_argument(
        '--dataset-dir',
        default='saved/safety_datasets/masked_balanced_v1')
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--target-failures', type=int, default=24)
    parser.add_argument('--target-successes', type=int, default=24)
    parser.add_argument('--max-episodes', type=int, default=80)
    parser.add_argument('--seed-start', type=int, default=1100)
    parser.add_argument(
        '--noise-levels', default='0.35,0.40,0.45,0.50',
        help='Comma-separated action noise std values for mixed collection.')
    parser.add_argument(
        '--success-noise-levels', default='0.00,0.05,0.10,0.15',
        help='Noise levels used once the failure quota is filled.')
    parser.add_argument(
        '--epsilons', default='0.10,0.15,0.20,0.30',
        help='Comma-separated safety-mask epsilon values.')
    parser.add_argument(
        '--success-epsilons', default='0.30,0.30,0.20,0.30',
        help='Epsilons used once the failure quota is filled.')
    return parser.parse_args()


def _counts(dataset_dir: Path) -> tuple[int, int]:
    rows = load_manifest(dataset_dir)
    failures = sum(1 for row in rows if row.get('outcome') == 'failure')
    successes = sum(1 for row in rows if row.get('outcome') == 'success')
    return failures, successes


def _used_seeds(dataset_dir: Path) -> set[int]:
    return {
        int(row['rollout_seed'])
        for row in load_manifest(dataset_dir)
        if 'rollout_seed' in row
    }


def _rewrite_manifest(dataset_dir: Path, rows: list[dict]) -> None:
    path = dataset_dir / 'manifest.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')


def _drop_latest_if_quota_exceeded(dataset_dir: Path,
                                   *,
                                   target_failures: int,
                                   target_successes: int) -> str | None:
    """Remove the newest artifact if it exceeds a filled class quota."""
    rows = load_manifest(dataset_dir)
    if not rows:
        return None
    latest = rows[-1]
    outcome = latest.get('outcome')
    failures = sum(1 for row in rows if row.get('outcome') == 'failure')
    successes = sum(1 for row in rows if row.get('outcome') == 'success')
    drop = (
        (outcome == 'failure' and failures > target_failures)
        or (outcome == 'success' and successes > target_successes))
    if not drop:
        return None
    artifact = dataset_dir / str(latest['path'])
    if artifact.exists():
        artifact.unlink()
    _rewrite_manifest(dataset_dir, rows[:-1])
    return outcome


def main() -> int:
    # Stage-2 shield withdrawn; keep file for historical reference only.
    print(
        'Masked collection is withdrawn. Use Q_safe-only '
        'safety_collect (no --safety-mask).',
        file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
