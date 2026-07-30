#!/usr/bin/env python3
"""Collect and merge P15 exact-state branches at every command speed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from learner.checkpoint import snapshot_agent_hash
from learner.counterfactual_dataset import (
    load_counterfactual_artifact,
    merge_counterfactual_artifacts,
    save_counterfactual_artifact,
)
from scripts.collect_mujoco_branches import DEFAULT_MODEL


def speed_bins(min_speed: float, max_speed: float,
               increment: float) -> list[float]:
    count = int(round((max_speed - min_speed) / increment)) + 1
    values = [
        round(min_speed + index * increment, 6)
        for index in range(count)
    ]
    if not values or abs(values[-1] - max_speed) > 1e-6:
        raise ValueError('speed range is not divisible by increment')
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cmd-min', type=float, default=0.30)
    parser.add_argument('--cmd-max', type=float, default=1.00)
    parser.add_argument('--cmd-increment', type=float, default=0.05)
    parser.add_argument('--natural-steps-per-speed', type=int, default=1600)
    parser.add_argument('--snapshots-per-speed', type=int, default=40)
    parser.add_argument('--min-episodes-per-speed', type=int, default=4)
    parser.add_argument('--episode-max-steps', type=int, default=400)
    parser.add_argument('--perturbation-count', type=int, default=8)
    parser.add_argument('--policy-sample-count', type=int, default=8)
    parser.add_argument('--perturbation-std', type=float, default=0.15)
    parser.add_argument('--contraction', type=float, default=0.90)
    parser.add_argument('--horizons', default='8,16,32')
    parser.add_argument('--settle-seconds', type=float, default=1.0)
    parser.add_argument(
        '--reuse-complete', action='store_true',
        help='Reuse per-speed artifacts after revalidating their manifest.')
    args = parser.parse_args()
    if args.natural_steps_per_speed < args.snapshots_per_speed:
        parser.error('natural steps must be >= snapshots per speed')
    if args.min_episodes_per_speed < 3:
        parser.error('at least three episodes are required for 70/15/15 split')

    checkpoint = args.checkpoint.expanduser().resolve()
    bins = speed_bins(args.cmd_min, args.cmd_max, args.cmd_increment)
    work_dir = (
        args.work_dir
        or args.output.parent / f'{args.output.stem}_parts')
    work_dir.mkdir(parents=True, exist_ok=True)
    snapshot_interval = max(
        1, args.natural_steps_per_speed // args.snapshots_per_speed)
    artifacts = []
    coverage = {}
    for speed_index, speed in enumerate(bins):
        tag = f'v{int(round(speed * 100)):03d}'
        part_path = work_dir / f'{tag}.pkl'
        artifact = None
        if args.reuse_complete and part_path.is_file():
            artifact = load_counterfactual_artifact(part_path)
        if artifact is None:
            command = [
                sys.executable,
                '-m', 'scripts.collect_mujoco_branches',
                '--config', args.config,
                '--checkpoint', str(checkpoint),
                '--model', args.model,
                '--move-speed', str(speed),
                '--output', str(part_path),
                '--seed', str(args.seed + speed_index * 1009),
                '--natural-steps', str(args.natural_steps_per_speed),
                '--max-episodes', '1000',
                '--natural-episode-max-steps',
                str(args.episode_max_steps),
                '--snapshot-interval', str(snapshot_interval),
                '--settle-seconds', str(args.settle_seconds),
                '--perturbation-count', str(args.perturbation_count),
                '--policy-sample-count', str(args.policy_sample_count),
                '--perturbation-std', str(args.perturbation_std),
                '--contraction', str(args.contraction),
                '--horizons', args.horizons,
            ]
            subprocess.run(command, check=True)
            artifact = load_counterfactual_artifact(part_path)
        metadata = dict(artifact.get('metadata') or {})
        completed = int(metadata.get('natural_steps_completed', 0))
        episodes = int(metadata.get('episodes_started', 0))
        snapshots = len(artifact.get('snapshots') or [])
        if completed < args.natural_steps_per_speed:
            raise RuntimeError(
                f'{speed:.2f} collected only {completed} natural transitions')
        if snapshots < args.snapshots_per_speed:
            raise RuntimeError(
                f'{speed:.2f} collected only {snapshots} snapshots')
        if episodes < args.min_episodes_per_speed:
            raise RuntimeError(
                f'{speed:.2f} collected only {episodes} episodes')
        coverage[f'{speed:.2f}'] = {
            'natural_transitions': completed,
            'episodes': episodes,
            'snapshots': snapshots,
            'branches': len(artifact.get('branches') or []),
            'falls': int(metadata.get('natural_failures', 0)),
            'path': str(part_path.resolve()),
        }
        artifacts.append(artifact)

    actor_hash = snapshot_agent_hash(checkpoint)
    merged = merge_counterfactual_artifacts(
        artifacts,
        metadata={
            'protocol': 'P15',
            'common_checkpoint': str(checkpoint),
            'common_actor_hash': actor_hash,
            'seed': args.seed,
            'speed_bins': bins,
            'coverage': coverage,
            'horizons': [
                int(value) for value in args.horizons.split(',')],
        })
    output = save_counterfactual_artifact(
        args.output,
        snapshots=merged['snapshots'],
        branches=merged['branches'],
        metadata=merged['metadata'])
    summary = {
        'output': str(output.resolve()),
        'common_actor_hash': actor_hash,
        'speed_bins': bins,
        'coverage': coverage,
        'snapshots': len(merged['snapshots']),
        'branches': len(merged['branches']),
    }
    summary_path = args.output.with_suffix('.json')
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
