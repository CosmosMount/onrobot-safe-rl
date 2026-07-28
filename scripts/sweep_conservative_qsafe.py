#!/usr/bin/env python3
"""Run reproducible offline conservative-Q_safe alpha sweeps.

Each alpha gets its own output directory.  The source SAC actor and episode
artifacts are read-only, so comparisons differ only in the safety objective.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _alphas(value: str) -> list[float]:
    result = [float(item) for item in value.split(',') if item.strip()]
    if not result or any(item < 0.0 for item in result):
        raise argparse.ArgumentTypeError('alphas must be non-negative')
    if 0.0 not in result:
        raise argparse.ArgumentTypeError('alpha sweep must include 0 baseline')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--alphas', type=_alphas, default=_alphas('0,0.01,0.03,0.1'))
    parser.add_argument('--retrain-steps', type=int, default=5000)
    parser.add_argument('--output-root',
                        default='saved/conservative_qsafe_sweep')
    parser.add_argument('--held-out-seeds', default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        'checkpoint': args.checkpoint,
        'dataset_dir': args.dataset_dir,
        'alphas': args.alphas,
        'runs': [],
    }
    for alpha in args.alphas:
        label = f'alpha_{alpha:g}'.replace('.', 'p')
        output = root / label
        command = [
            sys.executable, '-m', 'train',
            '--mode', 'safety_retrain',
            '--config', args.config,
            '--checkpoint', args.checkpoint,
            '--dataset-dir', args.dataset_dir,
            '--retrain-steps', str(args.retrain_steps),
            '--save-dir', str(output),
            '--safety-conservative-weight', str(alpha),
        ]
        if args.held_out_seeds:
            command.extend(['--held-out-seeds', args.held_out_seeds])
        print(f'[conservative-sweep] alpha={alpha:g} output={output}',
              flush=True)
        completed = subprocess.run(command, check=False)
        manifest['runs'].append({
            'alpha': alpha,
            'output': str(output),
            'returncode': completed.returncode,
        })
        if completed.returncode:
            (root / 'manifest.json').write_text(
                json.dumps(manifest, indent=2), encoding='utf-8')
            return completed.returncode
    (root / 'manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
