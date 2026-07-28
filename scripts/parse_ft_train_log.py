#!/usr/bin/env python3
"""Parse a finetune stage log for falls / velocity / convergence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_ROLL = re.compile(
    r'\[step (?P<step>\d+)\] rolling n=\d+ '
    r'forward_vel=(?P<vel>[-+eE0-9.]+).*?falls=(?P<falls>\d+)'
)


def parse_ft_log(text: str, *, start_step: int, ft_speed: float,
                 sustain_steps: int = 500) -> dict:
    fallen_episodes = len(re.findall(r'episode done \(fallen\)', text))
    rows: list[tuple[int, float, int]] = []
    for m in _ROLL.finditer(text):
        step = int(m.group('step'))
        if step < start_step:
            continue
        rows.append((step, float(m.group('vel')), int(m.group('falls'))))

    falls_total_end = rows[-1][2] if rows else None
    final_forward_vel = rows[-1][1] if rows else None
    threshold = 0.8 * float(ft_speed)
    converged_step = None
    if rows and sustain_steps > 0:
        # Approximate sustain using consecutive roll points (~100 step spacing).
        need = max(int(sustain_steps // 100), 1)
        streak = 0
        for step, vel, _ in rows:
            if vel >= threshold:
                streak += 1
                if streak >= need:
                    converged_step = step
                    break
            else:
                streak = 0

    return {
        'fallen_episodes': fallen_episodes,
        'falls_total_end': falls_total_end,
        'final_forward_vel': final_forward_vel,
        'vel_threshold': threshold,
        'converged_step': converged_step,
        'n_roll_points': len(rows),
        'start_step': start_step,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log_path')
    p.add_argument('--start-step', type=int, required=True)
    p.add_argument('--ft-speed', type=float, required=True)
    p.add_argument('--output', default=None)
    args = p.parse_args()
    text = Path(args.log_path).read_text(encoding='utf-8', errors='replace')
    info = parse_ft_log(text, start_step=args.start_step, ft_speed=args.ft_speed)
    info['log_path'] = args.log_path
    print(json.dumps(info, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(info, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
