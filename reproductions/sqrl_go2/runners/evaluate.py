"""Summarize target JSONL logs without changing the frozen experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict[str, float | int | str | None]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty metrics log: {path}")
    last = rows[-1]
    steps = int(last["step"])
    falls = int(last["falls"])
    masked = [row for row in rows if "mask/accepted" in row]
    episodes = [row for row in rows if "episode/return" in row]
    return {
        "branch": str(last.get("branch", last.get("phase", "unknown"))),
        "steps": steps,
        "falls": falls,
        "falls_per_1000_steps": 1000.0 * falls / max(steps, 1),
        "mean_episode_return": (
            sum(float(row["episode/return"]) for row in episodes) / len(episodes)
            if episodes else None),
        "mean_episode_velocity_tracking_error": (
            sum(float(row["episode/velocity_tracking_error"]) for row in episodes)
            / len(episodes) if episodes else None),
        "mask_intervention_rate": (
            sum(float(row["mask/intervened"]) for row in masked) / len(masked)
            if masked else 0.0),
        "mask_acceptance_rate": (
            sum(float(row["mask/accepted"]) for row in masked) / len(masked)
            if masked else 0.0),
        "no_safe_candidate_rate": (
            sum(float(row["mask/no_safe_candidate"]) for row in masked) / len(masked)
            if masked else 0.0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps([summarize(path) for path in args.metrics], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
