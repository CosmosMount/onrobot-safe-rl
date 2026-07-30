#!/usr/bin/env python3
"""Create the P16 0.30 m/s fall comparison from saved manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_group(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text())
    episodes_path = path / "episodes.json"
    episodes = (
        json.loads(episodes_path.read_text())
        if episodes_path.exists() else [])
    valid = [episode for episode in episodes
             if int(episode.get("policy_length", 0)) > 0]
    manifest["average_return"] = (
        mean(float(e["return"]) for e in valid) if valid else None)
    manifest["average_episode_length"] = (
        mean(float(e["policy_length"]) for e in valid) if valid else None)
    manifest["average_forward_velocity"] = (
        mean(float(e["forward_velocity_mean"]) for e in valid)
        if valid else None)
    manifest["average_tracking_error"] = (
        mean(float(e["tracking_error_mean"]) for e in valid)
        if valid else None)
    return manifest


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--sac", type=Path, required=True)
    parser.add_argument("--sqrl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = {
        "Q_safe logging source": json.loads(
            args.source_summary.read_text()),
        "Pure SAC": load_group(args.sac),
        "SAC + frozen Q_safe": load_group(args.sqrl),
    }
    sac = groups["Pure SAC"]
    sqrl = groups["SAC + frozen Q_safe"]
    fall_delta = int(sqrl["falls"]) - int(sac["falls"])
    rate_delta = (
        float(sqrl["falls_per_1000_policy_steps"])
        - float(sac["falls_per_1000_policy_steps"]))
    reduction = (
        (int(sac["falls"]) - int(sqrl["falls"])) / int(sac["falls"])
        if int(sac["falls"]) else None)

    lines = [
        "# P16: 0.30 m/s fall comparison",
        "",
        "| Group | Policy steps | Falls | Falls/1k | Episode fall rate | "
        "Avg return | Avg length | Avg velocity | Tracking error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, group in groups.items():
        lines.append(
            f"| {name} | {fmt(group.get('policy_steps'), 0)} "
            f"| {fmt(group.get('falls'), 0)} "
            f"| {fmt(group.get('falls_per_1000_policy_steps'))} "
            f"| {fmt(group.get('episode_fall_rate'))} "
            f"| {fmt(group.get('average_return'))} "
            f"| {fmt(group.get('average_episode_length'))} "
            f"| {fmt(group.get('average_forward_velocity'))} "
            f"| {fmt(group.get('average_tracking_error'))} |")

    replacements = int(sqrl.get("policy_replacements", 0))
    active_steps = int(sqrl.get("policy_safety_active_steps", 0))
    lines.extend([
        "",
        "## Primary paired result",
        "",
        f"- Fall difference (SQRL - SAC): {fall_delta:+d}",
        f"- Falls/1000 difference: {rate_delta:+.3f}",
        "- Fall reduction: "
        + ("undefined" if reduction is None else f"{100 * reduction:.1f}%"),
        f"- Replacements: {replacements}",
        f"- Replacement rate: "
        f"{100 * replacements / max(active_steps, 1):.2f}%",
        f"- No-safe rate: "
        f"{100 * sqrl.get('policy_no_safe_rate', 0):.2f}%",
        f"- False-negative falls (H=32): "
        f"{int(sqrl.get('false_negative_falls_h32', 0))}",
        f"- Replacement failure rate H=8/16/32: "
        + "/".join(
            f"{100 * int(sqrl.get('replacement_failures', {}).get(str(h), 0)) / max(int(sqrl.get('replacement_evaluated', {}).get(str(h), 0)), 1):.2f}%"
            for h in (8, 16, 32)),
        "",
        "The source run used the earlier runtime-loop budget; its normalized "
        "fall rate is diagnostic. The primary result is Pure SAC versus "
        "SAC + frozen Q_safe, both with exactly 15,000 policy transitions.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    args.output.with_suffix(".json").write_text(
        json.dumps(groups, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
