#!/usr/bin/env python3
"""Analyze candidate-space headroom before training Q_safe(s,a)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from safety_data.action_oracle_analysis import analyze_action_oracle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--role", choices=("development", "protected"),
                        default="development")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("oracle report was already published")
    report = analyze_action_oracle(
        args.inputs, role=args.role,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
