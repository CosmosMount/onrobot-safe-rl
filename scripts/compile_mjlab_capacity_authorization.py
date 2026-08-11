#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_data.mjlab_capacity_authorization import (
    compile_capacity_authorization,
    publish_capacity_authorization,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier-report", type=Path, action="append", required=True)
    parser.add_argument("--stability-report", type=Path, required=True)
    parser.add_argument("--production-envs", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_capacity_authorization(
        args.tier_report, args.stability_report,
        production_envs=args.production_envs)
    publish_capacity_authorization(args.output, report)
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
