#!/usr/bin/env python3
"""Create the frozen 600-state fresh Boundary option roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.short_option_roster import (
    records_from_fresh_archive, save_roster, select_fresh_boundary_roster,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed137-root", type=Path, required=True)
    parser.add_argument("--seed138-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = records_from_fresh_archive(
        args.seed137_root, collector_seed=137, rollout_seed=4137)
    rows += records_from_fresh_archive(
        args.seed138_root, collector_seed=138, rollout_seed=4138)
    selected = select_fresh_boundary_roster(rows)
    save_roster(args.output, selected)
    print(json.dumps({"states": len(selected), "seed137": 300, "seed138": 300,
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
