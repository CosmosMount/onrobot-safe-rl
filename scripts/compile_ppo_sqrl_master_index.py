#!/usr/bin/env python3
"""Compile six immutable PPO collector manifests into nested episode sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.ppo_sqrl_index import write_index_and_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.manifest) != 6:
        raise ValueError("master index requires two seeds times three stages")
    index, selection = write_index_and_selection(args.manifest, args.output)
    print(index.resolve())
    print(selection.resolve())


if __name__ == "__main__":
    main()
