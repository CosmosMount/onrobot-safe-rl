#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safety_data.natural_ppo_archive import validate_and_match_archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_and_match_archive(args.archive, args.output),
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
