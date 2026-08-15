"""Create or verify the immutable formal-v1 executable lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..formal_protocol import DEFAULT_OUTPUT_ROOT, verify_lock, write_lock


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock", type=Path,
        default=DEFAULT_OUTPUT_ROOT / "formal_protocol_lock.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    result = verify_lock(args.lock) if args.verify else write_lock(args.lock)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
