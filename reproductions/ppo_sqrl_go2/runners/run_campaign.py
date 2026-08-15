from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..orchestrator import Campaign


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("saved/reproductions/ppo_sqrl_go2"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    result = Campaign(
        args.root, resume=args.resume, retry_failed=args.retry_failed,
        device=args.device).run(prepare_only=args.prepare_only)
    if result is not None:
        print(json.dumps(result["flags"], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
