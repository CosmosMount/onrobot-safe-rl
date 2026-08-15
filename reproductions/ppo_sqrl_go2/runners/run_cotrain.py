from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..protocol import verify_protocol_lock
from ..training import run_cotrain


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    lock = verify_protocol_lock(args.lock)
    result = run_cotrain(
        seed=args.seed, output=args.output,
        protocol_bundle=lock["bundle_sha256"], device=args.device,
        iterations=args.iterations, resume_checkpoint=args.resume_checkpoint)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
