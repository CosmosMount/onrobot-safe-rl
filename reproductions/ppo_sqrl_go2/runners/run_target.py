from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..protocol import verify_protocol_lock
from ..training import run_target_branch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branch", choices=("ppo_transfer", "ppo_safe"), required=True)
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--pretrain-lock", type=Path, required=True)
    parser.add_argument("--target-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args(argv)
    pretrain_lock = verify_protocol_lock(args.pretrain_lock)
    target_lock = verify_protocol_lock(args.target_lock)
    result = run_target_branch(
        seed=args.seed, branch=args.branch,
        pretrain_checkpoint=args.pretrain_checkpoint,
        output=args.output,
        pretrain_protocol_bundle=pretrain_lock["bundle_sha256"],
        target_protocol_bundle=target_lock["bundle_sha256"],
        device=args.device, iterations=args.iterations,
        resume_checkpoint=args.resume_checkpoint)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
