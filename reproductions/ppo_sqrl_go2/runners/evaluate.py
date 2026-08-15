from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..evaluation import evaluate_checkpoint
from ..protocol import verify_protocol_lock


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--pretrain-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--branch", choices=("ppo_transfer", "ppo_safe"), required=True)
    parser.add_argument("--exposure", type=int, required=True)
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrain-lock", type=Path, required=True)
    parser.add_argument("--target-lock", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args(argv)
    pretrain_lock = verify_protocol_lock(args.pretrain_lock)
    target_lock = verify_protocol_lock(args.target_lock)
    result = evaluate_checkpoint(
        actor_checkpoint=args.actor_checkpoint,
        pretrain_checkpoint=args.pretrain_checkpoint,
        seed=args.seed, branch=args.branch, exposure=args.exposure,
        masked=args.masked, output=args.output,
        pretrain_protocol_bundle=pretrain_lock["bundle_sha256"],
        target_protocol_bundle=target_lock["bundle_sha256"],
        device=args.device, episodes=args.episodes)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
