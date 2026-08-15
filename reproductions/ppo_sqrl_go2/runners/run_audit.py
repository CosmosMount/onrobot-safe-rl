from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..audit import collect_audit_seed
from ..protocol import verify_protocol_lock


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=Path("/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml"))
    args = parser.parse_args(argv)
    verify_protocol_lock(args.lock)
    result = collect_audit_seed(
        seed=args.seed, formal_root=args.formal_root,
        output=args.output, model_path=args.model)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
