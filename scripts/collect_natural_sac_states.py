#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.natural_sac_states import collect_natural_sac_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--actor-seed", type=int, required=True)
    parser.add_argument("--training-step", type=int, required=True)
    parser.add_argument("--source-seed", type=int, required=True)
    parser.add_argument("--exposure", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path(
        "config/go2_50hz_sqrl_paper_sac_pretrain.yaml"))
    parser.add_argument("--model", type=Path, default=Path(
        "/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect_natural_sac_states(
        actor_checkpoint=args.actor,
        actor_seed=args.actor_seed,
        training_step=args.training_step,
        source_seed=args.source_seed,
        exposure_steps=args.exposure,
        config_path=args.config,
        model_path=args.model,
        output=args.output,
    ), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
