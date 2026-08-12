#!/usr/bin/env python3
"""Freeze an outcome-blind 400-state development diagnostic roster."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.counterfactual_diagnostic_roster import select_diagnostic_rows
from safety_data.counterfactual_firewall import assert_development_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(assert_development_artifact(args.development_dataset),
                 allow_pickle=False) as data:
        # Only preregistered metadata and frozen candidate/action fields are
        # accessed. h96_fall and first_fall_step are deliberately not read.
        rows = select_diagnostic_rows(
            data["state_id"], data["split"], data["collector_seed"],
            data["risk_stratum"])
        arrays = {
            "source_row": rows,
            "state_id": data["state_id"][rows],
            "episode_id": data["episode_id"][rows],
            "split": data["split"][rows],
            "risk_stratum": data["risk_stratum"][rows],
            "collector_seed": data["collector_seed"][rows],
            "candidate_id": data["candidate_id"][rows],
            "candidate_index": data["candidate_index"][rows],
            "candidate_distance": data["candidate_distance"][rows],
            "candidate_distance_bin": data["candidate_distance_bin"][rows],
            "action_requested": data["action_requested"][rows],
            "action_pre_projection": data["action_pre_projection"][rows],
            "critic_action": data["critic_action"][rows],
            "absolute_q_target": data["absolute_q_target"][rows],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.link(temporary, args.output)
    temporary.unlink()
    print({"states": len(rows), "output": str(args.output)})


if __name__ == "__main__":
    main()

