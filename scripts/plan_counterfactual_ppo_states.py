#!/usr/bin/env python3
"""Create development and protected identity plans from fresh 97-step archives."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.counterfactual_firewall import load_identity_denylist, reject_denied_identities
from safety_data.counterfactual_states import (
    assign_episode_disjoint_roster, episode_key, offset_for, save_roster,
    state_identity,
)


def _records(root: Path, seed: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted((root / "natural-falls").glob("falls-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for row in range(len(data["identity"])):
                length = int(data["trajectory_length"][row])
                if length < 96:
                    continue
                raw_identity = bytes(data["identity"][row])
                episode = episode_key(
                    seed, int(data["environment_id"][row]), int(data["episode_id"][row]))
                for stratum, low, high, namespace in (
                    ("boundary", 32, 64, b"qsafe.counterfactual.boundary.v2"),
                    ("medium", 65, 96, b"qsafe.counterfactual.medium.v2"),
                ):
                    offset = offset_for(raw_identity, low, high, namespace)
                    records.append({
                        "state_id": state_identity(episode, stratum, offset),
                        "episode_key": episode, "collector_seed": seed,
                        "risk_stratum": stratum, "offset": offset,
                        "archive_path": str(path.resolve()), "archive_row": row,
                        "trajectory_index": length - offset,
                    })
    for path in sorted((root / "natural-falls" / "normals").glob("normals-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for row in range(len(data["identity"])):
                if int(data["qualification_future_nonterminal_steps"][row]) < 96 or bool(
                        data["fall_within_96_steps"][row]):
                    continue
                episode = episode_key(
                    seed, int(data["environment_id"][row]), int(data["episode_id"][row]))
                records.append({
                    "state_id": state_identity(episode, "normal", 0),
                    "episode_key": episode, "collector_seed": seed,
                    "risk_stratum": "normal", "offset": 0,
                    "archive_path": str(path.resolve()), "archive_row": row,
                    "trajectory_index": -1,
                })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed137-root", type=Path, required=True)
    parser.add_argument("--seed138-root", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--protected-output", type=Path, required=True)
    parser.add_argument("--round-one-denylist", type=Path, required=True)
    args = parser.parse_args()
    rows = assign_episode_disjoint_roster(
        _records(args.seed137_root, 137) + _records(args.seed138_root, 138))
    reject_denied_identities(
        [str(row["state_id"]) for row in rows],
        load_identity_denylist(args.round_one_denylist))
    development = [row for row in rows if row["split"] != "protected"]
    protected = [row for row in rows if row["split"] == "protected"]
    save_roster(args.development_output, development)
    save_roster(args.protected_output, protected)
    print({"development_states": len(development), "protected_states": len(protected)})


if __name__ == "__main__":
    main()

