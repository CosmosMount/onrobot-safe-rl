#!/usr/bin/env python3
"""Seal the consumed round-one PPO cohort and publish its identity denylist."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal(source: Path, destination: Path, denylist: Path) -> None:
    if destination.exists() or denylist.exists():
        raise FileExistsError("seal destination or denylist already exists")
    outcome = source / "ppo-independent-200state-r8.npz"
    with np.load(outcome, allow_pickle=False) as data:
        identities = sorted(bytes(value).decode("ascii") for value in data["identity"])
    if len(identities) != 200 or len(set(identities)) != 200:
        raise RuntimeError("round-one protected cohort is not exactly 200 unique states")
    _atomic_json(denylist, {
        "schema_version": "qsafe.counterfactual_identity_denylist.v1",
        "reason": "consumed_round_one_ppo_action_ranking_protected_cohort",
        "identities": identities,
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(source, destination)
    for path in sorted(destination.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    destination.chmod(0o555)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--denylist", type=Path, required=True)
    args = parser.parse_args()
    seal(args.source, args.destination, args.denylist)


if __name__ == "__main__":
    main()

