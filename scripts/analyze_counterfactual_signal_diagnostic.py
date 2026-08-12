#!/usr/bin/env python3
"""Analyze R4/R8/R16, four horizons, and PPO candidate direction coverage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.counterfactual_firewall import assert_development_artifact
from safety_data.counterfactual_replica_extension import verify_frozen_candidate_identity
from safety_data.counterfactual_signal_analysis import (
    candidate_direction_analysis, horizon_analysis, replica_scaling_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--diagnostic-roster", type=Path, required=True)
    parser.add_argument("--replica-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(assert_development_artifact(args.diagnostic_roster),
                 allow_pickle=False) as roster_file:
        roster = {name: roster_file[name] for name in roster_file.files}
    rows = roster["source_row"].astype(np.int64)
    with np.load(assert_development_artifact(args.development_dataset),
                 allow_pickle=False) as source:
        first_r4 = source["first_fall_step"][rows]
        source_candidate_id = source["candidate_id"][rows]
        source_action = source["critic_action"][rows]
        observation_history = source["observation_history"][rows]
    verify_frozen_candidate_identity(
        source_candidate_id, roster["candidate_id"], source_action,
        roster["critic_action"])
    with np.load(assert_development_artifact(args.replica_extension),
                 allow_pickle=False) as extension:
        if not np.array_equal(extension["state_id"], roster["state_id"]):
            raise RuntimeError("replica extension state identity changed")
        verify_frozen_candidate_identity(
            roster["candidate_id"], extension["candidate_id"],
            roster["critic_action"], extension["critic_action"])
        if not np.array_equal(extension["replica_id"],
                              np.broadcast_to(np.arange(5, 17), (400, 16, 12))):
            raise RuntimeError("replica extension is not exactly R5--R16")
        first = np.concatenate((first_r4, extension["first_fall_step"]), axis=2)
    replica = replica_scaling_analysis(
        first, roster["risk_stratum"], roster["collector_seed"])
    horizons = horizon_analysis(
        first, roster["risk_stratum"], roster["collector_seed"])
    directions = candidate_direction_analysis(
        first, roster["critic_action"], roster["candidate_distance"],
        roster["candidate_distance_bin"], observation_history,
        roster["risk_stratum"], roster["collector_seed"], roster["state_id"])
    report = {
        "schema_version": "qsafe.counterfactual_signal_diagnostic.v1",
        "states": 400, "candidates": 16, "replicas": 16,
        "protected_outcomes_read_or_generated": False,
        "safety_critic_trained": False,
        "sac_transfer_run": False,
        "replica_scaling": replica,
        "horizon_analysis": horizons,
        "candidate_direction_analysis": directions,
        "flags": {
            "r4_label_noise_likely": replica["r4_label_noise_likely"],
            "h96_credit_dilution_likely": horizons["h96_credit_dilution_likely"],
            "candidate_direction_coverage_likely": directions[
                "candidate_direction_coverage_likely"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    os.link(temporary, args.output)
    temporary.unlink()
    print(json.dumps(report["flags"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
