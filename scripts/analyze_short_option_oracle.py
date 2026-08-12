#!/usr/bin/env python3
"""Analyze the frozen L1/L4/L8 short-option independent oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.counterfactual_firewall import assert_development_artifact
from safety_data.short_option_analysis import (
    analyze_short_option_oracle, validate_short_option_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(assert_development_artifact(args.dataset), allow_pickle=False) as data:
        validation = validate_short_option_dataset(
            {name: data[name] for name in data.files})
        report = analyze_short_option_oracle(
            h96_fall=data["h96_fall"],
            candidate_duration=data["candidate_duration"],
            collector_seed=data["collector_seed"],
            replacement_sum=data["replacement_magnitude_sum"],
            replacement_max=data["replacement_magnitude_max"],
            projection_saturation_count=data["projection_saturation_count"],
            joint_limit_saturation_count=data["joint_limit_saturation_count"],
            active_steps=data["option_active_steps_executed"],
            max_abs_roll=data["option_max_abs_roll"],
            max_abs_pitch=data["option_max_abs_pitch"],
            max_angular_velocity=data["option_max_angular_velocity"],
            min_base_height=data["option_min_base_height"],
        )
    report["dataset_validation"] = validation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "short_option_candidate_space_supported": report[
            "short_option_candidate_space_supported"],
        "one_step_action_timescale_insufficient": report[
            "one_step_action_timescale_insufficient"],
        "passing_long_families": report["passing_long_families"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
