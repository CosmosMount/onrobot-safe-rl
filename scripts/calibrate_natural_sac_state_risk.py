#!/usr/bin/env python3
"""Freeze SAC-only calibration for the natural-PPO Q_safe trigger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.natural_sac_calibration import calibrate_natural_sac_state_risk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    report = calibrate_natural_sac_state_risk(
        model_path=args.model,
        calibration_root=args.calibration_root,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
