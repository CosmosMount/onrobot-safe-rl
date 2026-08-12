#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from safety_data.natural_sac_recovery import build_selector_branch_plan

parser = argparse.ArgumentParser()
parser.add_argument("--calibration-root", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--samples-per-band", type=int, default=300)
parser.add_argument("--device")
args = parser.parse_args()
print(json.dumps(build_selector_branch_plan(
    calibration_root=args.calibration_root, calibrated_model=args.model,
    output=args.output, samples_per_band=args.samples_per_band,
    device=args.device), sort_keys=True, indent=2))
