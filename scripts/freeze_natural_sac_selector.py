#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from safety_data.natural_sac_recovery import freeze_selector_recovery

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--branch-plan", required=True)
parser.add_argument("--branch-file", action="append", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
print(json.dumps(freeze_selector_recovery(
    calibrated_model=args.model, branch_plan=args.branch_plan,
    branch_files=args.branch_file, output_model=args.output,
), sort_keys=True, indent=2))
