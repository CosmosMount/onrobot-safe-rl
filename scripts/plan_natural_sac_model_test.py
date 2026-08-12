#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from safety_data.natural_sac_recovery import build_protected_model_test_plan

parser = argparse.ArgumentParser()
parser.add_argument("--source-directory", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--target-states", type=int, default=1200)
parser.add_argument("--device")
args = parser.parse_args()
print(json.dumps(build_protected_model_test_plan(
    source_directory=args.source_directory, frozen_model=args.model,
    output=args.output, target_states=args.target_states, device=args.device,
), sort_keys=True, indent=2))
