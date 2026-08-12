#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from safety_data.natural_sac_recovery import evaluate_selector_recovery_source

parser = argparse.ArgumentParser()
parser.add_argument("--source-data", required=True)
parser.add_argument("--source-manifest", required=True)
parser.add_argument("--branch-plan", required=True)
parser.add_argument("--mature-checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument(
    "--candidate-set", choices=(
        "fixed_nonpolicy", "full_k9_development", "mature_short_development",
        "ppo_short_development", "attitude_feedback_development"),
    default="fixed_nonpolicy")
parser.add_argument("--ppo-checkpoint")
args = parser.parse_args()
print(json.dumps(evaluate_selector_recovery_source(
    source_data=args.source_data, source_manifest=args.source_manifest,
    branch_plan=args.branch_plan, mature_checkpoint=args.mature_checkpoint,
    output=args.output, candidate_set=args.candidate_set,
    ppo_checkpoint=args.ppo_checkpoint), sort_keys=True, indent=2))
