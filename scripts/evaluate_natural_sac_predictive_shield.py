#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from safety_data.natural_sac_predictive_shield import evaluate_predictive_plan_source

parser = argparse.ArgumentParser()
parser.add_argument("--source-data", required=True)
parser.add_argument("--source-manifest", required=True)
parser.add_argument("--plan", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--limit", type=int)
parser.add_argument("--device", default="cpu")
parser.add_argument("--lookahead-steps", type=int, default=3)
args = parser.parse_args()
print(json.dumps(evaluate_predictive_plan_source(
    source_data=args.source_data, source_manifest=args.source_manifest,
    plan_path=args.plan, model_path=args.model, output=args.output,
    limit=args.limit, device=args.device,
    lookahead_steps=args.lookahead_steps), sort_keys=True, indent=2))
