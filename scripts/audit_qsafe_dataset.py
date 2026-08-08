#!/usr/bin/env python3
"""Validate grouped Q_safe datasets and optionally score fixed predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from safety_data.metrics import evaluate_predictions
from safety_data.paths import assert_development_path
from safety_data.schema import GroupedBranchDataset, audit_split_disjointness


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="Development .npz split paths")
    parser.add_argument(
        "--predictions",
        help="Optional .npy [G,K] risks; valid only with one dataset")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--output", help="Optional development JSON report")
    args = parser.parse_args()

    datasets = [GroupedBranchDataset.load(path) for path in args.datasets]
    result: dict[str, Any] = {
        "datasets": [dataset.validate() for dataset in datasets],
        "split_audit": audit_split_disjointness(datasets),
    }
    if args.predictions:
        if len(datasets) != 1:
            parser.error("--predictions requires exactly one dataset")
        prediction_path = assert_development_path(args.predictions)
        prediction = np.load(prediction_path, allow_pickle=False)
        result["metrics"] = evaluate_predictions(
            datasets[0], prediction,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
    result = _json_value(result)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        output = assert_development_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
