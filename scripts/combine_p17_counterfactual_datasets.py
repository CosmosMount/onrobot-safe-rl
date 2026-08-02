#!/usr/bin/env python3
"""Combine counterfactual NPZ datasets while preserving state boundaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fields = (
        "observations", "actions", "failures", "failure_steps",
        "max_tilts", "min_heights",
        "state_ids", "candidate_kinds", "nominal_risks",
        "safety_contexts", "observation_histories")
    combined = {field: [] for field in fields}
    state_offset = 0
    for path in args.inputs:
        data = np.load(path)
        for field in fields:
            values = data[field]
            if field == "state_ids":
                values = values + state_offset
            combined[field].append(values)
        state_offset += len(np.unique(data["state_ids"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{
            field: np.concatenate(parts)
            for field, parts in combined.items()
        })
    print({
        "inputs": len(args.inputs),
        "states": state_offset,
        "rows": len(np.concatenate(combined["failures"])),
        "output": str(output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
