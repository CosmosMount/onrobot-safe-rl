#!/usr/bin/env python3
"""Combine counterfactual NPZ datasets while preserving state boundaries."""

from __future__ import annotations

import argparse
import os

import numpy as np

from safety_data.paths import (
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)
from scripts.collect_native_grouped_qsafe import (
    _prepare_staged_outputs,
    _publish_staged_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = assert_safe_evidence_output(args.output)
    if output.suffix != ".npz":
        parser.error("combined dataset output must use .npz")
    if os.path.lexists(os.fspath(output)):
        raise FileExistsError(f"refusing to overwrite output: {output}")
    fields = (
        "observations", "actions", "failures", "failure_steps",
        "max_tilts", "min_heights",
        "state_ids", "candidate_kinds", "nominal_risks",
        "safety_contexts", "observation_histories")
    combined = {field: [] for field in fields}
    state_offset = 0
    for path in args.inputs:
        source = require_v3_audit_consumed_or_safe_input(path)
        data = np.load(source)
        for field in fields:
            values = data[field]
            if field == "state_ids":
                values = values + state_offset
            combined[field].append(values)
        state_offset += len(np.unique(data["state_ids"]))
    staged = _prepare_staged_outputs((output,))
    staging = staged[0][0]
    try:
        np.savez_compressed(
            staging,
            **{
                field: np.concatenate(parts)
                for field, parts in combined.items()
            })
        _publish_staged_outputs(staged)
    finally:
        staging.unlink(missing_ok=True)
    print({
        "inputs": len(args.inputs),
        "states": state_offset,
        "rows": len(np.concatenate(combined["failures"])),
        "output": str(output),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
