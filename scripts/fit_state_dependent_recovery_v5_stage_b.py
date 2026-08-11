#!/usr/bin/env python3
"""Freeze the canonical V5 Stage-B development model artifacts."""

from __future__ import annotations

import argparse
import json

from safety_data.state_dependent_recovery_v5_stage_b_fit import (
    run_stage_b_development_fit,
)


def build_parser() -> argparse.ArgumentParser:
    # Deliberately no path, threshold, seed, epoch, or bootstrap argument.
    # Every production input and output is derived from the canonical V5
    # protocol, and this command has no Model-Test evidence surface.
    return argparse.ArgumentParser(description=__doc__)


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    result = run_stage_b_development_fit()
    print(json.dumps({
        "status": result.status,
        "selector_feasible": result.selector_feasible,
        "placebo_balanced": result.placebo_balanced,
        "frozen_artifact_sha256": dict(result.frozen_artifact_sha256),
        "failure_report": (
            None if result.failure_report is None else str(result.failure_report)
        ),
        "model_test_outcomes_read": False,
        "model_test_consumed": False,
    }, sort_keys=True, indent=2))
    return 0 if result.failure_report is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
