"""Guarded entry point for the post-success same-state diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-result", type=Path, required=True)
    args = parser.parse_args(argv)
    result = json.loads(args.gate_result.read_text(encoding="utf-8"))
    if not bool(result.get("sqrl_full_reproduction_gate_passed", False)):
        raise SystemExit(
            "same-state Q_safe audit is diagnostic-only and is locked until "
            "the SQRL-full fall/performance/mask reproduction gate passes")
    raise SystemExit(
        "gate passed; connect the frozen checkpoint to the existing same-state "
        "R8/R16 audit in a separately reviewed follow-up")


if __name__ == "__main__":
    main()
