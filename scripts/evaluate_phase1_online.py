#!/usr/bin/env python3
"""Compile a development-only Phase 1 three-arm evidence table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import yaml

from safety_data.paths import assert_development_path
from safety_data.phase1_stats import (
    CommonGateStatus,
    OnlineGateThresholds,
    RouteSpec,
    compile_phase1_evidence,
    evaluate_online_route,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Development JSON table")
    parser.add_argument("--output", required=True, help="New development JSON report")
    parser.add_argument(
        "--protocol", default="config/qsafe_evidence_protocol.yaml")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()

    input_path = assert_development_path(args.input)
    output_path = assert_development_path(args.output)
    protocol_path = assert_development_path(args.protocol)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evidence report: {output_path}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != 2:
        raise ValueError("Phase 1 compiler requires protocol version 2")

    common_payload = payload.get("common_gates")
    if not isinstance(common_payload, dict):
        raise ValueError("input requires a common_gates object")
    common = CommonGateStatus(**common_payload)
    route_payload = payload.get("routes")
    if not isinstance(route_payload, dict) or not route_payload:
        raise ValueError("input requires at least one Phase 1 route")

    online_gate = protocol["phase1"]["online_training_gate"]
    thresholds = OnlineGateThresholds(
        min_relative_fall_reduction=float(
            online_gate["min_relative_fall_reduction"]),
        min_absolute_falls_per_1000_reduction=float(
            online_gate["min_absolute_falls_per_1000_reduction"]),
        min_reduction_ci_low=float(online_gate["min_reduction_ci_low"]),
        min_treatment_vs_placebo_reduction_ci_low=float(
            online_gate[
                "min_treatment_vs_placebo_absolute_reduction_ci_low"]),
        min_return_ratio=float(online_gate["min_return_ratio"]),
        max_forward_velocity_error_increase_mps=float(
            online_gate["max_forward_velocity_error_increase_mps"]),
        max_runtime_deadline_miss_rate=float(
            online_gate["max_runtime_deadline_miss_rate"]),
        max_exact_label_swap_p_value=float(
            online_gate["exact_test"]["max_p_value"]),
    )
    expected_seeds = tuple(map(int, online_gate["confirmation"]["seeds"]))
    expected_exposure = int(
        online_gate["confirmation"]["policy_steps_per_seed"])
    reports = {}
    for route, value in route_payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"route {route!r} must be an object")
        allowed = {
            "runs", "starts_from_zero",
            "independently_finetuned_target_actor",
            "placebo_matching_verified",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"route {route!r} has unknown fields: {sorted(unknown)}")
        if not isinstance(value.get("runs"), list):
            raise ValueError(f"route {route!r} requires a runs list")
        spec = RouteSpec(
            route=route,
            expected_seeds=expected_seeds,
            expected_exposure_policy_steps=expected_exposure,
            starts_from_zero=value.get("starts_from_zero", False),
            independently_finetuned_target_actor=value.get(
                "independently_finetuned_target_actor", False),
            placebo_matching_verified=value.get(
                "placebo_matching_verified", False),
        )
        reports[route] = evaluate_online_route(
            value["runs"],
            spec,
            thresholds=thresholds,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed + len(reports),
        )
    decision = compile_phase1_evidence(common, reports)
    result = _json_safe({
        "schema_version": "qsafe.phase1.development_evidence.v1",
        "development_only": True,
        "generator_commit": _git_commit(),
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "route_reports": {
            route: report.to_dict() for route, report in reports.items()},
        "decision": decision.to_dict(),
        "phase1_pass": decision.phase1_pass,
        "phase2_authorized": False,
    })
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
