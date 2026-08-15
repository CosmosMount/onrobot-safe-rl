"""Audit the frozen three-seed development and one-seed target protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PRETRAIN_SEEDS = (0, 1, 2)
TARGET_BRANCHES = ("sac_transfer", "sqrl_mask", "sqrl_full")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _finite(rows: list[dict[str, Any]]) -> bool:
    return all(
        math.isfinite(float(value))
        for row in rows for value in row.values()
        if isinstance(value, (int, float))
    )


def development_checks(pretrain: dict[str, Any],
                       target: dict[str, Any]) -> dict[str, bool]:
    """Mechanical development admission; performance metrics are records only."""
    return {
        "three_pretrain_seeds_complete": all(row["complete"] for row in pretrain.values()),
        "pretrain_tail_fall_free": all(
            row["tail_falls"] == 0 for row in pretrain.values()),
        "pretrain_safety_signal_and_updates": all(
            row["safety_updates"] > 0 and row["safety_replay_falls"] > 0
            for row in pretrain.values()),
        "pretrain_mask_non_degenerate": all(
            0 < row["mask_acceptance_rate"] < 1
            and row["no_safe_candidate_rate"] < 0.95
            for row in pretrain.values()),
        "target_three_branches_complete": all(row["complete"] for row in target.values()),
        "target_lineage_valid": all(
            row["actor_lineage"] and row["safety_lineage"]
            for row in target.values()),
        "all_metrics_finite": all(row["finite"] for row in pretrain.values())
                              and all(row["finite"] for row in target.values()),
        "sqrl_full_dual_path_exercised": target["sqrl_full"]["dual_updates"] > 0,
    }


def audit(root: Path) -> dict[str, Any]:
    pretrain: dict[str, Any] = {}
    for seed in PRETRAIN_SEEDS:
        directory = root / f"seed_{seed}" / "pretrain_030"
        manifest = _json(directory / "manifest.json")
        rows = _rows(directory / "metrics.jsonl")
        masks = [row for row in rows if "mask/accepted" in row]
        tail = rows[-5000:]
        pretrain[str(seed)] = {
            "complete": manifest.get("status") == "finished"
                        and manifest.get("completed_steps") == 25_000
                        and len(rows) == 25_000,
            "falls": int(rows[-1]["falls"]),
            "tail_falls": int(rows[-1]["falls"] - rows[-5001]["falls"]),
            "tail_mean_reward": sum(float(row["reward"]) for row in tail) / len(tail),
            "tail_mean_tracking_error": sum(
                float(row["velocity_tracking_error"]) for row in tail) / len(tail),
            "mask_acceptance_rate": sum(float(row["mask/accepted"]) for row in masks) / len(masks),
            "no_safe_candidate_rate": sum(
                float(row["mask/no_safe_candidate"]) for row in masks) / len(masks),
            "safety_updates": sum("safety/loss" in row for row in rows),
            "safety_replay_falls": max(float(row.get("safety/replay_falls", 0)) for row in rows),
            "finite": _finite(rows),
            "actor_sha256": manifest["actor_sha256"],
            "safety_sha256": manifest["safety_sha256"],
        }

    source = _json(root / "seed_0" / "pretrain_030" / "manifest.json")
    target: dict[str, Any] = {}
    for branch in TARGET_BRANCHES:
        directory = root / "seed_0" / f"target_040_{branch}"
        manifest = _json(directory / "manifest.json")
        rows = _rows(directory / "metrics.jsonl")
        masks = [row for row in rows if "mask/accepted" in row]
        tail = rows[-5000:]
        target[branch] = {
            "complete": manifest.get("status") == "finished"
                        and manifest.get("completed_steps") == 10_000
                        and len(rows) == 10_000,
            "falls": int(rows[-1]["falls"]),
            "falls_per_1000_steps": float(rows[-1]["falls_per_1000_steps"]),
            "tail_mean_reward": sum(float(row["reward"]) for row in tail) / len(tail),
            "tail_mean_tracking_error": sum(
                float(row["velocity_tracking_error"]) for row in tail) / len(tail),
            "actor_lineage": manifest["initial_actor_sha256"] == source["actor_sha256"],
            "safety_lineage": (
                branch == "sac_transfer"
                or manifest["initial_safety_sha256"] == source["safety_sha256"]),
            "mask_acceptance_rate": (
                sum(float(row["mask/accepted"]) for row in masks) / len(masks)
                if masks else None),
            "no_safe_candidate_rate": (
                sum(float(row["mask/no_safe_candidate"]) for row in masks) / len(masks)
                if masks else None),
            "finite": _finite(rows),
        }
    full_rows = _rows(root / "seed_0" / "target_040_sqrl_full" / "metrics.jsonl")
    dual_rows = [row for row in full_rows if "sqrl/nu" in row]
    target["sqrl_full"].update({
        "dual_updates": len(dual_rows),
        "nu_min": min(float(row["sqrl/nu"]) for row in dual_rows),
        "nu_max": max(float(row["sqrl/nu"]) for row in dual_rows),
        "actor_violation_min": min(float(row["sqrl/actor_violation"]) for row in dual_rows),
        "actor_violation_max": max(float(row["sqrl/actor_violation"]) for row in dual_rows),
    })

    baseline = target["sac_transfer"]["falls"]
    for branch in ("sqrl_mask", "sqrl_full"):
        target[branch]["fall_reduction_vs_sac_transfer"] = (
            (baseline - target[branch]["falls"]) / baseline)

    checks = development_checks(pretrain, target)
    return {
        "protocol": {"N_pre": 25_000, "N_target": 10_000,
                     "development_seeds": list(PRETRAIN_SEEDS),
                     "target_seed": 0},
        "pretrain": pretrain,
        "target": target,
        "checks": checks,
        "development_gate_passed": all(checks.values()),
        "formal_reproduction_claim": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("saved/reproductions/sqrl_go2"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit(args.root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["development_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
