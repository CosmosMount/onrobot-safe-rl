"""Validate and summarize the locked ten-seed SQRL-Go2 formal experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..formal_protocol import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    BRANCHES,
    DEFAULT_OUTPUT_ROOT,
    FORMAL_SEEDS,
    NU_ACTIVE_THRESHOLD,
    PRETRAIN_STEPS,
    PROTOCOL_ID,
    TARGET_STEPS,
    verify_lock,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(rows: list[dict[str, Any]]) -> bool:
    return all(
        math.isfinite(float(value))
        for row in rows for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool))


def bootstrap_indices() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return rng.integers(
        0, len(FORMAL_SEEDS),
        size=(BOOTSTRAP_REPLICATES, len(FORMAL_SEEDS)), endpoint=False)


def paired_summary(baseline: np.ndarray, treatment: np.ndarray,
                   indices: np.ndarray | None = None) -> dict[str, Any]:
    baseline = np.asarray(baseline, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.float64)
    if baseline.shape != (len(FORMAL_SEEDS),) or treatment.shape != baseline.shape:
        raise ValueError("formal comparisons require exactly ten paired seed values")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(treatment)):
        raise ValueError("paired seed values must be finite")
    difference = baseline - treatment
    draws = bootstrap_indices() if indices is None else np.asarray(indices)
    if draws.shape != (BOOTSTRAP_REPLICATES, len(FORMAL_SEEDS)):
        raise ValueError("bootstrap indices must resample ten complete paired seeds")
    sampled = difference[draws]
    mean_draws = sampled.mean(axis=1)
    median_draws = np.median(sampled, axis=1)
    baseline_total = float(baseline.sum())
    relative = (
        float((baseline.sum() - treatment.sum()) / baseline.sum())
        if baseline_total > 0 else None)
    return {
        "paired_differences": difference.astype(int).tolist(),
        "mean_paired_reduction": float(difference.mean()),
        "median_paired_reduction": float(np.median(difference)),
        "mean_bootstrap_95_ci": np.quantile(
            mean_draws, [0.025, 0.975], method="linear").tolist(),
        "mean_one_sided_95_lcb": float(np.quantile(
            mean_draws, 0.05, method="linear")),
        "median_bootstrap_95_ci": np.quantile(
            median_draws, [0.025, 0.975], method="linear").tolist(),
        "positive_seeds": int(np.sum(difference > 0)),
        "tied_seeds": int(np.sum(difference == 0)),
        "negative_seeds": int(np.sum(difference < 0)),
        "baseline_total_falls": int(baseline.sum()),
        "treatment_total_falls": int(treatment.sum()),
        "pooled_relative_reduction": relative,
    }


def formal_flags(comparisons: dict[str, dict[str, Any]],
                 nu_active_seeds: int) -> dict[str, bool]:
    primary = comparisons["sqrl_full_vs_sac_transfer"]
    masking = comparisons["sqrl_mask_vs_sac_transfer"]
    lagrangian = comparisons["sqrl_full_vs_sqrl_mask"]
    formal = bool(
        primary["mean_paired_reduction"] > 0.0
        and primary["mean_one_sided_95_lcb"] > 0.0
        and primary["positive_seeds"] >= 8
        and primary["pooled_relative_reduction"] is not None
        and primary["pooled_relative_reduction"] >= 0.30)
    mask_supported = bool(
        masking["mean_one_sided_95_lcb"] > 0.0
        and masking["pooled_relative_reduction"] is not None
        and masking["pooled_relative_reduction"] >= 0.30)
    lagrangian_supported = bool(
        lagrangian["mean_one_sided_95_lcb"] > 0.0
        and nu_active_seeds >= 2)
    return {
        "formal_go2_sqrl_reproduced": formal,
        "sqrl_masking_effect_supported": mask_supported,
        "sqrl_lagrangian_effect_supported": lagrangian_supported,
        "successful_qsafe_positive_control_available": formal,
    }


def validate_lineage(pretrain: dict[str, Any], target: dict[str, Any],
                     *, seed: int, branch: str) -> None:
    if target["initial_actor_sha256"] != pretrain["actor_sha256"]:
        raise RuntimeError(f"actor lineage mismatch seed={seed} branch={branch}")
    if branch != "sac_transfer" and (
            target["initial_safety_sha256"] != pretrain["safety_sha256"]):
        raise RuntimeError(f"Q_safe lineage mismatch seed={seed} branch={branch}")


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return float(np.mean(values)) if values else None


def _max(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if key in row]
    return max(values) if values else None


def _validate_run(directory: Path, *, phase: str, seed: int,
                  branch: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _json(directory / "manifest.json")
    rows = _rows(directory / "metrics.jsonl")
    steps = PRETRAIN_STEPS if phase == "pretrain" else TARGET_STEPS
    expected = {
        "status": "finished", "phase": phase, "seed": seed,
        "completed_steps": steps, "protocol_id": PROTOCOL_ID,
    }
    if branch is not None:
        expected["branch"] = branch
    mismatch = {
        key: (manifest.get(key), value) for key, value in expected.items()
        if manifest.get(key) != value}
    if mismatch or len(rows) != steps or int(rows[-1].get("step", -1)) != steps:
        raise RuntimeError(f"incomplete or mismatched formal run {directory}: {mismatch}")
    if not _finite(rows):
        raise RuntimeError(f"non-finite formal metrics: {directory}")
    return manifest, rows


def _run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [row for row in rows if "episode/length" in row]
    masks = [row for row in rows if "mask/accepted" in row]
    dual = [row for row in rows if "sqrl/nu" in row]
    return {
        "falls": int(rows[-1]["falls"]),
        "falls_per_1000_steps": float(rows[-1]["falls_per_1000_steps"]),
        "mean_reward": _mean(rows, "reward"),
        "mean_tracking_error": _mean(rows, "velocity_tracking_error"),
        "mean_forward_velocity": _mean(rows, "forward_velocity"),
        "completed_episodes": len(episodes),
        "mean_completed_episode_length": _mean(episodes, "episode/length"),
        "mask_acceptance_rate": _mean(masks, "mask/accepted"),
        "no_safe_candidate_rate": _mean(masks, "mask/no_safe_candidate"),
        "mean_candidate_attempts": _mean(masks, "mask/candidate_count"),
        "mean_selected_qsafe": _mean(masks, "mask/risk"),
        "mean_candidate_qsafe_mean": _mean(masks, "safety/q_mean"),
        "mean_candidate_qsafe_p50": _mean(masks, "safety/q_p50"),
        "mean_candidate_qsafe_p90": _mean(masks, "safety/q_p90"),
        "dual_updates": len(dual),
        "nu_active_fraction": (
            float(np.mean([float(row["sqrl/nu"]) > NU_ACTIVE_THRESHOLD
                           for row in dual])) if dual else None),
        "mean_nu": _mean(dual, "sqrl/nu"),
        "max_nu": _max(dual, "sqrl/nu"),
        "mean_constraint_violation": _mean(dual, "sqrl/actor_violation"),
    }


def audit(root: Path) -> dict[str, Any]:
    lock = verify_lock(root / "formal_protocol_lock.json")
    table: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    qsafe: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        seed_root = root / f"seed_{seed}"
        pre_manifest, pre_rows = _validate_run(
            seed_root / "pretrain_030", phase="pretrain", seed=seed)
        pre_tail_falls = int(pre_rows[-1]["falls"] - pre_rows[-5001]["falls"])
        record: dict[str, Any] = {
            "seed": seed,
            "pretrain_falls": int(pre_rows[-1]["falls"]),
            "pretrain_final_5k_falls": pre_tail_falls,
            "pretrain_mean_reward": _mean(pre_rows, "reward"),
            "pretrain_mean_tracking_error": _mean(pre_rows, "velocity_tracking_error"),
            "pretrain_mean_forward_velocity": _mean(pre_rows, "forward_velocity"),
            "pretrain_safety_collection_falls": int(
                pre_rows[-1].get("safety/collection_falls", 0)),
            "pretrain_safety_replay_falls_max": _max(pre_rows, "safety/replay_falls"),
            "pretrain_qsafe_updates": sum("safety/loss" in row for row in pre_rows),
            "pretrain_mask_acceptance_rate": _mean(pre_rows, "mask/accepted"),
            "pretrain_no_safe_candidate_rate": _mean(
                pre_rows, "mask/no_safe_candidate"),
            "pretrain_mean_candidate_attempts": _mean(
                pre_rows, "mask/candidate_count"),
            "pretrain_qsafe_mean": _mean(pre_rows, "safety/q_mean"),
            "pretrain_qsafe_p50": _mean(pre_rows, "safety/q_p50"),
            "pretrain_qsafe_p90": _mean(pre_rows, "safety/q_p90"),
        }
        checkpoint = seed_root / "pretrain_030/final.pt"
        qsafe.append({
            "seed": seed, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "actor_sha256": pre_manifest["actor_sha256"],
            "safety_sha256": pre_manifest["safety_sha256"],
        })
        for branch in BRANCHES:
            manifest, rows = _validate_run(
                seed_root / f"target_040_{branch}", phase="target",
                seed=seed, branch=branch)
            validate_lineage(pre_manifest, manifest, seed=seed, branch=branch)
            summary = _run_summary(rows)
            for key, value in summary.items():
                record[f"{branch}_{key}"] = value
            curves.extend({
                "seed": seed, "branch": branch, "step": int(row["step"]),
                "cumulative_falls": int(row["falls"]),
            } for row in rows)
        table.append(record)

    indices = bootstrap_indices()
    sac = np.asarray([row["sac_transfer_falls"] for row in table])
    mask = np.asarray([row["sqrl_mask_falls"] for row in table])
    full = np.asarray([row["sqrl_full_falls"] for row in table])
    comparisons = {
        "sqrl_full_vs_sac_transfer": paired_summary(sac, full, indices),
        "sqrl_mask_vs_sac_transfer": paired_summary(sac, mask, indices),
        "sqrl_full_vs_sqrl_mask": paired_summary(mask, full, indices),
    }
    nu_active_seeds = sum(
        float(row["sqrl_full_nu_active_fraction"] or 0.0) > 0.0 for row in table)
    return {
        "schema_version": "sqrl_go2.formal_results.v1",
        "protocol_lock_bundle_sha256": lock["executable_bundle_sha256"],
        "seed_table": table,
        "comparisons": comparisons,
        "nu_active_seeds": nu_active_seeds,
        "flags": formal_flags(comparisons, nu_active_seeds),
        "qsafe_index": qsafe if formal_flags(
            comparisons, nu_active_seeds)["formal_go2_sqrl_reproduced"] else None,
        "_curves": curves,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# SQRL-Go2 formal paired-seed results", "",
        "Primary arm: SQRL-full vs SAC-transfer. Seed is the statistical unit.", "",
        "| Seed | Pretrain falls | SAC falls | Mask falls | Full falls | Mask rate | No-safe rate | Mean nu |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["seed_table"]:
        lines.append(
            f"| {row['seed']} | {row['pretrain_falls']} | "
            f"{row['sac_transfer_falls']} | {row['sqrl_mask_falls']} | "
            f"{row['sqrl_full_falls']} | "
            f"{row['sqrl_full_mask_acceptance_rate']:.3f} | "
            f"{row['sqrl_full_no_safe_candidate_rate']:.3f} | "
            f"{row['sqrl_full_mean_nu']:.6g} |")
    lines.extend(["", "## Comparisons", ""])
    for name, values in result["comparisons"].items():
        lines.extend([
            f"### {name}", "",
            f"- Mean paired reduction: {values['mean_paired_reduction']:.3f}",
            f"- Median paired reduction: {values['median_paired_reduction']:.3f}",
            f"- Mean bootstrap 95% CI: {values['mean_bootstrap_95_ci']}",
            f"- One-sided 95% LCB: {values['mean_one_sided_95_lcb']:.3f}",
            f"- Positive seeds: {values['positive_seeds']}/10",
            f"- Pooled relative reduction: {values['pooled_relative_reduction']}", "",
        ])
    lines.extend(["## Formal flags", ""] + [
        f"- `{key}={str(value).lower()}`" for key, value in result["flags"].items()])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_curves(path: Path, curves: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, branch in zip(axes, BRANCHES):
        for seed in FORMAL_SEEDS:
            rows = [row for row in curves
                    if row["branch"] == branch and row["seed"] == seed]
            axis.plot([row["step"] for row in rows],
                      [row["cumulative_falls"] for row in rows],
                      linewidth=1, alpha=0.75, label=str(seed))
        axis.set_title(branch)
        axis.set_xlabel("target fine-tuning step")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("cumulative falls")
    axes[-1].legend(title="seed", fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def publish(root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal report: {output}")
    result = audit(root)
    curves = result.pop("_curves")
    output.mkdir(parents=True)
    (output / "formal_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output / "formal_seed_table.csv", result["seed_table"])
    _write_csv(output / "formal_fall_curves.csv", curves)
    _write_report(output / "FORMAL_RESULTS.md", result)
    _plot_curves(output / "formal_cumulative_falls.png", curves)
    if result["qsafe_index"] is not None:
        (output / "successful_qsafe_index.json").write_text(
            json.dumps(result["qsafe_index"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or args.root / "report"
    result = publish(args.root, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
