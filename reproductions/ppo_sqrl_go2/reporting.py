"""Stage summaries, fixed flags, CSV tables, and human-readable reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .io import write_json_no_clobber
from .protocol import AUDIT_SEEDS, COTRAIN_SEEDS, POSITIVE_AUDIT_SEEDS, TARGET_SEEDS
from .stability import cotrain_stability
from .statistics import (
    risk_enrichment, state_cluster_bootstrap_difference, target_decision,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _svg_lines(path: Path, *, title: str, x_label: str, y_label: str,
               series: list[tuple[str, list[float], list[float]]]) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 90, 30, 60, 80
    xs = [value for _, x, _ in series for value in x]
    ys = [value for _, _, y in series for value in y]
    if not xs or not ys:
        raise ValueError(f"chart {title} has no data")
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmax = xmin + 1.0
    if ymin == ymax:
        ymax = ymin + 1.0
    colors = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c",
              "#0891b2", "#4f46e5", "#be123c", "#15803d", "#a16207")
    px = lambda value: left + (value - xmin) / (xmax - xmin) * (width-left-right)
    py = lambda value: top + (ymax - value) / (ymax - ymin) * (height-top-bottom)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-size="22">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle">{x_label}</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle">{y_label}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        xvalue = xmin + fraction * (xmax-xmin)
        yvalue = ymin + fraction * (ymax-ymin)
        parts.extend([
            f'<text x="{px(xvalue):.1f}" y="{height-bottom+22}" text-anchor="middle" font-size="11">{xvalue:.3g}</text>',
            f'<text x="{left-8}" y="{py(yvalue)+4:.1f}" text-anchor="end" font-size="11">{yvalue:.3g}</text>',
        ])
    for index, (name, xvalues, yvalues) in enumerate(series):
        color = colors[index % len(colors)]
        points = " ".join(
            f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xvalues, yvalues, strict=True))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x = left + (index % 4) * 220
        legend_y = height - 48 + (index // 4) * 15
        parts.append(
            f'<text x="{legend_x}" y="{legend_y}" fill="{color}" font-size="11">{name}</text>')
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(parts) + "\n")


def write_campaign_charts(campaign_root: str | Path,
                          output: str | Path) -> list[str]:
    root = Path(campaign_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    produced = []
    cotrain_series = []
    for seed in COTRAIN_SEEDS:
        rows = _read_jsonl(root / "cotrain_v1" / f"seed_{seed}" / "metrics.jsonl")
        cotrain_series.append((
            f"seed {seed}", [row["task_transitions"] for row in rows],
            [row["mask/candidate_safe_fraction"] for row in rows]))
    path = output / "cotrain_candidate_safe_fraction.svg"
    _svg_lines(path, title="Co-training candidate safe fraction",
               x_label="task transitions", y_label="safe fraction",
               series=cotrain_series)
    produced.append(path.name)
    for field, filename, title, ylabel in (
        ("reward/task_step_mean", "cotrain_reward.svg", "Co-training task reward", "mean step reward"),
        ("forward_velocity/task_mean", "cotrain_velocity.svg", "Co-training actual velocity", "m/s"),
        ("safety_buffer_retained_falls", "cotrain_buffer_falls.svg", "Recent safety-buffer fall supervision", "retained falls"),
    ):
        series = []
        for seed in COTRAIN_SEEDS:
            rows = _read_jsonl(root / "cotrain_v1" / f"seed_{seed}" / "metrics.jsonl")
            series.append((f"seed {seed}",
                           [row["task_transitions"] for row in rows],
                           [row[field] for row in rows]))
        path = output / filename
        _svg_lines(path, title=title, x_label="task transitions",
                   y_label=ylabel, series=series)
        produced.append(path.name)
    if (root / "target_v1" / "summary.json").exists():
        falls_series = []
        dual_series = []
        for seed in TARGET_SEEDS:
            for branch in ("ppo_transfer", "ppo_safe"):
                rows = _read_jsonl(
                    root / "target_v1" / f"seed_{seed}" / branch / "metrics.jsonl")
                name = f"seed {seed} {branch}"
                falls_series.append((name, [row["transitions"] for row in rows],
                                     [row["falls"] for row in rows]))
                if branch == "ppo_safe":
                    dual_series.append((f"seed {seed}",
                                        [row["transitions"] for row in rows],
                                        [row["nu"] for row in rows]))
        path = output / "target_cumulative_falls.svg"
        _svg_lines(path, title="Target cumulative falls", x_label="transitions",
                   y_label="cumulative falls", series=falls_series)
        produced.append(path.name)
        path = output / "target_dual.svg"
        _svg_lines(path, title="PPO-safe projected dual", x_label="transitions",
                   y_label="nu", series=dual_series)
        produced.append(path.name)
        for field, filename, title, ylabel in (
            ("reward/step_mean", "target_reward.svg", "Target reward", "mean step reward"),
            ("forward_velocity/mean", "target_velocity.svg", "Target actual velocity", "m/s"),
            ("tracking_error/mean", "target_tracking_error.svg", "Target tracking error", "absolute m/s"),
        ):
            series = []
            for seed in TARGET_SEEDS:
                for branch in ("ppo_transfer", "ppo_safe"):
                    rows = _read_jsonl(
                        root / "target_v1" / f"seed_{seed}" / branch / "metrics.jsonl")
                    series.append((f"seed {seed} {branch}",
                                   [row["transitions"] for row in rows],
                                   [row[field] for row in rows]))
            path = output / filename
            _svg_lines(path, title=title, x_label="transitions",
                       y_label=ylabel, series=series)
            produced.append(path.name)
    return produced


def summarize_mechanism(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root)
    reports = {}
    all_fall = []
    all_action = []
    all_state = []
    for seed in AUDIT_SEEDS:
        report = json.loads((root / f"seed_{seed}" / "manifest.json").read_text())
        with np.load(root / f"seed_{seed}" / "audit_arrays.npz", allow_pickle=False) as data:
            fall = data["fall"].astype(bool)
            action = np.broadcast_to(data["action_rejected"][:, :, None], fall.shape)
            state = np.broadcast_to(data["state_rejected"][:, :, None], fall.shape)
            all_fall.append(fall)
            all_action.append(action)
            all_state.append(state)
        reports[seed] = report
    fall = np.concatenate(all_fall)
    action_summary = risk_enrichment(fall, np.concatenate(all_action))
    state_summary = risk_enrichment(fall, np.concatenate(all_state))
    positive = sum(
        float(reports[seed]["action_conditioned"]["difference"] or 0.0) > 0
        for seed in POSITIVE_AUDIT_SEEDS)
    supported = bool(
        positive >= 2
        and float(action_summary["difference"] or 0.0) > 0
        and float(action_summary["risk_enrichment"] or 0.0)
        > float(state_summary["risk_enrichment"] or 0.0))
    result = {
        "schema_version": "ppo_sqrl_go2.mechanism_summary.v1",
        "sqrl_rejection_mechanism_supported": supported,
        "positive_reference_seeds_with_positive_difference": positive,
        "action_conditioned": action_summary,
        "state_only": state_summary,
        "action_difference_state_cluster_ci95": state_cluster_bootstrap_difference(
            np.concatenate(all_fall), np.concatenate(all_action),
            seed=20260814, replicates=100_000),
        "state_only_difference_state_cluster_ci95": state_cluster_bootstrap_difference(
            np.concatenate(all_fall), np.concatenate(all_state),
            seed=20260814, replicates=100_000),
        "seeds": {str(seed): reports[seed] for seed in AUDIT_SEEDS},
        "blocks_cotrain": False,
    }
    write_json_no_clobber(Path(output), result)
    return result


def summarize_cotrain(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root)
    rows = [json.loads((root / f"seed_{seed}" / "manifest.json").read_text())
            for seed in COTRAIN_SEEDS]
    decision = cotrain_stability(rows)
    result = {
        "schema_version": "ppo_sqrl_go2.cotrain_summary.v1",
        **decision, "seeds": rows,
    }
    write_json_no_clobber(Path(output), result)
    return result


def summarize_target(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root)
    rows = []
    for seed in TARGET_SEEDS:
        transfer = json.loads((root / f"seed_{seed}" / "ppo_transfer" /
                               "manifest.json").read_text())
        safe = json.loads((root / f"seed_{seed}" / "ppo_safe" /
                           "manifest.json").read_text())
        if transfer["source_actor_sha256"] != safe["source_actor_sha256"] or (
                transfer["source_safety_sha256"] != safe["source_safety_sha256"]):
            raise ValueError(f"target seed {seed} branch lineage differs")
        if (int(transfer["transitions"]) != 10_000_000
                or int(safe["transitions"]) != 10_000_000
                or not transfer["all_numerics_finite"]
                or not safe["all_numerics_finite"]):
            raise ValueError(f"target seed {seed} data integrity failed")
        rows.append({
            "seed": seed,
            "ppo_transfer_falls": transfer["falls"],
            "ppo_safe_falls": safe["falls"],
            "source_actor_sha256": transfer["source_actor_sha256"],
            "source_safety_sha256": transfer["source_safety_sha256"],
        })
    result = {
        "schema_version": "ppo_sqrl_go2.target_summary.v1",
        **target_decision(rows), "seeds": rows,
    }
    write_json_no_clobber(Path(output), result)
    return result


def write_final_report(
    *, mechanism: dict[str, Any], cotrain: dict[str, Any],
    target: dict[str, Any] | None, output: str | Path,
) -> dict[str, Any]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    target_observed = bool(
        target and target.get("ppo_sqrl_target_benefit_observed", False))
    flags = {
        "sqrl_rejection_mechanism_supported": bool(
            mechanism["sqrl_rejection_mechanism_supported"]),
        "ppo_sqrl_cotrain_stable": bool(cotrain["ppo_sqrl_cotrain_stable"]),
        "ppo_sqrl_target_benefit_observed": target_observed,
        "ppo_sqrl_worth_formalizing": bool(
            cotrain["ppo_sqrl_cotrain_stable"] and target_observed),
    }
    result = {
        "schema_version": "ppo_sqrl_go2.final_report.v1",
        "flags": flags, "mechanism": mechanism,
        "cotrain": cotrain, "target": target,
        "target_not_run_reason": (
            "cotrain_stability_gate_failed" if target is None else None),
    }
    write_json_no_clobber(root / "final_results.json", result)
    if target is not None:
        with (root / "target_seed_table.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(target["seeds"][0]))
            writer.writeheader()
            writer.writerows(target["seeds"])
    lines = ["# PPO-SQRL Go2 three-stage validation", "", "## Flags", ""]
    lines.extend(f"- `{name}={str(value).lower()}`" for name, value in flags.items())
    lines.extend(["", "## Target paired result", ""])
    if target is None:
        lines.append("Target round was not run because co-training was not stable.")
    else:
        lines.extend([
            f"- Positive seeds: {target['positive_seeds']}/6",
            f"- PPO-transfer total falls: {target['transfer_total_falls']}",
            f"- PPO-safe total falls: {target['safe_total_falls']}",
            f"- Pooled relative reduction: {target['pooled_relative_reduction']}",
            f"- Mean paired difference: {target['bootstrap']['mean']}",
            f"- 95% CI: {target['bootstrap']['ci95']}",
            f"- One-sided 95% LCB: {target['bootstrap']['lcb95']}",
        ])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
