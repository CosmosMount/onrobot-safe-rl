"""Non-invasive Q_safe rollout evaluation and dependency-free artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class SafetyEvalRecord:
    episode: int
    step: int
    q_safe: float
    unsafe: bool
    boundary: bool
    intervention: bool
    termination_reason: int
    reward: float
    mask_rejected_fraction: float = 0.0
    no_safe_candidate: bool = False
    selected_q_safe: float = float('nan')
    selected_action_delta: float = 0.0
    fallback_previous: bool = False
    fallback_min_risk: bool = False
    future_failure: bool = False
    time_to_failure: int = -1


def label_future_failures(records: list[SafetyEvalRecord],
                          horizon: int) -> None:
    """Label the H policy steps preceding each actual unsafe termination."""
    by_episode: dict[int, list[SafetyEvalRecord]] = {}
    for record in records:
        by_episode.setdefault(record.episode, []).append(record)
    for episode_records in by_episode.values():
        failure_indices = [
            i for i, record in enumerate(episode_records) if record.unsafe
        ]
        for failure_index in failure_indices:
            first = max(0, failure_index - horizon)
            for index in range(first, failure_index + 1):
                distance = failure_index - index
                record = episode_records[index]
                if (not record.future_failure
                        or distance < record.time_to_failure):
                    record.future_failure = True
                    record.time_to_failure = distance


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float('nan')
    comparisons = (
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean())
    return float(comparisons)


def analyze_records(records: list[SafetyEvalRecord], horizon: int,
                    min_auc: float = 0.70,
                    min_warning_delta: float = 0.05) -> dict[str, object]:
    label_future_failures(records, horizon)
    scores = np.asarray([r.q_safe for r in records], dtype=np.float64)
    future = np.asarray([r.future_failure for r in records], dtype=bool)
    normal = np.asarray([
        not r.boundary and not r.unsafe for r in records
    ], dtype=bool)
    boundary = np.asarray([
        r.boundary and not r.unsafe for r in records
    ], dtype=bool)
    failure = np.asarray([r.unsafe for r in records], dtype=bool)
    interventions = np.asarray([r.intervention for r in records], dtype=bool)
    masked = np.asarray(
        [r.mask_rejected_fraction for r in records], dtype=np.float64)
    no_safe = np.asarray(
        [r.no_safe_candidate for r in records], dtype=bool)
    selected_risk = np.asarray(
        [r.selected_q_safe for r in records], dtype=np.float64)
    selected_delta = np.asarray(
        [r.selected_action_delta for r in records], dtype=np.float64)
    episode_ids = sorted({r.episode for r in records})
    episode_returns = [
        float(sum(r.reward for r in records if r.episode == episode))
        for episode in episode_ids
    ]
    episode_lengths = [
        int(sum(r.episode == episode for r in records))
        for episode in episode_ids
    ]

    def mean(mask):
        return float(scores[mask].mean()) if np.any(mask) else float('nan')

    time_curve = []
    for distance in range(horizon, -1, -1):
        mask = np.asarray([
            r.future_failure and r.time_to_failure == distance
            for r in records
        ])
        if np.any(mask):
            time_curve.append({
                'time_to_failure': distance,
                'mean_q_safe': float(scores[mask].mean()),
                'count': int(mask.sum()),
            })
    normal_mean = mean(normal)
    warning_mask = future & ~failure
    warning_mean = mean(warning_mask)
    auc = _auc(future, scores)
    warning_delta = (
        warning_mean - normal_mean
        if np.isfinite(warning_mean) and np.isfinite(normal_mean)
        else float('nan'))
    enough_classes = bool(np.any(future) and np.any(~future))
    passes = bool(
        enough_classes and np.isfinite(auc) and auc >= min_auc
        and np.isfinite(warning_delta) and warning_delta >= min_warning_delta)
    return {
        'num_steps': len(records),
        'num_episodes': len({r.episode for r in records}),
        'failures': int(failure.sum()),
        'fall_rate_per_episode': (
            float(failure.sum() / len(episode_ids)) if episode_ids else 0.0),
        'near_failures': int(boundary.sum()),
        'near_failure_rate_per_step': float(boundary.mean()),
        'interventions': int(interventions.sum()),
        'intervention_rate_per_step': float(interventions.mean()),
        'average_return': float(np.mean(episode_returns)),
        'average_episode_length': float(np.mean(episode_lengths)),
        'future_failure_positive_steps': int(future.sum()),
        'q_safe_auroc': auc,
        'normal_q_safe_mean': normal_mean,
        'boundary_q_safe_mean': mean(boundary),
        'failure_q_safe_mean': mean(failure),
        'pre_failure_q_safe_mean': warning_mean,
        'pre_failure_vs_normal_delta': warning_delta,
        'intervention_true_positive': int(np.sum(interventions & future)),
        'intervention_false_positive': int(np.sum(interventions & ~future)),
        'failure_without_intervention': int(np.sum(failure & ~interventions)),
        'mask_rate': float(masked.mean()),
        'mask_step_rate': float(np.mean(masked > 0.0)),
        'no_safe_candidate_rate': float(no_safe.mean()),
        'selected_q_safe_mean': (
            float(np.nanmean(selected_risk))
            if np.any(np.isfinite(selected_risk)) else float('nan')),
        'selected_action_delta_mean': float(selected_delta.mean()),
        'fallback_previous_rate': float(np.mean(
            [r.fallback_previous for r in records])),
        'fallback_min_risk_rate': float(np.mean(
            [r.fallback_min_risk for r in records])),
        'time_to_failure_curve': time_curve,
        'gate': {
            'minimum_auroc': min_auc,
            'minimum_warning_delta': min_warning_delta,
            'has_positive_and_negative_samples': enough_classes,
            'ready_for_shield': passes,
            'decision': (
                'PASS: Q_safe may proceed to shield experiments'
                if passes else
                'BLOCK: improve labels/data/Q_safe before action masking'),
        },
    }


def _write_svg(records: list[SafetyEvalRecord], report: dict[str, object],
               path: Path) -> None:
    width, height = 1000, 660
    colors = {'normal': '#4c78a8', 'boundary': '#f2cf5b',
              'failure': '#e45756'}
    groups = {
        'normal': [r.q_safe for r in records
                   if not r.boundary and not r.unsafe],
        'boundary': [r.q_safe for r in records
                     if r.boundary and not r.unsafe],
        'failure': [r.q_safe for r in records if r.unsafe],
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;fill:#222}.axis{stroke:#555}'
        '.grid{stroke:#ddd;stroke-width:1}</style>',
        '<text x="40" y="35" font-size="22">Non-invasive Q_safe evaluation</text>',
    ]
    # Panel 1: risk versus time to failure.
    x0, y0, pw, ph = 60, 80, 410, 230
    parts += [
        f'<text x="{x0}" y="{y0-15}" font-size="16">Q_safe before failure</text>',
        f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>',
        f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}"/>',
    ]
    curve = report['time_to_failure_curve']
    if curve:
        maximum_h = max(1, max(p['time_to_failure'] for p in curve))
        points = []
        for point in curve:
            x = x0 + pw * (1 - point['time_to_failure'] / maximum_h)
            y = y0 + ph * (1 - np.clip(point['mean_q_safe'], 0, 1))
            points.append(f'{x:.1f},{y:.1f}')
        parts.append(
            f'<polyline fill="none" stroke="#54a24b" stroke-width="3" '
            f'points="{" ".join(points)}"/>')
    # Panel 2: category means.
    bx, by, bw, bh = 550, 80, 390, 230
    parts += [
        f'<text x="{bx}" y="{by-15}" font-size="16">Q_safe distributions (mean)</text>',
        f'<line class="axis" x1="{bx}" y1="{by+bh}" x2="{bx+bw}" y2="{by+bh}"/>',
    ]
    for index, (name, values) in enumerate(groups.items()):
        value = float(np.mean(values)) if values else 0.0
        x = bx + 35 + index * 120
        bar_h = bh * np.clip(value, 0, 1)
        parts += [
            f'<rect x="{x}" y="{by+bh-bar_h:.1f}" width="65" '
            f'height="{bar_h:.1f}" fill="{colors[name]}"/>',
            f'<text x="{x}" y="{by+bh+22}" font-size="13">{name}</text>',
            f'<text x="{x}" y="{by+bh-bar_h-7:.1f}" font-size="12">'
            f'{value:.3f}</text>',
        ]
    # Panel 3: intervention/failure summary and gate.
    gate = report['gate']
    parts += [
        '<text x="60" y="375" font-size="16">Intervention vs actual failure</text>',
        f'<text x="75" y="410">true positive: '
        f'{report["intervention_true_positive"]}</text>',
        f'<text x="75" y="440">false positive: '
        f'{report["intervention_false_positive"]}</text>',
        f'<text x="75" y="470">failure without intervention: '
        f'{report["failure_without_intervention"]}</text>',
        '<text x="550" y="375" font-size="16">Shield readiness gate</text>',
        f'<text x="565" y="410">AUROC: {report["q_safe_auroc"]}</text>',
        f'<text x="565" y="440">warning delta: '
        f'{report["pre_failure_vs_normal_delta"]}</text>',
        f'<text x="565" y="480" font-size="15">{gate["decision"]}</text>',
        '</svg>',
    ]
    path.write_text('\n'.join(parts), encoding='utf-8')


def write_evaluation_artifacts(records: list[SafetyEvalRecord],
                               report: dict[str, object],
                               output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / 'safety_rollout.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=asdict(records[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    report_path = root / 'safety_evaluation.json'
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    report_path.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False),
        encoding='utf-8')
    figure_path = root / 'safety_evaluation.svg'
    _write_svg(records, report, figure_path)
    return {'csv': csv_path, 'report': report_path, 'figure': figure_path}
