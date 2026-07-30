"""Pure protocol checks for the P15 common-actor SAC/SQRL experiment."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class P15GateThresholds:
    min_natural_auroc: float = 0.80
    max_ece: float = 0.10
    max_brier: float = 0.15
    min_pair_accuracy: float = 0.65
    max_selected_false_safe_rate: float = 0.05
    min_selector_coverage: float = 0.30
    min_replacement_rate: float = 0.15
    min_replacement_failure_contribution: float = 0.0
    max_fallback_reduction_fraction: float = 0.50


def _passes(value: object, comparison) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric) and comparison(numeric))


def evaluate_p15_gate(
        natural_metrics: dict[str, object],
        control_metrics: dict[str, object],
        thresholds: P15GateThresholds | None = None,
) -> dict[str, object]:
    """Apply every per-speed P15 activation requirement."""
    limits = thresholds or P15GateThresholds()
    checks = {
        'natural_auroc': _passes(
            natural_metrics.get('Q_safe_AUROC'),
            lambda value: value >= limits.min_natural_auroc),
        'ece': _passes(
            natural_metrics.get('Q_safe_calibration_ece'),
            lambda value: value <= limits.max_ece),
        'brier': _passes(
            natural_metrics.get('Q_safe_brier'),
            lambda value: value <= limits.max_brier),
        'pair_accuracy': _passes(
            control_metrics.get(
                'control_pairwise_risk_ranking_accuracy'),
            lambda value: value >= limits.min_pair_accuracy),
        'selected_false_safe': _passes(
            control_metrics.get('control_selected_false_safe_rate'),
            lambda value: value
            <= limits.max_selected_false_safe_rate),
        'selector_coverage': _passes(
            control_metrics.get('control_coverage'),
            lambda value: value >= limits.min_selector_coverage),
        'replacement_rate': _passes(
            control_metrics.get('control_replacement_rate'),
            lambda value: value >= limits.min_replacement_rate),
        'replacement_failure_contribution': _passes(
            control_metrics.get(
                'control_replacement_failure_contribution'),
            lambda value: value
            > limits.min_replacement_failure_contribution),
        'fallback_reduction_fraction': _passes(
            control_metrics.get('control_fallback_reduction_fraction'),
            lambda value: value
            <= limits.max_fallback_reduction_fraction),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        'p15_gate_passed': not failed,
        'p15_gate_failed_checks': failed,
        'p15_gate_checks': checks,
        'p15_gate_thresholds': asdict(limits),
        'natural_metrics': dict(natural_metrics),
        'control_metrics': dict(control_metrics),
    }


def split_safety_items_by_speed_episode(
        items: Sequence[dict[str, object]],
        speed_bins: Sequence[float],
        *,
        train_fraction: float = 0.70,
        calibration_fraction: float = 0.15,
        seed: int = 0,
) -> tuple[list[dict[str, object]], list[dict[str, object]],
           list[dict[str, object]], dict[str, object]]:
    """Make leakage-free 70/15/15 splits independently in each speed bin."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError('train_fraction must be in (0, 1)')
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError('calibration_fraction must be in (0, 1)')
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError('split fractions must sum below one')
    bins = np.asarray(tuple(float(value) for value in speed_bins))
    if not len(bins):
        raise ValueError('speed_bins cannot be empty')
    rng = np.random.default_rng(seed)
    split_episodes: dict[str, dict[str, list[int]]] = {}
    assignments: dict[tuple[int, int], str] = {}
    for bin_id, speed in enumerate(bins):
        episodes = sorted({
            int(item.get('episode_ids', 0))
            for item in items
            if int(np.argmin(np.abs(
                bins - float(item.get('command_speeds', 0.0)))))
            == bin_id
        })
        if len(episodes) < 3:
            raise ValueError(
                f'speed {speed:.2f} has only {len(episodes)} episodes; '
                'at least three are required')
        shuffled = [int(value) for value in rng.permutation(episodes)]
        train_count = max(1, int(round(len(shuffled) * train_fraction)))
        calibration_count = max(
            1, int(round(len(shuffled) * calibration_fraction)))
        if train_count + calibration_count >= len(shuffled):
            train_count = max(
                1, len(shuffled) - calibration_count - 1)
        groups = {
            'train': shuffled[:train_count],
            'calibration': shuffled[
                train_count:train_count + calibration_count],
            'validation': shuffled[train_count + calibration_count:],
        }
        if not groups['validation']:
            raise ValueError(f'speed {speed:.2f} validation split is empty')
        split_episodes[f'{speed:.2f}'] = {
            name: sorted(values) for name, values in groups.items()}
        for name, values in groups.items():
            assignments.update({
                (bin_id, episode_id): name for episode_id in values})

    split_items = {'train': [], 'calibration': [], 'validation': []}
    for item in items:
        bin_id = int(np.argmin(np.abs(
            bins - float(item.get('command_speeds', 0.0)))))
        episode_id = int(item.get('episode_ids', 0))
        split_items[assignments[(bin_id, episode_id)]].append(item)
    payload = repr(sorted(
        (speed, name, tuple(values))
        for speed, groups in split_episodes.items()
        for name, values in groups.items()))
    manifest: dict[str, object] = {
        'speed_episode_splits': split_episodes,
        'train_items': len(split_items['train']),
        'calibration_items': len(split_items['calibration']),
        'validation_items': len(split_items['validation']),
        'fingerprint': hashlib.sha256(payload.encode('utf-8')).hexdigest(),
    }
    return (
        split_items['train'],
        split_items['calibration'],
        split_items['validation'],
        manifest,
    )
