"""Episode-disjoint splits and branch bootstrap utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Sequence

import numpy as np

from learner.counterfactual_dataset import CandidateBranch


def split_episode_roles(
        episode_ids: Sequence[int],
        *,
        validation_fraction: float = 0.2,
        temperature_fraction: float = 0.1,
        conformal_fraction: float = 0.1,
        seed: int = 0,
) -> dict[str, list[int]]:
    """Create mutually exclusive fit/temperature/conformal/validation roles."""
    unique = sorted({int(value) for value in episode_ids})
    if len(unique) < 8:
        raise ValueError('at least eight episodes are required')
    fractions = (
        validation_fraction, temperature_fraction, conformal_fraction)
    if any(value <= 0.0 for value in fractions):
        raise ValueError('held-out episode fractions must be positive')
    if sum(fractions) >= 1.0:
        raise ValueError('held-out episode fractions must sum to less than 1')
    rng = np.random.default_rng(seed)
    shuffled = [int(value) for value in rng.permutation(unique)]

    remaining = len(unique)
    counts = []
    for fraction in fractions:
        count = max(1, int(round(len(unique) * fraction)))
        counts.append(count)
        remaining -= count
    if remaining < 2:
        raise ValueError('episode split leaves fewer than two fit episodes')
    validation_count, temperature_count, conformal_count = counts
    cursor = 0
    validation = shuffled[cursor:cursor + validation_count]
    cursor += validation_count
    temperature = shuffled[cursor:cursor + temperature_count]
    cursor += temperature_count
    conformal = shuffled[cursor:cursor + conformal_count]
    cursor += conformal_count
    fit = shuffled[cursor:]
    return {
        'fit': sorted(fit),
        'temperature': sorted(temperature),
        'conformal': sorted(conformal),
        'validation': sorted(validation),
    }


def filter_branches_by_episode(
        branches: Sequence[CandidateBranch],
        snapshot_episode_ids: dict[int, int],
        episode_ids: Sequence[int],
) -> list[CandidateBranch]:
    selected = {int(value) for value in episode_ids}
    return [
        item for item in branches
        if int(snapshot_episode_ids[int(item.snapshot_index)]) in selected
    ]


def bootstrap_episode_branches(
        branches: Sequence[CandidateBranch],
        snapshot_episode_ids: dict[int, int],
        fit_episode_ids: Sequence[int],
        *,
        seed: int,
) -> tuple[list[CandidateBranch], dict[str, object]]:
    """Sample episodes with replacement and remap duplicate snapshot groups."""
    episodes = [int(value) for value in fit_episode_ids]
    if not episodes:
        raise ValueError('bootstrap needs fit episodes')
    grouped: dict[int, list[CandidateBranch]] = {
        episode_id: [] for episode_id in episodes
    }
    for item in branches:
        episode_id = int(
            snapshot_episode_ids[int(item.snapshot_index)])
        if episode_id in grouped:
            grouped[episode_id].append(item)
    missing = [
        episode_id for episode_id, items in grouped.items() if not items]
    if missing:
        raise ValueError(f'fit episodes have no branches: {missing}')

    rng = np.random.default_rng(seed)
    draws = [
        int(value) for value in rng.choice(
            episodes, size=len(episodes), replace=True)
    ]
    bootstrapped = []
    next_snapshot = 0
    for occurrence, episode_id in enumerate(draws):
        snapshot_remap: dict[int, int] = {}
        for item in grouped[episode_id]:
            original = int(item.snapshot_index)
            if original not in snapshot_remap:
                snapshot_remap[original] = next_snapshot
                next_snapshot += 1
            bootstrapped.append(replace(
                item, snapshot_index=snapshot_remap[original]))
    counts = Counter(draws)
    return bootstrapped, {
        'seed': int(seed),
        'draws': draws,
        'unique_episode_count': len(counts),
        'episode_multiplicity': {
            str(key): int(value) for key, value in sorted(counts.items())
        },
        'bootstrapped_snapshots': next_snapshot,
        'bootstrapped_branches': len(bootstrapped),
    }
