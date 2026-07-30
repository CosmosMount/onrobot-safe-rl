"""Sampling utilities for exact-state counterfactual Q_safe supervision."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from learner.counterfactual_dataset import CandidateBranch


def branch_severity(branch: CandidateBranch, horizon: int) -> float:
    outcome = branch.outcomes[horizon]
    if outcome.failure:
        return 1.0 + 0.25 * (
            horizon - max(outcome.time_to_failure, 1)) / max(horizon, 1)
    return 0.5 if outcome.near_failure else 0.0


def split_branch_snapshots(
        branches: Sequence[CandidateBranch],
        *,
        validation_fraction: float = 0.2,
        seed: int = 0,
) -> tuple[list[CandidateBranch], list[CandidateBranch], list[int]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError('validation_fraction must be in (0, 1)')
    snapshot_ids = sorted({int(item.snapshot_index) for item in branches})
    if len(snapshot_ids) < 2:
        raise ValueError('at least two snapshots are required')
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(snapshot_ids)
    count = min(
        len(snapshot_ids) - 1,
        max(1, int(round(len(snapshot_ids) * validation_fraction))))
    validation_ids = {int(value) for value in shuffled[:count]}
    train = [
        item for item in branches
        if int(item.snapshot_index) not in validation_ids]
    validation = [
        item for item in branches
        if int(item.snapshot_index) in validation_ids]
    return train, validation, sorted(validation_ids)


def split_branch_episodes(
        branches: Sequence[CandidateBranch],
        snapshot_episode_ids: dict[int, int],
        *,
        validation_fraction: float = 0.2,
        seed: int = 0,
) -> tuple[list[CandidateBranch], list[CandidateBranch],
           list[int], list[int]]:
    """Keep every snapshot from one natural rollout episode in one split."""
    episode_ids = sorted({
        int(snapshot_episode_ids[int(item.snapshot_index)])
        for item in branches
    })
    if len(episode_ids) < 2:
        raise ValueError('at least two episodes are required')
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(episode_ids)
    count = min(
        len(episode_ids) - 1,
        max(1, int(round(len(episode_ids) * validation_fraction))))
    validation_episodes = {int(value) for value in shuffled[:count]}
    validation_snapshots = {
        int(snapshot_id)
        for snapshot_id, episode_id in snapshot_episode_ids.items()
        if int(episode_id) in validation_episodes
    }
    train = [
        item for item in branches
        if int(item.snapshot_index) not in validation_snapshots]
    validation = [
        item for item in branches
        if int(item.snapshot_index) in validation_snapshots]
    return (
        train, validation, sorted(validation_snapshots),
        sorted(validation_episodes))


def split_branch_episodes_three_way(
        branches: Sequence[CandidateBranch],
        snapshot_episode_ids: dict[int, int],
        *,
        train_fraction: float = 0.70,
        calibration_fraction: float = 0.15,
        seed: int = 0,
) -> tuple[
        list[CandidateBranch],
        list[CandidateBranch],
        list[CandidateBranch],
        dict[str, list[int] | str]]:
    """Split complete episodes into train/calibration/validation subsets."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError('train_fraction must be in (0, 1)')
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError('calibration_fraction must be in (0, 1)')
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError('train and calibration fractions must sum below 1')
    episode_ids = sorted({
        int(snapshot_episode_ids[int(item.snapshot_index)])
        for item in branches
    })
    if len(episode_ids) < 3:
        raise ValueError('at least three episodes are required')
    rng = np.random.default_rng(seed)
    shuffled = [int(value) for value in rng.permutation(episode_ids)]
    train_count = max(1, int(round(len(shuffled) * train_fraction)))
    calibration_count = max(
        1, int(round(len(shuffled) * calibration_fraction)))
    if train_count + calibration_count >= len(shuffled):
        overflow = train_count + calibration_count - len(shuffled) + 1
        train_count = max(1, train_count - overflow)
    train_episodes = set(shuffled[:train_count])
    calibration_episodes = set(
        shuffled[train_count:train_count + calibration_count])
    validation_episodes = set(
        shuffled[train_count + calibration_count:])
    if not validation_episodes:
        raise ValueError('validation episode split is empty')

    def episode_for(item: CandidateBranch) -> int:
        return int(snapshot_episode_ids[int(item.snapshot_index)])

    train = [item for item in branches
             if episode_for(item) in train_episodes]
    calibration = [item for item in branches
                   if episode_for(item) in calibration_episodes]
    validation = [item for item in branches
                  if episode_for(item) in validation_episodes]
    split_payload = (
        f'seed={seed};train={sorted(train_episodes)};'
        f'calibration={sorted(calibration_episodes)};'
        f'validation={sorted(validation_episodes)}')
    manifest: dict[str, list[int] | str] = {
        'train_episode_ids': sorted(train_episodes),
        'calibration_episode_ids': sorted(calibration_episodes),
        'validation_episode_ids': sorted(validation_episodes),
        'train_snapshot_ids': sorted({
            int(item.snapshot_index) for item in train}),
        'calibration_snapshot_ids': sorted({
            int(item.snapshot_index) for item in calibration}),
        'validation_snapshot_ids': sorted({
            int(item.snapshot_index) for item in validation}),
        'fingerprint': hashlib.sha256(
            split_payload.encode('utf-8')).hexdigest(),
    }
    return train, calibration, validation, manifest


@dataclass
class BranchSupervisionDataset:
    branches: list[CandidateBranch]
    horizon: int
    seed: int = 0
    hard_negative_keys: set[tuple[int, int]] | None = None

    def __post_init__(self):
        if not self.branches:
            raise ValueError('branch supervision dataset is empty')
        if any(self.horizon not in item.outcomes for item in self.branches):
            raise ValueError(f'horizon {self.horizon} is absent')
        self.rng = np.random.default_rng(self.seed)
        self.observations = np.stack([
            item.observation for item in self.branches]).astype(np.float32)
        self.actions = np.stack([
            item.action for item in self.branches]).astype(np.float32)
        self.failure_labels = np.asarray([
            item.outcomes[self.horizon].failure
            for item in self.branches], dtype=np.float32)
        self.positive = np.flatnonzero(self.failure_labels >= 0.5)
        self.negative = np.flatnonzero(self.failure_labels < 0.5)
        if not len(self.positive) or not len(self.negative):
            raise ValueError(
                'branch point supervision requires failures and non-failures')
        hard_keys = self.hard_negative_keys or set()
        self.hard_negative = np.asarray([
            index for index, item in enumerate(self.branches)
            if (int(item.snapshot_index), int(item.candidate_index))
            in hard_keys
        ], dtype=np.int64)
        self._hard_negative_set = set(
            int(value) for value in self.hard_negative)
        self.ordinary_positive = np.asarray([
            index for index in self.positive
            if int(index) not in self._hard_negative_set
        ], dtype=np.int64)
        if not len(self.ordinary_positive):
            self.ordinary_positive = self.positive

        groups: dict[int, list[int]] = {}
        for index, item in enumerate(self.branches):
            groups.setdefault(int(item.snapshot_index), []).append(index)
        pairs = []
        for indices in groups.values():
            for offset, left in enumerate(indices):
                for right in indices[offset + 1:]:
                    left_severity = branch_severity(
                        self.branches[left], self.horizon)
                    right_severity = branch_severity(
                        self.branches[right], self.horizon)
                    if left_severity > right_severity:
                        pairs.append((left, right))
                    elif right_severity > left_severity:
                        pairs.append((right, left))
        if not pairs:
            raise ValueError('no comparable same-state branch pairs')
        self.pairs = np.asarray(pairs, dtype=np.int64)
        self.hard_pairs = np.asarray([
            index for index, (riskier, _) in enumerate(self.pairs)
            if int(riskier) in self._hard_negative_set
        ], dtype=np.int64)

    def sample(self, point_batch_size: int, pair_batch_size: int, *,
               hard_negative_fraction: float = 0.0,
               hard_negative_weight: float = 1.0) -> dict:
        if not 0.0 <= hard_negative_fraction <= 1.0:
            raise ValueError('hard_negative_fraction must be in [0, 1]')
        if hard_negative_weight < 1.0:
            raise ValueError('hard_negative_weight must be >= 1')
        positive_count = point_batch_size // 2
        negative_count = point_batch_size - positive_count
        hard_count = (
            min(positive_count, int(round(
                positive_count * hard_negative_fraction)))
            if len(self.hard_negative) else 0)
        ordinary_positive_count = positive_count - hard_count
        point_indices = np.concatenate([
            self.rng.choice(
                self.ordinary_positive, ordinary_positive_count,
                replace=True),
            self.rng.choice(
                self.hard_negative, hard_count, replace=True)
            if hard_count else np.empty(0, dtype=np.int64),
            self.rng.choice(
                self.negative, negative_count, replace=True),
        ])
        self.rng.shuffle(point_indices)
        hard_pair_count = (
            min(pair_batch_size, int(round(
                pair_batch_size * hard_negative_fraction)))
            if len(self.hard_pairs) else 0)
        pair_indices = np.concatenate([
            self.rng.choice(
                len(self.pairs), pair_batch_size - hard_pair_count,
                replace=True),
            self.rng.choice(
                self.hard_pairs, hard_pair_count, replace=True)
            if hard_pair_count else np.empty(0, dtype=np.int64),
        ])
        self.rng.shuffle(pair_indices)
        riskier, safer = self.pairs[pair_indices].T
        # A pair always comes from one exact simulator snapshot, so either
        # observation is equivalent. Keep the riskier one for clarity.
        return {
            'observations': self.observations[point_indices],
            'actions': self.actions[point_indices],
            'failure_labels': self.failure_labels[point_indices],
            'point_weights': np.asarray([
                hard_negative_weight
                if int(index) in self._hard_negative_set else 1.0
                for index in point_indices
            ], dtype=np.float32),
            'pair_observations': self.observations[riskier],
            'riskier_actions': self.actions[riskier],
            'safer_actions': self.actions[safer],
        }


def mine_selected_false_safe(
        branches: Sequence[CandidateBranch],
        predicted_risks,
        *,
        horizon: int,
        epsilon: float,
        support=None,
) -> tuple[set[tuple[int, int]], dict[str, float]]:
    """Mine actions A would accept as safe but branch rollout proves unsafe."""
    risks = np.asarray(predicted_risks, dtype=np.float64).reshape(-1)
    supported = (
        np.ones(len(branches), dtype=bool)
        if support is None else np.asarray(support, dtype=bool).reshape(-1))
    groups: dict[int, list[int]] = {}
    for index, item in enumerate(branches):
        groups.setdefault(int(item.snapshot_index), []).append(index)
    selected = []
    for indices in groups.values():
        nominal = next(
            index for index in indices
            if branches[index].candidate_family == 'nominal')
        eligible = [
            index for index in indices
            if supported[index] and risks[index] <= epsilon
        ]
        if nominal in eligible:
            selected.append(nominal)
        elif eligible:
            selected.append(min(eligible, key=lambda index: risks[index]))
    hard = {
        (
            int(branches[index].snapshot_index),
            int(branches[index].candidate_index),
        )
        for index in selected
        if branches[index].outcomes[horizon].failure
    }
    return hard, {
        'hard_negative_selected_count': float(len(selected)),
        'hard_negative_count': float(len(hard)),
        'hard_negative_rate': (
            float(len(hard) / len(selected)) if selected else 0.0),
        'hard_negative_snapshot_coverage': float(
            len(selected) / max(len(groups), 1)),
    }


def conformal_upper_offset(labels, scores, alpha: float) -> float:
    """Finite-sample one-sided additive risk bound."""
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must be in (0, 1)')
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not len(labels) or len(labels) != len(scores):
        raise ValueError('labels and scores must be non-empty and aligned')
    residuals = labels - scores
    rank = int(np.ceil((len(residuals) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(residuals))
    offset = float(np.partition(residuals, rank - 1)[rank - 1])
    return max(offset, 0.0)
