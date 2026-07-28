"""Multi-buffer replay for auxiliary safety-critic training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from common.transition import COST_KEYS, Transition


def _copy_item(transition: Transition,
               behavior_noise_std: float = 0.0,
               failure_horizons: tuple[int, ...] = (8, 16, 32),
               ) -> dict[str, object]:
    item = transition.safety_replay_dict()
    copied = {
        'observations': np.asarray(item['observations'], dtype=np.float32).copy(),
        'actions': np.asarray(item['actions'], dtype=np.float32).copy(),
        'rewards': float(item['rewards']),
        'masks': float(item['masks']),
        'dones': bool(item['dones']),
        'next_observations': np.asarray(
            item['next_observations'], dtype=np.float32).copy(),
        'costs': np.asarray(
            [item['costs'][key] for key in COST_KEYS], dtype=np.float32),
        'unsafe_labels': float(item['unsafe_labels']),
        'near_failure_labels': float(item['near_failure_labels']),
        'future_failure_labels': float(item['unsafe_labels']),
        'n_step_next_observations': np.asarray(
            item['next_observations'], dtype=np.float32).copy(),
        'n_step_unsafe_labels': float(item['unsafe_labels']),
        'n_step_masks': float(not transition.done),
        'n_step_steps': np.int32(1),
        'behavior_noise_std': float(behavior_noise_std),
        'termination_reasons': int(item['termination_reasons']),
        'intervention_masks': bool(item['intervention_masks']),
        'policy_versions': np.int64(item.get('policy_versions', 0)),
        'episode_ids': np.int64(item.get('episode_ids', 0)),
        'command_speeds': float(item.get('command_speeds', 0.0)),
        'time_to_failure_steps': np.int32(
            0 if float(item['unsafe_labels']) >= 0.5 else -1),
    }
    for horizon in failure_horizons:
        copied[f'future_failure_h{horizon}'] = float(item['unsafe_labels'])
    return copied


class SafetyBuffer:
    """Bounded object replay with compact serialization."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError('safety replay capacity must be positive')
        self.capacity = int(capacity)
        self._items: deque[dict[str, object]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    def insert(self, item: dict[str, object]) -> None:
        self._items.append(item)

    def extend(self, items: Iterable[dict[str, object]]) -> None:
        self._items.extend(items)

    def sample(self, size: int, rng: np.random.Generator) -> list[dict[str, object]]:
        if not self._items:
            raise ValueError('cannot sample from an empty safety buffer')
        indices = rng.integers(0, len(self._items), size=size)
        items = list(self._items)
        return [items[int(index)] for index in indices]

    def state_dict(self) -> dict[str, object]:
        return {'capacity': self.capacity, 'items': list(self._items)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state['capacity']) != self.capacity:
            raise ValueError(
                'Safety replay capacity mismatch: '
                f'snapshot={state["capacity"]} current={self.capacity}')
        self._items.clear()
        self._items.extend(state['items'])


@dataclass(frozen=True)
class SafetyReplaySizes:
    recent: int
    failure: int
    boundary: int
    recovery: int
    all: int


class SafetyReplayManager:
    """Routes transitions and produces SQRL mixed batches.

    Ordinary SAC transitions remain in the existing ReplayBuffer. This manager
    is a parallel safety-labeled view and never feeds recovery actions to the
    reward critic or actor.
    """

    SOURCE_NAMES = ('recent', 'boundary', 'failure', 'all')
    SOURCE_WEIGHTS = np.asarray([0.40, 0.30, 0.20, 0.10], dtype=np.float64)

    def __init__(self, *, recent_capacity: int, failure_capacity: int,
                 boundary_capacity: int, recovery_capacity: int,
                 all_capacity: int, failure_history: int,
                 n_step: int = 8,
                 failure_horizons: tuple[int, ...] = (8, 16, 32),
                 seed: int = 0):
        if failure_history < 0:
            raise ValueError('failure_history must be non-negative')
        if n_step <= 0:
            raise ValueError('safety n_step must be positive')
        horizons = tuple(sorted({int(value) for value in failure_horizons}))
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError('safety failure horizons must be positive')
        self.recent = SafetyBuffer(recent_capacity)
        self.failure = SafetyBuffer(failure_capacity)
        self.boundary = SafetyBuffer(boundary_capacity)
        self.recovery = SafetyBuffer(recovery_capacity)
        self.all = SafetyBuffer(all_capacity)
        self.failure_history = int(failure_history)
        self.failure_horizons = horizons
        self.n_step = int(n_step)
        self._history: deque[dict[str, object]] = deque(
            maxlen=self.failure_history)
        self._rng = np.random.default_rng(seed)
        self._nstep_history: deque[dict[str, object]] = deque(
            maxlen=max(0, self.n_step - 1))

    def __len__(self) -> int:
        return len(self.all)

    @property
    def sizes(self) -> SafetyReplaySizes:
        return SafetyReplaySizes(
            recent=len(self.recent), failure=len(self.failure),
            boundary=len(self.boundary), recovery=len(self.recovery),
            all=len(self.all))

    def insert(self, transition: Transition, *, policy_step: bool = True,
               behavior_noise_std: float = 0.0) -> None:
        item = _copy_item(
            transition, behavior_noise_std, self.failure_horizons)
        if not policy_step:
            self.recovery.insert(item)
            if transition.done:
                self._history.clear()
                self._nstep_history.clear()
            return

        for previous in self._nstep_history:
            previous['n_step_next_observations'] = np.asarray(
                transition.next_observation, dtype=np.float32).copy()
            previous['n_step_unsafe_labels'] = max(
                float(previous['n_step_unsafe_labels']),
                float(transition.unsafe_label))
            previous['n_step_steps'] = np.int32(
                int(previous['n_step_steps']) + 1)
            previous['n_step_masks'] = float(not transition.done)

        self.recent.insert(item)
        self.all.insert(item)
        if transition.near_failure_label and not transition.unsafe_label:
            self.boundary.insert(item)
        if transition.unsafe_label:
            for steps_to_failure, history_item in enumerate(
                    reversed(self._history), start=1):
                history_item['future_failure_labels'] = 1.0
                previous_steps = int(
                    history_item.get('time_to_failure_steps', -1))
                if previous_steps < 0 or steps_to_failure < previous_steps:
                    history_item['time_to_failure_steps'] = np.int32(
                        steps_to_failure)
                for horizon in self.failure_horizons:
                    if steps_to_failure <= horizon:
                        history_item[f'future_failure_h{horizon}'] = 1.0
            item['future_failure_labels'] = 1.0
            item['time_to_failure_steps'] = np.int32(0)
            for horizon in self.failure_horizons:
                item[f'future_failure_h{horizon}'] = 1.0
            self.failure.extend([*self._history, item])

        if transition.done:
            self._history.clear()
            self._nstep_history.clear()
        else:
            self._history.append(item)
            if self.n_step > 1:
                self._nstep_history.append(item)

    def sample_recent(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample only from the recent FIFO buffer (SQRL D_safe)."""
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if len(self.recent) == 0:
            raise ValueError('cannot sample from empty recent safety buffer')
        items = self.recent.sample(batch_size, self._rng)
        batch = {key: np.stack([item[key] for item in items])
                 for key in items[0]}
        # Match sample_mixed schema used by SafetyCritic.update.
        batch['source_ids'] = np.zeros(batch_size, dtype=np.int8)
        batch['importance_weights'] = np.ones(
            batch_size, dtype=np.float32)
        return batch

    def _attach_prior_weights(
            self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Correct a stratified batch back to the recent-policy label prior."""
        labels = np.asarray(
            batch['future_failure_labels'], dtype=np.float32)
        recent_items = list(self.recent._items)
        natural_prior = (
            float(np.mean([
                float(item['future_failure_labels'])
                for item in recent_items]))
            if recent_items else float(np.mean(labels)))
        sampled_prior = float(np.mean(labels))
        if (natural_prior <= 0.0 or natural_prior >= 1.0
                or sampled_prior <= 0.0 or sampled_prior >= 1.0):
            weights = np.ones_like(labels, dtype=np.float32)
        else:
            weights = np.where(
                labels >= 0.5,
                natural_prior / sampled_prior,
                (1.0 - natural_prior) / (1.0 - sampled_prior))
            weights = weights / max(float(np.mean(weights)), 1e-6)
        batch['importance_weights'] = weights.astype(np.float32)
        batch['natural_positive_prior'] = np.full(
            labels.shape, natural_prior, dtype=np.float32)
        return batch

    def sample_recent_balanced(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample recent FIFO with ~50% future-failure positives when available.

        Uniform ``sample_recent`` under from-scratch training quickly collapses
        ``Q_safe`` to 0 on all-negative batches; once collapsed, the
        rare (~5%) positives in a long FIFO cannot recover the head.
        """
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if len(self.recent) == 0:
            raise ValueError('cannot sample from empty recent safety buffer')
        items = list(self.recent._items)
        pos = [it for it in items if float(it['future_failure_labels']) >= 0.5]
        neg = [it for it in items if float(it['future_failure_labels']) < 0.5]
        if not pos or not neg:
            return self.sample_recent(batch_size)
        n_pos = max(batch_size // 2, 1)
        n_neg = batch_size - n_pos
        pos_idx = self._rng.integers(0, len(pos), size=n_pos)
        neg_idx = self._rng.integers(0, len(neg), size=n_neg)
        sampled = (
            [pos[int(i)] for i in pos_idx] + [neg[int(i)] for i in neg_idx])
        order = self._rng.permutation(len(sampled))
        sampled = [sampled[int(i)] for i in order]
        batch = {key: np.stack([item[key] for item in sampled])
                 for key in sampled[0]}
        # Keep source_ids=0 (recent); label metrics use future_failure_labels.
        batch['source_ids'] = np.zeros(batch_size, dtype=np.int8)
        return self._attach_prior_weights(batch)

    def sample_mixed(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        buffers = (self.recent, self.boundary, self.failure, self.all)
        available = np.asarray([len(buffer) > 0 for buffer in buffers])
        if not np.any(available):
            raise ValueError('cannot sample from empty safety replay')

        raw = self.SOURCE_WEIGHTS * batch_size
        counts = np.floor(raw).astype(np.int32)
        remainder_order = np.argsort(-(raw - counts))
        for index in remainder_order[:batch_size - int(np.sum(counts))]:
            counts[index] += 1

        deficit = int(np.sum(counts[~available]))
        counts[~available] = 0
        if deficit:
            fallback_weights = self.SOURCE_WEIGHTS * available
            fallback_weights /= np.sum(fallback_weights)
            allocations = self._rng.multinomial(deficit, fallback_weights)
            counts += allocations.astype(np.int32)

        sampled: list[dict[str, object]] = []
        source_ids: list[int] = []
        for source_id, (buffer, count) in enumerate(zip(buffers, counts)):
            if count:
                sampled.extend(buffer.sample(int(count), self._rng))
                source_ids.extend([source_id] * int(count))
        order = self._rng.permutation(len(sampled))
        keys = sampled[0].keys()
        batch = {
            key: np.stack([sampled[int(i)][key] for i in order])
            for key in keys
        }
        batch['source_ids'] = np.asarray(source_ids, dtype=np.int8)[order]
        return self._attach_prior_weights(batch)

    def sample_recovery(self, batch_size: int) -> dict[str, np.ndarray]:
        items = self.recovery.sample(batch_size, self._rng)
        return {key: np.stack([item[key] for item in items])
                for key in items[0]}

    def state_dict(self) -> dict[str, object]:
        return {
            'recent': self.recent.state_dict(),
            'failure': self.failure.state_dict(),
            'boundary': self.boundary.state_dict(),
            'recovery': self.recovery.state_dict(),
            'all': self.all.state_dict(),
            'failure_history': self.failure_history,
            'failure_horizons': self.failure_horizons,
            'n_step': self.n_step,
            'history': list(self._history),
            'nstep_history': list(self._nstep_history),
            'rng_state': self._rng.bit_generator.state,
        }

    def _migrate_item(self, item: dict[str, object]) -> dict[str, object]:
        item.setdefault(
            'n_step_next_observations',
            np.asarray(item['next_observations'], dtype=np.float32).copy())
        item.setdefault('n_step_unsafe_labels',
                        float(item['unsafe_labels']))
        item.setdefault('n_step_masks', float(not item['dones']))
        item.setdefault('n_step_steps', np.int32(1))
        item.setdefault('behavior_noise_std', 0.0)
        item.setdefault('future_failure_labels',
                        float(item['unsafe_labels']))
        item.setdefault('policy_versions', np.int64(0))
        item.setdefault('episode_ids', np.int64(0))
        observations = np.asarray(item['observations'])
        item.setdefault(
            'command_speeds',
            float(observations[-1]) if observations.size else 0.0)
        unsafe = float(item['unsafe_labels']) >= 0.5
        legacy_future = float(item['future_failure_labels']) >= 0.5
        item.setdefault(
            'time_to_failure_steps', np.int32(0 if unsafe else -1))
        for horizon in self.failure_horizons:
            # Old snapshots have only the H=failure_history label. Preserve
            # positives at that horizon without pretending to know a shorter
            # time-to-failure.
            default = float(
                unsafe or (
                    legacy_future and horizon >= self.failure_history))
            item.setdefault(f'future_failure_h{horizon}', default)
        return item

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state['failure_history']) != self.failure_history:
            raise ValueError(
                'Safety failure history mismatch: '
                f'snapshot={state["failure_history"]} '
                f'current={self.failure_history}')
        snapshot_horizons = tuple(
            int(value) for value in state.get(
                'failure_horizons', self.failure_horizons))
        if snapshot_horizons != self.failure_horizons:
            raise ValueError(
                'Safety failure horizon mismatch: '
                f'snapshot={snapshot_horizons} '
                f'current={self.failure_horizons}')
        snapshot_n_step = int(state.get('n_step', self.n_step))
        if snapshot_n_step != self.n_step:
            raise ValueError(
                f'Safety n_step mismatch: snapshot={snapshot_n_step} '
                f'current={self.n_step}')
        for name in ('recent', 'failure', 'boundary', 'recovery', 'all'):
            for item in state[name]['items']:
                self._migrate_item(item)
        for item in state.get('history', []):
            self._migrate_item(item)
        for name in ('recent', 'failure', 'boundary', 'recovery', 'all'):
            getattr(self, name).load_state_dict(state[name])
        self._history.clear()
        self._history.extend(state['history'])
        self._nstep_history.clear()
        self._nstep_history.extend(
            self._migrate_item(item)
            for item in state.get('nstep_history', []))
        self._rng.bit_generator.state = state['rng_state']

    def extend_from_state(self, state: dict[str, object]) -> int:
        """Append buffer contents from another snapshot without live cursors.

        Used by offline dataset merges. Episode n-step / future-failure labels
        are preserved as stored; live history windows are ignored.
        """
        snapshot_n_step = int(state.get('n_step', self.n_step))
        if snapshot_n_step != self.n_step:
            raise ValueError(
                f'Safety n_step mismatch: snapshot={snapshot_n_step} '
                f'current={self.n_step}')
        added = 0
        for name in ('recent', 'failure', 'boundary', 'recovery', 'all'):
            buffer = getattr(self, name)
            for item in state[name]['items']:
                buffer.insert(dict(self._migrate_item(item)))
                if name == 'all':
                    added += 1
        return added
