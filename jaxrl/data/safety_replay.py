"""Multi-buffer replay for auxiliary safety-critic training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from common.transition import COST_KEYS, Transition


def _copy_item(transition: Transition,
               behavior_noise_std: float = 0.0) -> dict[str, object]:
    item = transition.safety_replay_dict()
    return {
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
    }


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
                 n_step: int = 8, seed: int = 0):
        if failure_history < 0:
            raise ValueError('failure_history must be non-negative')
        if n_step <= 0:
            raise ValueError('safety n_step must be positive')
        self.recent = SafetyBuffer(recent_capacity)
        self.failure = SafetyBuffer(failure_capacity)
        self.boundary = SafetyBuffer(boundary_capacity)
        self.recovery = SafetyBuffer(recovery_capacity)
        self.all = SafetyBuffer(all_capacity)
        self.failure_history = int(failure_history)
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
        item = _copy_item(transition, behavior_noise_std)
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
            for history_item in self._history:
                history_item['future_failure_labels'] = 1.0
            item['future_failure_labels'] = 1.0
            self.failure.extend([*self._history, item])

        if transition.done:
            self._history.clear()
            self._nstep_history.clear()
        else:
            self._history.append(item)
            if self.n_step > 1:
                self._nstep_history.append(item)

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
        return batch

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
            'n_step': self.n_step,
            'history': list(self._history),
            'nstep_history': list(self._nstep_history),
            'rng_state': self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state['failure_history']) != self.failure_history:
            raise ValueError(
                'Safety failure history mismatch: '
                f'snapshot={state["failure_history"]} '
                f'current={self.failure_history}')
        snapshot_n_step = int(state.get('n_step', self.n_step))
        if snapshot_n_step != self.n_step:
            raise ValueError(
                f'Safety n_step mismatch: snapshot={snapshot_n_step} '
                f'current={self.n_step}')
        def migrate(item):
            item.setdefault(
                'n_step_next_observations',
                np.asarray(item['next_observations'], dtype=np.float32).copy())
            item.setdefault('n_step_unsafe_labels',
                            float(item['unsafe_labels']))
            item.setdefault('n_step_masks', float(not item['dones']))
            item.setdefault('n_step_steps', np.int32(1))
            item.setdefault('behavior_noise_std', 0.0)
            return item
        for name in ('recent', 'failure', 'boundary', 'recovery', 'all'):
            for item in state[name]['items']:
                migrate(item)
        for item in state.get('history', []):
            migrate(item)
        for name in ('recent', 'failure', 'boundary', 'recovery', 'all'):
            getattr(self, name).load_state_dict(state[name])
        self._history.clear()
        self._history.extend(state['history'])
        self._nstep_history.clear()
        self._nstep_history.extend(
            migrate(item) for item in state.get('nstep_history', []))
        self._rng.bit_generator.state = state['rng_state']
