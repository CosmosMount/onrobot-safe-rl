"""Rolling experiment summaries over valid policy transitions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RollingTrainingSummary:
    window: int
    action_dim: int
    _steps: deque = field(init=False)
    _episodes: deque = field(init=False)
    total_policy_steps: int = 0
    total_falls: int = 0
    total_recoveries: int = 0
    total_timeouts: int = 0

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError('rolling summary window must be positive')
        self._steps = deque(maxlen=self.window)
        self._episodes = deque(maxlen=self.window)

    def record_step(self, *, action: np.ndarray, info: dict,
                    timing: dict[str, float],
                    update_info: dict | None) -> None:
        if not info.get('policy_step', True):
            return
        action = np.asarray(action, dtype=np.float32)
        self.total_policy_steps += 1
        if info.get('terminated'):
            self.total_falls += 1
        if info.get('is_recovering'):
            self.total_recoveries += 1
        if info.get('truncated') or info.get('standup_timed_out'):
            self.total_timeouts += 1
        update = update_info or {}
        self._steps.append({
            'forward_velocity': float(info.get(
                'forward_velocity', info.get('x_velocity', 0.0))),
            'world_x': float(info.get('world_x', 0.0)),
            'upright': float(info.get('upright_gate', 1.0)),
            'action_frequency_hz': float(
                info.get('action_frequency_hz', np.nan)),
            'control_hold_overrun_ms': float(
                info.get('control_hold_overrun_ms', 0.0)),
            'action': action.copy(),
            'critic_loss': _finite_or_nan(update.get('critic_loss')),
            'q': _finite_or_nan(update.get('q')),
            'actor_loss': _finite_or_nan(update.get('actor_loss')),
            'entropy': _finite_or_nan(update.get('entropy')),
            'temperature': _finite_or_nan(update.get('temperature')),
            'step_ms': float(timing.get('timing/step_ms', np.nan)),
            'update_ms': float(timing.get('timing/update_ms', np.nan)),
            'loop_ms': float(timing.get('timing/loop_ms', np.nan)),
            'effective_hz': float(timing.get('timing/effective_hz', np.nan)),
        })

    def record_episode(self, episode_return: float, episode_length: int) -> None:
        self._episodes.append((float(episode_return), int(episode_length)))

    def metrics(self, replay_size: int) -> dict[str, float]:
        if not self._steps:
            return {}
        steps = list(self._steps)
        actions = np.stack([step['action'] for step in steps])
        metrics = {
            'rolling/window_steps': float(len(steps)),
            'rolling/total_policy_steps': float(self.total_policy_steps),
            'rolling/replay_size': float(replay_size),
            'rolling/forward_velocity_mean': _nanmean(
                [step['forward_velocity'] for step in steps]),
            'rolling/world_x_delta': float(
                steps[-1]['world_x'] - steps[0]['world_x']),
            'rolling/upright_ratio': _nanmean(
                [step['upright'] for step in steps]),
            'rolling/action_frequency_hz_mean': _nanmean(
                [step['action_frequency_hz'] for step in steps]),
            'rolling/control_hold_overrun_ms_mean': _nanmean(
                [step['control_hold_overrun_ms'] for step in steps]),
            'rolling/falls_total': float(self.total_falls),
            'rolling/recoveries_total': float(self.total_recoveries),
            'rolling/timeouts_total': float(self.total_timeouts),
            'rolling/action_mean': float(np.mean(actions)),
            'rolling/action_std': float(np.std(actions)),
            'rolling/action_saturation_rate': float(
                np.mean(np.abs(actions) >= 0.99)),
        }
        for name in ('critic_loss', 'q', 'actor_loss', 'entropy',
                     'temperature', 'step_ms', 'update_ms', 'loop_ms',
                     'effective_hz'):
            metrics[f'rolling/{name}_mean'] = _nanmean(
                [step[name] for step in steps])
        for joint in range(self.action_dim):
            metrics[f'rolling/action_mean_joint_{joint}'] = float(
                np.mean(actions[:, joint]))
            metrics[f'rolling/action_std_joint_{joint}'] = float(
                np.std(actions[:, joint]))
            metrics[f'rolling/action_saturation_joint_{joint}'] = float(
                np.mean(np.abs(actions[:, joint]) >= 0.99))
        if self._episodes:
            metrics['rolling/episode_return_mean'] = _nanmean(
                [episode[0] for episode in self._episodes])
            metrics['rolling/episode_length_mean'] = _nanmean(
                [episode[1] for episode in self._episodes])
        return metrics


def _finite_or_nan(value) -> float:
    if value is None:
        return float('nan')
    result = float(value) if hasattr(value, 'item') else float(value)
    return result if np.isfinite(result) else float('nan')


def _nanmean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float('nan')
