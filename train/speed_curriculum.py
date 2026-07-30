"""Performance-gated command-speed curriculum.

The curriculum advances only after the reward policy is demonstrably usable at
the current frontier speed.  This keeps SQRL comparisons from being confounded
by asking the policy to solve a new locomotion task and a new safety constraint
at the same time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurriculumUpdate:
    promoted: bool
    upper_speed: float
    frontier_episodes: int
    mean_velocity_ratio: float
    mean_episode_length: float
    fall_rate: float
    exploration_multiplier: float
    phase: str
    frontier_complete: bool
    coverage_complete: bool


class PerformanceSpeedCurriculum:
    """Advance a discrete speed frontier from episode outcomes."""

    def __init__(
            self, *,
            min_speed: float,
            max_speed: float,
            increment: float = 0.05,
            window: int = 8,
            min_episode_length: float = 300.0,
            min_velocity_ratio: float = 0.75,
            max_fall_rate: float = 0.125,
            new_stage_exploration_scale: float = 0.50,
            exploration_recovery_episodes: int = 4,
            initial_upper_speed: float | None = None,
            mode: str = 'performance',
            balance_min_transitions: int = 1600,
            balance_min_episodes: int = 4):
        if increment <= 0.0 or increment > 0.05 + 1e-9:
            raise ValueError('cmd speed increment must be in (0, 0.05]')
        if max_speed < min_speed:
            raise ValueError('max_speed must be >= min_speed')
        if window < 1:
            raise ValueError('window must be positive')
        if not 0.0 <= max_fall_rate <= 1.0:
            raise ValueError('max_fall_rate must be in [0, 1]')
        if not 0.0 < new_stage_exploration_scale <= 1.0:
            raise ValueError(
                'new_stage_exploration_scale must be in (0, 1]')
        if mode not in ('performance', 'performance_then_balanced'):
            raise ValueError(
                'cmd speed curriculum mode must be performance or '
                'performance_then_balanced')
        if balance_min_transitions < 1 or balance_min_episodes < 1:
            raise ValueError('balanced coverage minima must be positive')

        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.increment = float(increment)
        self.window = int(window)
        self.min_episode_length = float(min_episode_length)
        self.min_velocity_ratio = float(min_velocity_ratio)
        self.max_fall_rate = float(max_fall_rate)
        self.new_stage_exploration_scale = float(
            new_stage_exploration_scale)
        self.exploration_recovery_episodes = max(
            int(exploration_recovery_episodes), 1)
        self.mode = str(mode)
        self.balance_min_transitions = int(balance_min_transitions)
        self.balance_min_episodes = int(balance_min_episodes)

        initial = (
            self.min_speed
            if initial_upper_speed is None else float(initial_upper_speed))
        self.upper_speed = float(np.clip(
            initial, self.min_speed, self.max_speed))
        self._frontier = deque(maxlen=self.window)
        self._recovery_progress = self.exploration_recovery_episodes
        self.phase = 'performance'
        self.frontier_complete = False
        self.speed_bins = tuple(
            round(self.min_speed + index * self.increment, 6)
            for index in range(
                int(round(
                    (self.max_speed - self.min_speed) / self.increment)) + 1)
        )
        self._balanced_transitions = {
            speed: 0 for speed in self.speed_bins}
        self._balanced_episodes = {
            speed: 0 for speed in self.speed_bins}
        self._balanced_falls = {
            speed: 0 for speed in self.speed_bins}
        self._frontier_history: list[dict[str, float | bool]] = []

    @property
    def exploration_multiplier(self) -> float:
        progress = min(
            1.0,
            self._recovery_progress / self.exploration_recovery_episodes)
        return float(
            self.new_stage_exploration_scale
            + progress * (1.0 - self.new_stage_exploration_scale))

    def _is_frontier(self, command_speed: float) -> bool:
        return abs(float(command_speed) - self.upper_speed) <= (
            self.increment * 0.25 + 1e-6)

    def _speed_bin(self, command_speed: float) -> float:
        return min(
            self.speed_bins,
            key=lambda value: abs(value - float(command_speed)))

    @property
    def coverage_complete(self) -> bool:
        return (
            self.phase == 'balanced'
            and all(
                self._balanced_transitions[speed]
                >= self.balance_min_transitions
                and self._balanced_episodes[speed]
                >= self.balance_min_episodes
                for speed in self.speed_bins)
        )

    def record_transition(self, command_speed: float) -> None:
        if self.phase != 'balanced':
            return
        self._balanced_transitions[self._speed_bin(command_speed)] += 1

    def manifest(self) -> dict[str, object]:
        return {
            'mode': self.mode,
            'phase': self.phase,
            'min_speed': self.min_speed,
            'max_speed': self.max_speed,
            'increment': self.increment,
            'upper_speed': self.upper_speed,
            'frontier_complete': self.frontier_complete,
            'coverage_complete': self.coverage_complete,
            'balance_min_transitions': self.balance_min_transitions,
            'balance_min_episodes': self.balance_min_episodes,
            'speed_bins': list(self.speed_bins),
            'balanced_transitions': {
                f'{speed:.2f}': self._balanced_transitions[speed]
                for speed in self.speed_bins},
            'balanced_episodes': {
                f'{speed:.2f}': self._balanced_episodes[speed]
                for speed in self.speed_bins},
            'balanced_falls': {
                f'{speed:.2f}': self._balanced_falls[speed]
                for speed in self.speed_bins},
            'frontier_history': list(self._frontier_history),
        }

    def record_episode(
            self, *,
            command_speed: float,
            mean_forward_velocity: float,
            episode_length: int,
            fell: bool) -> CurriculumUpdate:
        """Record one episode and possibly promote the speed frontier."""
        if self.phase == 'balanced':
            speed = self._speed_bin(command_speed)
            self._balanced_episodes[speed] += 1
            self._balanced_falls[speed] += int(bool(fell))
            return CurriculumUpdate(
                promoted=False,
                upper_speed=self.upper_speed,
                frontier_episodes=0,
                mean_velocity_ratio=float('nan'),
                mean_episode_length=float('nan'),
                fall_rate=float('nan'),
                exploration_multiplier=1.0,
                phase=self.phase,
                frontier_complete=self.frontier_complete,
                coverage_complete=self.coverage_complete,
            )

        if self._is_frontier(command_speed):
            ratio = float(mean_forward_velocity) / max(
                abs(float(command_speed)), 1e-6)
            self._frontier.append(
                (ratio, float(episode_length), float(bool(fell))))

            episode_stable = (
                not fell
                and episode_length >= self.min_episode_length
                and ratio >= self.min_velocity_ratio)
            if self._recovery_progress < self.exploration_recovery_episodes:
                if episode_stable:
                    self._recovery_progress += 1
                else:
                    self._recovery_progress = 0

        if self._frontier:
            values = np.asarray(self._frontier, dtype=np.float64)
            velocity_ratio = float(np.mean(values[:, 0]))
            episode_length_mean = float(np.mean(values[:, 1]))
            fall_rate = float(np.mean(values[:, 2]))
        else:
            velocity_ratio = float('nan')
            episode_length_mean = float('nan')
            fall_rate = float('nan')

        gate_passed = (
            len(self._frontier) == self.window
            and velocity_ratio >= self.min_velocity_ratio
            and episode_length_mean >= self.min_episode_length
            and fall_rate <= self.max_fall_rate)
        evaluated_speed = self.upper_speed
        promoted = (
            gate_passed
            and self.upper_speed < self.max_speed - 1e-9)
        if promoted:
            self._frontier_history.append({
                'speed': float(evaluated_speed),
                'passed': True,
                'mean_velocity_ratio': velocity_ratio,
                'mean_episode_length': episode_length_mean,
                'fall_rate': fall_rate,
            })
            self.upper_speed = min(
                self.max_speed,
                round(self.upper_speed + self.increment, 6))
            self._frontier.clear()
            self._recovery_progress = 0
        elif gate_passed and self.upper_speed >= self.max_speed - 1e-9:
            self._frontier_history.append({
                'speed': float(evaluated_speed),
                'passed': True,
                'mean_velocity_ratio': velocity_ratio,
                'mean_episode_length': episode_length_mean,
                'fall_rate': fall_rate,
            })
            self.frontier_complete = True
            self._frontier.clear()
            if self.mode == 'performance_then_balanced':
                self.phase = 'balanced'

        return CurriculumUpdate(
            promoted=promoted,
            upper_speed=self.upper_speed,
            frontier_episodes=len(self._frontier),
            mean_velocity_ratio=velocity_ratio,
            mean_episode_length=episode_length_mean,
            fall_rate=fall_rate,
            exploration_multiplier=self.exploration_multiplier,
            phase=self.phase,
            frontier_complete=self.frontier_complete,
            coverage_complete=self.coverage_complete,
        )
