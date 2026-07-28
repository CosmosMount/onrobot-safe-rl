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
            initial_upper_speed: float | None = None):
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

        initial = (
            self.min_speed
            if initial_upper_speed is None else float(initial_upper_speed))
        self.upper_speed = float(np.clip(
            initial, self.min_speed, self.max_speed))
        self._frontier = deque(maxlen=self.window)
        self._recovery_progress = self.exploration_recovery_episodes

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

    def record_episode(
            self, *,
            command_speed: float,
            mean_forward_velocity: float,
            episode_length: int,
            fell: bool) -> CurriculumUpdate:
        """Record one episode and possibly promote the speed frontier."""
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

        promoted = (
            len(self._frontier) == self.window
            and self.upper_speed < self.max_speed - 1e-9
            and velocity_ratio >= self.min_velocity_ratio
            and episode_length_mean >= self.min_episode_length
            and fall_rate <= self.max_fall_rate)
        if promoted:
            self.upper_speed = min(
                self.max_speed,
                round(self.upper_speed + self.increment, 6))
            self._frontier.clear()
            self._recovery_progress = 0

        return CurriculumUpdate(
            promoted=promoted,
            upper_speed=self.upper_speed,
            frontier_episodes=len(self._frontier),
            mean_velocity_ratio=velocity_ratio,
            mean_episode_length=episode_length_mean,
            fall_rate=fall_rate,
            exploration_multiplier=self.exploration_multiplier,
        )
