"""Training loop timing and throughput metrics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepProfiler:
    control_dt: float
    utd_ratio: int = 1
    enabled: bool = True
    _step_ms: float = 0.0
    _sample_ms: float = 0.0
    _update_ms: float = 0.0
    _loop_elapsed: float = 0.0
    _updates_this_step: int = 0
    _total_updates: int = 0
    _total_loop_s: float = 0.0

    def begin_step(self) -> None:
        if not self.enabled:
            return
        self._step_ms = 0.0
        self._sample_ms = 0.0
        self._update_ms = 0.0
        self._updates_this_step = 0

    def record_step(self, elapsed_s: float) -> None:
        if not self.enabled:
            return
        self._step_ms = elapsed_s * 1000.0

    def record_sample(self, elapsed_s: float) -> None:
        if not self.enabled:
            return
        self._sample_ms = elapsed_s * 1000.0

    def record_update(self, elapsed_s: float) -> None:
        if not self.enabled:
            return
        self._update_ms += elapsed_s * 1000.0
        self._updates_this_step += 1
        self._total_updates += 1

    def end_loop(self, elapsed_s: float) -> None:
        if not self.enabled:
            return
        self._loop_elapsed = elapsed_s
        self._total_loop_s += elapsed_s

    def metrics(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        loop_ms = self._loop_elapsed * 1000.0
        effective_hz = 1.0 / self._loop_elapsed if self._loop_elapsed > 0 else 0.0
        real_time_factor = (
            self.control_dt / self._loop_elapsed
            if self._loop_elapsed > 0 else 0.0)
        critic_per_s = (
            self._updates_this_step * self.utd_ratio / self._loop_elapsed
            if self._loop_elapsed > 0 else 0.0)
        avg_hz = (
            (self._total_updates * self.utd_ratio / self._total_loop_s)
            if self._total_loop_s > 0 else 0.0)
        return {
            'timing/step_ms': self._step_ms,
            'timing/sample_ms': self._sample_ms,
            'timing/update_ms': self._update_ms,
            'timing/loop_ms': loop_ms,
            'timing/effective_hz': effective_hz,
            'timing/real_time_factor': real_time_factor,
            'timing/critic_updates_per_sec': critic_per_s,
            'timing/avg_critic_updates_per_sec': avg_hz,
        }
