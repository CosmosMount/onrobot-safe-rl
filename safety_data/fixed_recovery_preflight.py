"""Fail-closed prevention preflight for the original Go2 recovery sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from safety_data.fixed_recovery_motion import (
    FixedRecoveryConfig,
    FixedRecoveryExecutor,
    FixedRecoveryMotion,
)


@dataclass(frozen=True)
class FixedRecoveryPreflightResult:
    sequence_completed: bool
    low_level_ticks: int
    elapsed_seconds: float
    first_fall_tick: int | None
    first_fall_seconds: float | None
    first_fall_stage: str | None
    minimum_height_m: float
    maximum_tilt_rad: float
    final_height_m: float
    final_tilt_rad: float
    prevention_negative_control_pass: bool

    def manifest(self) -> dict[str, Any]:
        return {
            "sequence_completed": self.sequence_completed,
            "low_level_ticks": self.low_level_ticks,
            "elapsed_seconds": self.elapsed_seconds,
            "first_fall_tick": self.first_fall_tick,
            "first_fall_seconds": self.first_fall_seconds,
            "first_fall_stage": self.first_fall_stage,
            "minimum_height_m": self.minimum_height_m,
            "maximum_tilt_rad": self.maximum_tilt_rad,
            "final_height_m": self.final_height_m,
            "final_tilt_rad": self.final_tilt_rad,
            "prevention_negative_control_pass": self.prevention_negative_control_pass,
        }


def evaluate_standing_fixed_recovery(
    env: Any,
    config: FixedRecoveryConfig,
    *,
    maximum_seconds: float = 10.0,
    frame_callback: Any | None = None,
) -> FixedRecoveryPreflightResult:
    """Apply fixed recovery from stable standing without masking fall labels.

    A prevention controller must not manufacture the exact event it is meant
    to prevent when invoked on the stable negative control.  This is only a
    necessary preflight, not evidence of benefit on natural pre-fall states.
    """
    if not np.isfinite(maximum_seconds) or maximum_seconds <= 0.0:
        raise ValueError("maximum_seconds must be finite and positive")
    initial = env.measurement()
    if initial.failure:
        raise ValueError("standing preflight must start outside the fall predicate")
    executor = FixedRecoveryExecutor(FixedRecoveryMotion(config, control_hz=500.0))
    executor.start(env)
    maximum_ticks = int(np.ceil(maximum_seconds * 500.0))
    first_fall_tick = None
    first_fall_stage = None
    minimum_height = float(initial.height_m)
    maximum_tilt = float(initial.tilt_rad)
    final = initial
    ticks = 0
    for ticks in range(1, maximum_ticks + 1):
        execution = executor.tick(env)
        final = execution.rollout.measurement
        minimum_height = min(minimum_height, float(final.height_m))
        maximum_tilt = max(maximum_tilt, float(final.tilt_rad))
        if final.failure and first_fall_tick is None:
            first_fall_tick = ticks
            first_fall_stage = execution.motion.stage_executed.value
        if frame_callback is not None:
            frame_callback(ticks, env, execution)
        if executor.done:
            break
    elapsed = ticks / 500.0
    return FixedRecoveryPreflightResult(
        sequence_completed=executor.done,
        low_level_ticks=ticks,
        elapsed_seconds=elapsed,
        first_fall_tick=first_fall_tick,
        first_fall_seconds=(
            None if first_fall_tick is None else first_fall_tick / 500.0),
        first_fall_stage=first_fall_stage,
        minimum_height_m=minimum_height,
        maximum_tilt_rad=maximum_tilt,
        final_height_m=float(final.height_m),
        final_tilt_rad=float(final.tilt_rad),
        prevention_negative_control_pass=first_fall_tick is None,
    )


__all__ = ["FixedRecoveryPreflightResult", "evaluate_standing_fixed_recovery"]
