"""Diagnostic-only leg activity tracking.

This state never changes actions, recovery, or episode termination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robots.go2.joint_layout import LEG_NAMES, LEG_SLICES


@dataclass
class LegActivityState:
    action_delta_ema: np.ndarray
    joint_velocity_ema: np.ndarray
    inactive_steps: np.ndarray

    @classmethod
    def create(cls) -> "LegActivityState":
        return cls(np.zeros(4, np.float32), np.zeros(4, np.float32), np.zeros(4, np.int32))

    def reset(self) -> None:
        self.action_delta_ema.fill(0.0)
        self.joint_velocity_ema.fill(0.0)
        self.inactive_steps.fill(0)

    def update(self, action_delta: np.ndarray, joint_dq: np.ndarray, *,
               forward_velocity: float, beta: float, minimum_action_delta_rms: float,
               minimum_joint_velocity_rms: float, inactive_grace_steps: int) -> dict[str, float]:
        delta = np.asarray(action_delta, dtype=np.float32).reshape(12)
        dq = np.asarray(joint_dq, dtype=np.float32).reshape(12)
        action_rms = np.asarray([np.sqrt(np.mean(delta[LEG_SLICES[n]] ** 2)) for n in LEG_NAMES])
        velocity_rms = np.asarray([np.sqrt(np.mean(dq[LEG_SLICES[n]] ** 2)) for n in LEG_NAMES])
        self.action_delta_ema[:] = beta * self.action_delta_ema + (1.0 - beta) * action_rms
        self.joint_velocity_ema[:] = beta * self.joint_velocity_ema + (1.0 - beta) * velocity_rms
        peer_indices = (1, 0, 3, 2)
        active_peer = self.action_delta_ema[list(peer_indices)]
        moving = forward_velocity > 0.0
        inactive = moving & (self.action_delta_ema < minimum_action_delta_rms) \
            & (self.joint_velocity_ema < minimum_joint_velocity_rms) \
            & (active_peer >= minimum_action_delta_rms)
        self.inactive_steps[inactive] += 1
        self.inactive_steps[~inactive] = 0
        flags = self.inactive_steps >= int(inactive_grace_steps)
        return {
            **{f"leg_{name}_inactive": float(flags[i]) for i, name in enumerate(LEG_NAMES)},
            **{f"leg_{name}_inactive_steps": float(self.inactive_steps[i]) for i, name in enumerate(LEG_NAMES)},
        }

    def metrics(self) -> dict[str, float]:
        return {
            **{f"leg_{name}_action_delta_ema": float(self.action_delta_ema[i]) for i, name in enumerate(LEG_NAMES)},
            **{f"leg_{name}_joint_velocity_ema": float(self.joint_velocity_ema[i]) for i, name in enumerate(LEG_NAMES)},
            **{f"leg_{name}_inactive": float(self.inactive_steps[i] > 0) for i, name in enumerate(LEG_NAMES)},
            **{f"leg_{name}_inactive_steps": float(self.inactive_steps[i]) for i, name in enumerate(LEG_NAMES)},
        }
