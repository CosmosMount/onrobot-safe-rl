"""Shared transition schema for online robot learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping

import numpy as np


class TerminationReason(IntEnum):
    NONE = 0
    TIME_LIMIT = 1
    EXCESSIVE_TILT = 2
    JOINT_LIMIT = 3
    COMMAND_TIMEOUT = 4
    STATE_TIMEOUT = 5
    MOTOR_FAULT = 6
    MANUAL_ESTOP = 7
    RECOVERY_FAILED = 8


COST_KEYS = (
    'tilt_cost',
    'joint_limit_cost',
    'joint_velocity_cost',
    'torque_cost',
    'power_cost',
    'impact_cost',
    'slip_cost',
    'intervention_cost',
    'communication_cost',
)


def zero_costs() -> dict[str, float]:
    return {key: 0.0 for key in COST_KEYS}


@dataclass
class Transition:
    observation: np.ndarray
    requested_action: np.ndarray
    projected_action: np.ndarray
    executed_q_target: np.ndarray
    reward: float
    sent_q_target: np.ndarray = field(default_factory=lambda: np.empty(0))
    costs: Mapping[str, float] = field(default_factory=zero_costs)
    next_observation: np.ndarray = field(default_factory=lambda: np.empty(0))
    terminated: bool = False
    truncated: bool = False
    termination_reason: TerminationReason = TerminationReason.NONE
    intervention_mask: bool = False
    policy_version: int = 0

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def mask(self) -> float:
        return 0.0 if self.terminated else 1.0

    def replay_dict(self) -> dict[str, np.ndarray | float | bool]:
        """Return the shared replay fields consumed by all agent backends."""
        return {
            'observations': self.observation,
            'actions': self.requested_action,
            'rewards': float(self.reward),
            'masks': float(self.mask),
            'dones': bool(self.done),
            'terminateds': bool(self.terminated),
            'truncateds': bool(self.truncated),
            'next_observations': self.next_observation,
        }

    def flashsac_dict(self) -> dict[str, np.ndarray]:
        """Return the single-environment transition schema used by FlashSAC."""
        return {
            'observation': np.asarray(
                self.observation, dtype=np.float32)[None, ...],
            'action': np.asarray(
                self.requested_action, dtype=np.float32)[None, ...],
            'reward': np.asarray([self.reward], dtype=np.float32),
            'terminated': np.asarray([self.terminated], dtype=np.float32),
            'truncated': np.asarray([self.truncated], dtype=np.float32),
            'next_observation': np.asarray(
                self.next_observation, dtype=np.float32)[None, ...],
        }
