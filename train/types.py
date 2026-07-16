"""Shared datatypes (SI units)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUM_JOINTS = 12


@dataclass
class RobotState:
    joint_q: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_JOINTS, np.float32))       # rad
    joint_dq: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_JOINTS, np.float32))      # rad/s
    imu_quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    imu_gyro: np.ndarray = field(
        default_factory=lambda: np.zeros(3, np.float32))               # rad/s
    imu_accel: np.ndarray = field(
        default_factory=lambda: np.zeros(3, np.float32))               # m/s²
    body_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, np.float32))               # m/s
    timestamp: float = 0.0                                               # s
    low_state_timestamp: float = 0.0                                      # s
    sport_state_timestamp: float = 0.0                                    # s
    low_state_count: int = 0
    sport_state_count: int = 0
