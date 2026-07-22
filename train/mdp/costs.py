"""Go2 deploy-safe safety cost components."""

from __future__ import annotations

import numpy as np

from train.config import Go2Config
from train.types import RobotState


def soft_joint_limit_cost(state: RobotState, cfg: Go2Config) -> float:
    margin = 0.9
    center = 0.5 * (cfg.joint_min + cfg.joint_max)
    half_range = 0.5 * (cfg.joint_max - cfg.joint_min) * margin
    soft_min = center - half_range
    soft_max = center + half_range
    low = np.maximum(soft_min - state.joint_q, 0.0)
    high = np.maximum(state.joint_q - soft_max, 0.0)
    return float(np.mean(low * low + high * high))


def joint_velocity_cost(state: RobotState, safe_speed: float) -> float:
    excess = np.maximum(np.abs(state.joint_dq) - safe_speed, 0.0)
    return float(np.mean(excess * excess))
