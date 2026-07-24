"""Non-invasive safety signals derived from the existing Go2 state."""

from __future__ import annotations

import numpy as np

from common.transition import zero_costs
from train.config import Go2Config
from train.obs import body_up_cos, tilt_from_upright
from train.types import RobotState


def safety_signals(state: RobotState, cfg: Go2Config, *, terminated: bool,
                   recovering: bool, intervention_mask: bool,
                   communication_fault: bool = False) -> dict[str, object]:
    """Compute bounded costs and SQRL labels without affecting control."""
    tilt = tilt_from_upright(state.imu_quat)
    up_cos = body_up_cos(state.imu_quat)
    joint_min = np.asarray(getattr(
        cfg, 'joint_min', np.full(state.joint_q.shape, -np.inf)))
    joint_max = np.asarray(getattr(
        cfg, 'joint_max', np.full(state.joint_q.shape, np.inf)))
    joint_distance = np.minimum(state.joint_q - joint_min,
                                joint_max - state.joint_q)
    joint_margin = max(float(getattr(
        cfg, 'safety_joint_limit_margin_rad', 0.10)), 1e-6)
    joint_limit_cost = float(np.clip(
        np.max((joint_margin - joint_distance) / joint_margin), 0.0, 1.0))
    max_joint_velocity = float(np.max(np.abs(state.joint_dq)))
    max_torque = float(np.max(np.abs(state.joint_tau)))
    max_power = float(np.max(np.abs(state.joint_tau * state.joint_dq)))
    angular_velocity = float(np.linalg.norm(state.imu_gyro))
    accel_norm = float(np.linalg.norm(state.imu_accel))
    impact_excess = abs(accel_norm - 9.81)
    success_orientation = float(getattr(
        cfg, 'success_orientation_rad', np.pi / 6))
    fallen_risk = float(getattr(cfg, 'fallen_risk_rad', np.pi / 9))
    upright_up_cos = float(getattr(cfg, 'imu_upright_up_cos', 0.5))
    upside_down_up_cos = float(getattr(
        cfg, 'imu_upside_down_up_cos', -0.7))
    joint_velocity_limit = float(getattr(
        cfg, 'safety_joint_velocity_rad_s', 25.0))
    torque_limit = float(getattr(cfg, 'safety_torque_nm', 35.0))
    power_limit = float(getattr(cfg, 'safety_power_w', 350.0))
    impact_limit = float(getattr(
        cfg, 'safety_impact_accel_m_s2', 8.0))
    angular_velocity_limit = float(getattr(
        cfg, 'safety_angular_velocity_rad_s', 5.0))
    base_height_limit = float(getattr(cfg, 'safety_base_height_m', 0.18))
    hard_fall = bool(terminated and tilt > success_orientation)
    belly_up = up_cos < upside_down_up_cos

    costs = zero_costs()
    costs.update({
        'tilt_cost': float(np.clip(
            tilt / max(success_orientation, 1e-6), 0.0, 1.0)),
        'joint_limit_cost': joint_limit_cost,
        'joint_velocity_cost': float(np.clip(
            max_joint_velocity / max(joint_velocity_limit, 1e-6),
            0.0, 1.0)),
        'torque_cost': float(np.clip(
            max_torque / max(torque_limit, 1e-6), 0.0, 1.0)),
        'power_cost': float(np.clip(
            max_power / max(power_limit, 1e-6), 0.0, 1.0)),
        'impact_cost': float(max(hard_fall, np.clip(
            impact_excess / max(impact_limit, 1e-6),
            0.0, 1.0))),
        # No foot contact/velocity is present in the current DDS state.
        'slip_cost': 0.0,
        'intervention_cost': float(recovering or intervention_mask),
        'communication_cost': float(communication_fault),
    })

    near_failure = bool(
        tilt >= fallen_risk
        or up_cos < upright_up_cos
        or angular_velocity >= angular_velocity_limit
        or float(state.world_position[2]) <= base_height_limit
        or joint_limit_cost > 0.0
        or recovering
    )
    unsafe = bool(hard_fall or belly_up)
    return {
        'costs': costs,
        'unsafe_label': unsafe,
        'near_failure_label': near_failure or unsafe,
        'tilt_rad': tilt,
        'angular_velocity_norm': angular_velocity,
        'max_joint_velocity': max_joint_velocity,
        'max_torque': max_torque,
        'max_power': max_power,
        'base_height': float(state.world_position[2]),
        'joint_limit_margin_min': float(np.min(joint_distance)),
        'hard_fall': hard_fall,
    }
