"""Observation and reward helpers for the walk task.

All physical quantities use SI unless noted dimensionless:
  angles rad, angular rates rad/s, lengths m, speeds m/s, acceleration m/s², time s.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from runtime.inference.state import RobotState


def quat_to_euler_xyz(quat: np.ndarray) -> tuple[float, float, float]:
    """Body roll, pitch, yaw (rad) from IMU quaternion, XYZ euler convention."""
    w, x, y, z = normalize_quat(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return float(roll), float(pitch), float(yaw)


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def sanitize_observation(obs: np.ndarray, cfg: Any) -> np.ndarray:
    """Replace non-finite values and normalize quaternion in obs."""
    out = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0).astype(
        np.float32)
    quat_start = 2 * cfg.num_joints + 6
    out[quat_start:quat_start + 4] = normalize_quat(out[quat_start:quat_start +
                                                          4])
    return np.clip(out, -100.0, 100.0)


def build_observation(state: RobotState,
                      previous_requested_action: np.ndarray,
                      cfg: Any
                      ) -> np.ndarray:
    quat = normalize_quat(state.imu_quat)
    obs = np.concatenate([
        state.joint_q.astype(np.float32),
        state.joint_dq.astype(np.float32),
        state.imu_gyro.astype(np.float32),
        state.body_velocity.astype(np.float32),
        quat,
        previous_requested_action.astype(np.float32),
    ])
    assert obs.shape == (cfg.obs_dim,), obs.shape
    return sanitize_observation(obs, cfg)


def get_run_reward(x_velocity: float,
                   move_speed: float,
                   cos_pitch: float,
                   dyaw: float,
                   *,
                   min_forward_vel: float | None = None) -> float:
    """Run task reward (sim/tasks/run.py).

    Args:
        x_velocity: body-frame forward speed (m/s).
        move_speed: target speed (m/s).
        cos_pitch: cos(body pitch) (dimensionless).
        dyaw: yaw rate (rad/s).
        min_forward_vel: optional no-reward gate below this forward speed (m/s).
            None matches upstream walk_in_the_park.
    """
    forward_vel = cos_pitch * x_velocity
    if min_forward_vel is not None and forward_vel < min_forward_vel:
        forward_term = 0.0
    else:
        forward_term = _tolerance(
            forward_vel,
            bounds=(move_speed, 2 * move_speed),
            margin=2 * move_speed,
            value_at_margin=0.0,
        )
    reward = forward_term - 0.1 * abs(dyaw)
    return float(10.0 * reward)


def _tolerance( x: float,
                bounds: tuple[float, float],
                margin: float,
                value_at_margin: float = 0.0) -> float:

    lower, upper = bounds
    if lower > upper:
        lower, upper = upper, lower

    if lower <= x <= upper:
        return 1.0

    if margin <= 0:
        return 0.0

    d = lower - x if x < lower else x - upper

    if d >= margin:
        return 0.0

    return 1.0 - (d / margin) * (1.0 - value_at_margin)


def get_run_reward_from_state(
        state: RobotState,
        cfg: Any) -> tuple[float, dict[str, float]]:
    _, pitch, _ = quat_to_euler_xyz(state.imu_quat)
    cos_pitch = float(np.cos(pitch))
    x_velocity = float(state.body_velocity[0])
    droll = float(state.imu_gyro[0])
    dpitch = float(state.imu_gyro[1])
    dyaw = float(state.imu_gyro[2])
    forward_vel = cos_pitch * x_velocity
    if (cfg.reward_min_forward_vel is not None
            and forward_vel < cfg.reward_min_forward_vel):
        forward_term = 0.0
    else:
        forward_term = _tolerance(
            forward_vel,
            bounds=(cfg.move_speed, 2 * cfg.move_speed),
            margin=2 * cfg.move_speed,
            value_at_margin=0.0,
        )
    reward_raw = forward_term - 0.1 * abs(dyaw)
    reward = float(10.0 * reward_raw)
    info = {
        'x_velocity': x_velocity,
        'forward_velocity': forward_vel,
        'cos_pitch': cos_pitch,
        'dyaw': dyaw,
        'dpitch': dpitch,
        'droll': droll,
        'forward_term': forward_term,
        'reward_raw': reward_raw,
        'task_reward': reward,
    }
    return reward, info


def get_terminal_penalty(*, terminated: bool, cfg: Any) -> float:
    """Apply failure penalty only to true MDP terminations, never time limits."""
    return float(cfg.fall_terminal_penalty if terminated else 0.0)


def is_pose_stable(state: RobotState,
                   cfg: Any,
                   *,
                   joint_tolerance: float | None = None) -> bool:
    """Joint pose is close enough to the nominal standing pose."""
    joint_err = float(np.linalg.norm(state.joint_q - cfg.init_qpos))
    tolerance = (cfg.joint_tolerance
                 if joint_tolerance is None else joint_tolerance)
    return joint_err < tolerance
