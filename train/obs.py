"""Observation, reward, and fall detection for walk task.

All physical quantities use SI unless noted dimensionless:
  angles rad, angular rates rad/s, lengths m, speeds m/s, acceleration m/s², time s.
"""

from __future__ import annotations

import numpy as np

from train.config import Go2Config
from train.types import RobotState


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


def body_up_cos(quat: np.ndarray) -> float:
    """Cosine between body +Z and world +Z. 1=upright, 0=on side, -1=belly-up."""
    _, x, y, _ = normalize_quat(quat)
    return float(1.0 - 2.0 * (x * x + y * y))


def tilt_from_upright(quat: np.ndarray) -> float:
    """Tilt angle from vertical (rad). 0=upright, π/2=on side, π=belly-up."""
    up = body_up_cos(quat)
    return float(np.arccos(np.clip(up, -1.0, 1.0)))


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def sanitize_observation(obs: np.ndarray, cfg: Go2Config) -> np.ndarray:
    """Replace non-finite values and normalize quaternion in obs."""
    out = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0).astype(
        np.float32)
    quat_start = 4 * cfg.num_joints + 6
    out[quat_start:quat_start + 4] = normalize_quat(out[quat_start:quat_start +
                                                          4])
    return np.clip(out, -100.0, 100.0)


def build_observation(state: RobotState,
                      previous_requested_action: np.ndarray,
                      cfg: Go2Config,
                      previous_executed_action: np.ndarray | None = None
                      ) -> np.ndarray:
    if previous_executed_action is None:
        previous_executed_action = previous_requested_action
    quat = normalize_quat(state.imu_quat)
    obs = np.concatenate([
        state.joint_q.astype(np.float32),
        state.joint_dq.astype(np.float32),
        previous_requested_action.astype(np.float32),
        previous_executed_action.astype(np.float32),
        state.imu_gyro.astype(np.float32),
        state.body_velocity.astype(np.float32),
        quat,
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
        cfg: Go2Config) -> tuple[float, dict[str, float]]:
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
    reward = get_run_reward(
        x_velocity,
        cfg.move_speed,
        cos_pitch,
        dyaw,
        min_forward_vel=cfg.reward_min_forward_vel,
    )
    info = {
        'x_velocity': x_velocity,
        'forward_velocity': forward_vel,
        'cos_pitch': cos_pitch,
        'dyaw': dyaw,
        'dpitch': dpitch,
        'droll': droll,
        'forward_term': forward_term,
        'reward_raw': reward_raw,
    }
    return reward, info


def gravity_acc_z(state: RobotState) -> float:
    """Body-frame IMU accelerometer Z (m/s²). At rest ≈ gravity component on body Z."""
    return float(state.imu_accel[2])


def gravity_up_cos(state: RobotState) -> float:
    """cos(tilt from vertical) from gravity vector. 1=upright, -1=belly-up."""
    acc = state.imu_accel
    norm = float(np.linalg.norm(acc))
    if norm < 1.0:
        return body_up_cos(state.imu_quat)
    return float(acc[2] / norm)


def is_flipped_back(state: RobotState, cfg: Go2Config) -> bool:
    up = gravity_up_cos(state)
    if up > cfg.imu_upright_up_cos:
        return True
    return gravity_acc_z(state) > cfg.imu_upright_acc_z


def is_belly_up(state: RobotState, cfg: Go2Config) -> bool:
    """Clearly inverted (belly-up). Uses quaternion only — acc_z flickers while walking."""
    return body_up_cos(state.imu_quat) < cfg.imu_upside_down_up_cos


def is_fallen(state: RobotState, cfg: Go2Config) -> bool:
    """Match sim Run.after_step: |roll| or |pitch| > terminate_pitch_roll."""
    roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
    limit = cfg.success_orientation_rad
    return abs(roll) > limit or abs(pitch) > limit


def is_fallen_risk(state: RobotState, cfg: Go2Config) -> bool:
    """Early tilt warning before episode termination (quat-based, stable while walking)."""
    roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
    limit = cfg.fallen_risk_rad
    return abs(roll) > limit or abs(pitch) > limit


def is_pose_stable(state: RobotState, cfg: Go2Config) -> bool:
    """Standing pose recovered enough to resume policy."""
    if is_fallen_risk(state, cfg):
        return False
    joint_err = float(np.linalg.norm(state.joint_q - cfg.init_qpos))
    return joint_err < cfg.joint_tolerance
