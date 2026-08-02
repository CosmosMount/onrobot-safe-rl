"""Observation and straight-walk reward helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from runtime.inference.state import RobotState


@dataclass(frozen=True)
class RewardContext:
    action_requested: np.ndarray
    action_requested_previous: np.ndarray
    leg_action_delta_rms: np.ndarray
    leg_joint_velocity_rms: np.ndarray


def quat_to_euler_xyz(quat: np.ndarray) -> tuple[float, float, float]:
    """Body roll, pitch, yaw (rad) from an XYZ-convention quaternion."""
    w, x, y, z = normalize_quat(quat)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def sanitize_observation(obs: np.ndarray, cfg: Any) -> np.ndarray:
    out = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0).astype(np.float32)
    quat_start = 2 * cfg.num_joints + 6
    out[quat_start:quat_start + 4] = normalize_quat(out[quat_start:quat_start + 4])
    return np.clip(out, -100.0, 100.0)


def build_observation(
        state: RobotState,
        previous_action_q_target: np.ndarray,
        cfg: Any) -> np.ndarray:
    obs = np.concatenate([
        state.joint_q.astype(np.float32),
        state.joint_dq.astype(np.float32),
        state.imu_gyro.astype(np.float32),
        state.body_velocity.astype(np.float32),
        normalize_quat(state.imu_quat),
        previous_action_q_target.astype(np.float32),
    ])
    assert obs.shape == (cfg.obs_dim,), obs.shape
    return sanitize_observation(obs, cfg)


def _tolerance(
        value: float,
        bounds: tuple[float, float],
        margin: float,
        value_at_margin: float = 0.0) -> float:
    lower, upper = sorted(bounds)
    if lower <= value <= upper:
        return 1.0
    if margin <= 0.0:
        return 0.0
    distance = lower - value if value < lower else value - upper
    if distance >= margin:
        return 0.0
    return 1.0 - (distance / margin) * (1.0 - value_at_margin)


def get_run_reward(
        x_velocity: float,
        move_speed: float,
        cos_pitch: float,
        dyaw: float,
        *,
        min_forward_vel: float | None = None) -> float:
    """Legacy upstream scalar reward kept for compatibility callers."""
    forward_velocity = cos_pitch * x_velocity
    if min_forward_vel is not None and forward_velocity < min_forward_vel:
        forward_term = 0.0
    else:
        forward_term = _tolerance(
            forward_velocity,
            bounds=(move_speed, 2.0 * move_speed),
            margin=2.0 * move_speed,
        )
    return float(10.0 * (forward_term - 0.1 * abs(dyaw)))


def get_run_reward_from_state(
        state: RobotState,
        cfg: Any,
        context: RewardContext | None = None) -> tuple[float, dict[str, float]]:
    if getattr(cfg, 'reward_profile', 'upstream') == 'locomotion_straight':
        return _get_locomotion_straight_reward(state, cfg, context)

    _, pitch, _ = quat_to_euler_xyz(state.imu_quat)
    vx = float(state.body_velocity[0])
    dyaw = float(state.imu_gyro[2])
    forward_velocity = float(np.cos(pitch) * vx)
    if cfg.reward_min_forward_vel is not None and forward_velocity < cfg.reward_min_forward_vel:
        forward_term = 0.0
    else:
        forward_term = _tolerance(
            forward_velocity,
            bounds=(cfg.move_speed, 2.0 * cfg.move_speed),
            margin=2.0 * cfg.move_speed,
        )
    reward_raw = forward_term - 0.1 * abs(dyaw)
    reward = float(10.0 * reward_raw)
    return reward, {
        'x_velocity': vx,
        'forward_velocity': forward_velocity,
        'cos_pitch': float(np.cos(pitch)),
        'dyaw': dyaw,
        'dpitch': float(state.imu_gyro[1]),
        'droll': float(state.imu_gyro[0]),
        'forward_term': forward_term,
        'reward_raw': reward_raw,
        'task_reward': reward,
    }


def _body_up(quat: np.ndarray) -> float:
    w, x, y, z = normalize_quat(quat)
    return float(1.0 - 2.0 * (x * x + y * y))


def _leg_balance(values: np.ndarray, epsilon: float) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float32).reshape(4)
    front = float(abs(values[0] - values[1]) / (values[0] + values[1] + epsilon))
    rear = float(abs(values[2] - values[3]) / (values[2] + values[3] + epsilon))
    return front, rear, 0.5 * (front + rear)


def _empty_context() -> RewardContext:
    zeros = np.zeros(12, dtype=np.float32)
    return RewardContext(zeros, zeros, np.zeros(4, np.float32), np.zeros(4, np.float32))


def _get_locomotion_straight_reward(
        state: RobotState,
        cfg: Any,
        context: RewardContext | None) -> tuple[float, dict[str, float]]:
    """Fixed-vx Genesis-style shaping for straight walking only."""
    context = _empty_context() if context is None else context
    velocity = np.asarray(state.body_velocity, dtype=np.float32)
    vx, vy, vz = (float(velocity[i]) for i in range(3))
    roll, pitch, _ = quat_to_euler_xyz(state.imu_quat)
    droll, dpitch, dyaw = (float(state.imu_gyro[i]) for i in range(3))
    body_up = _body_up(state.imu_quat)

    forward_bounds = (cfg.move_speed, 2.0 * cfg.move_speed)
    forward_margin = 2.0 * cfg.move_speed
    forward_raw = _tolerance(vx, forward_bounds, forward_margin)
    rest_baseline = _tolerance(0.0, forward_bounds, forward_margin)
    forward_zero_based = float(np.clip(
        (forward_raw - rest_baseline) / (1.0 - rest_baseline + 1e-6), 0.0, 1.0))
    if vx < 0.0:
        forward_zero_based = 0.0

    upright_gate = float(np.clip(
        (body_up - cfg.reward_upright_min_cos)
        / (1.0 - cfg.reward_upright_min_cos + 1e-6), 0.0, 1.0
    ) ** cfg.reward_upright_exponent)

    tracking_sigma = float(cfg.reward_tracking_sigma)
    linear_error = (cfg.reward_command_vx - vx) ** 2 + vy ** 2
    linear_tracking = float(np.exp(-linear_error / tracking_sigma))
    stationary_linear = float(np.exp(-(cfg.reward_command_vx ** 2) / tracking_sigma))
    linear_tracking_zero_based = float(np.clip(
        (linear_tracking - stationary_linear)
        / (1.0 - stationary_linear + 1e-6), 0.0, 1.0))
    angular_tracking = float(np.exp(-(dyaw ** 2) / tracking_sigma))
    yaw_tracking_penalty = float(np.clip(1.0 - angular_tracking, 0.0, 1.0))

    action_delta = (
        np.asarray(context.action_requested, dtype=np.float32)
        - np.asarray(context.action_requested_previous, dtype=np.float32))
    action_rate_penalty = float(np.clip(
        np.mean(np.square(action_delta)) / (cfg.reward_action_rate_scale ** 2), 0.0, 1.0))
    roll_pitch_penalty = float(np.clip(
        (droll * droll + dpitch * dpitch) / (cfg.reward_angular_rate_scale ** 2), 0.0, 1.0))
    lateral_penalty = float(np.clip(vy * vy / (cfg.reward_lateral_velocity_scale ** 2), 0.0, 1.0))
    vertical_penalty = float(np.clip(vz * vz / (cfg.reward_vertical_velocity_scale ** 2), 0.0, 1.0))

    front_balance, rear_balance, leg_balance = _leg_balance(
        context.leg_action_delta_rms, cfg.reward_leg_activity_epsilon)
    if not (vx > cfg.reward_leg_balance_speed_gate or forward_zero_based > 0.01):
        front_balance = rear_balance = leg_balance = 0.0

    pose_penalty = float(np.sum(np.abs(
        np.asarray(state.joint_q, dtype=np.float32)
        - np.asarray(cfg.init_qpos, dtype=np.float32))))
    base_height = float(np.asarray(state.world_position, dtype=np.float32)[2])
    base_height_penalty = float((base_height - cfg.reward_base_height_target) ** 2)
    orientation_penalty = float(np.clip(1.0 - body_up * body_up, 0.0, 1.0))
    forward_tilt_penalty = float(np.clip(
        max(0.0, pitch - 0.10) / 0.30, 0.0, 1.0) ** 2)
    forward_pitch_rate_penalty = float(np.clip(
        max(0.0, dpitch) / 1.0, 0.0, 1.0) ** 2)
    dof_velocity_penalty = float(np.clip(
        np.mean(np.square(np.asarray(state.joint_dq, dtype=np.float32)))
        / (cfg.reward_dof_velocity_scale ** 2), 0.0, 1.0))

    joint_q = np.asarray(state.joint_q, dtype=np.float32)
    joint_min = np.asarray(cfg.joint_min, dtype=np.float32)
    joint_max = np.asarray(cfg.joint_max, dtype=np.float32)
    joint_width = np.maximum(joint_max - joint_min, 1e-6)
    limit_margin = cfg.reward_joint_limit_margin * joint_width
    near_lower = np.clip((joint_min + limit_margin - joint_q) / limit_margin, 0.0, 1.0)
    near_upper = np.clip((joint_q - (joint_max - limit_margin)) / limit_margin, 0.0, 1.0)
    joint_limit_penalty = float(np.mean(np.maximum(near_lower, near_upper)))

    dense_total = (
        cfg.reward_forward_weight * forward_zero_based * upright_gate
        + cfg.reward_tracking_lin_vel_weight * linear_tracking_zero_based * upright_gate
        - cfg.reward_tracking_ang_vel_weight * yaw_tracking_penalty
        - cfg.reward_roll_pitch_rate_weight * roll_pitch_penalty
        - cfg.reward_lateral_velocity_weight * lateral_penalty
        - cfg.reward_vertical_velocity_weight * vertical_penalty
        - cfg.reward_action_rate_weight * action_rate_penalty
        - cfg.reward_similar_to_default_weight * pose_penalty
        - cfg.reward_base_height_weight * base_height_penalty
        - cfg.reward_leg_activity_balance_weight * leg_balance
        - cfg.reward_orientation_weight * orientation_penalty
        - cfg.reward_dof_velocity_weight * dof_velocity_penalty
        - cfg.reward_joint_limit_weight * joint_limit_penalty
        - cfg.reward_forward_tilt_weight * forward_tilt_penalty
        - cfg.reward_forward_pitch_rate_weight * forward_pitch_rate_penalty
    )
    info = {
        'reward/forward_raw': float(forward_raw),
        'reward/forward_zero_based': forward_zero_based,
        'reward/body_up': body_up,
        'reward/upright_gate': upright_gate,
        'reward/linear_velocity_tracking': linear_tracking,
        'reward/linear_tracking_zero_based': linear_tracking_zero_based,
        'reward/angular_velocity_tracking': angular_tracking,
        'reward/yaw_tracking_penalty': yaw_tracking_penalty,
        'reward/roll_pitch_rate_penalty': roll_pitch_penalty,
        'reward/lateral_velocity_penalty': lateral_penalty,
        'reward/vertical_velocity_penalty': vertical_penalty,
        'reward/action_rate_penalty': action_rate_penalty,
        'reward/similar_to_default_penalty': pose_penalty,
        'reward/base_height_penalty': base_height_penalty,
        'reward/orientation_penalty': orientation_penalty,
        'reward/dof_velocity_penalty': dof_velocity_penalty,
        'reward/joint_limit_penalty': joint_limit_penalty,
        'reward/forward_tilt_penalty': forward_tilt_penalty,
        'reward/forward_pitch_rate_penalty': forward_pitch_rate_penalty,
        'reward/leg_activity_balance_penalty': leg_balance,
        'reward/front_activity_balance': front_balance,
        'reward/rear_activity_balance': rear_balance,
        'reward/dense_total': float(dense_total),
        'reward/terminal_penalty': 0.0,
        'reward/total': float(dense_total),
        'env/command_vx': float(cfg.reward_command_vx),
        'env/body_height': base_height,
        'env/body_up': body_up,
        'env/roll': roll,
        'env/pitch': pitch,
        'env/droll': droll,
        'env/dpitch': dpitch,
        'env/dyaw': dyaw,
        'env/vx': vx,
        'env/vy': vy,
        'env/vz': vz,
        'x_velocity': vx,
        'forward_velocity': vx,
        'cos_pitch': float(np.cos(pitch)),
        'dyaw': dyaw,
        'dpitch': dpitch,
        'droll': droll,
    }
    return float(dense_total), info


def get_terminal_penalty(*, terminated: bool, cfg: Any) -> float:
    """Apply failure penalty only to true MDP terminations."""
    return float(cfg.fall_terminal_penalty if terminated else 0.0)


def is_pose_stable(
        state: RobotState,
        cfg: Any,
        *,
        joint_tolerance: float | None = None) -> bool:
    """Joint pose is close enough to the nominal standing pose."""
    joint_err = float(np.linalg.norm(state.joint_q - cfg.init_qpos))
    tolerance = cfg.joint_tolerance if joint_tolerance is None else joint_tolerance
    return joint_err < tolerance
