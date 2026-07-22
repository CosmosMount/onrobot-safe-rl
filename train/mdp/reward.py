"""Go2 reward and safety cost functions."""

from __future__ import annotations

import numpy as np

from common.transition import zero_costs
from train.config import Go2Config, TrainConfig
from train.mdp.costs import joint_velocity_cost, soft_joint_limit_cost
from train.mdp.observation import projected_gravity
from train.types import RobotState


def _smooth_tracking(value: float, target: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-6)
    return float(np.exp(-((value - target) ** 2) / (sigma * sigma)))


def compute_reward(
    state: RobotState,
    cfg: Go2Config,
    train_cfg: TrainConfig,
    *,
    command: np.ndarray,
    requested_action: np.ndarray,
    previous_sent_action: np.ndarray,
    terminated: bool,
) -> tuple[float, dict[str, float], dict[str, float]]:
    dt = 1.0 / float(train_cfg.control_frequency)
    cmd = np.asarray(command, dtype=np.float32)
    body_velocity = np.asarray(state.body_velocity, dtype=np.float32)
    gyro = np.asarray(state.imu_gyro, dtype=np.float32)
    gravity = projected_gravity(state.imu_quat)

    lin_tracking = _smooth_tracking(
        float(body_velocity[0]), float(cmd[0]), train_cfg.reward_tracking_sigma)
    yaw_tracking = _smooth_tracking(
        float(gyro[2]), float(cmd[2]), train_cfg.reward_tracking_sigma)
    orientation_cost = float(np.sum(np.square(gravity[:2])))
    action_rate_cost = float(np.mean(np.square(
        np.asarray(requested_action, dtype=np.float32)
        - np.asarray(previous_sent_action, dtype=np.float32))))
    joint_limit_cost = soft_joint_limit_cost(state, cfg)
    joint_velocity_penalty = joint_velocity_cost(
        state, train_cfg.joint_velocity_safe_rad_s)

    task_rate = (
        train_cfg.reward_linear_velocity_weight * lin_tracking
        + train_cfg.reward_yaw_tracking_weight * yaw_tracking)
    safety_rate = (
        train_cfg.reward_orientation_cost_weight * orientation_cost
        + train_cfg.reward_action_rate_cost_weight * action_rate_cost
        + train_cfg.reward_joint_limit_cost_weight * joint_limit_cost
        + train_cfg.reward_joint_velocity_cost_weight * joint_velocity_penalty)
    terminal_penalty = (
        float(train_cfg.terminal_reward) if terminated else 0.0)
    reward = float(dt * (task_rate - safety_rate) + terminal_penalty)

    costs = zero_costs()
    costs.update({
        'tilt_cost': orientation_cost,
        'joint_limit_cost': joint_limit_cost,
        'joint_velocity_cost': joint_velocity_penalty,
        'intervention_cost': 0.0,
    })
    info = {
        'task_reward': float(dt * task_rate),
        'safety_penalty': float(dt * safety_rate),
        'terminal_penalty': terminal_penalty,
        'tracking_lin_vel': lin_tracking,
        'tracking_yaw': yaw_tracking,
        'orientation_cost': orientation_cost,
        'action_rate_cost': action_rate_cost,
        'joint_limit_cost': joint_limit_cost,
        'joint_velocity_cost': joint_velocity_penalty,
        'x_velocity': float(body_velocity[0]),
        'forward_velocity': float(body_velocity[0]),
        'dyaw': float(gyro[2]),
        'body_up_cos': float(-gravity[2]),
        'upright_gate': 1.0,
        'forward_term': lin_tracking,
        'reward_raw': reward,
    }
    return reward, info, costs
