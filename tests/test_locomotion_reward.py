from types import SimpleNamespace

import numpy as np

from runtime.inference.observations import (
    RewardContext,
    _leg_balance,
    get_run_reward_from_state,
    get_terminal_penalty,
)
from runtime.inference.state import RobotState


def cfg():
    return SimpleNamespace(
        reward_profile="locomotion_straight", reward_command_vx=0.5,
        reward_tracking_sigma=0.25, reward_upright_min_cos=0.94,
        reward_upright_exponent=2.0, reward_tracking_lin_vel_weight=8.0,
        reward_tracking_ang_vel_weight=0.25, reward_roll_pitch_rate_weight=0.2,
        reward_angular_rate_scale=2.0, reward_lateral_velocity_weight=0.2,
        reward_lateral_velocity_scale=0.35, reward_vertical_velocity_weight=0.3,
        reward_vertical_velocity_scale=0.4, reward_vertical_velocity_penalty_max=4.0,
        reward_action_rate_weight=0.05, reward_action_rate_scale=0.25,
        reward_action_rate_penalty_max=4.0, reward_action_magnitude_weight=0.02,
        reward_action_magnitude_scale=0.6, reward_action_magnitude_penalty_max=2.0,
        reward_leg_activity_epsilon=0.01, reward_leg_balance_speed_gate=0.05,
        reward_leg_activity_balance_weight=0.05,
        reward_leg_action_activity_scale=0.05, reward_leg_joint_velocity_scale=1.0,
        reward_similar_to_default_weight=0.05, reward_base_height_weight=15.0,
        reward_base_height_target=0.445, reward_orientation_weight=1.0,
        reward_orientation_penalty_max=4.0, reward_dof_velocity_weight=0.05,
        reward_dof_velocity_scale=4.0, reward_joint_limit_weight=0.2,
        reward_joint_limit_margin=0.1, reward_forward_tilt_weight=2.0,
        reward_pitch_free_rad=0.1, reward_pitch_danger_rad=0.4,
        reward_forward_pitch_rate_weight=0.5, reward_pitch_rate_scale=1.0,
        reward_pitch_rate_penalty_max=2.0, fall_terminal_penalty=-100.0,
        move_speed=0.5, init_qpos=np.zeros(12, np.float32),
        joint_min=-np.ones(12, np.float32), joint_max=np.ones(12, np.float32),
    )


def state(vx=0.0, vy=0.0, vz=0.0, quat=None, gyro=None):
    return RobotState(
        body_velocity=np.array([vx, vy, vz], np.float32),
        imu_quat=np.array([1, 0, 0, 0], np.float32) if quat is None else np.asarray(quat, np.float32),
        imu_gyro=np.zeros(3, np.float32) if gyro is None else np.asarray(gyro, np.float32),
        world_position=np.array([0, 0, 0.445], np.float32),
    )


def context(action, previous=None, dq=None):
    action = np.asarray(action, np.float32)
    return RewardContext(action, action if previous is None else np.asarray(previous, np.float32),
                         np.zeros(4, np.float32), np.zeros(4, np.float32) if dq is None else np.asarray(dq, np.float32))


def reward_info(s, c=None):
    return get_run_reward_from_state(s, cfg(), c or context(np.zeros(12)))[1]


def test_x_tracking_is_zero_based_and_x_only():
    assert reward_info(state(vx=0.0))["reward/x_tracking_zero_based"] == 0.0
    assert reward_info(state(vx=0.5))["reward/x_tracking_zero_based"] > 0.99
    assert reward_info(state(vx=-0.2))["reward/x_tracking_zero_based"] == 0.0
    assert reward_info(state(vx=1.0))["reward/x_tracking_zero_based"] < reward_info(state(vx=0.5))["reward/x_tracking_zero_based"]
    assert reward_info(state(vx=0.5, vy=0.3))["reward/x_tracking_zero_based"] == reward_info(state(vx=0.5))["reward/x_tracking_zero_based"]
    assert reward_info(state(vx=0.5, vy=0.3))["reward/lateral_velocity_penalty"] > 0.0


def test_orientation_vertical_action_and_pitch_risks():
    upright = reward_info(state())
    assert upright["reward/orientation_penalty"] == 0.0
    assert reward_info(state(quat=[2**-.5, 2**-.5, 0, 0]))["reward/orientation_penalty"] > 0.0
    assert reward_info(state(quat=[0, 1, 0, 0]))["reward/orientation_penalty"] == 4.0
    plus_pitch = reward_info(state(quat=[np.cos(0.2), 0, np.sin(0.2), 0]))
    minus_pitch = reward_info(state(quat=[np.cos(0.2), 0, -np.sin(0.2), 0]))
    assert plus_pitch["reward/pitch_risk_penalty"] == minus_pitch["reward/pitch_risk_penalty"]
    plus_rate = reward_info(state(gyro=[0, 0.4, 0]))
    minus_rate = reward_info(state(gyro=[0, -0.4, 0]))
    assert plus_rate["reward/pitch_rate_risk_penalty"] == minus_rate["reward/pitch_rate_risk_penalty"]
    small = reward_info(state(vz=0.2))["reward/vertical_velocity_penalty"]
    large = reward_info(state(vz=0.6))["reward/vertical_velocity_penalty"]
    assert small == reward_info(state(vz=-0.2))["reward/vertical_velocity_penalty"]
    assert large > small


def test_action_magnitude_and_leg_activity_are_distinct():
    same = reward_info(state(vx=0.5), context(np.full(12, 0.8), np.full(12, 0.8)))
    changed = reward_info(state(vx=0.5), context(np.full(12, 0.8), np.zeros(12)))
    assert same["reward/action_rate_penalty"] == 0.0
    assert same["reward/action_magnitude_penalty"] > 0.0
    assert changed["reward/action_rate_penalty"] > 0.0
    balanced = _leg_balance(np.array([0.01, 0.01, 0.01, 0.01]), np.array([1, 1, 1, 1]), .05, 1.0, .01)
    assert balanced[2] == 0.0


def test_terminal_penalty_distinguishes_termination_and_timeout():
    assert get_terminal_penalty(terminated=True, cfg=cfg()) == -100.0
    assert get_terminal_penalty(terminated=False, cfg=cfg()) == 0.0
