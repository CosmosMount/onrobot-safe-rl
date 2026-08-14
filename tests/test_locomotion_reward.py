import numpy as np

from train.env import Go2State, _locomotion_straight_reward


def state(*, quat=None, gyro=None):
    return Go2State(
        policy_sequence=1,
        applied_action_id=1,
        event=0,
        event_action_id=0,
        event_confirm_ms=0,
        timestamp=0.0,
        joint_q=np.zeros(12, np.float32),
        joint_dq=np.zeros(12, np.float32),
        imu_quat=np.asarray([1, 0, 0, 0] if quat is None else quat, np.float32),
        imu_gyro=np.asarray([0, 0, 0] if gyro is None else gyro, np.float32),
        imu_accel=np.zeros(3, np.float32),
        velocity=np.zeros(3, np.float32),
        position=np.zeros(3, np.float32),
        q_target=np.zeros(12, np.float32),
        phase=3,
    )


def reward(s, velocity):
    return _locomotion_straight_reward(
        s, np.zeros(12), np.zeros(12), np.asarray(velocity, np.float32),
        False, return_info=True)


def test_reward_matches_walk_in_the_park_forward_term():
    value, info = reward(state(), [0.5, 0.0, 0.0])
    assert np.isclose(value, 10.0)
    assert np.isclose(info["forward_reward"], 10.0)
    assert np.isclose(info["yaw_penalty"], 0.0)


def test_reward_uses_pitch_corrected_x_velocity_only():
    upright, _ = reward(state(), [0.5, 0.0, 0.0])
    tilted, _ = reward(state(quat=[np.cos(0.3), 0, np.sin(0.3), 0]),
                       [0.5, 0.0, 0.0])
    lateral, _ = reward(state(), [0.5, 2.0, 2.0])
    assert tilted < upright
    assert np.isclose(lateral, upright)


def test_reward_has_only_the_original_yaw_penalty_and_no_terminal_bonus():
    value, info = reward(state(gyro=[0, 0, 0.5]), [0.5, 0.0, 0.0])
    assert np.isclose(value, 10.0 * (1.0 - 0.1 * 0.5))
    terminated, _ = _locomotion_straight_reward(
        state(gyro=[0, 0, 0.5]), np.zeros(12), np.zeros(12),
        np.array([0.5, 0, 0]),
        True, return_info=True)
    assert np.isclose(terminated, value)


def test_stationary_robot_uses_original_forward_tolerance():
    value, info = reward(state(), [0.0, 0.0, 0.0])
    # WITP's tolerance has value_at_margin=0 and target=0.5, margin=1.0.
    # Therefore zero velocity receives 0.5 before the original x10 scale.
    assert np.isclose(value, 5.0)
    assert np.isclose(info["forward_reward"], 5.0)
