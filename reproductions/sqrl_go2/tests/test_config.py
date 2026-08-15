from reproductions.sqrl_go2.config import load_config
from train.config import load_app_config


def test_frozen_pretrain_and_target_tasks_differ_only_in_command_and_phase():
    pre = load_config("reproductions/sqrl_go2/config/pretrain_030.yaml")
    target = load_config("reproductions/sqrl_go2/config/target_040.yaml")
    assert pre.phase == "pretrain" and target.phase == "target"
    assert pre.move_speed == 0.30 and target.move_speed == 0.40
    assert pre.environment == target.environment
    assert pre.replay == target.replay
    assert pre.sqrl == target.sqrl
    assert pre.training == target.training
    assert pre.development_protocol == target.development_protocol
    assert pre.development_protocol.pretrain_steps == 25_000
    assert pre.development_protocol.target_steps == 10_000
    assert pre.development_protocol.pretrain_seeds == (0, 1, 2)
    assert pre.stacked_observation_dim == 230


def test_phase_files_are_also_exact_runtime_overlays():
    pre_robot, pre_train, _ = load_app_config(
        path="reproductions/sqrl_go2/config/pretrain_030.yaml")
    target_robot, target_train, _ = load_app_config(
        path="reproductions/sqrl_go2/config/target_040.yaml")
    assert pre_robot.move_speed == pre_robot.reward_command_vx == 0.30
    assert target_robot.move_speed == target_robot.reward_command_vx == 0.40
    assert pre_robot.reward_profile == target_robot.reward_profile == "locomotion_straight"
    assert pre_train.control_frequency == target_train.control_frequency == 50.0
    assert pre_train.max_episode_steps == target_train.max_episode_steps == 500
    assert not pre_train.use_action_filter and not target_train.use_action_filter
    assert pre_train.max_joint_delta is None and target_train.max_joint_delta is None
