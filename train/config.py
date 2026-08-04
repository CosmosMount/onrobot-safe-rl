"""Load layered Python runtime/training config (SI units)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from robots.go2.joint_layout import validate_joint_vector

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / 'config/go2.yaml'

PAPER_ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4, dtype=np.float32)


def _load_float_array(node, name: str) -> np.ndarray:
    return validate_joint_vector(name, node)


@dataclass(frozen=True)
class Go2Config:
    init_qpos: np.ndarray          # rad
    action_offset: np.ndarray      # rad
    joint_min: np.ndarray            # rad
    joint_max: np.ndarray            # rad
    kp: np.ndarray                    # Nm/rad, explicit 12-vector view
    kd: np.ndarray                    # Nms/rad, explicit 12-vector view
    reward_profile: str
    reward_forward_weight: float  # deprecated; unused by locomotion_straight
    reward_upright_exponent: float
    reward_roll_pitch_rate_weight: float
    reward_lateral_velocity_weight: float
    reward_vertical_velocity_weight: float
    reward_action_rate_weight: float
    reward_leg_activity_balance_weight: float
    reward_angular_rate_scale: float
    reward_lateral_velocity_scale: float
    reward_vertical_velocity_scale: float
    reward_vertical_velocity_penalty_max: float
    reward_action_rate_scale: float
    reward_action_rate_penalty_max: float
    reward_action_magnitude_weight: float
    reward_action_magnitude_scale: float
    reward_action_magnitude_penalty_max: float
    reward_leg_activity_epsilon: float
    reward_leg_balance_speed_gate: float
    reward_leg_action_activity_scale: float
    reward_leg_joint_velocity_scale: float
    leg_activity_ema_beta: float
    leg_activity_minimum_action_delta_rms: float
    leg_activity_minimum_joint_velocity_rms: float
    leg_activity_inactive_grace_steps: int
    reward_tracking_lin_vel_weight: float
    reward_tracking_ang_vel_weight: float
    reward_similar_to_default_weight: float
    reward_base_height_weight: float
    reward_tracking_sigma: float
    reward_base_height_target: float
    reward_command_vx: float
    reward_orientation_weight: float
    reward_dof_velocity_weight: float
    reward_dof_velocity_scale: float
    reward_joint_limit_weight: float
    reward_joint_limit_margin: float
    reward_forward_tilt_weight: float
    reward_forward_pitch_rate_weight: float
    reward_orientation_penalty_max: float
    reward_pitch_free_rad: float
    reward_pitch_danger_rad: float
    reward_pitch_rate_scale: float
    reward_pitch_rate_penalty_max: float
    ipc_socket: str
    state_socket: str
    runtime_action_shm: str
    runtime_state_shm: str
    domain_id: int
    interface: str
    control_hz: float                # Hz
    success_orientation_rad: float   # rad
    fallen_risk_rad: float           # rad
    fallen_orientation_rad: float    # rad
    imu_upright_acc_z: float         # m/s²
    imu_upside_down_acc_z: float     # m/s²
    imu_upright_up_cos: float        # dimensionless
    imu_upside_down_up_cos: float    # dimensionless
    joint_tolerance: float           # rad (L2 joint error)
    move_speed: float                # m/s
    reward_min_forward_vel: Optional[float]  # m/s; None matches upstream reward
    reward_upright_min_cos: float    # minimum body-up cosine for forward reward
    fall_terminal_penalty: float     # reward added on true failure termination
    action_filter_highcut: float     # Hz
    sport_state_max_age_ms: float = 250.0
    sport_velocity_world_frame: bool = True  # unitree_mujoco framelinvel is world frame

    @property
    def num_joints(self) -> int:
        return int(self.init_qpos.shape[0])

    @property
    def obs_dim(self) -> int:
        # joint_q, joint_dq, gyro, body_velocity, quaternion,
        # previous_action_q_target (absolute joint target, not normalized action).
        return 3 * self.num_joints + 10

    @property
    def action_joint_min(self) -> np.ndarray:
        return self.init_qpos - self.action_offset

    @property
    def action_joint_max(self) -> np.ndarray:
        return self.init_qpos + self.action_offset


@dataclass
class TrainConfig:
    experiment_name: str = 'default'
    seed: int = 42
    control_frequency: float = 50.0
    max_episode_steps: int = 400
    reset_grace_steps: int = 20
    reset_hold_steps: int = 220
    reset_joint_tolerance: float = 0.30
    recovery_stable_steps: int = 10
    standup_timeout_steps: int = 200
    abort_on_unstable_reset: bool = True
    max_joint_delta: np.ndarray | None = None
    use_action_filter: bool = True
    explore_action_scale: float = 0.2
    max_steps: int = 1_000_000
    start_training: int = 1000
    batch_size: int = 256
    utd_ratio: int = 20
    buffer_size: int = 1_000_000
    terminal_replay_repeats: int = 1
    async_collection: bool = False
    inference_sync_updates: int = 20
    async_transition_queue_capacity: int = 8192
    log_interval: int = 100
    metrics_interval: int = 1
    rolling_summary_window: int = 1000
    save_dir: str = 'saved/checkpoints'
    checkpoint_interval: int = 1000
    use_tqdm: bool = True
    save_checkpoints: bool = True
    resume_checkpoint: bool = False
    benchmark_only: bool = False
    benchmark_steps: int = 200
    wandb: bool = False
    wandb_project: str = 'go2_walk'
    wandb_run_name: str | None = None
    agent: str = 'flashsac'


def _optional_float(value: Any) -> float | None:
    if value is None or value == 'null':
        return None
    return float(value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _load_layered_config(*paths: str | Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path in paths:
        root = _deep_merge(root, _load_yaml(path))
    return root


def _load_angle_rad(root: dict[str, Any], rad_key: str, deg_key: str,
                    default_rad: float) -> float:
    if rad_key in root:
        return float(root[rad_key])
    if deg_key in root:
        return float(math.radians(root[deg_key]))
    return default_rad


def _parse_robot(root: dict[str, Any]) -> Go2Config:
    init_qpos = _load_float_array(root['init_qpos'], 'init_qpos')
    joint_min = _load_float_array(root['joint_min'], 'joint_min')
    joint_max = _load_float_array(root['joint_max'], 'joint_max')
    if 'action_offset' in root:
        action_offset = _load_float_array(root['action_offset'], 'action_offset')
    else:
        action_offset = PAPER_ACTION_OFFSET.copy()
    if np.any(np.abs(action_offset) < 1e-6):
        raise ValueError('action_offset must not contain zero or near-zero values')
    if np.any(joint_min >= joint_max):
        raise ValueError('joint_min must be strictly less than joint_max')
    if np.any((init_qpos < joint_min) | (init_qpos > joint_max)):
        raise ValueError('init_qpos must lie within joint_min and joint_max')

    kp = validate_joint_vector('kp', root.get('kp', 60.0), allow_scalar=True)
    kd = validate_joint_vector('kd', root.get('kd', 10.0), allow_scalar=True)
    reward_profile = str(root.get('reward_profile', 'upstream'))
    if reward_profile not in {'baseline', 'upstream', 'locomotion_straight'}:
        raise ValueError(f'Unknown reward profile: {reward_profile}')
    def positive(name: str, default: float) -> float:
        value = float(root.get(name, default))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and non-negative')
        return value
    def scale(name: str, default: float) -> float:
        value = float(root.get(name, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        return value
    def finite_value(name: str, default: float) -> float:
        value = float(root.get(name, default))
        if not np.isfinite(value):
            raise ValueError(f'{name} must be finite')
        return value
    upright_exponent = positive('reward_upright_exponent', 1.0)
    upright_min = float(root.get('reward_upright_min_cos', -1.0))
    if not np.isfinite(upright_min) or not -1.0 <= upright_min <= 1.0:
        raise ValueError('reward_upright_min_cos must be in [-1, 1]')
    ema_beta = float(root.get('leg_activity', {}).get('ema_beta', 0.95))
    if not np.isfinite(ema_beta) or not 0.0 <= ema_beta < 1.0:
        raise ValueError('leg_activity.ema_beta must be in [0, 1)')
    grace = int(root.get('leg_activity', {}).get('inactive_grace_steps', 25))
    if grace < 0:
        raise ValueError('leg_activity.inactive_grace_steps must be non-negative')

    imu_node = root.get('imu') or {}

    return Go2Config(
        init_qpos=init_qpos,
        action_offset=action_offset,
        joint_min=joint_min,
        joint_max=joint_max,
        kp=kp,
        kd=kd,
        reward_profile=reward_profile,
        reward_forward_weight=positive('reward_forward_weight', 10.0),
        reward_upright_exponent=upright_exponent,
        reward_roll_pitch_rate_weight=positive('reward_roll_pitch_rate_weight', 0.20),
        reward_lateral_velocity_weight=positive('reward_lateral_velocity_weight', 0.10),
        reward_vertical_velocity_weight=positive('reward_vertical_velocity_weight', 0.10),
        reward_action_rate_weight=positive('reward_action_rate_weight', 0.05),
        reward_leg_activity_balance_weight=positive('reward_leg_activity_balance_weight', 0.05),
        reward_angular_rate_scale=scale('reward_angular_rate_scale', 2.0),
        reward_lateral_velocity_scale=scale('reward_lateral_velocity_scale', 0.5),
        reward_vertical_velocity_scale=scale('reward_vertical_velocity_scale', 0.5),
        reward_vertical_velocity_penalty_max=scale('reward_vertical_velocity_penalty_max', 4.0),
        reward_action_rate_scale=scale('reward_action_rate_scale', 0.25),
        reward_action_rate_penalty_max=scale('reward_action_rate_penalty_max', 4.0),
        reward_action_magnitude_weight=positive('reward_action_magnitude_weight', 0.02),
        reward_action_magnitude_scale=scale('reward_action_magnitude_scale', 0.60),
        reward_action_magnitude_penalty_max=scale('reward_action_magnitude_penalty_max', 2.0),
        reward_leg_activity_epsilon=scale('reward_leg_activity_epsilon', 0.01),
        reward_leg_balance_speed_gate=positive('reward_leg_balance_speed_gate', 0.05),
        reward_leg_action_activity_scale=scale('reward_leg_action_activity_scale', 0.05),
        reward_leg_joint_velocity_scale=scale('reward_leg_joint_velocity_scale', 1.0),
        leg_activity_ema_beta=ema_beta,
        leg_activity_minimum_action_delta_rms=positive('leg_activity_minimum_action_delta_rms', 0.01),
        leg_activity_minimum_joint_velocity_rms=positive('leg_activity_minimum_joint_velocity_rms', 0.05),
        leg_activity_inactive_grace_steps=grace,
        reward_tracking_lin_vel_weight=positive('reward_tracking_lin_vel_weight', 10.0),
        reward_tracking_ang_vel_weight=positive('reward_tracking_ang_vel_weight', 0.5),
        reward_similar_to_default_weight=positive('reward_similar_to_default_weight', 0.02),
        reward_base_height_weight=positive('reward_base_height_weight', 5.0),
        reward_tracking_sigma=scale('reward_tracking_sigma', 0.25),
        reward_base_height_target=positive('reward_base_height_target', 0.445),
        reward_command_vx=finite_value('reward_command_vx', float(root.get('move_speed', 0.5))),
        reward_orientation_weight=positive('reward_orientation_weight', 5.0),
        reward_dof_velocity_weight=positive('reward_dof_velocity_weight', 0.05),
        reward_dof_velocity_scale=scale('reward_dof_velocity_scale', 4.0),
        reward_joint_limit_weight=positive('reward_joint_limit_weight', 0.20),
        reward_joint_limit_margin=scale('reward_joint_limit_margin', 0.10),
        reward_forward_tilt_weight=positive('reward_forward_tilt_weight', 8.0),
        reward_forward_pitch_rate_weight=positive('reward_forward_pitch_rate_weight', 1.0),
        reward_orientation_penalty_max=scale('reward_orientation_penalty_max', 4.0),
        reward_pitch_free_rad=positive('reward_pitch_free_rad', 0.10),
        reward_pitch_danger_rad=positive('reward_pitch_danger_rad', 0.40),
        reward_pitch_rate_scale=scale('reward_pitch_rate_scale', 1.0),
        reward_pitch_rate_penalty_max=scale('reward_pitch_rate_penalty_max', 2.0),
        ipc_socket=root.get('ipc_socket', '/tmp/go2_policy.sock'),
        state_socket=root.get('state_socket', '/tmp/go2_policy.sock.state'),
        runtime_action_shm=root.get(
            'runtime_action_shm',
            root.get('runtime_action_socket', 'go2_runtime_action')),
        runtime_state_shm=root.get(
            'runtime_state_shm',
            root.get('runtime_state_socket', 'go2_runtime_state')),
        domain_id=int(root.get('domain_id', 1)),
        interface=str(root.get('interface', 'lo')),
        control_hz=float(root.get('control_hz', 500.0)),
        success_orientation_rad=_load_angle_rad(
            root, 'success_orientation_rad', 'success_orientation_deg',
            math.pi / 6),
        fallen_risk_rad=_load_angle_rad(
            root, 'fallen_risk_rad', 'fallen_risk_deg', math.pi / 9),
        fallen_orientation_rad=_load_angle_rad(
            root, 'fallen_orientation_rad', 'fallen_orientation_deg',
            math.pi / 3),
        imu_upright_acc_z=float(imu_node.get('upright_acc_z', 3.0)),
        imu_upside_down_acc_z=float(
            imu_node.get('upside_down_acc_z_on',
                         imu_node.get('upside_down_acc_z', -3.0))),
        imu_upright_up_cos=float(imu_node.get('upright_up_cos', 0.5)),
        imu_upside_down_up_cos=float(imu_node.get('upside_down_up_cos', -0.5)),
        joint_tolerance=float(root.get('joint_tolerance', 0.20)),
        move_speed=float(root.get('move_speed', 0.5)),
        reward_min_forward_vel=_optional_float(
            root.get('reward_min_forward_vel', None)),
        reward_upright_min_cos=float(
            root.get('reward_upright_min_cos',
                     math.cos(math.pi / 6))),
        fall_terminal_penalty=float(
            root.get('fall_terminal_penalty', -10.0)),
        action_filter_highcut=float(root.get('action_filter_highcut', 4.0)),
        sport_state_max_age_ms=float(root.get('sport_state_max_age_ms',
                                              250.0)),
        sport_velocity_world_frame=bool(
            root.get('sport_velocity_world_frame', True)),
    )


_FLASHSAC_DEFAULTS: dict[str, Any] = {
    'normalize_reward': False,
    'normalized_G_max': 10.0,
    'asymmetric_observation': False,
    'device_type': 'cuda',
    'buffer_device_type': 'cuda',
    'buffer_min_length': 1000,
    'learning_rate_init': 1.0e-6,
    'learning_rate_peak': 3.0e-4,
    'learning_rate_end': 1.0e-6,
    'learning_rate_warmup_rate': 1.0,
    'learning_rate_warmup_step': 1000,
    'learning_rate_decay_rate': 1.0,
    'learning_rate_decay_step': 1_000_000,
    'actor_num_blocks': 2,
    'actor_hidden_dim': 256,
    'actor_bc_alpha': 0.0,
    'actor_noise_zeta_mu': 2.0,
    'actor_noise_zeta_max': 10,
    'actor_update_period': 1,
    'actor_update_interval': 2,
    'actor_update_unit': 'critic_step',
    'critic_num_blocks': 2,
    'critic_hidden_dim': 256,
    'critic_num_bins': 101,
    'critic_min_v': -100.0,
    'critic_max_v': 1000.0,
    'critic_target_update_tau': 0.01,
    'temp_initial_value': 0.1,
    'temp_target_sigma': 0.2,
    'temp_target_entropy': 0.0,
    'gamma': 0.99,
    'n_step': 1,
    'use_compile': False,
    'compile_mode': 'reduce-overhead',
    'use_amp': False,
    'load_optimizer': True,
    'load_reward_normalizer': True,
}


_DROQ_DEFAULTS: dict[str, Any] = {
    'device_type': 'cuda',
    'buffer_device_type': 'cuda',
    'buffer_min_length': 1000,
    'actor_lr': 3.0e-4,
    'critic_lr': 3.0e-4,
    'temp_lr': 3.0e-4,
    'hidden_dims': [256, 256],
    'gamma': 0.99,
    'n_step': 1,
    'critic_target_update_tau': 0.005,
    'num_qs': 5,
    'num_min_qs': 2,
    'critic_dropout_rate': 0.01,
    'critic_layer_norm': True,
    'sampled_backup': True,
    'target_entropy': None,
    'temp_initial_value': 0.1,
    'actor_q_reduction': 'min',
    'target_q_min': -100.0,
    'target_q_max': 1000.0,
    'asymmetric_observation': False,
    'actor_update_period': 1,
    'actor_update_interval': 1,
    'actor_update_unit': 'policy_step',
    'use_compile': False,
    'compile_mode': 'reduce-overhead',
    'use_amp': False,
    'load_optimizer': True,
}

_LIVESAC_DEFAULTS: dict[str, Any] = {
    'device_type': 'cuda', 'buffer_device_type': 'cpu', 'buffer_min_length': 1000,
    'actor_lr': 3.0e-4, 'critic_lr': 3.0e-4, 'temp_lr': 3.0e-4,
    'actor_hidden_dims': [256, 256], 'critic_hidden_dim': 256, 'critic_expansion': 2,
    'critic_num_blocks': 1, 'critic_num_qs': 2, 'critic_num_bins': 101, 'critic_dropout_rate': 0.01,
    'critic_min_v': -5.0, 'critic_max_v': 5.0, 'critic_target_update_tau': 0.005,
    # These are the non-distributional parts of the DroQ update contract.
    'num_min_qs': 2, 'sampled_backup': True, 'target_q_min': -100.0,
    'target_q_max': 1000.0, 'actor_q_reduction': 'min',
    'normalize_reward': True, 'normalized_G_max': 5.0, 'gamma': 0.99, 'n_step': 1,
    'target_entropy': None, 'temp_initial_value': 0.1, 'asymmetric_observation': False,
    'actor_update_interval': 1, 'actor_update_unit': 'policy_step', 'use_compile': False,
    'compile_mode': 'reduce-overhead', 'use_amp': False, 'load_optimizer': True,
    'load_reward_normalizer': True,
}


def _parse_train(node: dict[str, Any]) -> tuple[TrainConfig, dict[str, dict[str, Any]]]:
    train_node = dict(node)
    flashsac = train_node.pop('flashsac', {})
    droq = train_node.pop('droq', {})
    safe_droq = train_node.pop('safe_droq', {})
    paper_sqrl = train_node.pop('paper_sqrl', {})
    livesac = train_node.pop('livesac', {})

    cfg = TrainConfig()
    for key, value in train_node.items():
        if not hasattr(cfg, key):
            raise ValueError(f'Unknown train config key: {key}')
        if key == 'max_joint_delta':
            value = (None if value is None or value == 'null'
                     else validate_joint_vector('max_joint_delta', value, allow_scalar=True))
        elif key == 'wandb_run_name' and value == 'null':
            value = None
        setattr(cfg, key, value)

    if cfg.utd_ratio <= 0:
        raise ValueError('utd_ratio must be positive')
    if cfg.max_joint_delta is not None and (
            not np.all(np.isfinite(cfg.max_joint_delta)) or np.any(cfg.max_joint_delta <= 0)):
        raise ValueError('max_joint_delta must be finite and positive when enabled')
    if cfg.async_collection and str(cfg.agent).lower() not in {
            'droq', 'flashsac', 'safe_droq', 'paper_sqrl', 'livesac'}:
        raise ValueError(
            'train.async_collection requires agent= droq, flashsac, '
            'safe_droq, paper_sqrl, or livesac')
    if str(cfg.agent).lower() == 'livesac' and int(cfg.utd_ratio) != 5:
        raise ValueError('LiveSAC v1.0 requires train.utd_ratio=5')

    return cfg, {
        'flashsac': dict(flashsac),
        'droq': dict(droq),
        'safe_droq': dict(safe_droq),
        'paper_sqrl': dict(paper_sqrl),
        'livesac': dict(livesac),
    }


def _parse_agent(train_cfg: TrainConfig, agent_nodes: dict[str, dict[str, Any]]) -> DictConfig:
    agent_type = str(train_cfg.agent).lower()
    if agent_type == 'flashsac':
        values = dict(_FLASHSAC_DEFAULTS)
        node = agent_nodes.get('flashsac', {})
        values.update(node)
        if 'actor_update_interval' not in node and 'actor_update_period' in node:
            values['actor_update_interval'] = values['actor_update_period']
            values['actor_update_unit'] = 'critic_step'
    elif agent_type == 'droq':
        values = dict(_DROQ_DEFAULTS)
        node = agent_nodes.get('droq', {})
        values.update(node)
        if 'actor_update_interval' not in node and 'actor_update_period' in node:
            values['actor_update_interval'] = values['actor_update_period']
            values['actor_update_unit'] = 'policy_step'
    elif agent_type == 'safe_droq':
        values = dict(_DROQ_DEFAULTS)
        values.update({
            'safety_mode': 'logging',
            'safety_hidden_dims': [256, 256],
            'safety_lr': 3.0e-4,
            'safety_gamma': 0.99,
            'safety_target_update_tau': 0.005,
            'safety_buffer_max_length': 100_000,
            'safety_buffer_min_length': 1000,
            'safety_batch_size': 256,
            'safety_failure_horizon': 32,
            # Legacy compatibility fields. New scheduling uses the explicit
            # safety_update_interval/unit pair below when configured.
            'safety_update_period': 5,
            'safety_update_interval': 1,
            'safety_update_unit': 'policy_step',
            'safety_updates_per_event': 1,
            'safety_future_loss_weight': 0.5,
            'safety_num_candidates': 32,
            'safety_epsilon': 0.20,
            'safety_activation_step': 1000,
            'safety_masking_ramp_steps': 0,
            'safety_min_risk_improvement': 0.0,
            'safety_max_action_rms': 2.0,
            'safety_contract_candidates': False,
            'safety_reward_q_margin': None,
            'safety_pretrained_path': None,
            'freeze_safety_critic': False,
        })
        # DroQ optimizer/network overrides remain shared, while safety-only
        # settings live under train.safe_droq.
        values.update(agent_nodes.get('droq', {}))
        values.update(agent_nodes.get('safe_droq', {}))
        safe_node = agent_nodes.get('safe_droq', {})
        if ('safety_update_interval' not in safe_node
                and 'safety_update_period' in safe_node):
            values['safety_update_interval'] = values['safety_update_period']
            values['safety_update_unit'] = 'critic_step'
    elif agent_type == 'paper_sqrl':
        values = dict(_DROQ_DEFAULTS)
        values.update({
            'sqrl_phase': 'pretrain',
            'safety_hidden_dims': [256, 256],
            'safety_lr': 3.0e-4,
            'safety_gamma': 0.7,
            'safety_target_update_tau': 0.005,
            'safety_replay_trajectories': 10,
            # Algorithm 1 updates after k complete trajectories; a trajectory
            # that terminates quickly in failure must still be trainable.
            'safety_replay_min_transitions': 1,
            'safety_batch_size': 256,
            'safety_update_period': 1,
            'safety_updates_per_cycle': 1,
            'safety_num_candidates': 100,
            'safety_boundary_pool_multiplier': 4,
            'safety_epsilon': 0.1,
            'pretrain_task_steps_per_cycle': 1000,
            'pretrain_safety_episodes_per_cycle': 1,
            'safety_lagrange_lr': 3.0e-4,
            'safety_lagrange_initial': 1.0,
            'safety_lagrange_max': 100.0,
            'finetune_update_safety_critic': False,
        })
        values.update(agent_nodes.get('droq', {}))
        values.update(agent_nodes.get('paper_sqrl', {}))
    elif agent_type == 'livesac':
        values = dict(_LIVESAC_DEFAULTS)
        values.update(agent_nodes.get('livesac', {}))
        if int(train_cfg.utd_ratio) != 5:
            raise ValueError('LiveSAC v1.0 requires train.utd_ratio=5')
    else:
        raise ValueError(f'Unsupported train.agent={train_cfg.agent!r}')

    values.update({
        'agent_type': agent_type,
        'seed': train_cfg.seed,
        'buffer_max_length': train_cfg.buffer_size,
        'sample_batch_size': train_cfg.batch_size,
    })
    if values.get('temp_target_entropy') == 'null':
        values['temp_target_entropy'] = 0.0
    if values.get('target_entropy') == 'null':
        values['target_entropy'] = None
    if values.get('num_min_qs') == 'null':
        values['num_min_qs'] = None
    if values.get('target_q_min') == 'null':
        values['target_q_min'] = None
    if values.get('target_q_max') == 'null':
        values['target_q_max'] = None
    if values.get('safety_pretrained_path') == 'null':
        values['safety_pretrained_path'] = None
    if values.get('safety_reward_q_margin') == 'null':
        values['safety_reward_q_margin'] = None
    if int(values.get('actor_update_interval', 1)) <= 0:
        raise ValueError('actor_update_interval must be positive')
    if values.get('actor_update_unit') not in {'policy_step', 'critic_step'}:
        raise ValueError('actor_update_unit must be policy_step or critic_step')
    if 'safety_update_interval' in values:
        if int(values['safety_update_interval']) <= 0:
            raise ValueError('safety_update_interval must be positive')
        if values.get('safety_update_unit') not in {'policy_step', 'critic_step'}:
            raise ValueError('safety_update_unit must be policy_step or critic_step')
        if int(values.get('safety_updates_per_event', 0)) < 0:
            raise ValueError('safety_updates_per_event must be non-negative')
    return OmegaConf.create(values)


def parse_app_config(root: dict[str, Any]) -> tuple[Go2Config, TrainConfig,
                                                    DictConfig]:
    train_node = root.get('train')
    if not train_node:
        raise ValueError('train section missing in config')

    robot_cfg = _parse_robot(root)
    train_cfg, agent_nodes = _parse_train(dict(train_node))
    return robot_cfg, train_cfg, _parse_agent(train_cfg, agent_nodes)


def _load_profile_root(overlay_path: Path) -> dict[str, Any]:
    overlay = _load_yaml(overlay_path)
    reward_profile = str(overlay.get('reward_profile', 'upstream'))
    if reward_profile not in {'baseline', 'upstream', 'locomotion_straight'}:
        raise ValueError(f'Unknown reward profile: {reward_profile}')
    return _load_layered_config(
        REPO_ROOT / 'config/common.yaml',
        REPO_ROOT / f'config/rewards/{reward_profile}.yaml',
        overlay_path,
    )


def load_app_config(
        path: str | Path | None = None,
        *,
        profile: str = 'go2',
        agent: str | None = None) -> tuple[Go2Config, TrainConfig, DictConfig]:
    if path is not None:
        root = _load_profile_root(Path(path))
        if agent is not None:
            root.setdefault('train', {})['agent'] = agent
        return parse_app_config(root)

    profile_paths = {
        'go2': DEFAULT_CONFIG_PATH,
        'simulation': REPO_ROOT / 'config/simulation.yaml',
        'real_robot': REPO_ROOT / 'config/real_robot.yaml',
        'go2_livesac': REPO_ROOT / 'config/go2_livesac.yaml',
    }
    if profile not in profile_paths:
        raise ValueError(f'Unknown config profile: {profile}')
    root = _load_profile_root(profile_paths[profile])
    if agent is not None:
        root.setdefault('train', {})['agent'] = agent
    return parse_app_config(root)
