"""Load layered Python runtime/training config (SI units)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / 'config/go2.yaml'

PAPER_ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4, dtype=np.float32)


def _load_float_array(node, name: str) -> np.ndarray:
    if not isinstance(node, list) or len(node) != 12:
        raise ValueError(f'{name} must be a YAML sequence of length 12')
    return np.asarray(node, dtype=np.float32)


@dataclass(frozen=True)
class Go2Config:
    init_qpos: np.ndarray          # rad
    action_offset: np.ndarray      # rad
    joint_min: np.ndarray            # rad
    joint_max: np.ndarray            # rad
    ipc_socket: str
    state_socket: str
    runtime_action_shm: str
    runtime_state_shm: str
    domain_id: int
    interface: str
    control_hz: float                # Hz
    success_orientation_rad: float   # rad
    fallen_risk_rad: float           # rad
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
        # joint_q, joint_dq, previous_requested_action, gyro, body_velocity,
        # quaternion.
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
    reset_joint_tolerance: float = 0.30
    recovery_stable_steps: int = 10
    standup_timeout_steps: int = 200
    max_joint_delta: float | None = None
    use_action_filter: bool = True
    explore_action_scale: float = 0.2
    max_steps: int = 1_000_000
    start_training: int = 1000
    batch_size: int = 256
    utd_ratio: int = 20
    buffer_size: int = 1_000_000
    terminal_replay_repeats: int = 1
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

    imu_node = root.get('imu') or {}

    return Go2Config(
        init_qpos=init_qpos,
        action_offset=action_offset,
        joint_min=joint_min,
        joint_max=joint_max,
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


def _parse_train(node: dict[str, Any]) -> tuple[TrainConfig, dict[str, Any]]:
    train_node = dict(node)
    flashsac = train_node.pop('flashsac', {})

    cfg = TrainConfig()
    for key, value in train_node.items():
        if not hasattr(cfg, key):
            raise ValueError(f'Unknown train config key: {key}')
        if key == 'max_joint_delta':
            value = _optional_float(value)
        elif key == 'wandb_run_name' and value == 'null':
            value = None
        setattr(cfg, key, value)

    return cfg, dict(flashsac)


def _parse_agent(train_cfg: TrainConfig,
                 flashsac_node: dict[str, Any]) -> DictConfig:
    if train_cfg.agent != 'flashsac':
        raise ValueError(f'Unsupported train.agent={train_cfg.agent!r}')
    values = dict(_FLASHSAC_DEFAULTS)
    values.update(flashsac_node)
    values.update({
        'agent_type': 'flashsac',
        'seed': train_cfg.seed,
        'buffer_max_length': train_cfg.buffer_size,
        'sample_batch_size': train_cfg.batch_size,
    })
    if values.get('temp_target_entropy') == 'null':
        values['temp_target_entropy'] = 0.0
    return OmegaConf.create(values)


def parse_app_config(root: dict[str, Any]) -> tuple[Go2Config, TrainConfig,
                                                    DictConfig]:
    train_node = root.get('train')
    if not train_node:
        raise ValueError('train section missing in config')

    robot_cfg = _parse_robot(root)
    train_cfg, flashsac_node = _parse_train(dict(train_node))
    return robot_cfg, train_cfg, _parse_agent(train_cfg, flashsac_node)


def _load_profile_root(overlay_path: Path) -> dict[str, Any]:
    overlay = _load_yaml(overlay_path)
    reward_profile = str(overlay.get('reward_profile', 'baseline'))
    if reward_profile not in {'baseline', 'upstream'}:
        raise ValueError(f'Unknown reward profile: {reward_profile}')
    return _load_layered_config(
        REPO_ROOT / 'config/common.yaml',
        REPO_ROOT / f'config/rewards/{reward_profile}.yaml',
        overlay_path,
    )


def load_app_config(
        path: str | Path | None = None,
        *,
        profile: str = 'go2') -> tuple[Go2Config, TrainConfig, DictConfig]:
    if path is not None:
        return parse_app_config(_load_profile_root(Path(path)))

    profile_paths = {
        'go2': DEFAULT_CONFIG_PATH,
        'simulation': REPO_ROOT / 'config/simulation.yaml',
        'real_robot': REPO_ROOT / 'config/real_robot.yaml',
    }
    if profile not in profile_paths:
        raise ValueError(f'Unknown config profile: {profile}')
    return parse_app_config(_load_profile_root(profile_paths[profile]))
