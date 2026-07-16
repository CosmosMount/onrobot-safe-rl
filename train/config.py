"""Load all settings from config/go2.yaml (SI units)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from common.config_schema import load_layered_config

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
    action_filter_highcut: float     # Hz
    sport_state_max_age_ms: float = 250.0
    sport_velocity_world_frame: bool = True  # unitree_mujoco framelinvel is world frame

    @property
    def num_joints(self) -> int:
        return int(self.init_qpos.shape[0])

    @property
    def obs_dim(self) -> int:
        # joint_q, joint_dq, previous_requested_action,
        # previous_executed_action, gyro, body_velocity, quaternion.
        return 4 * self.num_joints + 10

    @property
    def action_joint_min(self) -> np.ndarray:
        return self.init_qpos - self.action_offset

    @property
    def action_joint_max(self) -> np.ndarray:
        return self.init_qpos + self.action_offset


@dataclass
class TrainConfig:
    seed: int = 42
    control_frequency: float = 20.0
    max_episode_steps: int = 400
    reset_grace_steps: int = 20
    reset_hold_steps: int = 100
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
    log_interval: int = 100
    eval_interval: int = 1000
    eval_episodes: int = 1
    no_eval: bool = True
    save_dir: str = 'saved/checkpoints'
    use_tqdm: bool = True
    save_checkpoints: bool = True
    warmup: bool = True
    profile: bool = False
    benchmark_only: bool = False
    benchmark_steps: int = 200
    wandb: bool = False
    wandb_project: str = 'go2_walk'
    wandb_run_name: str | None = None


def _optional_float(value: Any) -> float | None:
    if value is None or value == 'null':
        return None
    return float(value)


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
        domain_id=int(root.get('domain_id', 1)),
        interface=str(root.get('interface', 'lo')),
        control_hz=float(root.get('control_hz', 500.0)),
        success_orientation_rad=_load_angle_rad(
            root, 'success_orientation_rad', 'success_orientation_deg',
            math.pi / 6),
        fallen_risk_rad=_load_angle_rad(
            root, 'fallen_risk_rad', 'fallen_risk_deg', math.pi / 9),
        imu_upright_acc_z=float(imu_node.get('upright_acc_z', 3.0)),
        imu_upside_down_acc_z=float(imu_node.get('upside_down_acc_z', -3.0)),
        imu_upright_up_cos=float(imu_node.get('upright_up_cos', 0.5)),
        imu_upside_down_up_cos=float(imu_node.get('upside_down_up_cos', -0.5)),
        joint_tolerance=float(root.get('joint_tolerance', 0.20)),
        move_speed=float(root.get('move_speed', 0.5)),
        reward_min_forward_vel=_optional_float(
            root.get('reward_min_forward_vel', None)),
        action_filter_highcut=float(root.get('action_filter_highcut', 4.0)),
        sport_state_max_age_ms=float(root.get('sport_state_max_age_ms',
                                              250.0)),
        sport_velocity_world_frame=bool(
            root.get('sport_velocity_world_frame', True)),
    )


def _parse_train(node: dict[str, Any]) -> tuple[TrainConfig, dict[str, Any]]:
    train_node = dict(node)
    droq = train_node.pop('droq', {})
    if not droq:
        raise ValueError('train.droq section missing in config/go2.yaml')

    cfg = TrainConfig()
    for key, value in train_node.items():
        if not hasattr(cfg, key):
            raise ValueError(f'Unknown train config key: {key}')
        if key == 'max_joint_delta':
            value = _optional_float(value)
        elif key == 'wandb_run_name' and value == 'null':
            value = None
        setattr(cfg, key, value)

    droq_cfg = dict(droq)
    if 'hidden_dims' in droq_cfg:
        droq_cfg['hidden_dims'] = tuple(droq_cfg['hidden_dims'])
    if droq_cfg.get('target_entropy') == 'null':
        droq_cfg['target_entropy'] = None
    return cfg, droq_cfg


def parse_app_config(root: dict[str, Any]) -> tuple[Go2Config, TrainConfig,
                                                    dict[str, Any]]:
    train_node = root.get('train')
    if not train_node:
        raise ValueError('train section missing in config/go2.yaml')

    robot_cfg = _parse_robot(root)
    train_cfg, droq_cfg = _parse_train(dict(train_node))
    return robot_cfg, train_cfg, droq_cfg


def load_app_config(
        path: str | Path | None = None,
        *,
        profile: str = 'go2') -> tuple[Go2Config, TrainConfig, dict[str, Any]]:
    if path is not None:
        config_path = Path(path)
        with config_path.open(encoding='utf-8') as f:
            return parse_app_config(yaml.safe_load(f))

    if profile == 'simulation':
        root = load_layered_config(REPO_ROOT / 'config/common.yaml',
                                   REPO_ROOT / 'config/simulation.yaml')
        return parse_app_config(root)
    if profile == 'real_robot':
        root = load_layered_config(REPO_ROOT / 'config/common.yaml',
                                   REPO_ROOT / 'config/real_robot.yaml')
        return parse_app_config(root)
    if profile != 'go2':
        raise ValueError(f'Unknown config profile: {profile}')

    with DEFAULT_CONFIG_PATH.open(encoding='utf-8') as f:
        root = yaml.safe_load(f)
    return parse_app_config(root)
