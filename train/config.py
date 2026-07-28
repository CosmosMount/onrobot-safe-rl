"""Load all settings from config/go2.yaml (SI units)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from common.config_schema import load_layered_config, load_yaml

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
    reward_upright_min_cos: float    # minimum body-up cosine for forward reward
    fall_terminal_penalty: float     # reward added on true failure termination
    action_filter_highcut: float     # Hz
    safety_joint_limit_margin_rad: float = 0.10
    safety_joint_velocity_rad_s: float = 25.0
    safety_torque_nm: float = 35.0
    safety_power_w: float = 350.0
    safety_impact_accel_m_s2: float = 8.0
    safety_angular_velocity_rad_s: float = 5.0
    safety_base_height_m: float = 0.18
    sport_state_max_age_ms: float = 250.0
    sport_velocity_world_frame: bool = True  # unitree_mujoco framelinvel is world frame
    cmd_speed_obs_scale: float = 1.0  # obs stores move_speed / scale

    @property
    def num_joints(self) -> int:
        return int(self.init_qpos.shape[0])

    @property
    def obs_dim(self) -> int:
        # joint_q, joint_dq, previous_requested_action,
        # previous_executed_action, gyro, body_velocity, quaternion,
        # cmd_speed (normalized).
        return 4 * self.num_joints + 11

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
    control_frequency: float = 20.0
    max_episode_steps: int = 400
    reset_grace_steps: int = 20
    reset_hold_steps: int = 220
    reset_joint_tolerance: float = 0.30
    recovery_stable_steps: int = 10
    standup_timeout_steps: int = 200
    abort_on_unstable_reset: bool = True
    max_joint_delta: float | None = None
    use_action_filter: bool = True
    explore_action_scale: float = 0.2
    max_steps: int = 1_000_000
    start_training: int = 1000
    batch_size: int = 256
    utd_ratio: int = 20
    buffer_size: int = 1_000_000
    safety_replay_enabled: bool = True
    safety_recent_capacity: int = 20_000
    safety_failure_capacity: int = 20_000
    safety_boundary_capacity: int = 20_000
    safety_recovery_capacity: int = 10_000
    safety_failure_history: int = 32
    safety_failure_horizons: tuple[int, ...] = (8, 16, 32)
    safety_critic_enabled: bool = True
    safety_critic_batch_size: int = 256
    safety_critic_update_interval: int = 1
    safety_critic_learning_rate: float = 3e-4
    safety_critic_hidden_dims: tuple[int, ...] = (256, 256)
    safety_critic_ensemble_size: int = 1
    safety_discount: float = 0.99
    safety_critic_tau: float = 0.005
    safety_critic_n_step: int = 8
    safety_future_loss_weight: float = 0.5
    # CQL-style risk overestimation. Zero preserves the Stage-1 baseline.
    safety_conservative_weight: float = 0.0
    safety_conservative_num_actions: int = 4
    safety_calibration_interval: int = 100
    safety_calibration_min_samples: int = 128
    safety_eval_horizon: int = 32
    safety_eval_output_dir: str = 'saved/safety_evaluation'
    safety_eval_min_auroc: float = 0.70
    safety_eval_min_warning_delta: float = 0.05
    safety_collection_action_noise_std: float = 0.35
    # Legacy Stage-2 heuristic shield knobs (withdrawn).
    safety_mask_num_candidates: int = 32
    safety_mask_epsilon: float = 0.30
    safety_mask_local_action_std: float = 0.15
    safety_mask_risk_penalty: float = 1.0
    safety_mask_action_delta_penalty: float = 1.0
    safety_mask_fallback_contraction: float = 0.9
    safety_mask_fallback_emergency_risk: float = 0.5
    # SQRL Route A (constrained sampling + optional Lagrange actor).
    sqrl_enabled: bool = False
    sqrl_phase: str = 'pretrain'  # pretrain | finetune
    sqrl_epsilon: float = 0.20
    sqrl_num_candidates: int = 64
    sqrl_lagrange_lr: float = 1.0e-4
    sqrl_qsafe_recent_only: bool = True
    # After warm-start, train Q_safe without constraining pi for this many
    # steps so a fresh critic is not 100% no-safe (which collapses locomotion).
    sqrl_activation_steps: int = 1000
    # Do not constrain pi until the safety critic separates labels on the
    # training batch (prevents SQRL with a collapsed Q_safe).
    sqrl_min_auroc: float = 0.70
    sqrl_min_pos_neg_gap: float = 0.05
    sqrl_max_ece: float = 0.15
    sqrl_max_brier: float = 0.20
    sqrl_min_gate_samples: int = 128
    sqrl_gate_candidate_window: int = 128
    sqrl_max_no_safe_rate: float = 0.50
    sqrl_min_candidate_range: float = 0.02
    # Linearly anneal epsilon from this value down to sqrl_epsilon over
    # sqrl_epsilon_anneal_steps after activation (1.0 = no filtering).
    sqrl_epsilon_start: float = 0.80
    sqrl_epsilon_anneal_steps: int = 1000
    # Match held-out candidate-noise eval during constrained rollouts.
    sqrl_train_candidate_noise_std: float = 0.0
    sqrl_local_candidate_count: int = 8
    sqrl_local_action_std: float = 0.10
    sqrl_fallback_contraction: float = 0.90
    sqrl_fallback_emergency_risk: float = 0.80
    sqrl_uncertainty_penalty: float = 1.0
    sqrl_support_gate_enabled: bool = False
    sqrl_min_behavior_log_prob_per_dim: float = -4.0
    sqrl_max_nominal_action_distance: float = 1.0
    # Independent B critic validates (but never searches over) A's choice.
    sqrl_double_critic_enabled: bool = False
    sqrl_validation_improvement_margin: float = 0.02
    # Offline MuJoCo branch evidence is required before action control.
    sqrl_control_gate_required: bool = True
    sqrl_control_metrics_path: str | None = None
    sqrl_control_min_pairwise_accuracy: float = 0.65
    sqrl_control_max_false_safe_rate: float = 0.10
    sqrl_control_min_coverage: float = 0.10
    sqrl_control_min_failure_reduction: float = 0.01
    warm_start_checkpoint: str | None = None
    log_interval: int = 100
    metrics_interval: int = 1
    rolling_summary_window: int = 1000
    eval_interval: int = 1000
    eval_episodes: int = 1
    no_eval: bool = True
    save_dir: str = 'saved/checkpoints'
    checkpoint_interval: int = 1000
    use_tqdm: bool = True
    save_checkpoints: bool = True
    resume_checkpoint: bool = False
    warmup: bool = True
    profile: bool = False
    pipeline_updates: bool = False
    split_update_interval_steps: int = 4
    split_max_pending_updates: int = 2
    benchmark_only: bool = False
    benchmark_steps: int = 200
    wandb: bool = False
    wandb_project: str = 'go2_walk'
    wandb_run_name: str | None = None
    # Multi-speed command curriculum (samples move_speed each episode).
    cmd_speed_curriculum: bool = False
    cmd_speed_min: float = 0.30
    cmd_speed_max: float = 1.0
    # Legacy linear-curriculum field retained for config compatibility.
    cmd_speed_curriculum_steps: int = 12_000
    cmd_speed_increment: float = 0.05
    cmd_speed_frontier_probability: float = 0.50
    cmd_speed_promotion_window: int = 8
    cmd_speed_min_episode_length: float = 300.0
    cmd_speed_min_velocity_ratio: float = 0.75
    cmd_speed_max_fall_rate: float = 0.125
    cmd_speed_new_stage_exploration_scale: float = 0.50
    cmd_speed_exploration_recovery_episodes: int = 4


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

    safety_node = root.get('safety_logging') or {}

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
        reward_upright_min_cos=float(
            root.get('reward_upright_min_cos',
                     math.cos(math.pi / 6))),
        fall_terminal_penalty=float(
            root.get('fall_terminal_penalty', -10.0)),
        action_filter_highcut=float(root.get('action_filter_highcut', 4.0)),
        safety_joint_limit_margin_rad=float(
            safety_node.get('joint_limit_margin_rad', 0.10)),
        safety_joint_velocity_rad_s=float(
            safety_node.get('joint_velocity_rad_s', 25.0)),
        safety_torque_nm=float(safety_node.get('torque_nm', 35.0)),
        safety_power_w=float(safety_node.get('power_w', 350.0)),
        safety_impact_accel_m_s2=float(
            safety_node.get('impact_accel_m_s2', 8.0)),
        safety_angular_velocity_rad_s=float(
            safety_node.get('angular_velocity_rad_s', 5.0)),
        safety_base_height_m=float(
            safety_node.get('base_height_m', 0.18)),
        sport_state_max_age_ms=float(root.get('sport_state_max_age_ms',
                                              250.0)),
        sport_velocity_world_frame=bool(
            root.get('sport_velocity_world_frame', True)),
        cmd_speed_obs_scale=float(root.get('cmd_speed_obs_scale', 1.0)),
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
        elif key in (
                'wandb_run_name', 'warm_start_checkpoint',
                'sqrl_control_metrics_path') and value in ('null', None):
            value = None
        elif key == 'safety_critic_hidden_dims':
            value = tuple(value)
        elif key == 'safety_failure_horizons':
            value = tuple(int(item) for item in value)
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
        overlay_path = REPO_ROOT / 'config/simulation.yaml'
        overlay = load_yaml(overlay_path)
        reward_profile = str(overlay.get('reward_profile', 'baseline'))
        if reward_profile not in {'baseline', 'upstream'}:
            raise ValueError(f'Unknown reward profile: {reward_profile}')
        root = load_layered_config(
            REPO_ROOT / 'config/common.yaml',
            REPO_ROOT / f'config/rewards/{reward_profile}.yaml',
            overlay_path,
        )
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
