"""Deployment-compatible Go2 observation builders."""

from __future__ import annotations

import numpy as np

from train.config import Go2Config
from train.mdp.spec import OBSERVATION_SPECS, observation_dim
from train.obs import normalize_quat
from train.types import RobotState


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(quat)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
         2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float32)


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """World gravity [0, 0, -1] expressed in the body frame."""
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    rot_body_to_world = _quat_to_rotation_matrix(quat)
    return (rot_body_to_world.T @ gravity_world).astype(np.float32)


def _sanitize(obs: np.ndarray) -> np.ndarray:
    out = np.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0)
    return np.clip(out.astype(np.float32), -100.0, 100.0)


def build_observation(
    state: RobotState,
    previous_requested_action: np.ndarray,
    previous_sent_action: np.ndarray,
    cfg: Go2Config,
    *,
    spec: str,
    command: np.ndarray,
) -> np.ndarray:
    if spec == 'legacy_58':
        quat = normalize_quat(state.imu_quat)
        obs = np.concatenate([
            state.joint_q.astype(np.float32),
            state.joint_dq.astype(np.float32),
            previous_requested_action.astype(np.float32),
            previous_sent_action.astype(np.float32),
            state.imu_gyro.astype(np.float32),
            state.body_velocity.astype(np.float32),
            quat,
        ])
    elif spec == 'deploy_safe_57':
        command_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
        obs = np.concatenate([
            state.imu_gyro.astype(np.float32) * 0.25,
            projected_gravity(state.imu_quat),
            np.asarray(command, dtype=np.float32) * command_scale,
            (state.joint_q - cfg.init_qpos).astype(np.float32),
            state.joint_dq.astype(np.float32) * 0.05,
            previous_requested_action.astype(np.float32),
            previous_sent_action.astype(np.float32),
        ])
    else:
        raise ValueError(f'Unknown observation spec: {spec}')

    expected = observation_dim(spec, cfg)
    assert obs.shape == (expected,), (obs.shape, expected, spec)
    return _sanitize(obs)
