"""Common normalized action mapping for Go2."""

from __future__ import annotations

import numpy as np

from train.config import Go2Config


def action_to_qpos(action: np.ndarray, cfg: Go2Config) -> np.ndarray:
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    q_target = cfg.init_qpos + action * cfg.action_offset
    return np.clip(q_target, cfg.action_joint_min, cfg.action_joint_max)


def qpos_to_action(q_target: np.ndarray, cfg: Go2Config) -> np.ndarray:
    action = (
        np.asarray(q_target, dtype=np.float32) - cfg.init_qpos
    ) / np.maximum(cfg.action_offset, 1e-6)
    return np.clip(action, -1.0, 1.0).astype(np.float32)
