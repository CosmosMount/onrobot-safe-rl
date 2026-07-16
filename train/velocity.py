"""Body-frame velocity helpers for DDS state."""

from __future__ import annotations

import numpy as np


def quat_world_to_body(v_world: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the IMU/body frame (w, x, y, z)."""
    w, x, y, z = quat
    norm = float(np.linalg.norm(quat))
    if norm < 1e-6:
        return np.asarray(v_world, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # Rotation matrix body->world, then transpose for world->body.
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    rot = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
                   dtype=np.float32)
    return (rot.T @ np.asarray(v_world, dtype=np.float32)).astype(np.float32)
