"""Canonical Go2 policy/controller joint layout.

Every Python and C++ boundary uses this policy order.  Keeping the mapping
explicit makes an accidental FR-only channel permutation visible at startup.
"""

from __future__ import annotations

from typing import Any

import numpy as np

JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)
LEG_NAMES = ("FR", "FL", "RR", "RL")
LEG_SLICES = {name: slice(i * 3, i * 3 + 3) for i, name in enumerate(LEG_NAMES)}
JOINT_TYPE_INDICES = {
    "hip": (0, 3, 6, 9),
    "thigh": (1, 4, 7, 10),
    "calf": (2, 5, 8, 11),
}

PYTHON_TO_CONTROLLER_INDEX = np.arange(12, dtype=np.int64)
CONTROLLER_TO_PYTHON_INDEX = np.arange(12, dtype=np.int64)


def validate_joint_vector(name: str, value: Any, *, allow_scalar: bool = False) -> np.ndarray:
    """Return a finite, explicit 12-element float32 joint vector."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 and allow_scalar:
        array = np.full(12, float(array), dtype=np.float32)
    elif array.size != 12:
        raise ValueError(f"{name} must have length 12 (got shape {array.shape})")
    else:
        array = array.reshape(12)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.astype(np.float32, copy=False)


def mapping_summary() -> str:
    return "\n".join(
        f"policy[{i}] {name:<9} -> controller[{int(PYTHON_TO_CONTROLLER_INDEX[i])}]"
        for i, name in enumerate(JOINT_NAMES)
    )


def leg_activity(values: np.ndarray) -> np.ndarray:
    """Per-leg RMS of a 12-vector, preserving the canonical leg order."""
    vector = validate_joint_vector("leg activity input", values)
    return np.asarray(
        [np.sqrt(np.mean(vector[LEG_SLICES[name]] ** 2)) for name in LEG_NAMES],
        dtype=np.float32,
    )
