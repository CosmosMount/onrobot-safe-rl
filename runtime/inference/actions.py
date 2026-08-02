"""Action projection and filtering for runtime policy commands."""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter

from robots.go2.joint_layout import (
    CONTROLLER_TO_PYTHON_INDEX,
    PYTHON_TO_CONTROLLER_INDEX,
    validate_joint_vector,
)


class ActionFilterButter:
    """Low-pass Butterworth filter on absolute joint position commands."""

    def __init__(
        self,
        num_joints: int,
        sampling_rate: float,
        highcut: float = 4.0,
        order: int = 2,
    ):
        self.num_joints = num_joints
        self._hist_len = order
        nyq = 0.5 * sampling_rate
        high = highcut / nyq
        b, a = butter(order, high, btype="low")
        self._b = np.stack([b] * num_joints)
        self._a = np.stack([a] * num_joints)
        self._b /= self._a[:, :1]
        self._a /= self._a[:, :1]
        self._xhist: collections.deque[np.ndarray] = collections.deque(maxlen=self._hist_len)
        self._yhist: collections.deque[np.ndarray] = collections.deque(maxlen=self._hist_len)
        self.reset()

    def reset(self) -> None:
        self._xhist.clear()
        self._yhist.clear()
        for _ in range(self._hist_len):
            self._xhist.appendleft(np.zeros(self.num_joints, dtype=np.float32))
            self._yhist.appendleft(np.zeros(self.num_joints, dtype=np.float32))

    def init_history(self, qpos: np.ndarray) -> None:
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        for i in range(self._hist_len):
            self._xhist[i] = q.copy()
            self._yhist[i] = q.copy()

    def filter(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        xs = np.stack(list(self._xhist), axis=-1)
        ys = np.stack(list(self._yhist), axis=-1)
        y = (
            self._b[:, 0] * x
            + np.sum(self._b[:, 1:] * xs, axis=-1)
            - np.sum(self._a[:, 1:] * ys, axis=-1)
        )
        self._xhist.appendleft(x.copy())
        self._yhist.appendleft(y.copy())
        return y.astype(np.float32)


def action_to_qpos(
    action: np.ndarray,
    *,
    init_qpos: np.ndarray,
    action_offset: np.ndarray,
    joint_min: np.ndarray,
    joint_max: np.ndarray,
) -> np.ndarray:
    """Map normalized policy action [-1, 1] to clipped joint targets."""
    clipped = np.clip(validate_joint_vector("action", action), -1.0, 1.0)
    desired_policy = init_qpos + clipped * action_offset
    desired = desired_policy[PYTHON_TO_CONTROLLER_INDEX][CONTROLLER_TO_PYTHON_INDEX]
    action_min = np.maximum(joint_min, init_qpos - action_offset)
    action_max = np.minimum(joint_max, init_qpos + action_offset)
    return np.clip(desired, action_min, action_max).astype(np.float32)


def qpos_to_action(
    q_target: np.ndarray,
    *,
    init_qpos: np.ndarray,
    action_offset: np.ndarray,
) -> np.ndarray:
    """Map absolute joint targets back into normalized policy action space."""
    action = (np.asarray(q_target, dtype=np.float32) - init_qpos) / np.maximum(action_offset, 1e-6)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class ActionProjection:
    action_requested: np.ndarray
    action_executed: np.ndarray
    action_q_target: np.ndarray


@dataclass
class ActionApplier:
    """Convert normalized policy actions into executable joint targets."""

    init_qpos: np.ndarray
    action_offset: np.ndarray
    joint_min: np.ndarray
    joint_max: np.ndarray
    max_joint_delta: float | None = None
    action_filter: ActionFilterButter | None = None

    def reset_filter(self) -> None:
        if self.action_filter is not None:
            self.action_filter.reset()
            self.action_filter.init_history(self.init_qpos)

    def init_filter_history(self, qpos: np.ndarray) -> None:
        if self.action_filter is not None:
            self.action_filter.init_history(qpos)

    def project(self, action: np.ndarray, current_qpos: np.ndarray) -> ActionProjection:
        action_requested = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        q_desired = action_to_qpos(
            action_requested,
            init_qpos=self.init_qpos,
            action_offset=self.action_offset,
            joint_min=self.joint_min,
            joint_max=self.joint_max,
        )
        if self.max_joint_delta is not None:
            delta = np.clip(q_desired - current_qpos, -self.max_joint_delta, self.max_joint_delta)
            q_send = current_qpos + delta
        else:
            q_send = q_desired
        if self.action_filter is not None:
            q_send = self.action_filter.filter(q_send)
        action_q_target = q_send.astype(np.float32)
        return ActionProjection(
            action_requested=action_requested.copy(),
            action_executed=self.executed_action(action_q_target),
            action_q_target=action_q_target.copy(),
        )

    def executed_action(self, q_target: np.ndarray) -> np.ndarray:
        return qpos_to_action(
            q_target,
            init_qpos=self.init_qpos,
            action_offset=self.action_offset,
        )
