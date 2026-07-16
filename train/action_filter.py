"""Butterworth low-pass filter on absolute joint targets (walk_in_the_park)."""

from __future__ import annotations

import collections

import numpy as np
from scipy.signal import butter


class ActionFilterButter:
    """Low-pass Butterworth filter on joint position commands."""

    def __init__(self,
                 num_joints: int,
                 sampling_rate: float,
                 highcut: float = 4.0,
                 order: int = 2):
        self.num_joints = num_joints
        self._hist_len = order
        nyq = 0.5 * sampling_rate
        high = highcut / nyq
        b, a = butter(order, high, btype='low')
        self._b = np.stack([b] * num_joints)
        self._a = np.stack([a] * num_joints)
        self._b /= self._a[:, :1]
        self._a /= self._a[:, :1]
        self._xhist: collections.deque[np.ndarray] = collections.deque(
            maxlen=self._hist_len)
        self._yhist: collections.deque[np.ndarray] = collections.deque(
            maxlen=self._hist_len)
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
        y = (self._b[:, 0] * x
             + np.sum(self._b[:, 1:] * xs, axis=-1)
             - np.sum(self._a[:, 1:] * ys, axis=-1))
        self._xhist.appendleft(x.copy())
        self._yhist.appendleft(y.copy())
        return y.astype(np.float32)
