"""Observation adaptation shared by the lossless SQRL collector."""

from __future__ import annotations

from collections import deque

import numpy as np


class ObservationStack:
    def __init__(self, frames: int, observation_dim: int):
        self.frames = int(frames)
        self.observation_dim = int(observation_dim)
        self._values: deque[np.ndarray] = deque(maxlen=self.frames)

    def reset(self, observation: np.ndarray) -> np.ndarray:
        value = self._validate(observation)
        self._values.clear()
        self._values.extend(value.copy() for _ in range(self.frames))
        return self.value()

    def append(self, observation: np.ndarray) -> np.ndarray:
        if not self._values:
            return self.reset(observation)
        self._values.append(self._validate(observation))
        return self.value()

    def value(self) -> np.ndarray:
        if len(self._values) != self.frames:
            raise RuntimeError("observation stack has not been reset")
        return np.concatenate(tuple(self._values)).astype(np.float32)

    def _validate(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(-1)
        if value.shape != (self.observation_dim,) or not np.all(np.isfinite(value)):
            raise ValueError("malformed raw Go2 observation")
        return value
