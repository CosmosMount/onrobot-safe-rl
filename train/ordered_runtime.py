"""Ordered runtime protocol used by the asynchronous 50 Hz collector.

This module only owns transport and protocol validation.  Policy inference and
learning deliberately live above it so a slow learner cannot make this layer
drop a state or a one-cycle terminal message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from runtime.inference.transport import (
    SharedMemoryRingQueue,
    SharedMemorySender,
)


class RuntimeProtocolError(RuntimeError):
    """The runtime stream is missing, reordered, or incorrectly attributed."""


@dataclass(frozen=True)
class RuntimeEnvelope:
    runtime_step_id: int
    episode_id: int
    episode_step: int
    applied_action_id: int
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class OrderedRuntimeChannel:
    """Single-consumer ordered state stream plus latest-action command path."""

    def __init__(self, action_key: str, state_key: str, *,
                 capacity: int = 2048, slot_size: int = 16 * 1024):
        self._action_tx = SharedMemorySender(action_key)
        self._state_rx = SharedMemoryRingQueue(
            f"{state_key}.ordered", capacity=capacity, slot_size=slot_size)
        self._next_action_id = 0
        self._last_runtime_step_id: int | None = None
        self._last_episode_id: int | None = None
        self._last_applied_action_id = -1

    def connect(self, *, timeout: float = 120.0) -> None:
        self._action_tx.wait_ready(timeout=timeout)
        self._state_rx.wait_ready(timeout=timeout)

    def send_action(self, action: np.ndarray) -> int:
        action_id = self._next_action_id
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains a non-finite value")
        self._action_tx.send({
            "action": np.clip(action, -1.0, 1.0),
            "action_id": action_id,
        })
        self._next_action_id += 1
        return action_id

    def clear_action(self) -> None:
        self._action_tx.send({"command": "clear"})

    def recv(self, *, timeout: float = 10.0) -> RuntimeEnvelope:
        message = self._state_rx.recv(timeout=timeout)
        info = dict(message.get("info", {}))
        required = (
            "runtime_step_id", "episode_id", "episode_step",
            "applied_action_id")
        missing = [key for key in required if key not in info]
        if missing:
            raise RuntimeProtocolError(
                f"runtime message is missing protocol fields: {missing}")
        envelope = RuntimeEnvelope(
            runtime_step_id=int(info["runtime_step_id"]),
            episode_id=int(info["episode_id"]),
            episode_step=int(info["episode_step"]),
            applied_action_id=int(info["applied_action_id"]),
            observation=np.asarray(message["observation"], dtype=np.float32),
            reward=float(message["reward"]),
            done=bool(message["done"]),
            info=info,
        )
        self._validate(envelope)
        return envelope

    def _validate(self, message: RuntimeEnvelope) -> None:
        if (
            self._last_runtime_step_id is not None
            and message.runtime_step_id != self._last_runtime_step_id + 1
        ):
            raise RuntimeProtocolError(
                "runtime step gap/reordering: "
                f"previous={self._last_runtime_step_id}, "
                f"current={message.runtime_step_id}")
        if (
            self._last_episode_id is not None
            and message.episode_id < self._last_episode_id
        ):
            raise RuntimeProtocolError(
                "episode id moved backwards: "
                f"previous={self._last_episode_id}, current={message.episode_id}")
        if (
            message.applied_action_id >= 0
            and message.applied_action_id < self._last_applied_action_id
        ):
            raise RuntimeProtocolError(
                "applied action id moved backwards: "
                f"previous={self._last_applied_action_id}, "
                f"current={message.applied_action_id}")
        if (
            message.applied_action_id >= 0
            and message.applied_action_id >= self._next_action_id
        ):
            raise RuntimeProtocolError(
                "runtime attributed a state to an action not sent by this "
                f"collector: applied={message.applied_action_id}, "
                f"next_local={self._next_action_id}")
        self._last_runtime_step_id = message.runtime_step_id
        self._last_episode_id = message.episode_id
        if message.applied_action_id >= 0:
            self._last_applied_action_id = message.applied_action_id

    @property
    def queue_depth(self) -> int:
        return self._state_rx.depth()

    def close(self) -> None:
        self.clear_action()
        self._action_tx.close()
        self._state_rx.close()
