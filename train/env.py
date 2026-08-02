"""Gym-style client for the standalone Go2 runtime process."""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from runtime.inference.dds import DdsConfig
from runtime.inference.transport import SharedMemoryReceiver, SharedMemorySender
from train.config import Go2Config


class Go2Env:
    def __init__(
        self,
        dds_config: DdsConfig,
        go2_config: Go2Config,
        control_frequency: float,
        max_episode_steps: int,
        seed: int = 42,
        **_,
    ):
        del dds_config, max_episode_steps
        self.cfg = go2_config
        self.control_frequency = float(control_frequency)
        self.control_dt = 1.0 / self.control_frequency
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.cfg.num_joints, dtype=np.float32),
            high=np.ones(self.cfg.num_joints, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space.seed(int(seed))
        self._action_tx = SharedMemorySender(self.cfg.runtime_action_shm)
        self._state_rx = SharedMemoryReceiver(self.cfg.runtime_state_shm)
        self._action_sequence = 0

    def reset(self, **_) -> np.ndarray:
        self._state_rx.bind()
        self._action_tx.wait_ready()
        self.clear_action()
        # The transport keeps only the latest value. Wait until runtime has
        # consumed the clear command before publishing the first action, or a
        # new client can inherit the previous client's episode counter.
        deadline = time.monotonic() + 10.0
        while True:
            message = self._recv_step()
            step_count = int(message.get("info", {}).get("step_count", 0))
            if step_count <= 1:
                return message["observation"]
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "runtime did not acknowledge episode reset")

    def step(self, action: np.ndarray, *, interaction_step: int | None = None,
             policy_update_step: int | None = None) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        self._state_rx.recv_latest()
        self._action_sequence += 1
        self._action_tx.send({
            "action": np.clip(action, -1.0, 1.0),
            "action_sequence": self._action_sequence,
            "action_interaction_step": -1 if interaction_step is None else int(interaction_step),
            "action_policy_update_step": -1 if policy_update_step is None else int(policy_update_step),
            "action_sent_time_ns": time.monotonic_ns(),
        })
        message = self._recv_step()
        return (
            np.asarray(message["observation"], dtype=np.float32),
            float(message["reward"]),
            bool(message["done"]),
            dict(message["info"]),
        )

    def sample_action(self) -> np.ndarray:
        return self.action_space.sample().astype(np.float32)

    def clear_action(self) -> None:
        # This must be a real mailbox payload. The transport-level {clear:
        # True} operation only erases the mailbox and cannot notify runtime.
        self._action_tx.send({"command": "clear"})

    def close(self) -> None:
        self.clear_action()
        self._state_rx.clear()
        self._action_tx.close()
        self._state_rx.close()

    def _recv_step(self) -> dict:
        message = self._state_rx.recv(timeout=10.0)
        obs = np.asarray(message["observation"], dtype=np.float32)
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                f"runtime observation shape {obs.shape} != {self.observation_space.shape}")
        message["observation"] = obs
        return message
