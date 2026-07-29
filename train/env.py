"""Gym-style client for the standalone Go2 runtime process."""

from __future__ import annotations

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
        self._action_tx = SharedMemorySender(self.cfg.runtime_action_shm)
        self._state_rx = SharedMemoryReceiver(self.cfg.runtime_state_shm)

    def reset(self, **_) -> np.ndarray:
        self._state_rx.bind()
        self._action_tx.wait_ready()
        return self._recv_step()["observation"]

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        self._state_rx.recv_latest()
        self._action_tx.send({"action": np.clip(action, -1.0, 1.0)})
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
        self._action_tx.send({"clear": True})

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
