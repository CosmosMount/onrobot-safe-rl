"""Replay-buffer adapters owned by the training entrypoint."""

from __future__ import annotations

import os
from collections import deque
from typing import MutableMapping, Optional

import gymnasium as gym
import numpy as np
import torch

from rl.buffers.numpy_buffer import NpyUniformBuffer
from rl.utils.types import NDArray, Tensor


def _torch_device(device_type: str) -> torch.device:
    if device_type.startswith("cuda"):
        return torch.device(device_type if ":" in device_type else "cuda:0")
    return torch.device("cpu")


def _npy_replay_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    return root if ext == ".pt" else path


class FlashSACNumpyReplay(NpyUniformBuffer):
    """Use numpy storage while returning torch samples for FlashSAC updates."""

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
        device_type: str,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            n_step=n_step,
            gamma=gamma,
            max_length=max_length,
            min_length=min_length,
            sample_batch_size=sample_batch_size,
        )
        self._sample_device = _torch_device(device_type)

    def add_batch(self, transitions: MutableMapping[str, Tensor]) -> None:
        batch = self._validate_batch(transitions)
        batch_size = batch["observation"].shape[0]
        while len(self._pending) < batch_size:
            self._pending.append(deque())

        ready: list[dict[str, NDArray]] = []
        for env_idx in range(batch_size):
            queue = self._pending[env_idx]
            queue.append(self._make_single_env_transition(batch, env_idx))
            while len(queue) >= self._n_step:
                ready.append(self._build_n_step_transition(queue))
                queue.popleft()
            if not queue:
                continue
            latest = queue[-1]
            if bool(latest["terminated"]) or bool(latest["truncated"]):
                while queue:
                    ready.append(self._build_n_step_transition(queue))
                    queue.popleft()

        self._append_ready(ready)

    def sample(
        self,
        batch_size: Optional[int] = None,
        sample_idxs: Optional[Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch = super().sample(batch_size=batch_size, sample_idxs=sample_idxs)
        return {
            key: torch.as_tensor(value, device=self._sample_device)
            for key, value in batch.items()
        }

    def save(self, path: str) -> None:
        super().save(_npy_replay_path(path))

    def load(self, path: str) -> None:
        super().load(_npy_replay_path(path))


def install_flashsac_numpy_replay(agent, env, cfg) -> bool:
    if str(getattr(cfg, "agent_type", "")).lower() != "flashsac":
        return False
    agent._replay_buffer = FlashSACNumpyReplay(
        observation_space=env.observation_space,
        action_space=env.action_space,
        n_step=int(cfg.n_step),
        gamma=float(cfg.gamma),
        max_length=int(cfg.buffer_max_length),
        min_length=int(cfg.buffer_min_length),
        sample_batch_size=int(cfg.sample_batch_size),
        device_type=str(cfg.buffer_device_type),
    )
    return True
