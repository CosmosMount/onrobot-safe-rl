from __future__ import annotations

import os
from collections import deque
from typing import MutableMapping

import numpy as np
import torch

from rl.utils.types import Tensor


class RecentTrajectoryReplay:
    """The small on-policy trajectory buffer D_safe from SQRL.

    It retains the latest complete constrained-policy trajectories. Sampling
    is uniform over their transitions: there is deliberately no class
    balancing, failure-window relabeling, or near-failure auxiliary target.
    """

    def __init__(self, *, max_trajectories: int, min_transitions: int,
                 batch_size: int, device: torch.device, seed: int):
        if max_trajectories <= 0 or min_transitions < 0 or batch_size <= 0:
            raise ValueError("invalid SQRL safety replay sizes")
        self.max_trajectories = int(max_trajectories)
        self.min_transitions = int(min_transitions)
        self.batch_size = int(batch_size)
        self.device = device
        self._trajectories: deque[list[dict[str, np.ndarray | float]]] = (
            deque(maxlen=self.max_trajectories))
        self._current: list[dict[str, np.ndarray | float]] = []
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return sum(len(trajectory) for trajectory in self._trajectories)

    @property
    def trajectory_count(self) -> int:
        return len(self._trajectories)

    @property
    def failure_count(self) -> int:
        return sum(
            int(float(item["unsafe"]) >= 0.5)
            for trajectory in self._trajectories for item in trajectory)

    def add_batch(self, transition: MutableMapping[str, Tensor]) -> bool:
        repeat = int(np.asarray(
            transition.get("replay_repeat_index", [0])).reshape(-1)[0])
        if repeat:
            return False
        terminated = bool(np.asarray(transition["terminated"]).reshape(-1)[0])
        truncated = bool(np.asarray(transition["truncated"]).reshape(-1)[0])
        self._current.append({
            "observation": np.asarray(
                transition["observation"], dtype=np.float32)[0].copy(),
            "action": np.asarray(
                transition["action"], dtype=np.float32)[0].copy(),
            "next_observation": np.asarray(
                transition["next_observation"], dtype=np.float32)[0].copy(),
            # I(s') is used because the environment reports the failure on
            # the transition entering the terminal unsafe state.
            "unsafe": float(np.asarray(
                transition.get("unsafe_label", transition["terminated"])
            ).reshape(-1)[0]),
            "done": float(terminated or truncated),
        })
        if not (terminated or truncated):
            return False
        self._trajectories.append(self._current)
        self._current = []
        return True

    def can_sample(self) -> bool:
        return len(self) >= self.min_transitions and self.trajectory_count > 0

    def sample(self) -> dict[str, torch.Tensor]:
        if not self.can_sample():
            raise ValueError("SQRL safety replay is not ready")
        items = [item for trajectory in self._trajectories
                 for item in trajectory]
        indices = self._rng.choice(
            len(items), self.batch_size, replace=len(items) < self.batch_size)

        def tensor(key: str) -> torch.Tensor:
            return torch.as_tensor(
                np.stack([np.asarray(items[int(i)][key]) for i in indices]),
                dtype=torch.float32, device=self.device)

        return {
            "observation": tensor("observation"),
            "action": tensor("action"),
            "next_observation": tensor("next_observation"),
            "unsafe": tensor("unsafe"),
            "done": tensor("done"),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "trajectories": list(self._trajectories),
            "current": self._current,
            "rng_state": self._rng.bit_generator.state,
        }, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._trajectories.clear()
        self._trajectories.extend(payload["trajectories"])
        self._current = list(payload.get("current", []))
        self._rng.bit_generator.state = payload["rng_state"]
