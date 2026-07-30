from __future__ import annotations

import os
from collections import deque
from typing import MutableMapping

import numpy as np
import torch

from rl.utils.types import Tensor


class SafetyReplay:
    """Episode-aware replay that retains transitions preceding a failure."""

    def __init__(
        self,
        *,
        capacity: int,
        min_length: int,
        batch_size: int,
        failure_horizon: int,
        device: torch.device,
        seed: int,
    ):
        if capacity <= 0 or min_length < 0 or batch_size <= 0:
            raise ValueError("invalid safety replay sizes")
        if failure_horizon <= 0:
            raise ValueError("failure_horizon must be positive")
        self.capacity = int(capacity)
        self.min_length = int(min_length)
        self.batch_size = int(batch_size)
        self.failure_horizon = int(failure_horizon)
        self.device = device
        self._items: deque[dict[str, np.ndarray | float]] = deque(
            maxlen=self.capacity)
        self._episode: list[dict[str, np.ndarray | float]] = []
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def positive_count(self) -> int:
        return sum(
            float(item["future_failure"]) >= 0.5 for item in self._items)

    def add_batch(self, transition: MutableMapping[str, Tensor]) -> None:
        repeat = int(np.asarray(
            transition.get("replay_repeat_index", [0])).reshape(-1)[0])
        if repeat:
            return
        item: dict[str, np.ndarray | float] = {
            "observation": np.asarray(
                transition["observation"], dtype=np.float32)[0].copy(),
            "action": np.asarray(
                transition["action"], dtype=np.float32)[0].copy(),
            "next_observation": np.asarray(
                transition["next_observation"], dtype=np.float32)[0].copy(),
            "unsafe": float(np.asarray(
                transition.get("unsafe_label", transition["terminated"])
            ).reshape(-1)[0]),
            "near_failure": float(np.asarray(
                transition.get("near_failure_label", [0.0])
            ).reshape(-1)[0]),
            "done": float(
                bool(np.asarray(transition["terminated"]).reshape(-1)[0])
                or bool(np.asarray(transition["truncated"]).reshape(-1)[0])),
        }
        self._episode.append(item)
        if item["done"] < 0.5:
            return
        failed = item["unsafe"] >= 0.5
        failure_index = len(self._episode) - 1
        for index, episode_item in enumerate(self._episode):
            future_failure = float(
                failed
                and 0 <= failure_index - index < self.failure_horizon)
            stored = dict(episode_item)
            stored["future_failure"] = future_failure
            self._items.append(stored)
        self._episode.clear()

    def can_sample(self) -> bool:
        return (
            len(self._items) >= self.min_length
            and self.positive_count > 0
            and self.positive_count < len(self._items)
        )

    def sample(self) -> dict[str, torch.Tensor]:
        if not self.can_sample():
            raise ValueError("safety replay is not ready")
        items = list(self._items)
        positives = np.asarray([
            i for i, item in enumerate(items)
            if float(item["future_failure"]) >= 0.5])
        negatives = np.asarray([
            i for i, item in enumerate(items)
            if float(item["future_failure"]) < 0.5])
        positive_size = self.batch_size // 2
        indices = np.concatenate([
            self._rng.choice(positives, positive_size, replace=True),
            self._rng.choice(
                negatives, self.batch_size - positive_size, replace=True),
        ])
        self._rng.shuffle(indices)

        def tensor(key: str) -> torch.Tensor:
            return torch.as_tensor(
                np.stack([np.asarray(items[int(i)][key]) for i in indices]),
                dtype=torch.float32,
                device=self.device)

        return {
            "observation": tensor("observation"),
            "action": tensor("action"),
            "next_observation": tensor("next_observation"),
            "unsafe": tensor("unsafe"),
            "future_failure": tensor("future_failure"),
            "done": tensor("done"),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "items": list(self._items),
            "episode": self._episode,
            "rng_state": self._rng.bit_generator.state,
        }, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self._items.clear()
        self._items.extend(payload["items"])
        self._episode = list(payload.get("episode", []))
        self._rng.bit_generator.state = payload["rng_state"]
