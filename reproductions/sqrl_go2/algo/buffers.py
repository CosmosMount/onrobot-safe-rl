"""Independent task and recent constrained-policy replay buffers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    critic_action: np.ndarray
    reward: float
    next_observation: np.ndarray
    cost: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if self.terminated and self.truncated:
            raise ValueError("transition cannot be both terminated and truncated")
        if self.cost not in (0.0, 1.0):
            raise ValueError("SQRL cost must be sparse binary failure")
        if bool(self.cost) != bool(self.terminated):
            raise ValueError("cost must equal first-fall termination")


class ReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple[int, ...],
                 action_dim: int, seed: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._observations = np.empty((capacity, *observation_shape), np.float32)
        self._actions = np.empty((capacity, action_dim), np.float32)
        self._rewards = np.empty(capacity, np.float32)
        self._next_observations = np.empty((capacity, *observation_shape), np.float32)
        self._costs = np.empty(capacity, np.float32)
        self._terminated = np.empty(capacity, np.float32)
        self._truncated = np.empty(capacity, np.float32)
        self._size = 0
        self._cursor = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        i = self._cursor
        self._observations[i] = transition.observation
        self._actions[i] = transition.critic_action
        self._rewards[i] = transition.reward
        self._next_observations[i] = transition.next_observation
        self._costs[i] = transition.cost
        self._terminated[i] = transition.terminated
        self._truncated[i] = transition.truncated
        self._cursor = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        if self._size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        indices = self._rng.integers(0, self._size, size=int(batch_size))
        tensor = lambda value: torch.as_tensor(value[indices], device=device)
        return {
            "observation": tensor(self._observations),
            "action": tensor(self._actions),
            "reward": tensor(self._rewards),
            "next_observation": tensor(self._next_observations),
            "cost": tensor(self._costs),
            "terminated": tensor(self._terminated),
            "truncated": tensor(self._truncated),
        }


class SafetyReplayBuffer:
    """Small FIFO of the latest complete bar-pi trajectories."""

    def __init__(self, max_trajectories: int, seed: int):
        if max_trajectories <= 0:
            raise ValueError("max_trajectories must be positive")
        self._trajectories: deque[list[Transition]] = deque(maxlen=max_trajectories)
        self._current: list[Transition] = []
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return sum(map(len, self._trajectories))

    @property
    def trajectory_count(self) -> int:
        return len(self._trajectories)

    @property
    def fall_count(self) -> int:
        return sum(int(item.cost) for trajectory in self._trajectories for item in trajectory)

    def add(self, transition: Transition) -> bool:
        self._current.append(transition)
        if not (transition.terminated or transition.truncated):
            return False
        self._trajectories.append(self._current)
        self._current = []
        return True

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        items = [item for trajectory in self._trajectories for item in trajectory]
        if not items:
            raise ValueError("no complete safety trajectory is available")
        indices = self._rng.integers(0, len(items), size=int(batch_size))
        selected = [items[int(index)] for index in indices]
        return {
            "observation": torch.as_tensor(np.stack([x.observation for x in selected]), device=device),
            "action": torch.as_tensor(np.stack([x.critic_action for x in selected]), device=device),
            "next_observation": torch.as_tensor(np.stack([x.next_observation for x in selected]), device=device),
            "cost": torch.as_tensor([x.cost for x in selected], dtype=torch.float32, device=device),
            "terminated": torch.as_tensor([x.terminated for x in selected], dtype=torch.float32, device=device),
            "truncated": torch.as_tensor([x.truncated for x in selected], dtype=torch.float32, device=device),
        }
