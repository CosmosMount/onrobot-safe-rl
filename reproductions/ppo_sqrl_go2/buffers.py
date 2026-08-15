"""Strictly separated on-policy task storage and recent safety replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SafetyTransition:
    observation: np.ndarray
    policy_observation: np.ndarray
    action: np.ndarray
    next_observation: np.ndarray
    next_policy_observation: np.ndarray
    cost: bool
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if self.terminated and self.truncated:
            raise ValueError("a transition cannot terminate and truncate")
        if self.cost != self.terminated:
            raise ValueError("cost must equal first-fall termination")


class RecentSafetyBuffer:
    def __init__(self, trajectories: int, seed: int):
        if trajectories <= 0:
            raise ValueError("trajectory capacity must be positive")
        self.capacity = int(trajectories)
        self._complete: deque[list[SafetyTransition]] = deque(maxlen=self.capacity)
        self._pending: dict[int, list[SafetyTransition]] = {}
        self._rng = np.random.default_rng(seed)
        self.total_falls = 0

    def add(self, environment_id: int, transition: SafetyTransition) -> bool:
        pending = self._pending.setdefault(int(environment_id), [])
        pending.append(transition)
        if not (transition.terminated or transition.truncated):
            return False
        self._complete.append(pending)
        self._pending[int(environment_id)] = []
        self.total_falls += int(transition.cost)
        return True

    def __len__(self) -> int:
        return sum(len(trajectory) for trajectory in self._complete)

    @property
    def trajectory_count(self) -> int:
        return len(self._complete)

    @property
    def retained_falls(self) -> int:
        return sum(
            int(item.cost) for trajectory in self._complete for item in trajectory)

    def sample(self, size: int, device: torch.device) -> dict[str, torch.Tensor]:
        items = [item for trajectory in self._complete for item in trajectory]
        if not items:
            raise ValueError("no complete safety trajectory is available")
        selected = [items[int(index)] for index in self._rng.integers(
            0, len(items), size=int(size))]
        tensor = lambda name: torch.as_tensor(
            np.stack([getattr(item, name) for item in selected]), device=device)
        return {
            "observation": tensor("observation").float(),
            "policy_observation": tensor("policy_observation").float(),
            "action": tensor("action").float(),
            "next_observation": tensor("next_observation").float(),
            "next_policy_observation": tensor("next_policy_observation").float(),
            "cost": tensor("cost").float(),
            "terminated": tensor("terminated").float(),
            "truncated": tensor("truncated").float(),
        }


class VectorRecentSafetyBuffer:
    """GPU episode assembler retaining the latest complete trajectories."""

    def __init__(self, environments: int, max_episode_steps: int,
                 trajectories: int, observation_dim: int,
                 policy_observation_dim: int, action_dim: int,
                 *, device: torch.device, seed: int):
        self.environments = int(environments)
        self.max_episode_steps = int(max_episode_steps)
        self.capacity = int(trajectories)
        self.device = device
        shape = (self.environments, self.max_episode_steps)
        self.observation = torch.empty((*shape, observation_dim), device=device)
        self.policy_observation = torch.empty(
            (*shape, policy_observation_dim), device=device)
        self.action = torch.empty((*shape, action_dim), device=device)
        self.next_observation = torch.empty((*shape, observation_dim), device=device)
        self.next_policy_observation = torch.empty(
            (*shape, policy_observation_dim), device=device)
        self.cost = torch.empty(shape, device=device)
        self.terminated = torch.empty(shape, device=device)
        self.truncated = torch.empty(shape, device=device)
        self.length = torch.zeros(self.environments, dtype=torch.long, device=device)
        self._complete: deque[dict[str, torch.Tensor]] = deque(maxlen=self.capacity)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.total_transitions = 0
        self.total_falls = 0

    def add_batch(self, *, observation: torch.Tensor,
                  policy_observation: torch.Tensor, action: torch.Tensor,
                  next_observation: torch.Tensor,
                  next_policy_observation: torch.Tensor,
                  cost: torch.Tensor, terminated: torch.Tensor,
                  truncated: torch.Tensor) -> int:
        values = (observation, policy_observation, action, next_observation,
                  next_policy_observation, cost, terminated, truncated)
        if any(len(value) != self.environments for value in values):
            raise ValueError("safety batch must contain every safety environment")
        if bool(torch.any(self.length >= self.max_episode_steps).item()):
            raise RuntimeError("safety episode exceeded configured maximum")
        row = torch.arange(self.environments, device=self.device)
        slot = self.length
        for destination, value in (
            (self.observation, observation),
            (self.policy_observation, policy_observation),
            (self.action, action), (self.next_observation, next_observation),
            (self.next_policy_observation, next_policy_observation),
            (self.cost, cost.float()), (self.terminated, terminated.float()),
            (self.truncated, truncated.float()),
        ):
            destination[row, slot] = value
        self.length += 1
        self.total_transitions += self.environments
        self.total_falls += int(cost.sum().item())
        done = terminated.to(torch.bool) | truncated.to(torch.bool)
        completed_ids = done.nonzero(as_tuple=False).flatten().tolist()
        for environment_id in completed_ids:
            count = int(self.length[environment_id].item())
            self._complete.append({
                "observation": self.observation[environment_id, :count].clone(),
                "policy_observation": self.policy_observation[
                    environment_id, :count].clone(),
                "action": self.action[environment_id, :count].clone(),
                "next_observation": self.next_observation[
                    environment_id, :count].clone(),
                "next_policy_observation": self.next_policy_observation[
                    environment_id, :count].clone(),
                "cost": self.cost[environment_id, :count].clone(),
                "terminated": self.terminated[environment_id, :count].clone(),
                "truncated": self.truncated[environment_id, :count].clone(),
            })
            self.length[environment_id] = 0
        return len(completed_ids)

    def __len__(self) -> int:
        return sum(len(trajectory["cost"]) for trajectory in self._complete)

    @property
    def trajectory_count(self) -> int:
        return len(self._complete)

    @property
    def retained_falls(self) -> int:
        return int(sum(float(item["cost"].sum()) for item in self._complete))

    def sample(self, size: int) -> dict[str, torch.Tensor]:
        if not self._complete:
            raise ValueError("no complete safety trajectory is available")
        merged = {
            name: torch.cat([item[name] for item in self._complete], dim=0)
            for name in self._complete[0]
        }
        indices = torch.randint(
            len(merged["cost"]), (int(size),), device=self.device,
            generator=self.generator)
        return {name: value[indices] for name, value in merged.items()}

    def state_dict(self) -> dict[str, object]:
        fields = (
            "observation", "policy_observation", "action",
            "next_observation", "next_policy_observation", "cost",
            "terminated", "truncated")
        return {
            "pending": {name: getattr(self, name).clone() for name in fields},
            "length": self.length.clone(),
            "complete": [
                {name: value.clone() for name, value in item.items()}
                for item in self._complete],
            "generator_state": self.generator.get_state(),
            "total_transitions": self.total_transitions,
            "total_falls": self.total_falls,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        pending = state["pending"]
        for name, value in pending.items():  # type: ignore[union-attr]
            getattr(self, name).copy_(value)
        self.length.copy_(state["length"])  # type: ignore[arg-type]
        self._complete.clear()
        for item in state["complete"]:  # type: ignore[union-attr]
            self._complete.append({name: value.to(self.device) for name, value in item.items()})
        self.generator.set_state(state["generator_state"])  # type: ignore[arg-type]
        self.total_transitions = int(state["total_transitions"])
        self.total_falls = int(state["total_falls"])


class TaskRollout:
    """Fixed-size PPO rollout; masked safety samples cannot be inserted."""

    def __init__(self, steps: int, environments: int, actor_obs_dim: int,
                 critic_obs_dim: int, qsafe_obs_dim: int, action_dim: int,
                 device: torch.device):
        self.steps = int(steps)
        self.environments = int(environments)
        self.device = device
        shape = (self.steps, self.environments)
        self.actor_observation = torch.empty(
            (*shape, actor_obs_dim), device=device)
        self.critic_observation = torch.empty(
            (*shape, critic_obs_dim), device=device)
        self.qsafe_observation = torch.empty(
            (*shape, qsafe_obs_dim), device=device)
        self.action = torch.empty((*shape, action_dim), device=device)
        self.log_probability = torch.empty(shape, device=device)
        self.value = torch.empty(shape, device=device)
        self.reward = torch.empty(shape, device=device)
        self.done = torch.empty(shape, dtype=torch.bool, device=device)
        self.mean = torch.empty((*shape, action_dim), device=device)
        self.std = torch.empty((*shape, action_dim), device=device)
        self.return_ = torch.empty(shape, device=device)
        self.advantage = torch.empty(shape, device=device)
        self.cursor = 0

    def add(self, *, actor_observation: torch.Tensor,
            critic_observation: torch.Tensor,
            qsafe_observation: torch.Tensor, action: torch.Tensor,
            log_probability: torch.Tensor, value: torch.Tensor,
            reward: torch.Tensor, done: torch.Tensor,
            mean: torch.Tensor, std: torch.Tensor, source: str = "task") -> None:
        if source != "task":
            raise ValueError("masked safety transitions cannot enter PPO storage")
        if self.cursor >= self.steps:
            raise RuntimeError("task rollout is full")
        index = self.cursor
        for destination, value_ in (
            (self.actor_observation, actor_observation),
            (self.critic_observation, critic_observation),
            (self.qsafe_observation, qsafe_observation),
            (self.action, action), (self.log_probability, log_probability),
            (self.value, value), (self.reward, reward), (self.done, done),
            (self.mean, mean), (self.std, std),
        ):
            destination[index].copy_(value_)
        self.cursor += 1

    def finish(self, last_value: torch.Tensor, gamma: float, lam: float) -> None:
        if self.cursor != self.steps:
            raise RuntimeError("cannot finish an incomplete task rollout")
        gae = torch.zeros(self.environments, device=self.device)
        for step in reversed(range(self.steps)):
            next_value = last_value if step == self.steps - 1 else self.value[step + 1]
            live = (~self.done[step]).float()
            delta = self.reward[step] + gamma * live * next_value - self.value[step]
            gae = delta + gamma * lam * live * gae
            self.advantage[step] = gae
            self.return_[step] = gae + self.value[step]
        self.advantage.sub_(self.advantage.mean()).div_(self.advantage.std() + 1e-8)

    def batches(self, mini_batches: int, epochs: int,
                generator: torch.Generator):
        total = self.steps * self.environments
        if total % mini_batches:
            raise ValueError("rollout size must divide evenly into minibatches")
        flattened = {
            "actor_observation": self.actor_observation.reshape(total, -1),
            "critic_observation": self.critic_observation.reshape(total, -1),
            "qsafe_observation": self.qsafe_observation.reshape(total, -1),
            "action": self.action.reshape(total, -1),
            "old_log_probability": self.log_probability.reshape(total),
            "old_value": self.value.reshape(total),
            "return": self.return_.reshape(total),
            "advantage": self.advantage.reshape(total),
            "old_mean": self.mean.reshape(total, -1),
            "old_std": self.std.reshape(total, -1),
        }
        width = total // mini_batches
        for _ in range(epochs):
            order = torch.randperm(total, generator=generator, device=self.device)
            for start in range(0, total, width):
                index = order[start:start + width]
                yield {name: value[index] for name, value in flattened.items()}
