import os
from collections import deque
from typing import Any, MutableMapping, Optional

import gymnasium as gym
import numpy as np
import torch

from rl.buffers.buffer import BaseBuffer, Batch
from rl.utils.types import NDArray, Tensor

_TRANSITION_KEYS = (
    "observation",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_observation",
)
_ACTION_DIAGNOSTIC_FIELDS = (
    "action_nominal", "action_executed", "action_q_target",
    "action_safety_intervened", "action_safety_intervention_norm",
    "action_runtime_intervened", "action_runtime_intervention_norm",
    "action_total_intervened", "action_total_intervention_norm",
)

_NP_TO_TORCH_DTYPE: dict[np.dtype[Any], torch.dtype] = {
    np.dtype(np.float64): torch.float32,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.bool_): torch.bool,
    np.dtype(np.uint8): torch.uint8,
}


def _numpy_dtype_to_torch(dtype: Any) -> torch.dtype:
    """Convert a NumPy dtype to a torch dtype, enforcing float32 for float64."""
    dtype = np.dtype(dtype)
    if dtype in _NP_TO_TORCH_DTYPE:
        return _NP_TO_TORCH_DTYPE[dtype]
    return torch.float32


class TorchUniformBuffer(BaseBuffer):
    """A uniform n-step transition replay buffer using PyTorch tensors."""

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
        super(TorchUniformBuffer, self).__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
        )
        device_type = (
            device_type
            if device_type.startswith("cuda") and ":" in device_type
            else ("cuda:0" if device_type.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_type)
        self.reset()

    def __len__(self) -> int:
        return self._num_in_buffer

    def reset(self) -> None:
        if not isinstance(self._observation_space, gym.spaces.Box):
            raise TypeError("TorchUniformBuffer only supports gymnasium.spaces.Box observations.")
        if not isinstance(self._action_space, gym.spaces.Box):
            raise TypeError("TorchUniformBuffer only supports gymnasium.spaces.Box actions.")

        m = self._max_length
        pin = self._device.type == "cpu" and torch.cuda.is_available()
        observation_shape = tuple(self._observation_space.shape)
        action_shape = tuple(self._action_space.shape)
        observation_dtype = _numpy_dtype_to_torch(self._observation_space.dtype or np.float32)
        action_dtype = _numpy_dtype_to_torch(self._action_space.dtype or np.float32)

        self._observations = torch.empty(
            (m,) + observation_shape, dtype=observation_dtype, device=self._device, pin_memory=pin
        )
        self._actions = torch.empty((m,) + action_shape, dtype=action_dtype, device=self._device, pin_memory=pin)
        self._rewards = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._discounts = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._terminateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._truncateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._next_observations = torch.empty(
            (m,) + observation_shape, dtype=observation_dtype, device=self._device, pin_memory=pin
        )
        self._actions_nominal = torch.empty((m,) + action_shape, dtype=torch.float32, device=self._device)
        self._actions_executed = torch.empty((m,) + action_shape, dtype=torch.float32, device=self._device)
        self._actions_q_target = torch.empty((m,) + action_shape, dtype=torch.float32, device=self._device)
        for key in _ACTION_DIAGNOSTIC_FIELDS[3:]:
            setattr(self, f"_{key.replace('action_', 'actions_')}", torch.empty((m,), dtype=torch.float32, device=self._device))

        self._pending: list[deque[dict[str, torch.Tensor]]] = []
        self._num_in_buffer = 0
        self._current_idx = 0

    def _to_tensor(self, value: Tensor) -> torch.Tensor:
        """Convert a value to a tensor on the buffer device."""
        if isinstance(value, torch.Tensor):
            return value.detach().to(self._device, copy=True)
        return torch.as_tensor(value, device=self._device)

    def _validate_batch(self, transitions: MutableMapping[str, Tensor]) -> dict[str, torch.Tensor]:
        missing = [key for key in _TRANSITION_KEYS if key not in transitions]
        if missing:
            raise ValueError(f"Transition is missing required fields: {missing}.")

        batch = {key: self._to_tensor(transitions[key]) for key in _TRANSITION_KEYS}
        observation_shape = self._observations.shape[1:]
        action_shape = self._actions.shape[1:]

        if batch["observation"].ndim < 1:
            raise ValueError("Batched observation must include a leading batch dimension.")
        batch_size = batch["observation"].shape[0]
        expected_shapes = {
            "observation": (batch_size,) + observation_shape,
            "action": (batch_size,) + action_shape,
            "reward": (batch_size,),
            "terminated": (batch_size,),
            "truncated": (batch_size,),
            "next_observation": (batch_size,) + observation_shape,
        }
        for key, expected_shape in expected_shapes.items():
            if tuple(batch[key].shape) != expected_shape:
                raise ValueError(f"{key} has shape {tuple(batch[key].shape)}, expected {expected_shape}.")
        for key in _ACTION_DIAGNOSTIC_FIELDS:
            if key not in transitions:
                continue
            value = self._to_tensor(transitions[key])
            expected = (batch_size,) + action_shape if key in _ACTION_DIAGNOSTIC_FIELDS[:3] else (batch_size,)
            if tuple(value.shape) != expected:
                raise ValueError(f"{key} has shape {tuple(value.shape)}, expected {expected}.")
            batch[key] = value

        return batch

    def _make_single_env_transition(
        self, batch: dict[str, torch.Tensor], env_idx: int
    ) -> dict[str, torch.Tensor]:
        result = {
            "observation": batch["observation"][env_idx].clone(),
            "action": batch["action"][env_idx].clone(),
            "reward": batch["reward"][env_idx].to(dtype=torch.float32).clone(),
            "terminated": batch["terminated"][env_idx].to(dtype=torch.float32).clone(),
            "truncated": batch["truncated"][env_idx].to(dtype=torch.float32).clone(),
            "next_observation": batch["next_observation"][env_idx].clone(),
        }
        defaults = {
            "action_nominal": batch["action"], "action_executed": batch["action"],
            "action_q_target": torch.full_like(batch["action"], float("nan")),
            **{key: torch.zeros((batch["action"].shape[0],), device=self._device) for key in _ACTION_DIAGNOSTIC_FIELDS[3:]},
        }
        for key in _ACTION_DIAGNOSTIC_FIELDS:
            result[key] = batch.get(key, defaults[key])[env_idx].clone()
        return result

    def _build_n_step_transition(self, queue: deque[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        first = queue[0]
        reward = torch.zeros((), dtype=torch.float32, device=self._device)
        discount = torch.ones((), dtype=torch.float32, device=self._device)
        terminated = torch.zeros((), dtype=torch.float32, device=self._device)
        truncated = torch.zeros((), dtype=torch.float32, device=self._device)
        next_observation = queue[-1]["next_observation"].clone()

        for transition in queue:
            reward = reward + discount * transition["reward"]
            terminated = transition["terminated"].to(dtype=torch.float32)
            truncated = transition["truncated"].to(dtype=torch.float32)
            next_observation = transition["next_observation"].clone()
            discount = discount * self._gamma
            if bool(terminated.item()) or bool(truncated.item()):
                break

        if bool(terminated.item()):
            discount = torch.zeros((), dtype=torch.float32, device=self._device)

        return {
            "observation": first["observation"],
            "action": first["action"],
            "reward": reward.to(dtype=torch.float32),
            "discount": discount.to(dtype=torch.float32),
            "terminated": terminated.to(dtype=torch.float32),
            "truncated": truncated.to(dtype=torch.float32),
            "next_observation": next_observation,
            **{key: first[key] for key in _ACTION_DIAGNOSTIC_FIELDS},
        }

    def _write_batch(self, transitions: dict[str, torch.Tensor]) -> None:
        add_batch_size = transitions["observation"].shape[0]
        end_idx = self._current_idx + add_batch_size
        if end_idx <= self._max_length:
            idxs: Any = slice(self._current_idx, end_idx)
        else:
            idxs = (torch.arange(add_batch_size, device=self._device) + self._current_idx) % self._max_length

        self._observations[idxs] = transitions["observation"].to(dtype=self._observations.dtype)
        self._actions[idxs] = transitions["action"].to(dtype=self._actions.dtype)
        self._rewards[idxs] = transitions["reward"].to(dtype=torch.float32)
        self._discounts[idxs] = transitions["discount"].to(dtype=torch.float32)
        self._terminateds[idxs] = transitions["terminated"].to(dtype=torch.float32)
        self._truncateds[idxs] = transitions["truncated"].to(dtype=torch.float32)
        self._next_observations[idxs] = transitions["next_observation"].to(dtype=self._next_observations.dtype)
        self._actions_nominal[idxs] = transitions["action_nominal"].to(dtype=torch.float32)
        self._actions_executed[idxs] = transitions["action_executed"].to(dtype=torch.float32)
        self._actions_q_target[idxs] = transitions["action_q_target"].to(dtype=torch.float32)
        for key in _ACTION_DIAGNOSTIC_FIELDS[3:]:
            getattr(self, f"_{key.replace('action_', 'actions_')}")[idxs] = transitions[key].to(dtype=torch.float32)

        self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
        self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def _append_ready(self, ready: list[dict[str, torch.Tensor]]) -> None:
        if not ready:
            return
        self._write_batch({key: torch.stack([transition[key] for transition in ready]) for key in ready[0]})

    def add(self, transition: MutableMapping[str, Tensor]) -> None:
        batch = {key: self._to_tensor(value).unsqueeze(0) for key, value in transition.items()}
        self.add_batch(batch)

    def add_batch(self, transitions: MutableMapping[str, Tensor]) -> None:
        batch = self._validate_batch(transitions)
        batch_size = batch["observation"].shape[0]
        while len(self._pending) < batch_size:
            self._pending.append(deque())

        ready: list[dict[str, torch.Tensor]] = []
        for env_idx in range(batch_size):
            queue = self._pending[env_idx]
            queue.append(self._make_single_env_transition(batch, env_idx))
            while len(queue) >= self._n_step:
                ready.append(self._build_n_step_transition(queue))
                queue.popleft()
            if not queue:
                continue
            latest = queue[-1]
            if bool(latest["terminated"].item()) or bool(latest["truncated"].item()):
                while queue:
                    ready.append(self._build_n_step_transition(queue))
                    queue.popleft()

        self._append_ready(ready)

    def flush(self) -> None:
        ready: list[dict[str, torch.Tensor]] = []
        for queue in self._pending:
            while queue:
                ready.append(self._build_n_step_transition(queue))
                queue.popleft()
        self._append_ready(ready)

    def can_sample(self) -> bool:
        return self._num_in_buffer >= self._min_length

    def sample(
        self,
        batch_size: Optional[int] = None,
        sample_idxs: Optional[Tensor] = None,
    ) -> Batch:
        if batch_size is not None and sample_idxs is not None:
            raise ValueError("batch_size and sample_idxs cannot be provided together.")
        if sample_idxs is None:
            size = self._sample_batch_size if batch_size is None else batch_size
            if size <= 0:
                raise ValueError("batch_size must be greater than 0.")
            idxs = torch.randint(0, self._num_in_buffer, (size,), device=self._device)
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device, dtype=torch.long)

        batch: Batch = {}
        batch["observation"] = self._observations[idxs]
        batch["action"] = self._actions[idxs]
        batch["reward"] = self._rewards[idxs]
        batch["discount"] = self._discounts[idxs]
        batch["terminated"] = self._terminateds[idxs]
        batch["truncated"] = self._truncateds[idxs]
        batch["next_observation"] = self._next_observations[idxs]
        batch.update({key: getattr(self, f"_{key.replace('action_', 'actions_')}")[idxs]
                      for key in _ACTION_DIAGNOSTIC_FIELDS[3:]})
        batch.update({
            "action_nominal": self._actions_nominal[idxs],
            "action_executed": self._actions_executed[idxs],
            "action_q_target": self._actions_q_target[idxs],
        })
        return batch

    def save(self, path: str) -> None:
        """Save buffer contents and metadata."""
        self.flush()
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        n = self._num_in_buffer
        dataset: dict[str, Any] = {
            "observation": self._observations[:n],
            "action": self._actions[:n],
            "reward": self._rewards[:n],
            "discount": self._discounts[:n],
            "terminated": self._terminateds[:n],
            "truncated": self._truncateds[:n],
            "next_observation": self._next_observations[:n],
            "action_nominal": self._actions_nominal[:n],
            "action_executed": self._actions_executed[:n],
            "action_q_target": self._actions_q_target[:n],
            **{key: getattr(self, f"_{key.replace('action_', 'actions_')}")[:n]
               for key in _ACTION_DIAGNOSTIC_FIELDS[3:]},
            "num_in_buffer": self._num_in_buffer,
            "current_idx": self._current_idx,
        }
        torch.save(dataset, path)

    def load(self, path: str) -> None:
        """Load buffer contents and metadata."""
        dataset = torch.load(path, map_location=self._device)
        n = int(dataset["num_in_buffer"])

        self._observations[:n] = dataset["observation"]
        self._actions[:n] = dataset["action"]
        self._rewards[:n] = dataset["reward"]
        if "discount" in dataset:
            self._discounts[:n] = dataset["discount"]
        else:
            self._discounts[:n] = (self._gamma**self._n_step) * (1.0 - dataset["terminated"])
        self._terminateds[:n] = dataset["terminated"]
        self._truncateds[:n] = dataset["truncated"]
        self._next_observations[:n] = dataset["next_observation"]
        defaults = {
            "action_nominal": dataset["action"],
            "action_executed": dataset["action"],
            "action_q_target": torch.full((n,) + self._actions.shape[1:], float("nan"), device=self._device),
            **{key: torch.zeros((n,), device=self._device) for key in _ACTION_DIAGNOSTIC_FIELDS[3:]},
        }
        for key in _ACTION_DIAGNOSTIC_FIELDS:
            value = dataset.get(key, defaults[key])
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value, device=self._device)
            getattr(self, f"_{key.replace('action_', 'actions_')}")[:n] = value

        self._num_in_buffer = n
        self._current_idx = int(dataset["current_idx"])
        self._pending.clear()

    def get_observations(self) -> torch.Tensor:
        return self._observations[: self._num_in_buffer]
