import os
import pickle
from collections import deque
from typing import MutableMapping, Optional

import gymnasium as gym
import numpy as np

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


class NpyUniformBuffer(BaseBuffer):
    """A uniform n-step transition replay buffer using NumPy arrays."""

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
    ):
        super(NpyUniformBuffer, self).__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
        )
        self.reset()

    def __len__(self) -> int:
        return self._num_in_buffer

    def reset(self) -> None:
        if not isinstance(self._observation_space, gym.spaces.Box):
            raise TypeError("NpyUniformBuffer only supports gymnasium.spaces.Box observations.")
        if not isinstance(self._action_space, gym.spaces.Box):
            raise TypeError("NpyUniformBuffer only supports gymnasium.spaces.Box actions.")

        m = self._max_length
        observation_shape = tuple(self._observation_space.shape)
        action_shape = tuple(self._action_space.shape)
        observation_dtype = self._observation_space.dtype or np.float32
        action_dtype = self._action_space.dtype or np.float32

        if observation_dtype == np.float64:
            observation_dtype = np.float32
        if action_dtype == np.float64:
            action_dtype = np.float32

        self._observations = np.empty((m,) + observation_shape, dtype=observation_dtype)
        self._actions = np.empty((m,) + action_shape, dtype=action_dtype)
        self._rewards = np.empty((m,), dtype=np.float32)
        self._discounts = np.empty((m,), dtype=np.float32)
        self._terminateds = np.empty((m,), dtype=np.float32)
        self._truncateds = np.empty((m,), dtype=np.float32)
        self._next_observations = np.empty((m,) + observation_shape, dtype=observation_dtype)

        self._pending: list[deque[dict[str, NDArray]]] = []
        self._num_in_buffer = 0
        self._current_idx = 0

    def _validate_batch(self, transitions: MutableMapping[str, Tensor]) -> dict[str, NDArray]:
        missing = [key for key in _TRANSITION_KEYS if key not in transitions]
        if missing:
            raise ValueError(f"Transition is missing required fields: {missing}.")

        batch = {key: np.asarray(transitions[key]) for key in _TRANSITION_KEYS}
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
            if batch[key].shape != expected_shape:
                raise ValueError(f"{key} has shape {batch[key].shape}, expected {expected_shape}.")

        return batch

    def _make_single_env_transition(self, batch: dict[str, NDArray], env_idx: int) -> dict[str, NDArray]:
        return {
            "observation": np.array(batch["observation"][env_idx], copy=True),
            "action": np.array(batch["action"][env_idx], copy=True),
            "reward": np.asarray(batch["reward"][env_idx], dtype=np.float32),
            "terminated": np.asarray(batch["terminated"][env_idx], dtype=np.float32),
            "truncated": np.asarray(batch["truncated"][env_idx], dtype=np.float32),
            "next_observation": np.array(batch["next_observation"][env_idx], copy=True),
        }

    def _build_n_step_transition(self, queue: deque[dict[str, NDArray]]) -> dict[str, NDArray]:
        first = queue[0]
        reward = np.float32(0.0)
        discount = np.float32(1.0)
        terminated = np.float32(0.0)
        truncated = np.float32(0.0)
        next_observation = np.array(queue[-1]["next_observation"], copy=True)

        for transition in queue:
            reward = np.float32(reward + discount * transition["reward"])
            terminated = np.float32(transition["terminated"])
            truncated = np.float32(transition["truncated"])
            next_observation = np.array(transition["next_observation"], copy=True)
            discount = np.float32(discount * self._gamma)
            if bool(terminated) or bool(truncated):
                break

        if bool(terminated):
            discount = np.float32(0.0)

        return {
            "observation": first["observation"],
            "action": first["action"],
            "reward": np.asarray(reward, dtype=np.float32),
            "discount": np.asarray(discount, dtype=np.float32),
            "terminated": np.asarray(terminated, dtype=np.float32),
            "truncated": np.asarray(truncated, dtype=np.float32),
            "next_observation": next_observation,
        }

    def _write_batch(self, transitions: dict[str, NDArray]) -> None:
        add_batch_size = transitions["observation"].shape[0]
        add_idxs = (np.arange(add_batch_size) + self._current_idx) % self._max_length

        self._observations[add_idxs] = transitions["observation"].astype(self._observations.dtype, copy=False)
        self._actions[add_idxs] = transitions["action"].astype(self._actions.dtype, copy=False)
        self._rewards[add_idxs] = transitions["reward"].astype(np.float32, copy=False)
        self._discounts[add_idxs] = transitions["discount"].astype(np.float32, copy=False)
        self._terminateds[add_idxs] = transitions["terminated"].astype(np.float32, copy=False)
        self._truncateds[add_idxs] = transitions["truncated"].astype(np.float32, copy=False)
        self._next_observations[add_idxs] = transitions["next_observation"].astype(
            self._next_observations.dtype, copy=False
        )

        self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
        self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def _append_ready(self, ready: list[dict[str, NDArray]]) -> None:
        if not ready:
            return
        self._write_batch({key: np.stack([transition[key] for transition in ready]) for key in ready[0]})

    def add(self, transition: MutableMapping[str, Tensor]) -> None:
        batch = {key: np.expand_dims(np.asarray(value), axis=0) for key, value in transition.items()}
        self.add_batch(batch)

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
            latest = queue[-1]
            if bool(latest["terminated"]) or bool(latest["truncated"]):
                while queue:
                    ready.append(self._build_n_step_transition(queue))
                    queue.popleft()

        self._append_ready(ready)

    def flush(self) -> None:
        ready: list[dict[str, NDArray]] = []
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
            idxs = np.random.randint(0, self._num_in_buffer, size=size)
        else:
            idxs = np.asarray(sample_idxs, dtype=np.int64)

        batch: Batch = {}
        batch["observation"] = self._observations[idxs]
        batch["action"] = self._actions[idxs]
        batch["reward"] = self._rewards[idxs]
        batch["discount"] = self._discounts[idxs]
        batch["terminated"] = self._terminateds[idxs]
        batch["truncated"] = self._truncateds[idxs]
        batch["next_observation"] = self._next_observations[idxs]
        return batch

    def save(self, path: str) -> None:
        self.flush()
        os.makedirs(path, exist_ok=True)
        n = self._num_in_buffer
        dataset = {
            "observation": self._observations[:n],
            "action": self._actions[:n],
            "reward": self._rewards[:n],
            "discount": self._discounts[:n],
            "terminated": self._terminateds[:n],
            "truncated": self._truncateds[:n],
            "next_observation": self._next_observations[:n],
            "num_in_buffer": self._num_in_buffer,
            "current_idx": self._current_idx,
        }
        with open(os.path.join(path, "dataset.pickle"), "wb") as f:
            pickle.dump(dataset, f)

    def load(self, path: str) -> None:
        dataset_path = os.path.join(path, "dataset.pickle") if os.path.isdir(path) else path
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        n = int(dataset.get("num_in_buffer", len(dataset["observation"])))
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

        self._num_in_buffer = n
        self._current_idx = int(dataset.get("current_idx", n % self._max_length))
        self._pending.clear()

    def get_observations(self) -> Tensor:
        return self._observations[: self._num_in_buffer]
