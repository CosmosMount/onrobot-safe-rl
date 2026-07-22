from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from rl.droq._gym import gym
from rl.droq.data.dataset import Dataset, DatasetDict


def _init_replay_dict(obs_space: gym.Space,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict,
                        insert_index: int):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 capacity: int,
                 next_observation_space: Optional[gym.Space] = None):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space,
                                                  capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape),
                             dtype=action_space.dtype),
            rewards=np.empty((capacity, ), dtype=np.float32),
            masks=np.empty((capacity, ), dtype=np.float32),
            dones=np.empty((capacity, ), dtype=bool),
            terminateds=np.empty((capacity, ), dtype=bool),
            truncateds=np.empty((capacity, ), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample_jax(self, batch_size: int):
        batch = self.sample(batch_size)
        return jax.tree_util.tree_map(jnp.asarray, batch)

    def state_dict(self) -> dict:
        return {
            'capacity': self._capacity,
            'size': self._size,
            'insert_index': self._insert_index,
            'dataset_dict': _slice_recursively(self.dataset_dict, self._size),
        }

    def load_state_dict(self, state: dict) -> 'ReplayBuffer':
        if int(state['capacity']) != self._capacity:
            raise ValueError(
                f"capacity mismatch: checkpoint={state['capacity']} "
                f'current={self._capacity}')
        self._size = int(state['size'])
        self._insert_index = int(state['insert_index'])
        _load_recursively(self.dataset_dict, state['dataset_dict'], self._size)
        return self


def _slice_recursively(dataset_dict: DatasetDict,
                       size: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[:size].copy()
    if isinstance(dataset_dict, dict):
        return {
            key: _slice_recursively(value, size)
            for key, value in dataset_dict.items()
        }
    raise TypeError()


def _load_recursively(target: DatasetDict, source: DatasetDict, size: int) -> None:
    if isinstance(target, np.ndarray) and isinstance(source, np.ndarray):
        target[:size] = source[:size]
    elif isinstance(target, dict) and isinstance(source, dict):
        for key in target.keys():
            if key in source:
                _load_recursively(target[key], source[key], size)
            elif key in ('terminateds', 'truncateds'):
                target[key][:size] = False
            else:
                raise KeyError(f'Missing replay field in checkpoint: {key}')
    else:
        raise TypeError()
