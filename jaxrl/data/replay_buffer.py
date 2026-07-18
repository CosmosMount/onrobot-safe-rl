from typing import Iterable, Optional, Union

import jax
import numpy as np

from jaxrl.data.dataset import Dataset, DatasetDict
from jaxrl.env.specs import BoxSpec, DictSpec, SpaceSpec


def _init_replay_dict(space: SpaceSpec,
                      capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(space, BoxSpec):
        return np.empty((capacity, *space.shape), dtype=space.dtype)
    elif isinstance(space, DictSpec):
        data_dict = {}
        for k, v in space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError(f'Unsupported space type: {type(space)}')


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
                 observation_spec: SpaceSpec,
                 action_spec: BoxSpec,
                 capacity: int,
                 next_observation_spec: Optional[SpaceSpec] = None):
        if next_observation_spec is None:
            next_observation_spec = observation_spec

        observation_data = _init_replay_dict(observation_spec, capacity)
        next_observation_data = _init_replay_dict(next_observation_spec,
                                                  capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_spec.shape),
                             dtype=action_spec.dtype),
            rewards=np.empty((capacity, ), dtype=np.float32),
            masks=np.empty((capacity, ), dtype=np.float32),
            dones=np.empty((capacity, ), dtype=bool),
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

    def state_dict(self) -> dict:
        """Return a compact snapshot containing only initialized replay rows."""
        def compact(value):
            if isinstance(value, np.ndarray):
                return value[:self._size].copy()
            if isinstance(value, dict):
                return {key: compact(item) for key, item in value.items()}
            raise TypeError(f'Unsupported replay value: {type(value)}')

        return {
            'dataset_dict': compact(self.dataset_dict),
            'size': self._size,
            'capacity': self._capacity,
            'insert_index': self._insert_index,
            'rng': self._rng,
            'seed': self._seed,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a compact snapshot into this preallocated replay buffer."""
        if int(state['capacity']) != self._capacity:
            raise ValueError(
                'Replay capacity mismatch: '
                f'snapshot={state["capacity"]} current={self._capacity}')
        size = int(state['size'])
        if not 0 <= size <= self._capacity:
            raise ValueError(f'Invalid replay size in snapshot: {size}')

        def restore(destination, source):
            if isinstance(destination, np.ndarray):
                if source.shape != (size, *destination.shape[1:]):
                    raise ValueError(
                        'Replay array shape mismatch: '
                        f'snapshot={source.shape} current={destination.shape}')
                destination[:size] = source
                return
            if isinstance(destination, dict) and isinstance(source, dict):
                if destination.keys() != source.keys():
                    raise ValueError('Replay snapshot keys do not match')
                for key in destination:
                    restore(destination[key], source[key])
                return
            raise TypeError('Unsupported replay snapshot structure')

        restore(self.dataset_dict, state['dataset_dict'])
        self._size = size
        self._insert_index = int(state['insert_index'])
        self._rng = state['rng']
        self._seed = int(state['seed'])

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None):
        if self._size == 0:
            raise ValueError('Cannot sample from an empty replay buffer.')
        if indx is None:
            self._rng, key = jax.random.split(self._rng)
            # With-replacement sampling over filled slots (walk_in_the_park).
            indx = np.asarray(
                jax.random.randint(key, (batch_size,), 0, self._size))
        return super().sample(batch_size, keys=keys, indx=indx)
