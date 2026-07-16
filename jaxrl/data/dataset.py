from typing import Dict, Iterable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict

from jaxrl.utils.utils import freeze_jnp_batch
from jaxrl.utils.types import DataType

DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict,
                   dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, 'Inconsistent item lengths in the dataset.'
        else:
            raise TypeError('Unsupported type.')
    return dataset_len


def _sample(dataset_dict: Union[np.ndarray, DatasetDict],
            indx: np.ndarray) -> DatasetDict:
    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):

    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)
        self._rng = jax.random.PRNGKey(0)
        self._seed = 0
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None) -> int:
        if seed is None:
            seed = int(np.random.randint(0, 2**31 - 1))
        self._seed = seed
        self._rng = jax.random.PRNGKey(seed)
        return seed

    def __len__(self) -> int:
        return self.dataset_len

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        if indx is None:
            self._rng, key = jax.random.split(self._rng)
            indx = np.asarray(
                jax.random.randint(key, (batch_size,), 0, len(self)))

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

    def sample_jax(self,
                   batch_size: int,
                   keys: Optional[Iterable[str]] = None,
                   indx: Optional[np.ndarray] = None) -> frozen_dict.FrozenDict:
        return freeze_jnp_batch(dict(self.sample(batch_size, keys, indx)))
