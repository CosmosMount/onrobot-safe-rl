from abc import ABC, abstractmethod
from typing import MutableMapping, Optional

import gymnasium as gym

from rl.utils.types import NDArray, Tensor

Batch = MutableMapping[str, Tensor]


class BaseBuffer(ABC):
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
        """A generic transition replay buffer."""
        if n_step < 1:
            raise ValueError("n_step must be greater than or equal to 1.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in the range [0.0, 1.0].")
        if max_length <= 0:
            raise ValueError("max_length must be greater than 0.")
        if min_length < 0 or min_length > max_length:
            raise ValueError("min_length must satisfy 0 <= min_length <= max_length.")
        if sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be greater than 0.")

        self._observation_space = observation_space
        self._action_space = action_space
        self._max_length = max_length
        self._min_length = min_length
        self._n_step = n_step
        self._gamma = gamma
        self._sample_batch_size = sample_batch_size

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def add(self, transition: MutableMapping[str, Tensor]) -> None:
        """Add one transition without a leading batch dimension."""
        pass

    @abstractmethod
    def add_batch(self, transitions: MutableMapping[str, Tensor]) -> None:
        """Add transitions with an explicit leading batch dimension."""
        pass

    @abstractmethod
    def can_sample(self) -> bool:
        pass

    @abstractmethod
    def sample(
        self,
        batch_size: Optional[int] = None,
        sample_idxs: Optional[Tensor] = None,
    ) -> Batch:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def get_observations(self) -> Tensor:
        pass
