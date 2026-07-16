"""Environment space specifications without gym."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

SpaceSpec = Union['BoxSpec', 'DictSpec']


@dataclass(frozen=True)
class BoxSpec:
    shape: Tuple[int, ...]
    dtype: np.dtype = np.float32
    low: Optional[np.ndarray] = None
    high: Optional[np.ndarray] = None

    def __post_init__(self):
        object.__setattr__(self, 'dtype', np.dtype(self.dtype))
        if self.low is not None:
            object.__setattr__(self, 'low',
                               np.asarray(self.low, dtype=self.dtype))
        if self.high is not None:
            object.__setattr__(self, 'high',
                               np.asarray(self.high, dtype=self.dtype))

    def sample(self, rng: jax.Array) -> np.ndarray:
        rng, key = jax.random.split(rng)
        if self.low is not None and self.high is not None:
            value = jax.random.uniform(key,
                                       shape=self.shape,
                                       minval=self.low,
                                       maxval=self.high)
        else:
            value = jax.random.normal(key, shape=self.shape)
        return np.asarray(value, dtype=self.dtype)

    def zeros(self) -> jnp.ndarray:
        return jnp.zeros(self.shape, dtype=jnp.float32)


@dataclass(frozen=True)
class DictSpec:
    spaces: Dict[str, BoxSpec]

    def sample(self, rng: jax.Array) -> Dict[str, np.ndarray]:
        samples = {}
        for key, space in self.spaces.items():
            rng, key_rng = jax.random.split(rng)
            samples[key] = space.sample(key_rng)
        return samples

    def flatten_spec(self) -> BoxSpec:
        size = sum(int(np.prod(space.shape)) for space in self.spaces.values())
        return BoxSpec(shape=(size,), dtype=np.float32)


@dataclass(frozen=True)
class EnvSpec:
    observation: SpaceSpec
    action: BoxSpec
    max_episode_steps: Optional[int] = None


def space_to_spec(space) -> SpaceSpec:
    """Convert duck-typed space objects to rl specs."""
    if isinstance(space, (BoxSpec, DictSpec)):
        return space
    if hasattr(space, 'spaces'):
        return DictSpec(
            {key: space_to_spec(value) for key, value in space.spaces.items()})
    if hasattr(space, 'shape'):
        low = getattr(space, 'low', None)
        high = getattr(space, 'high', None)
        return BoxSpec(shape=tuple(space.shape),
                       dtype=np.dtype(space.dtype),
                       low=low,
                       high=high)
    raise TypeError(f'Unsupported space type: {type(space)}')


def env_to_spec(env) -> EnvSpec:
    return EnvSpec(observation=space_to_spec(env.observation_space),
                   action=space_to_spec(env.action_space))
