"""Environment preprocessing without gym wrappers."""

from typing import Any, Dict, Optional, Tuple, Union

import jax
import numpy as np

from jaxrl.env.specs import BoxSpec, DictSpec, SpaceSpec, space_to_spec

ObsType = Union[np.ndarray, Dict[str, np.ndarray]]


def _to_float32(obs: ObsType) -> ObsType:
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32) if obs.dtype == np.float64 else obs
    return {key: _to_float32(value) for key, value in obs.items()}


def _flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for key in sorted(obs.keys()):
        value = obs[key]
        parts.append(value.reshape(-1) if value.ndim > 0 else np.asarray([value]))
    return np.concatenate(parts).astype(np.float32)


def _rescale_action(action: np.ndarray, low: np.ndarray,
                    high: np.ndarray) -> np.ndarray:
    return low + (action + 1.0) * 0.5 * (high - low)


class BaseEnv:
    """Applies obs preprocessing and action rescale/clip for RL training."""

    def __init__(self,
                 env: Any,
                 rescale_actions: bool = True,
                 seed: Optional[int] = None):
        self._env = env
        self._rescale = rescale_actions

        raw_obs_spec = space_to_spec(env.observation_space)
        raw_action_spec = space_to_spec(env.action_space)
        if not isinstance(raw_action_spec, BoxSpec):
            raise TypeError('Action space must be Box-like.')

        self._raw_action_spec = raw_action_spec
        if isinstance(raw_obs_spec, DictSpec):
            self.observation_spec = raw_obs_spec.flatten_spec()
            self._flatten_obs = True
        else:
            self.observation_spec = raw_obs_spec
            self._flatten_obs = False

        self.action_spec = raw_action_spec
        self._policy_action_spec = BoxSpec(shape=raw_action_spec.shape)
        self._rng = jax.random.PRNGKey(0)
        if seed is not None:
            self.seed(seed)

    @property
    def observation_space(self) -> BoxSpec:
        return self.observation_spec

    @property
    def action_space(self) -> BoxSpec:
        if self._rescale:
            return self._policy_action_spec
        return self.action_spec

    def seed(self, seed: int) -> int:
        self._rng = jax.random.PRNGKey(seed)
        if hasattr(self._env, 'seed'):
            self._env.seed(seed)
        return seed

    def sample_action(self) -> np.ndarray:
        self._rng, key = jax.random.split(self._rng)
        if self._rescale:
            return np.asarray(
                jax.random.uniform(key,
                                   shape=self._policy_action_spec.shape,
                                   minval=-1.0,
                                   maxval=1.0),
                dtype=np.float32)
        if (self.action_spec.low is not None
                and self.action_spec.high is not None):
            return np.asarray(
                jax.random.uniform(key,
                                   shape=self.action_spec.shape,
                                   minval=self.action_spec.low,
                                   maxval=self.action_spec.high),
                dtype=np.float32)
        return np.asarray(jax.random.normal(key, shape=self.action_spec.shape),
                          dtype=np.float32)

    def _process_obs(self, obs: ObsType) -> np.ndarray:
        obs = _to_float32(obs)
        if self._flatten_obs:
            if not isinstance(obs, dict):
                raise TypeError('Expected dict observation before flattening.')
            obs = _flatten_obs(obs)
        return np.asarray(obs, dtype=np.float32)

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if self._rescale:
            action = np.clip(action, -1.0, 1.0)
            if (self._raw_action_spec.low is not None
                    and self._raw_action_spec.high is not None):
                action = _rescale_action(action, self._raw_action_spec.low,
                                         self._raw_action_spec.high)
        return action

    def reset(self, **kwargs) -> np.ndarray:
        return self._process_obs(self._env.reset(**kwargs))

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(self._process_action(action))
        return self._process_obs(obs), reward, done, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        if hasattr(self._env, 'close'):
            self._env.close()


def prepare_env(env: Any,
                rescale_actions: bool = True,
                seed: Optional[int] = None) -> BaseEnv:
    return BaseEnv(env,
                    rescale_actions=rescale_actions,
                    seed=seed)
