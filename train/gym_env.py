"""Gymnasium environment preprocessing wrappers."""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

import gymnasium as gym

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

        raw_obs_space = env.observation_space
        raw_action_space = env.action_space
        if not isinstance(raw_action_space, gym.spaces.Box):
            raise TypeError('Action space must be Box-like.')

        self._raw_action_space = raw_action_space
        if isinstance(raw_obs_space, gym.spaces.Dict):
            flat_dim = sum(int(np.prod(space.shape))
                           for space in raw_obs_space.spaces.values())
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(flat_dim,),
                dtype=np.float32,
            )
            self._flatten_obs = True
        else:
            self.observation_space = raw_obs_space
            self._flatten_obs = False

        if self._rescale:
            self.action_space = gym.spaces.Box(
                low=-np.ones(raw_action_space.shape, dtype=np.float32),
                high=np.ones(raw_action_space.shape, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = raw_action_space
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int) -> int:
        if hasattr(self._env, 'seed'):
            self._env.seed(seed)
        if hasattr(self.observation_space, 'seed'):
            self.observation_space.seed(seed)
        if hasattr(self.action_space, 'seed'):
            self.action_space.seed(seed)
        return seed

    def sample_action(self) -> np.ndarray:
        return np.asarray(self.action_space.sample(), dtype=np.float32)

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
            action = _rescale_action(action, self._raw_action_space.low,
                                     self._raw_action_space.high)
        return action

    def reset(self, **kwargs) -> tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self._env.reset(**kwargs)
        return self._process_obs(obs), dict(info)

    def step(
        self, action: np.ndarray, during_hold=None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self._env.step(
            self._process_action(action), during_hold=during_hold)
        return self._process_obs(obs), reward, terminated, truncated, info

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
