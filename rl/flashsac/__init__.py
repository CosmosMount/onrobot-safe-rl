from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch

from rl.flashsac.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig


_DEFAULT_CONFIG: dict[str, Any] = {
    'normalize_reward': False,
    'normalized_G_max': 10.0,
    'asymmetric_observation': False,
    'device_type': 'cpu',
    'buffer_max_length': 100000,
    'buffer_min_length': 1,
    'buffer_device_type': 'cpu',
    'sample_batch_size': 256,
    'learning_rate_init': 1.0e-6,
    'learning_rate_peak': 3.0e-4,
    'learning_rate_end': 1.0e-6,
    'learning_rate_warmup_rate': 0.0,
    'learning_rate_warmup_step': 1000,
    'learning_rate_decay_rate': 0.0,
    'learning_rate_decay_step': 1000000,
    'actor_num_blocks': 2,
    'actor_hidden_dim': 256,
    'actor_bc_alpha': 0.0,
    'actor_noise_zeta_mu': 2.0,
    'actor_noise_zeta_max': 10,
    'actor_update_period': 1,
    'critic_num_blocks': 2,
    'critic_hidden_dim': 256,
    'critic_num_bins': 101,
    'critic_min_v': -100.0,
    'critic_max_v': 1000.0,
    'critic_target_update_tau': 0.005,
    'temp_initial_value': 0.1,
    'temp_target_sigma': 0.2,
    'temp_target_entropy': 0.0,
    'gamma': 0.99,
    'n_step': 1,
    'use_compile': False,
    'compile_mode': 'reduce-overhead',
    'use_amp': False,
    'load_optimizer': True,
    'load_reward_normalizer': False,
}


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, 'device_buffer') or value.__class__.__module__.startswith('jax'):
        return np.asarray(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class FlashSACBackend:
    agent_type = 'flashsac'
    owns_replay_buffer = True

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        *,
        seed: int = 0,
        env_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        cfg_dict = {**_DEFAULT_CONFIG, **kwargs, 'seed': seed}
        if 'buffer_device_type' not in kwargs:
            cfg_dict['buffer_device_type'] = cfg_dict['device_type']
        cfg = FlashSACConfig(**cfg_dict)
        self._agent = FlashSACAgent(
            observation_space,
            action_space,
            env_info or {},
            cfg,
        )
        self._interaction_step = 0

    def sample_actions(self, observation: np.ndarray) -> tuple[np.ndarray, 'FlashSACBackend']:
        transition = {
            'next_observation': np.asarray(observation, dtype=np.float32)[None, ...],
        }
        action = self._agent.sample_actions(
            self._interaction_step,
            transition,
            training=True,
        )[0]
        self._interaction_step += 1
        return np.asarray(action, dtype=np.float32), self

    def eval_actions(self, observation: np.ndarray) -> np.ndarray:
        transition = {
            'next_observation': np.asarray(observation, dtype=np.float32)[None, ...],
        }
        action = self._agent.sample_actions(
            self._interaction_step,
            transition,
            training=False,
        )[0]
        return np.asarray(action, dtype=np.float32)

    def process_transition(self, transition: Mapping[str, Any]) -> None:
        self._agent.process_transition(
            self._normalize_transition_shape(transition))

    def can_start_training(self) -> bool:
        return self._agent.can_start_training()

    def replay_size(self) -> int:
        return len(self._agent._replay_buffer)

    def update(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        if args or kwargs:
            raise RuntimeError(
                'FlashSAC updates from its internal replay buffer. '
                'Call process_transition() once per environment step, '
                'then call update() without external batches.')
        if not self._agent.can_start_training():
            return {}
        return self._agent.update()

    def state_dict(self) -> dict:
        agent = self._agent
        state = {
            'interaction_step': self._interaction_step,
            'update_step': agent._update_step,
            'actor': self._network_state(agent._actor),
            'critic': self._network_state(agent._critic),
            'target_critic': self._network_state(agent._target_critic),
            'temperature': self._network_state(agent._temperature),
            'grad_scaler': deepcopy(agent._grad_scaler.state_dict()),
            'replay_buffer': agent._replay_buffer.state_dict(),
        }
        if agent.reward_normalizer is not None:
            state['reward_normalizer'] = deepcopy(agent.reward_normalizer.state_dict())
        return state

    def load_state_dict(self, state: dict) -> 'FlashSACBackend':
        agent = self._agent
        self._interaction_step = int(state.get('interaction_step', 0))
        agent._update_step = int(state.get('update_step', 0))
        self._load_network_state(agent._actor, state['actor'])
        self._load_network_state(agent._critic, state['critic'])
        self._load_network_state(agent._target_critic, state['target_critic'])
        self._load_network_state(agent._temperature, state['temperature'])
        agent._grad_scaler.load_state_dict(state.get('grad_scaler', {}))
        if 'replay_buffer' in state:
            agent._replay_buffer.load_state_dict(state['replay_buffer'])
        if (agent.reward_normalizer is not None
                and 'reward_normalizer' in state):
            agent.reward_normalizer.load_state_dict(state['reward_normalizer'])
        return self

    @staticmethod
    def _normalize_transition_shape(
            transition: Mapping[str, Any]) -> dict[str, np.ndarray]:
        normalized: dict[str, np.ndarray] = {}
        for key in (
                'observation',
                'action',
                'reward',
                'terminated',
                'truncated',
                'next_observation'):
            if key not in transition:
                raise KeyError(f'Missing FlashSAC transition field: {key}')
            value = _as_numpy(transition[key]).astype(np.float32)
            if key in {'observation', 'action', 'next_observation'}:
                if value.ndim == 1:
                    value = value[None, ...]
            elif value.ndim == 0:
                value = value[None]
            normalized[key] = value
        return normalized

    @staticmethod
    def _network_state(network) -> dict:
        return {
            'network': deepcopy(network.network.state_dict()),
            'optimizer': deepcopy(network.optimizer.state_dict())
            if network.optimizer is not None else None,
            'scheduler': deepcopy(network.scheduler.state_dict())
            if network.scheduler is not None else None,
            'update_step': network.update_step,
        }

    @staticmethod
    def _load_network_state(network, state: dict) -> None:
        network.network.load_state_dict(state['network'])
        if network.optimizer is not None and state.get('optimizer') is not None:
            network.optimizer.load_state_dict(state['optimizer'])
        if network.scheduler is not None and state.get('scheduler') is not None:
            network.scheduler.load_state_dict(state['scheduler'])
        network.update_step = int(state.get('update_step', 0))


__all__ = ['FlashSACAgent', 'FlashSACBackend', 'FlashSACConfig']
