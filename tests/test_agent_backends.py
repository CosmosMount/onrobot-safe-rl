import unittest

import jax
import jax.numpy as jnp
import numpy as np

import gymnasium as gym
from rl.droq import DroQLearner
from rl.flashsac import FlashSACBackend


def _spaces(obs_dim=5, action_dim=3):
    return (
        gym.spaces.Box(low=-np.inf,
                       high=np.inf,
                       shape=(obs_dim,),
                       dtype=np.float32),
        gym.spaces.Box(low=-1.0,
                       high=1.0,
                       shape=(action_dim,),
                       dtype=np.float32),
    )


def _batch(total_batch_size=8, obs_dim=5, action_dim=3):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 5)
    return {
        'observations': jax.random.normal(keys[0], (total_batch_size, obs_dim)),
        'actions': jnp.tanh(
            jax.random.normal(keys[1], (total_batch_size, action_dim))),
        'rewards': jax.random.normal(keys[2], (total_batch_size,)),
        'masks': jnp.ones((total_batch_size,), dtype=jnp.float32),
        'dones': jax.random.bernoulli(keys[3], 0.1,
                                      (total_batch_size,)),
        'next_observations': jax.random.normal(
            keys[4], (total_batch_size, obs_dim)),
    }


class AgentBackendTest(unittest.TestCase):

    def _assert_backend_contract(self, agent):
        obs = np.zeros((5,), dtype=np.float32)
        action, next_agent = agent.sample_actions(obs)
        self.assertIsNotNone(next_agent)
        self.assertEqual(action.shape, (3,))
        self.assertTrue(np.all(action <= 1.0))
        self.assertTrue(np.all(action >= -1.0))
        eval_action = next_agent.eval_actions(obs)
        self.assertEqual(eval_action.shape, (3,))
        updated, metrics = next_agent.update(_batch(), utd_ratio=2)
        self.assertIsNotNone(updated)
        self.assertIsInstance(metrics, dict)
        self.assertTrue(metrics)
        restored = updated.load_state_dict(updated.state_dict())
        self.assertEqual(restored.agent_type, updated.agent_type)

    def test_droq_backend_contract(self):
        obs_space, action_space = _spaces()
        agent = DroQLearner.create(
            0,
            obs_space,
            action_space,
            hidden_dims=(16, 16),
            critic_dropout_rate=0.0,
            critic_layer_norm=False,
        )
        self._assert_backend_contract(agent)

    def test_flashsac_backend_contract(self):
        obs_space, action_space = _spaces()
        agent = FlashSACBackend(
            obs_space,
            action_space,
            seed=0,
            device_type='cpu',
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            actor_num_blocks=1,
            critic_num_blocks=1,
            critic_num_bins=21,
            use_compile=False,
        )
        self._assert_backend_contract(agent)


if __name__ == '__main__':
    unittest.main()
