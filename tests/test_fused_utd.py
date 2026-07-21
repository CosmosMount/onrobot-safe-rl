import unittest

import jax
import jax.numpy as jnp
import numpy as np

import gymnasium as gym
from rl.droq import DroQLearner


def _batch(total_batch_size, obs_dim, action_dim):
    key = jax.random.PRNGKey(123)
    keys = jax.random.split(key, 5)
    return {
        'observations': jax.random.normal(
            keys[0], (total_batch_size, obs_dim)),
        'actions': jnp.tanh(jax.random.normal(
            keys[1], (total_batch_size, action_dim))),
        'rewards': jax.random.normal(keys[2], (total_batch_size,)),
        'masks': jax.random.bernoulli(
            keys[3], 0.8, (total_batch_size,)).astype(jnp.float32),
        'dones': jax.random.bernoulli(
            keys[4], 0.2, (total_batch_size,)),
        'next_observations': jax.random.normal(
            jax.random.fold_in(key, 5), (total_batch_size, obs_dim)),
    }


def _sequential_update(agent, batch, utd_ratio):
    new_agent = agent
    mini_batch = None
    critic_info = None
    for index in range(utd_ratio):
        def take_slice(x):
            mini_batch_size = x.shape[0] // utd_ratio
            start = mini_batch_size * index
            return x[start:start + mini_batch_size]

        mini_batch = jax.tree_util.tree_map(take_slice, batch)
        new_agent, critic_info = DroQLearner.update_critic(
            new_agent, mini_batch)

    new_agent, actor_info = DroQLearner.update_actor(new_agent, mini_batch)
    new_agent, temp_info = DroQLearner.update_temperature(
        new_agent, actor_info['entropy'])
    return new_agent, {**actor_info, **critic_info, **temp_info}


class WalkStyleDroQTest(unittest.TestCase):

    def setUp(self):
        self.obs_dim = 5
        self.action_dim = 3
        self.utd_ratio = 4
        self.mini_batch_size = 6
        self.agent = DroQLearner.create(
            42,
            gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_dim,),
                dtype=np.float32,
            ),
            gym.spaces.Box(
                low=-np.ones(self.action_dim, dtype=np.float32),
                high=np.ones(self.action_dim, dtype=np.float32),
                dtype=np.float32,
            ),
            hidden_dims=(16, 16),
            critic_dropout_rate=0.01,
            critic_layer_norm=True,
            init_temperature=0.1,
        )
        self.batch = _batch(
            self.utd_ratio * self.mini_batch_size,
            self.obs_dim,
            self.action_dim,
        )

    def test_update_matches_walk_sequential_reference(self):
        expected_agent, expected_info = _sequential_update(
            self.agent, self.batch, self.utd_ratio)
        actual_agent, actual_info = self.agent.update(
            self.batch, self.utd_ratio)
        jax.block_until_ready(actual_agent)

        expected_leaves = jax.tree_util.tree_leaves(expected_agent)
        actual_leaves = jax.tree_util.tree_leaves(actual_agent)
        self.assertEqual(len(expected_leaves), len(actual_leaves))
        for expected, actual in zip(expected_leaves, actual_leaves):
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected),
                rtol=2e-6, atol=2e-6)

        self.assertEqual(expected_info.keys(), actual_info.keys())
        for key in expected_info:
            np.testing.assert_allclose(
                np.asarray(actual_info[key]),
                np.asarray(expected_info[key]),
                rtol=2e-6, atol=2e-6)

    def test_update_counts_preserve_utd_semantics(self):
        updated, _ = self.agent.update(self.batch, self.utd_ratio)
        jax.block_until_ready(updated)

        self.assertEqual(int(updated.critic.step), self.utd_ratio)
        self.assertEqual(int(updated.actor.step), 1)
        self.assertEqual(int(updated.temp.step), 1)
        self.assertEqual(int(updated.target_critic.step), 0)

    def test_batch_must_split_evenly(self):
        bad_batch = jax.tree_util.tree_map(lambda x: x[:-1], self.batch)
        with self.assertRaises(AssertionError):
            self.agent.update(bad_batch, self.utd_ratio)


if __name__ == '__main__':
    unittest.main()
