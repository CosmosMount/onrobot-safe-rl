import numpy as np
import jax.numpy as jnp

from jaxsac import create_agent
from jaxsac.batch import Batch
from jaxsac.config import AlgorithmConfig
from jaxsac.critic import update as update_critic


def _agent_and_batch(discounts, discount=0.99):
    config = AlgorithmConfig(
        num_qs=2,
        critic_dropout_rate=0.0,
        critic_layer_norm=False,
        hidden_dims=(8,),
        init_temperature=0.1,
        discount=discount,
        sampled_backup=False,
    )
    agent = create_agent((3,), (2,), config, seed=11)
    batch = Batch(
        observations=jnp.zeros((4, 3), dtype=jnp.float32),
        actions=jnp.zeros((4, 2), dtype=jnp.float32),
        rewards=jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        discounts=jnp.asarray(discounts, dtype=jnp.float32),
        next_observations=jnp.ones((4, 3), dtype=jnp.float32),
    )
    return agent, config, batch


def test_critic_applies_configured_discount_to_bootstrap():
    agent_a, config_a, batch_a = _agent_and_batch([1.0] * 4, discount=0.99)
    agent_b, config_b, batch_b = _agent_and_batch([1.0] * 4, discount=0.95)

    _, _, _, target_a = update_critic(
        agent_a, batch_a, config_a, jnp.asarray([1, 2], dtype=jnp.uint32))
    _, _, _, target_b = update_critic(
        agent_b, batch_b, config_b, jnp.asarray([1, 2], dtype=jnp.uint32))

    # The target critic is initialized identically, so the bootstrap ratios
    # must equal the configured discount ratio.
    rewards = np.asarray(batch_a.rewards)
    bootstrap_a = np.asarray(target_a) - rewards
    bootstrap_b = np.asarray(target_b) - rewards
    np.testing.assert_allclose(
        bootstrap_a / bootstrap_b, np.full(4, 0.99 / 0.95), rtol=1e-4,
        atol=1e-4)


def test_terminal_mask_removes_bootstrap_but_preserves_reward():
    agent, config, batch = _agent_and_batch([0.0] * 4, discount=0.99)
    _, _, _, target = update_critic(
        agent, batch, config, jnp.asarray([3, 4], dtype=jnp.uint32))
    np.testing.assert_allclose(np.asarray(target), np.asarray(batch.rewards),
                               rtol=1e-5, atol=1e-5)
