"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from jaxrl.agents.agent import Agent
from jaxrl.agents.sac.temperature import Temperature
from jaxrl.data.dataset import DatasetDict
from jaxrl.distributions import TanhNormal
from jaxrl.networks import MLP, Ensemble, StateActionValue
from jaxrl.networks.common import soft_target_update
from jaxrl.env.specs import BoxSpec, DictSpec, SpaceSpec


def _flatten_obs_spec(observation_spec: SpaceSpec) -> BoxSpec:
    if isinstance(observation_spec, DictSpec):
        return observation_spec.flatten_spec()
    return observation_spec


def _normalize_agent_tree(agent: 'DroQLearner') -> 'DroQLearner':
    """Strip JAX weak types so jitted updates do not recompile every step."""

    def _cast(leaf):
        if isinstance(leaf, jnp.ndarray):
            return jnp.array(leaf, dtype=leaf.dtype)
        return leaf

    return jax.tree.map(_cast, agent)


class DroQLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False)  # See M in RedQ https://arxiv.org/abs/2101.05982
    sampled_backup: bool = struct.field(pytree_node=False)

    @classmethod
    def create(cls,
               seed: int,
               observation_spec: SpaceSpec,
               action_spec: BoxSpec,
               actor_lr: float = 3e-4,
               critic_lr: float = 3e-4,
               temp_lr: float = 3e-4,
               hidden_dims: Sequence[int] = (256, 256),
               discount: float = 0.99,
               tau: float = 0.005,
               num_qs: int = 2,
               num_min_qs: Optional[int] = None,
               critic_dropout_rate: Optional[float] = None,
               critic_layer_norm: bool = False,
               target_entropy: Optional[float] = None,
               init_temperature: float = 1.0,
               sampled_backup: bool = True):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        observation_spec = _flatten_obs_spec(observation_spec)
        action_dim = action_spec.shape[-1]
        observations = observation_spec.zeros()
        actions = jnp.zeros((action_dim, ), dtype=jnp.float32)

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_base_cls = partial(MLP,
                                 hidden_dims=hidden_dims,
                                 activate_final=True)
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.chain(
                                      optax.clip_by_global_norm(1.0),
                                      optax.adam(learning_rate=actor_lr),
                                  ))

        critic_base_cls = partial(MLP,
                                  hidden_dims=hidden_dims,
                                  activate_final=True,
                                  dropout_rate=critic_dropout_rate,
                                  use_layer_norm=critic_layer_norm)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations,
                                        actions)['params']
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.chain(
                                       optax.clip_by_global_norm(1.0),
                                       optax.adam(learning_rate=critic_lr),
                                   ))
        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(apply_fn=target_critic_def.apply,
                                          params=critic_params,
                                          tx=optax.GradientTransformation(
                                              lambda _: None, lambda _: None))

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.chain(
                                     optax.clip_by_global_norm(1.0),
                                     optax.adam(learning_rate=temp_lr),
                                 ))

        return _normalize_agent_tree(
            cls(rng=rng,
                actor=actor,
                critic=critic,
                target_critic=target_critic,
                temp=temp,
                target_entropy=target_entropy,
                tau=tau,
                discount=discount,
                num_qs=num_qs,
                num_min_qs=num_min_qs,
                sampled_backup=sampled_backup))

    @staticmethod
    @jax.jit
    def update_actor(agent,
                     batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(agent.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(
                actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = agent.actor.apply_fn({'params': actor_params},
                                        batch['observations'])
            actions, log_probs = dist.sample_and_log_prob(seed=key)
            qs = agent.critic.apply_fn({'params': agent.critic.params},
                                       batch['observations'],
                                       actions,
                                       True,
                                       rngs={'dropout': key2})  # training=True
            q = qs.mean(axis=0)
            actor_loss = (log_probs *
                          agent.temp.apply_fn({'params': agent.temp.params}) -
                          q).mean()
            return actor_loss, {
                'actor_loss': actor_loss,
                'entropy': -log_probs.mean()
            }

        grads, actor_info = jax.grad(actor_loss_fn,
                                     has_aux=True)(agent.actor.params)
        actor = agent.actor.apply_gradients(grads=grads)

        agent = agent.replace(actor=actor, rng=rng)

        return agent, actor_info

    @staticmethod
    @jax.jit
    def update_temperature(agent,
                           entropy: float) -> Tuple[Agent, Dict[str, float]]:

        def temperature_loss_fn(temp_params):
            temperature = agent.temp.apply_fn({'params': temp_params})
            temp_loss = temperature * (entropy - agent.target_entropy).mean()
            return temp_loss, {
                'temperature': temperature,
                'temperature_loss': temp_loss
            }

        grads, temp_info = jax.grad(temperature_loss_fn,
                                    has_aux=True)(agent.temp.params)
        temp = agent.temp.apply_gradients(grads=grads)

        agent = agent.replace(temp=temp)

        return agent, temp_info

    @staticmethod
    @jax.jit
    def update_critic(
            agent, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        dist = agent.actor.apply_fn({'params': agent.actor.params},
                                    batch['next_observations'])

        rng = agent.rng

        if agent.sampled_backup:
            key, rng = jax.random.split(rng)
            next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
        else:
            next_actions = dist.mode()

        key2, rng = jax.random.split(rng)

        if agent.num_min_qs is None:
            target_params = agent.target_critic.params
        else:
            all_indx = jnp.arange(0, agent.num_qs)
            rng, key = jax.random.split(rng)
            indx = jax.random.choice(key,
                                     a=all_indx,
                                     shape=(agent.num_min_qs, ),
                                     replace=False)
            target_params = jax.tree_util.tree_map(lambda param: param[indx],
                                                   agent.target_critic.params)

        next_qs = agent.target_critic.apply_fn({'params': target_params},
                                               batch['next_observations'],
                                               next_actions,
                                               True,
                                               rngs={'dropout':
                                                     key2})  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch['rewards'] + agent.discount * batch['masks'] * next_q

        if agent.sampled_backup:
            target_q -= agent.discount * batch['masks'] * agent.temp.apply_fn(
                {'params': agent.temp.params}) * next_log_probs

        key3, rng = jax.random.split(rng)

        def critic_loss_fn(
                critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn({'params': critic_params},
                                       batch['observations'],
                                       batch['actions'],
                                       True,
                                       rngs={'dropout': key3})  # training=True
            critic_loss = ((qs - target_q)**2).mean()
            return critic_loss, {'critic_loss': critic_loss, 'q': qs.mean()}

        grads, info = jax.grad(critic_loss_fn,
                               has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)

        target_critic_params = soft_target_update(critic.params,
                                                  agent.target_critic.params,
                                                  agent.tau)
        target_critic = agent.target_critic.replace(
            params=target_critic_params)

        new_agent = agent.replace(critic=critic,
                                  target_critic=target_critic,
                                  rng=rng)

        return new_agent, info

    @staticmethod
    @partial(jax.jit, static_argnames=('utd_ratio', ))
    def _update_fused(agent, batch: DatasetDict, utd_ratio: int):
        """Run one UTD update as a single compiled program."""

        def split_utd_axis(x):
            assert x.shape[0] % utd_ratio == 0
            mini_batch_size = x.shape[0] // utd_ratio
            return x.reshape((utd_ratio, mini_batch_size, *x.shape[1:]))

        mini_batches = jax.tree_util.tree_map(split_utd_axis, batch)

        def critic_step(carry, mini_batch):
            return DroQLearner.update_critic(carry, mini_batch)

        new_agent, critic_infos = jax.lax.scan(
            critic_step, agent, mini_batches)
        critic_info = jax.tree_util.tree_map(lambda x: x[-1], critic_infos)
        actor_batch = jax.tree_util.tree_map(lambda x: x[-1], mini_batches)

        new_agent, actor_info = DroQLearner.update_actor(
            new_agent, actor_batch)
        new_agent, temp_info = DroQLearner.update_temperature(
            new_agent, actor_info['entropy'])

        return new_agent, {**actor_info, **critic_info, **temp_info}

    def update(self, batch: DatasetDict, utd_ratio: int):
        new_agent, info = DroQLearner._update_fused(
            self, batch, utd_ratio)
        return _normalize_agent_tree(new_agent), info
