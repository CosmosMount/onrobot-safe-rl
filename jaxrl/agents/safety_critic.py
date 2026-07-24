"""Auxiliary SQRL-style safety critic.

This state is deliberately separate from the reward learner: updating it cannot
change the SAC actor, reward critics, temperature, or executed action.
"""

from __future__ import annotations

from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState

from jaxrl.networks import MLP
from jaxrl.networks.common import soft_target_update
from jaxrl.networks.common import default_init


class SafetyStateActionValue(nn.Module):
    """Failure probability head used by the SQRL Bellman recursion."""
    base_cls: object

    @nn.compact
    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        features = self.base_cls()(inputs)
        logits = nn.Dense(1, kernel_init=default_init())(features)
        return jnp.squeeze(jax.nn.sigmoid(logits), -1)


class SafetyCritic(struct.PyTreeNode):
    critic: TrainState
    target_critic: TrainState
    rng: jax.Array
    discount: float
    tau: float
    future_loss_weight: float = struct.field(pytree_node=False)

    @classmethod
    def create(cls, seed: int, observation_dim: int, action_dim: int,
               hidden_dims: Sequence[int] = (256, 256),
               learning_rate: float = 3e-4, discount: float = 0.99,
               tau: float = 0.005,
               future_loss_weight: float = 0.5) -> 'SafetyCritic':
        rng = jax.random.PRNGKey(seed)
        rng, init_key = jax.random.split(rng)
        base = partial(MLP, hidden_dims=tuple(hidden_dims),
                       activate_final=True)
        critic_def = SafetyStateActionValue(base_cls=base)
        observations = jnp.zeros((observation_dim,), dtype=jnp.float32)
        actions = jnp.zeros((action_dim,), dtype=jnp.float32)
        params = critic_def.init(init_key, observations, actions)['params']
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=params,
            tx=optax.chain(optax.clip_by_global_norm(1.0),
                           optax.adam(learning_rate)))
        target = TrainState.create(
            apply_fn=critic_def.apply, params=params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None))
        return cls(critic=critic, target_critic=target, rng=rng,
                   discount=float(discount), tau=float(tau),
                   future_loss_weight=float(future_loss_weight))

    @staticmethod
    @jax.jit
    def update(safety: 'SafetyCritic', actor: TrainState,
               batch: dict[str, jax.Array]):
        key, noise_key, rng = jax.random.split(safety.rng, 3)
        next_dist = actor.apply_fn({'params': actor.params},
                                   batch['n_step_next_observations'])
        next_actions = next_dist.sample(seed=key)
        noise_std = batch['behavior_noise_std']
        next_actions = jnp.clip(
            next_actions + jax.random.normal(
                noise_key, next_actions.shape) * noise_std[:, None],
            -1.0, 1.0)
        next_q = safety.target_critic.apply_fn(
            {'params': safety.target_critic.params},
            batch['n_step_next_observations'], next_actions)
        unsafe = batch['n_step_unsafe_labels']
        discount = jnp.power(safety.discount, batch['n_step_steps'])
        target = (
            unsafe
            + (1.0 - unsafe) * discount * batch['n_step_masks'] * next_q)
        target = jax.lax.stop_gradient(target)

        def loss_fn(params):
            q = safety.critic.apply_fn({'params': params},
                                       batch['observations'],
                                       batch['actions'])
            td_loss = jnp.mean(jnp.square(q - target))
            future_labels = batch['future_failure_labels']
            future_bce = -jnp.mean(
                future_labels * jnp.log(jnp.clip(q, 1e-6, 1.0))
                + (1.0 - future_labels)
                * jnp.log(jnp.clip(1.0 - q, 1e-6, 1.0)))
            loss = td_loss + safety.future_loss_weight * future_bce
            source = batch['source_ids']
            failure_mask = source == 2
            boundary_mask = source == 1
            normal_mask = jnp.logical_not(
                jnp.logical_or(failure_mask, boundary_mask))

            def masked_mean(values, mask):
                count = jnp.maximum(jnp.sum(mask), 1)
                return jnp.sum(jnp.where(mask, values, 0.0)) / count

            return loss, {
                'safety_critic_loss': loss,
                'safety_td_loss': td_loss,
                'safety_future_bce': future_bce,
                'safety_n_step_mean': jnp.mean(batch['n_step_steps']),
                'safety_backup_noise_std_mean': jnp.mean(noise_std),
                'mean_Q_safe': jnp.mean(q),
                'Q_safe_failure': masked_mean(q, failure_mask),
                'Q_safe_boundary': masked_mean(q, boundary_mask),
                'Q_safe_normal': masked_mean(q, normal_mask),
                'safety_target_mean': jnp.mean(target),
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(safety.critic.params)
        critic = safety.critic.apply_gradients(grads=grads)
        target_params = soft_target_update(
            critic.params, safety.target_critic.params, safety.tau)
        return safety.replace(
            critic=critic,
            target_critic=safety.target_critic.replace(params=target_params),
            rng=rng), info

    def predict(self, observations, actions) -> np.ndarray:
        q = self.critic.apply_fn({'params': self.critic.params},
                                 observations, actions)
        return np.asarray(q)


def binary_prediction_metrics(labels, scores) -> dict[str, float]:
    """Dependency-free AUROC, average precision, and calibration error."""
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    result = {'Q_safe_AUROC': np.nan, 'Q_safe_average_precision': np.nan,
              'Q_safe_calibration_ece': np.nan}
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives and negatives:
        order = np.argsort(scores)
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1)
        result['Q_safe_AUROC'] = float(
            (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
            / (positives * negatives))
        descending = np.argsort(-scores)
        sorted_labels = labels[descending]
        precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
        result['Q_safe_average_precision'] = float(
            precision[sorted_labels == 1].mean())
    probabilities = np.clip(scores, 0.0, 1.0)
    ece = 0.0
    for low in np.linspace(0.0, 0.9, 10):
        mask = (probabilities >= low) & (probabilities < low + 0.1)
        if np.any(mask):
            ece += (np.mean(mask) * abs(
                probabilities[mask].mean() - labels[mask].mean()))
    result['Q_safe_calibration_ece'] = float(ece)
    return result
