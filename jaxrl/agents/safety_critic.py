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
    ensemble_size: int = 1

    @nn.compact
    def __call__(self, observations, actions, *, return_logits: bool = False,
                 return_members: bool = False):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        features = self.base_cls()(inputs)
        logits = nn.Dense(
            self.ensemble_size, kernel_init=default_init())(features)
        if return_members:
            return logits
        logits = jnp.mean(logits, axis=-1)
        if return_logits:
            return logits
        return jax.nn.sigmoid(logits)


class SafetyCritic(struct.PyTreeNode):
    critic: TrainState
    target_critic: TrainState
    rng: jax.Array
    discount: float
    tau: float
    calibration_temperature: jax.Array
    future_loss_weight: float = struct.field(pytree_node=False)
    ensemble_size: int = struct.field(pytree_node=False)
    conservative_weight: float = struct.field(pytree_node=False)
    conservative_num_actions: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, seed: int, observation_dim: int, action_dim: int,
               hidden_dims: Sequence[int] = (256, 256),
               learning_rate: float = 3e-4, discount: float = 0.99,
               tau: float = 0.005,
               future_loss_weight: float = 0.5,
               ensemble_size: int = 1,
               conservative_weight: float = 0.0,
               conservative_num_actions: int = 4) -> 'SafetyCritic':
        if ensemble_size < 1:
            raise ValueError('safety critic ensemble_size must be positive')
        if conservative_weight < 0.0:
            raise ValueError('conservative_weight must be non-negative')
        if conservative_num_actions < 1:
            raise ValueError('conservative_num_actions must be positive')
        rng = jax.random.PRNGKey(seed)
        rng, init_key = jax.random.split(rng)
        base = partial(MLP, hidden_dims=tuple(hidden_dims),
                       activate_final=True)
        critic_def = SafetyStateActionValue(
            base_cls=base, ensemble_size=int(ensemble_size))
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
                   future_loss_weight=float(future_loss_weight),
                   calibration_temperature=jnp.asarray(
                       1.0, dtype=jnp.float32),
                   ensemble_size=int(ensemble_size),
                   conservative_weight=float(conservative_weight),
                   conservative_num_actions=int(conservative_num_actions))

    @staticmethod
    @jax.jit
    def update(safety: 'SafetyCritic', actor: TrainState,
               batch: dict[str, jax.Array]):
        key, noise_key, bootstrap_key, rng = jax.random.split(
            safety.rng, 4)
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

        # Draw current-policy actions at the same states.  This branch is
        # compile-time disabled at alpha=0, preserving Stage-1 behavior and
        # avoiding extra actor/critic work in the baseline.
        if safety.conservative_weight > 0.0:
            conservative_key = jax.random.fold_in(safety.rng, 0xC05A)
            count = safety.conservative_num_actions
            policy_observations = jnp.repeat(
                batch['observations'], count, axis=0)
            policy_dist = actor.apply_fn(
                {'params': actor.params}, policy_observations)
            policy_actions = policy_dist.sample(seed=conservative_key)
        else:
            policy_observations = batch['observations'][:1]
            policy_actions = batch['actions'][:1]

        def loss_fn(params):
            logits = safety.critic.apply_fn(
                {'params': params}, batch['observations'], batch['actions'],
                return_members=True)
            q = jax.nn.sigmoid(logits)
            weights = batch.get(
                'importance_weights', jnp.ones_like(target))[:, None]
            weights = weights / jnp.maximum(jnp.mean(weights), 1e-6)
            bootstrap = jax.random.bernoulli(
                bootstrap_key, 0.8, logits.shape).astype(jnp.float32)
            # A one-head critic remains exactly backward-compatible.
            bootstrap = jnp.where(
                safety.ensemble_size == 1,
                jnp.ones_like(bootstrap), bootstrap)
            effective_weights = weights * bootstrap
            target_members = target[:, None]
            td_loss = (
                jnp.sum(effective_weights * jnp.square(
                    q - target_members))
                / jnp.maximum(jnp.sum(effective_weights), 1.0))
            future_labels = batch['future_failure_labels']
            # BCE-with-logits: keeps gradient when sigmoid saturates near 0/1.
            # (Clipping probabilities before log zeros the grad and can freeze
            # a collapsed Q_safe head.)
            future_bce_values = optax.sigmoid_binary_cross_entropy(
                logits, future_labels[:, None])
            future_bce = (
                jnp.sum(effective_weights * future_bce_values)
                / jnp.maximum(jnp.sum(effective_weights), 1.0))
            data_risk = jnp.mean(q)
            if safety.conservative_weight > 0.0:
                policy_logits = safety.critic.apply_fn(
                    {'params': params}, policy_observations, policy_actions,
                    return_members=True)
                policy_risk_values = jax.nn.sigmoid(policy_logits)
                policy_risk = jnp.mean(policy_risk_values)
                conservative_raw = conservative_risk_regularizer(
                    data_risk, policy_risk)
                conservative_loss = (
                    safety.conservative_weight * conservative_raw)
                saturation_rate = jnp.mean(jnp.logical_or(
                    policy_risk_values <= 0.01,
                    policy_risk_values >= 0.99))
            else:
                policy_risk = data_risk
                conservative_raw = jnp.asarray(0.0, dtype=q.dtype)
                conservative_loss = jnp.asarray(0.0, dtype=q.dtype)
                saturation_rate = jnp.mean(jnp.logical_or(
                    q <= 0.01, q >= 0.99))
            loss = (
                td_loss
                + safety.future_loss_weight * future_bce
                + conservative_loss)
            source = batch['source_ids']
            failure_mask = source == 2
            boundary_mask = source == 1
            normal_mask = jnp.logical_not(
                jnp.logical_or(failure_mask, boundary_mask))

            def masked_mean(values, mask):
                count = jnp.maximum(jnp.sum(mask), 1)
                return jnp.sum(jnp.where(mask, values, 0.0)) / count

            q_mean = jnp.mean(q, axis=-1)
            return loss, {
                'safety_critic_loss': loss,
                'safety_td_loss': td_loss,
                'safety_future_bce': future_bce,
                'safety_conservative_loss': conservative_loss,
                'safety_conservative_raw': conservative_raw,
                'safety_data_risk': data_risk,
                'safety_policy_risk': policy_risk,
                'safety_conservative_risk_gap': policy_risk - data_risk,
                'safety_risk_saturation_rate': saturation_rate,
                'safety_n_step_mean': jnp.mean(batch['n_step_steps']),
                'safety_backup_noise_std_mean': jnp.mean(noise_std),
                'mean_Q_safe': jnp.mean(q_mean),
                'Q_safe_failure': masked_mean(q_mean, failure_mask),
                'Q_safe_boundary': masked_mean(q_mean, boundary_mask),
                'Q_safe_normal': masked_mean(q_mean, normal_mask),
                'Q_safe_ensemble_disagreement': jnp.mean(
                    jnp.std(q, axis=-1)),
                'safety_target_mean': jnp.mean(target),
                'safety_importance_weight_std': jnp.std(weights),
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
        probabilities, _ = self.predict_with_uncertainty(
            observations, actions)
        return probabilities

    def predict_with_uncertainty(
            self, observations, actions) -> tuple[np.ndarray, np.ndarray]:
        member_logits = self.critic.apply_fn(
            {'params': self.critic.params}, observations, actions,
            return_members=True)
        temperature = max(
            float(np.asarray(self.calibration_temperature)), 1e-3)
        members = jax.nn.sigmoid(member_logits / temperature)
        return (
            np.asarray(jnp.mean(members, axis=-1)),
            np.asarray(jnp.std(members, axis=-1)))

    def predict_logits(self, observations, actions) -> np.ndarray:
        member_logits = self.critic.apply_fn(
            {'params': self.critic.params}, observations, actions,
            return_members=True)
        return np.asarray(jnp.mean(member_logits, axis=-1))

    def calibrate(self, labels, logits) -> tuple['SafetyCritic', dict[str, float]]:
        temperature, before, after = fit_temperature(labels, logits)
        return self.replace(
            calibration_temperature=jnp.asarray(
                temperature, dtype=jnp.float32)), {
                    'Q_safe_calibration_temperature': temperature,
                    'Q_safe_calibration_log_loss_before': before,
                    'Q_safe_calibration_log_loss_after': after,
                }


def conservative_risk_regularizer(data_risk, policy_risk):
    """CQL-style loss for a critic where larger values mean greater risk.

    Minimizing E_D[Q_safe] - E_pi[Q_safe] pushes sampled policy/OOD actions
    upward relative to dataset actions.  The commonly quoted reward-CQL sign
    is reversed because Q_safe is a cost/risk rather than a reward.
    """
    return jnp.mean(data_risk) - jnp.mean(policy_risk)


def fit_temperature(labels, logits) -> tuple[float, float, float]:
    """Fit one positive temperature on a natural-distribution holdout."""
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.all(labels == labels[0]):
        return 1.0, np.nan, np.nan

    def log_loss(temperature):
        scaled = np.clip(logits / temperature, -30.0, 30.0)
        return float(np.mean(
            np.logaddexp(0.0, scaled) - labels * scaled))

    # A dense log grid is deterministic, dependency-free and robust for a
    # scalar calibration parameter.
    temperatures = np.exp(np.linspace(np.log(0.05), np.log(20.0), 401))
    losses = np.asarray([log_loss(value) for value in temperatures])
    index = int(np.argmin(losses))
    return float(temperatures[index]), log_loss(1.0), float(losses[index])


def binary_prediction_metrics(labels, scores) -> dict[str, float]:
    """Dependency-free AUROC, average precision, and calibration error."""
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    result = {
        'Q_safe_AUROC': np.nan,
        'Q_safe_average_precision': np.nan,
        'Q_safe_calibration_ece': np.nan,
        'Q_safe_brier': np.nan,
        'Q_safe_log_loss': np.nan,
        'Q_safe_num_samples': float(labels.size),
        'Q_safe_positive_rate': (
            float(np.mean(labels)) if labels.size else np.nan),
    }
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
    for index, low in enumerate(np.linspace(0.0, 0.9, 10)):
        high = low + 0.1
        mask = (
            (probabilities >= low) & (probabilities < high)
            if index < 9 else
            (probabilities >= low) & (probabilities <= high))
        if np.any(mask):
            ece += (np.mean(mask) * abs(
                probabilities[mask].mean() - labels[mask].mean()))
    result['Q_safe_calibration_ece'] = float(ece)
    if labels.size:
        result['Q_safe_brier'] = float(np.mean(
            np.square(probabilities - labels)))
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        result['Q_safe_log_loss'] = float(-np.mean(
            labels * np.log(clipped)
            + (1 - labels) * np.log(1 - clipped)))
    return result
