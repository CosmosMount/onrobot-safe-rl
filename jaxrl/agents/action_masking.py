"""Legacy SQRL-style action masking (withdrawn from the active pipeline).

Kept for unit tests and historical experiments. Runtime entrypoints reject
``--safety-mask``; use Q_safe-only ``safety_eval`` / ``safety_collect``.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


@partial(jax.jit, static_argnames=('num_candidates', 'allow_min_risk_fallback'))
def _select(agent, safety_critic, observation, rng, *,
            num_candidates: int, epsilon_safe: float,
            action_noise_std: float, local_action_std: float,
            risk_penalty: float, action_delta_penalty: float,
            fallback_contraction: float, fallback_emergency_risk: float,
            allow_min_risk_fallback: bool,
            previous_action):
    sample_key, local_key, noise_key, rng = jax.random.split(rng, 4)
    observations = jnp.repeat(
        jnp.asarray(observation)[None, :], num_candidates, axis=0)
    dist = agent.actor.apply_fn(
        {'params': agent.actor.params}, observations)
    candidates = dist.sample(seed=sample_key)
    policy_means = dist.mode()
    local_count = max(0, (num_candidates - 3) // 2)
    # Always retain deterministic and temporally conservative proposals.
    candidates = candidates.at[0].set(policy_means[0])
    candidates = candidates.at[1].set(previous_action)
    candidates = candidates.at[2].set(
        fallback_contraction * previous_action)
    if local_count:
        local = jnp.clip(
            policy_means[3:3 + local_count]
            + local_action_std * jax.random.normal(
                local_key, (local_count, candidates.shape[-1])),
            -1.0, 1.0)
        candidates = candidates.at[3:3 + local_count].set(local)
    candidates = jnp.clip(
        candidates + action_noise_std * jax.random.normal(
            noise_key, (candidates.shape[-1],))[None, :], -1.0, 1.0)
    risks = safety_critic.critic.apply_fn(
        {'params': safety_critic.critic.params}, observations, candidates)
    reward_qs = agent.critic.apply_fn(
        {'params': agent.critic.params}, observations, candidates,
        False)
    reward_q = jnp.mean(reward_qs, axis=0)
    reward_q_normalized = (
        reward_q - jnp.mean(reward_q)) / (jnp.std(reward_q) + 1e-6)
    action_delta = jnp.mean(
        jnp.square(candidates - previous_action[None, :]), axis=-1)
    score = (
        reward_q_normalized
        - risk_penalty * risks
        - action_delta_penalty * action_delta)
    safe = risks <= epsilon_safe
    safe_score = jnp.where(safe, score, -jnp.inf)
    safe_index = jnp.argmax(safe_score)
    min_risk_index = jnp.argmin(risks)
    no_safe = ~jnp.any(safe)
    mean_ok = risks[0] <= fallback_emergency_risk
    previous_ok = risks[1] <= fallback_emergency_risk
    contracted_ok = risks[2] <= fallback_emergency_risk
    # Prefer temporally conservative recoveries over jumping to an arbitrary
    # min-risk candidate that may still be far above epsilon_safe.
    conservative_index = jnp.where(
        previous_ok, jnp.asarray(1),
        jnp.where(
            contracted_ok, jnp.asarray(2),
            jnp.where(mean_ok, jnp.asarray(0), jnp.asarray(1))))
    if allow_min_risk_fallback:
        no_safe_index = jnp.where(
            previous_ok | contracted_ok | mean_ok,
            conservative_index, min_risk_index)
        used_min_risk = no_safe & ~(previous_ok | contracted_ok | mean_ok)
        used_hold = jnp.asarray(False)
    else:
        no_safe_index = conservative_index
        used_min_risk = jnp.asarray(False)
        used_hold = no_safe & ~(previous_ok | contracted_ok | mean_ok)
    selected_index = jnp.where(no_safe, no_safe_index, safe_index)
    return (
        candidates[selected_index],
        {
            'mask_rejected_fraction': 1.0 - jnp.mean(safe),
            'no_safe_candidate': no_safe.astype(jnp.float32),
            'selected_Q_safe': risks[selected_index],
            'candidate_Q_safe_mean': jnp.mean(risks),
            'safe_candidate_count': jnp.sum(safe),
            'selected_reward_Q': reward_q[selected_index],
            'selected_action_delta': jnp.sqrt(
                jnp.sum(jnp.square(
                    candidates[selected_index] - previous_action))),
            'fallback_previous': (
                no_safe & previous_ok).astype(jnp.float32),
            'fallback_contracted': (
                no_safe & ~previous_ok & contracted_ok).astype(jnp.float32),
            'fallback_policy_mean': (
                no_safe & ~previous_ok & ~contracted_ok & mean_ok
            ).astype(jnp.float32),
            'fallback_hold_previous': used_hold.astype(jnp.float32),
            'fallback_min_risk': used_min_risk.astype(jnp.float32),
        },
        rng,
    )


def select_masked_action(agent, safety_critic, observation, rng, *,
                         num_candidates: int = 16,
                         epsilon_safe: float = 0.30,
                         action_noise_std: float = 0.0,
                         previous_action=None,
                         local_action_std: float = 0.15,
                         risk_penalty: float = 1.0,
                         action_delta_penalty: float = 1.0,
                         fallback_contraction: float = 0.9,
                         fallback_emergency_risk: float = 0.5,
                         allow_min_risk_fallback: bool = False):
    if num_candidates < 3:
        raise ValueError('num_candidates must be at least 3')
    if previous_action is None:
        previous_action = np.zeros_like(
            agent.eval_actions(observation), dtype=np.float32)
    action, info, rng = _select(
        agent, safety_critic, observation, rng,
        num_candidates=num_candidates,
        epsilon_safe=float(epsilon_safe),
        action_noise_std=float(action_noise_std),
        local_action_std=float(local_action_std),
        risk_penalty=float(risk_penalty),
        action_delta_penalty=float(action_delta_penalty),
        fallback_contraction=float(fallback_contraction),
        fallback_emergency_risk=float(fallback_emergency_risk),
        allow_min_risk_fallback=bool(allow_min_risk_fallback),
        previous_action=jnp.asarray(previous_action))
    return np.asarray(action), {
        key: float(np.asarray(value)) for key, value in info.items()
    }, rng
