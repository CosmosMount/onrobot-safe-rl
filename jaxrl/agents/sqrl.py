"""SQRL constrained action selection with structured safe proposals."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _group_metrics(prefix, risks, safe):
    return {
        f'{prefix}_candidate_Q_safe_min': jnp.min(risks),
        f'{prefix}_candidate_Q_safe_mean': jnp.mean(risks),
        f'{prefix}_candidate_Q_safe_max': jnp.max(risks),
        f'{prefix}_candidate_Q_safe_std': jnp.std(risks),
        f'{prefix}_safe_coverage': jnp.mean(safe.astype(jnp.float32)),
    }


def double_critic_replacement_decision(
        selector_risks, validator_risks, support, *,
        selected_index, nominal_index, abstain_index,
        epsilon_safe, improvement_margin):
    """Validate A's one selected action with B, without searching using B."""
    nominal_safe = (
        support[nominal_index]
        & (selector_risks[nominal_index] <= epsilon_safe)
        & (validator_risks[nominal_index] <= epsilon_safe))
    selected_safe = (
        support[selected_index]
        & (selector_risks[selected_index] <= epsilon_safe)
        & (validator_risks[selected_index] <= epsilon_safe))
    selector_improvement = (
        selector_risks[nominal_index] - selector_risks[selected_index])
    validator_improvement = (
        validator_risks[nominal_index] - validator_risks[selected_index])
    is_replacement = selected_index != nominal_index
    replacement_valid = (
        is_replacement
        & selected_safe
        & (selector_improvement >= improvement_margin)
        & (validator_improvement >= improvement_margin))
    final_index = jnp.where(
        nominal_safe, nominal_index,
        jnp.where(replacement_valid, selected_index, abstain_index))
    validation_reject = ~nominal_safe & ~replacement_valid
    return (
        final_index, nominal_safe, replacement_valid, validation_reject,
        selector_improvement, validator_improvement)


@partial(
    jax.jit,
    static_argnames=(
        'num_candidates', 'local_candidate_count',
        'support_gate_enabled'),
)
def _select_sqrl(
        agent, safety_critic, validation_critic, observation, rng, *,
        num_candidates: int,
        local_candidate_count: int,
        epsilon_safe: float,
        candidate_noise_std: float,
        local_action_std: float,
        fallback_contraction: float,
        fallback_emergency_risk: float,
        uncertainty_penalty: float,
        support_gate_enabled: bool,
        min_behavior_log_prob_per_dim: float,
        max_nominal_action_distance: float,
        validation_improvement_margin: float,
        previous_action,
        proposal_action):
    sample_key, noise_key, local_key, structured_noise_key, choice_key, rng = (
        jax.random.split(rng, 6))
    observation = jnp.asarray(observation, dtype=jnp.float32)
    action_dim = previous_action.shape[-1]

    policy_obs = jnp.repeat(observation[None, :], num_candidates, axis=0)
    policy_dist = agent.actor.apply_fn(
        {'params': agent.actor.params}, policy_obs)
    policy, policy_log_probs = policy_dist.sample_and_log_prob(seed=sample_key)
    if policy_log_probs.ndim > 1:
        policy_log_probs = jnp.sum(policy_log_probs, axis=-1)
    policy = jnp.clip(
        policy + candidate_noise_std * jax.random.normal(
            noise_key, policy.shape),
        -1.0, 1.0)

    single_dist = agent.actor.apply_fn(
        {'params': agent.actor.params}, observation[None, :])
    policy_mean = single_dist.mode()[0]
    contracted = fallback_contraction * previous_action
    structured = jnp.stack(
        [policy_mean, previous_action, contracted, proposal_action], axis=0)
    structured = jnp.clip(
        structured + candidate_noise_std * jax.random.normal(
            structured_noise_key, structured.shape),
        -1.0, 1.0)

    local_bases = jnp.where(
        (jnp.arange(local_candidate_count) % 2)[:, None] == 0,
        policy_mean[None, :],
        previous_action[None, :])
    local = jnp.clip(
        local_bases + local_action_std * jax.random.normal(
            local_key, (local_candidate_count, action_dim)),
        -1.0, 1.0)
    # candidate_noise_std models the same execution-side disturbance for every
    # proposal family, not only policy samples.
    local = jnp.clip(
        local + candidate_noise_std * jax.random.normal(
            noise_key, local.shape),
        -1.0, 1.0)

    candidates = jnp.concatenate([policy, local, structured], axis=0)
    observations = jnp.repeat(
        observation[None, :], candidates.shape[0], axis=0)
    behavior_dist = agent.actor.apply_fn(
        {'params': agent.actor.params}, observations)
    behavior_log_probs = behavior_dist.log_prob(candidates)
    if behavior_log_probs.ndim > 1:
        behavior_log_probs = jnp.sum(behavior_log_probs, axis=-1)
    behavior_log_prob_per_dim = behavior_log_probs / action_dim
    nominal_action_distance = jnp.sqrt(jnp.mean(
        jnp.square(candidates - policy_mean[None, :]), axis=-1))
    support = jnp.logical_and(
        behavior_log_prob_per_dim >= min_behavior_log_prob_per_dim,
        nominal_action_distance <= max_nominal_action_distance)
    if not support_gate_enabled:
        support = jnp.ones_like(support)

    risk_logits = safety_critic.critic.apply_fn(
        {'params': safety_critic.critic.params}, observations, candidates,
        return_members=True)
    member_risks = jax.nn.sigmoid(
        risk_logits / jnp.maximum(
            safety_critic.calibration_temperature, 1e-3))
    risk_mean = jnp.mean(member_risks, axis=-1)
    risk_std = jnp.std(member_risks, axis=-1)
    risks = jnp.clip(
        risk_mean + uncertainty_penalty * risk_std, 0.0, 1.0)
    risk_safe = risks <= epsilon_safe
    safe = jnp.logical_and(risk_safe, support)

    policy_end = num_candidates
    local_end = policy_end + local_candidate_count
    structured_start = local_end
    policy_risks, policy_safe = risks[:policy_end], safe[:policy_end]
    local_risks, local_safe = (
        risks[policy_end:local_end], safe[policy_end:local_end])
    structured_risks, structured_safe = (
        risks[structured_start:], safe[structured_start:])

    any_safe = jnp.any(safe)
    any_policy_safe = jnp.any(policy_safe)
    any_local_safe = jnp.any(local_safe)
    masked_logits = jnp.where(policy_safe, policy_log_probs, -1.0e9)
    policy_index = jax.random.categorical(choice_key, masked_logits)
    local_safe_index = policy_end + jnp.argmin(
        jnp.where(local_safe, local_risks, jnp.inf))
    # Non-invasive priority: retain the deterministic policy action whenever
    # it is safe, then temporal continuity, before sampling a different gait.
    mean_safe = structured_safe[0]
    previous_safe = structured_safe[1]
    contracted_safe = structured_safe[2]
    proposal_safe = structured_safe[3]
    structured_priority_index = jnp.where(
        mean_safe, structured_start,
        jnp.where(
            previous_safe, structured_start + 1,
            jnp.where(
                contracted_safe, structured_start + 2,
                structured_start + 3)))
    any_structured_safe = jnp.any(structured_safe)
    safe_index = jnp.where(
        any_structured_safe, structured_priority_index,
        jnp.where(
            any_policy_safe, policy_index, local_safe_index))

    # No-safe fallback never jumps to an arbitrary sampled action. Prefer the
    # least-risk structured proposal; if all are beyond the emergency limit,
    # hold a contracted previous action and notify the supervisor.
    best_structured = jnp.argmin(structured_risks)
    best_structured_index = structured_start + best_structured
    support_abstain = support_gate_enabled & ~any_safe
    emergency = (
        ~any_safe
        & (structured_risks[best_structured] > fallback_emergency_risk))
    contracted_index = structured_start + 2
    fallback_index = jnp.where(
        support_abstain | emergency,
        contracted_index, best_structured_index)
    selected_index = jnp.where(any_safe, safe_index, fallback_index)

    selector_selected_index = selected_index
    if validation_critic is not None:
        validation_logits = validation_critic.critic.apply_fn(
            {'params': validation_critic.critic.params},
            observations, candidates, return_members=True)
        validation_members = jax.nn.sigmoid(
            validation_logits / jnp.maximum(
                validation_critic.calibration_temperature, 1e-3))
        validation_mean = jnp.mean(validation_members, axis=-1)
        validation_std = jnp.std(validation_members, axis=-1)
        validation_risks = jnp.clip(
            validation_mean + uncertainty_penalty * validation_std,
            0.0, 1.0)
        (
            selected_index, double_nominal_safe, double_replacement,
            validation_reject, selector_improvement,
            validator_improvement,
        ) = double_critic_replacement_decision(
            risks, validation_risks, support,
            selected_index=selector_selected_index,
            nominal_index=structured_start,
            abstain_index=contracted_index,
            epsilon_safe=epsilon_safe,
            improvement_margin=validation_improvement_margin)
    else:
        validation_risks = risks
        double_nominal_safe = jnp.asarray(False)
        double_replacement = jnp.asarray(False)
        validation_reject = jnp.asarray(False)
        selector_improvement = jnp.asarray(0.0, dtype=risks.dtype)
        validator_improvement = jnp.asarray(0.0, dtype=risks.dtype)

    info = {
        'sqrl_rejected_fraction':
            1.0 - jnp.mean(safe.astype(jnp.float32)),
        'sqrl_no_safe_candidate': (~any_safe).astype(jnp.float32),
        'selected_Q_safe': risks[selected_index],
        'candidate_Q_safe_mean': jnp.mean(risks),
        'candidate_Q_safe_min': jnp.min(risks),
        'candidate_Q_safe_max': jnp.max(risks),
        'candidate_Q_safe_std': jnp.std(risks),
        'candidate_Q_safe_range': jnp.max(risks) - jnp.min(risks),
        'candidate_Q_safe_disagreement_mean': jnp.mean(risk_std),
        'candidate_Q_safe_disagreement_max': jnp.max(risk_std),
        'safe_candidate_count': jnp.sum(safe.astype(jnp.float32)),
        'sqrl_fallback_min_risk': jnp.asarray(0.0, dtype=jnp.float32),
        'sqrl_fallback_structured':
            ((~any_safe) & ~emergency).astype(jnp.float32),
        'sqrl_emergency_supervisor': emergency.astype(jnp.float32),
        'sqrl_support_coverage': jnp.mean(
            support.astype(jnp.float32)),
        'sqrl_unsupported_candidate_rate': 1.0 - jnp.mean(
            support.astype(jnp.float32)),
        'sqrl_support_abstention': jnp.asarray(
            support_abstain, dtype=jnp.float32),
        'selected_behavior_log_prob_per_dim':
            behavior_log_prob_per_dim[selected_index],
        'selected_nominal_action_distance':
            nominal_action_distance[selected_index],
        'sqrl_double_critic_enabled': jnp.asarray(
            validation_critic is not None, dtype=jnp.float32),
        'sqrl_double_nominal_safe': double_nominal_safe.astype(jnp.float32),
        'sqrl_double_replacement': double_replacement.astype(jnp.float32),
        'sqrl_validation_reject': validation_reject.astype(jnp.float32),
        'selected_Q_safe_A': risks[selected_index],
        'selected_Q_safe_B': validation_risks[selected_index],
        'nominal_Q_safe_A': risks[structured_start],
        'nominal_Q_safe_B': validation_risks[structured_start],
        'sqrl_A_B_selected_disagreement': jnp.abs(
            risks[selected_index] - validation_risks[selected_index]),
        'sqrl_selector_improvement': selector_improvement,
        'sqrl_validator_improvement': validator_improvement,
        'sqrl_selected_group': jnp.where(
            selected_index < policy_end, 0.0,
            jnp.where(selected_index < local_end, 1.0, 2.0)),
    }
    info.update(_group_metrics('policy', policy_risks, policy_safe))
    info.update(_group_metrics('local', local_risks, local_safe))
    info.update(_group_metrics(
        'structured', structured_risks, structured_safe))
    return candidates[selected_index], info, rng


def select_sqrl_action(
        agent, safety_critic, observation, rng, *,
        validation_critic=None,
        num_candidates: int = 64,
        epsilon_safe: float = 0.10,
        candidate_noise_std: float = 0.0,
        previous_action=None,
        proposal_action=None,
        local_candidate_count: int = 8,
        local_action_std: float = 0.10,
        fallback_contraction: float = 0.90,
        fallback_emergency_risk: float = 0.80,
        uncertainty_penalty: float = 1.0,
        support_gate_enabled: bool = False,
        min_behavior_log_prob_per_dim: float = -4.0,
        max_nominal_action_distance: float = 1.0,
        validation_improvement_margin: float = 0.02):
    """Select from policy, local and temporally stable action proposals."""
    if num_candidates < 1:
        raise ValueError('num_candidates must be at least 1')
    if local_candidate_count < 1:
        raise ValueError('local_candidate_count must be at least 1')
    if previous_action is None:
        previous_action = np.asarray(
            agent.eval_actions(observation), dtype=np.float32)
    if proposal_action is None:
        proposal_action = np.zeros_like(previous_action, dtype=np.float32)
    action, info, rng = _select_sqrl(
        agent, safety_critic, validation_critic, observation, rng,
        num_candidates=int(num_candidates),
        local_candidate_count=int(local_candidate_count),
        epsilon_safe=float(epsilon_safe),
        candidate_noise_std=float(candidate_noise_std),
        local_action_std=float(local_action_std),
        fallback_contraction=float(fallback_contraction),
        fallback_emergency_risk=float(fallback_emergency_risk),
        uncertainty_penalty=float(uncertainty_penalty),
        support_gate_enabled=bool(support_gate_enabled),
        min_behavior_log_prob_per_dim=float(
            min_behavior_log_prob_per_dim),
        max_nominal_action_distance=float(
            max_nominal_action_distance),
        validation_improvement_margin=float(
            validation_improvement_margin),
        previous_action=jnp.asarray(previous_action, dtype=jnp.float32),
        proposal_action=jnp.asarray(proposal_action, dtype=jnp.float32),
    )
    return np.asarray(action, dtype=np.float32), {
        key: float(np.asarray(value)) for key, value in info.items()
    }, rng
