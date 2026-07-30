#!/usr/bin/env python3
"""Collect exact-state MuJoCo counterfactual branches from a frozen SAC policy."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from jaxrl.agents import DroQLearner
from jaxrl.agents.safety_critic import SafetyCritic
from jaxrl.env.specs import BoxSpec
from learner.active_branch_sampling import (
    ActiveBranchSampler,
    build_active_snapshot_signals,
)
from learner.checkpoint import restore_training_snapshot
from learner.counterfactual_dataset import (
    evaluate_snapshot_candidates,
    make_candidate_actions,
    save_counterfactual_artifact,
)
from train.config import load_app_config
from train.main import apply_move_speed
from train.mujoco_branch import MujocoBranchBackend


DEFAULT_MODEL = (
    '/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml')


def _load_agent(checkpoint: Path, robot_cfg, train_cfg, droq_cfg):
    observation_spec = BoxSpec(
        shape=(robot_cfg.obs_dim,), low=-np.inf, high=np.inf)
    action_spec = BoxSpec(
        shape=(robot_cfg.num_joints,),
        low=-np.ones(robot_cfg.num_joints, np.float32),
        high=np.ones(robot_cfg.num_joints, np.float32))
    template = DroQLearner.create(
        train_cfg.seed, observation_spec, action_spec, **droq_cfg)
    return restore_training_snapshot(
        checkpoint, agent=template)['agent']


def _checkpoint_ensemble_size(path: Path) -> int:
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    return int(np.asarray(
        payload['safety_critic_state']['critic']['params'][
            'Dense_0']['bias']).shape[-1])


def _load_safety(checkpoint: Path, robot_cfg, train_cfg, droq_cfg):
    observation_spec = BoxSpec(
        shape=(robot_cfg.obs_dim,), low=-np.inf, high=np.inf)
    action_spec = BoxSpec(
        shape=(robot_cfg.num_joints,),
        low=-np.ones(robot_cfg.num_joints, np.float32),
        high=np.ones(robot_cfg.num_joints, np.float32))
    agent_template = DroQLearner.create(
        train_cfg.seed, observation_spec, action_spec, **droq_cfg)
    safety_template = SafetyCritic.create(
        train_cfg.seed + 10_000,
        robot_cfg.obs_dim,
        robot_cfg.num_joints,
        hidden_dims=train_cfg.safety_critic_hidden_dims,
        learning_rate=train_cfg.safety_critic_learning_rate,
        discount=train_cfg.safety_discount,
        tau=train_cfg.safety_critic_tau,
        future_loss_weight=train_cfg.safety_future_loss_weight,
        ensemble_size=_checkpoint_ensemble_size(checkpoint),
        conservative_weight=train_cfg.safety_conservative_weight,
        conservative_num_actions=(
            train_cfg.safety_conservative_num_actions))
    snapshot = restore_training_snapshot(
        checkpoint, agent=agent_template,
        safety_critic=safety_template)
    if 'safety_critic' not in snapshot:
        raise RuntimeError(
            f'checkpoint has no safety critic: {checkpoint}')
    return snapshot['safety_critic']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument(
        '--move-speed', type=float,
        help='Override the fixed command speed used by observations and reward.')
    parser.add_argument(
        '--output',
        default='saved/safety_datasets/counterfactual_branches_v1.pkl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--natural-steps', type=int, default=400)
    parser.add_argument(
        '--natural-action-noise-std', type=float, default=0.0,
        help=('Gaussian behavior noise for diverse natural episodes. '
              'Zero preserves the historical deterministic collector.'))
    parser.add_argument(
        '--max-episodes', type=int, default=100,
        help=('Stop after this many natural rollout episodes; episodes end '
              'on failure or --natural-episode-max-steps.'))
    parser.add_argument(
        '--natural-episode-max-steps', type=int, default=0,
        help=('Optional data-collection horizon. A non-failure rollout is '
              'reset at this length without creating a failure label.'))
    parser.add_argument('--snapshot-interval', type=int, default=10)
    parser.add_argument('--settle-seconds', type=float, default=1.0)
    parser.add_argument('--perturbation-count', type=int, default=8)
    parser.add_argument(
        '--policy-sample-count', type=int, default=0,
        help='Additional candidates sampled directly from pi(a|s).')
    parser.add_argument('--perturbation-std', type=float, default=0.15)
    parser.add_argument('--contraction', type=float, default=0.90)
    parser.add_argument('--horizons', default='8,16,32')
    parser.add_argument(
        '--active-selector-checkpoint', type=Path,
        help='Enable decision-boundary snapshot selection with frozen critic A.')
    parser.add_argument(
        '--active-validator-checkpoint', type=Path,
        help='Optional independent critic B used for disagreement probes.')
    parser.add_argument('--active-epsilon', type=float, default=0.20)
    parser.add_argument('--active-risk-boundary-width',
                        type=float, default=0.05)
    parser.add_argument('--active-disagreement-threshold',
                        type=float, default=0.15)
    parser.add_argument('--active-support-boundary-width',
                        type=float, default=0.10)
    parser.add_argument('--active-quota-per-reason', type=int, default=40)
    parser.add_argument('--active-normal-quota', type=int, default=80)
    parser.add_argument('--active-min-snapshot-gap', type=int, default=2)
    parser.add_argument('--active-normal-interval', type=int, default=10)
    parser.add_argument('--active-normal-risk-max', type=float, default=0.10)
    args = parser.parse_args()
    if args.natural_action_noise_std < 0.0:
        parser.error('--natural-action-noise-std must be non-negative')
    if args.natural_episode_max_steps < 0:
        parser.error('--natural-episode-max-steps must be non-negative')

    robot_cfg, train_cfg, droq_cfg = load_app_config(args.config)
    if args.move_speed is not None:
        robot_cfg = apply_move_speed(robot_cfg, args.move_speed)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    agent = _load_agent(
        checkpoint, robot_cfg, train_cfg, droq_cfg)
    backend = MujocoBranchBackend(
        args.model, robot_cfg,
        policy_frequency=train_cfg.control_frequency)
    backend.reset_standing(settle_seconds=args.settle_seconds)
    candidate_rng = np.random.default_rng(args.seed)
    natural_rng = np.random.default_rng(args.seed + 90_000)
    horizons = tuple(int(value) for value in args.horizons.split(','))
    selector = None
    validator = None
    active_sampler = None
    if args.active_selector_checkpoint is not None:
        selector_path = args.active_selector_checkpoint.expanduser().resolve()
        selector = _load_safety(
            selector_path, robot_cfg, train_cfg, droq_cfg)
        if args.active_validator_checkpoint is not None:
            validator_path = (
                args.active_validator_checkpoint.expanduser().resolve())
            validator = _load_safety(
                validator_path, robot_cfg, train_cfg, droq_cfg)
        active_sampler = ActiveBranchSampler(
            quota_per_reason=args.active_quota_per_reason,
            normal_quota=args.active_normal_quota,
            risk_boundary_width=args.active_risk_boundary_width,
            disagreement_threshold=args.active_disagreement_threshold,
            support_boundary_width=args.active_support_boundary_width,
            min_snapshot_gap=args.active_min_snapshot_gap,
            normal_interval=args.active_normal_interval)

    def policy(observation):
        return np.clip(
            np.asarray(agent.eval_actions(observation), dtype=np.float32),
            -1.0, 1.0)

    def candidate_actions(observation, nominal, step):
        candidates = make_candidate_actions(
            nominal, previous, rng=candidate_rng,
            perturbation_count=args.perturbation_count,
            perturbation_std=args.perturbation_std,
            contraction=args.contraction)
        if args.policy_sample_count > 0:
            policy_observations = np.repeat(
                observation[None, :],
                args.policy_sample_count, axis=0)
            distribution = agent.actor.apply_fn(
                {'params': agent.actor.params},
                policy_observations)
            sampled = np.asarray(distribution.sample(
                seed=jax.random.fold_in(
                    jax.random.PRNGKey(args.seed + 70_000),
                    step)), dtype=np.float32)
            candidates.extend([
                ('policy_sample', np.clip(action, -1.0, 1.0))
                for action in sampled
            ])
        return candidates

    snapshots = []
    branches = []
    snapshot_diagnostics = []
    previous = np.zeros(robot_cfg.num_joints, dtype=np.float32)
    executed = previous.copy()
    total_steps = 0
    episode_id = 0
    episode_step = 0
    natural_failures = 0
    previous_near_failure = False
    natural_episode_summaries = []
    episode_action_sum = np.zeros(
        robot_cfg.num_joints, dtype=np.float64)
    episode_action_square_sum = np.zeros_like(episode_action_sum)
    while (total_steps < args.natural_steps
           and episode_id < args.max_episodes):
        observation = backend.observation(
            previous, executed, robot_cfg.move_speed)
        nominal = policy(observation)
        if args.natural_action_noise_std > 0.0:
            nominal = np.clip(
                nominal + natural_rng.normal(
                    0.0, args.natural_action_noise_std,
                    size=nominal.shape),
                -1.0, 1.0).astype(np.float32)
        candidates = None
        selection_reason = None
        selection_triggers = []
        probe = None
        if active_sampler is not None:
            candidates = candidate_actions(
                observation, nominal, total_steps)
            candidate_array = np.stack([
                action for _, action in candidates]).astype(np.float32)
            probe_observations = np.repeat(
                observation[None, :], len(candidates), axis=0)
            selector_risks = np.asarray(selector.predict(
                probe_observations, candidate_array))
            validator_risks = (
                np.asarray(validator.predict(
                    probe_observations, candidate_array))
                if validator is not None else selector_risks)
            distribution = agent.actor.apply_fn(
                {'params': agent.actor.params}, probe_observations)
            log_probability = np.asarray(
                distribution.log_prob(candidate_array))
            if log_probability.ndim > 1:
                log_probability = np.sum(log_probability, axis=-1)
            log_probability_per_dim = (
                log_probability / candidate_array.shape[-1])
            action_distances = np.sqrt(np.mean(np.square(
                candidate_array - np.asarray(distribution.mode())),
                axis=-1))
            supported = (
                log_probability_per_dim
                >= train_cfg.sqrl_min_behavior_log_prob_per_dim
            ) & (
                action_distances
                <= train_cfg.sqrl_max_nominal_action_distance)
            signals = build_active_snapshot_signals(
                selector_risks,
                validator_risks=validator_risks,
                supported=supported,
                behavior_log_prob_per_dim=log_probability_per_dim,
                action_distances=action_distances,
                epsilon=args.active_epsilon,
                min_behavior_log_prob_per_dim=(
                    train_cfg.sqrl_min_behavior_log_prob_per_dim),
                max_nominal_action_distance=(
                    train_cfg.sqrl_max_nominal_action_distance),
                improvement_margin=(
                    train_cfg.sqrl_validation_improvement_margin),
                normal_risk_max=args.active_normal_risk_max,
                near_failure=previous_near_failure)
            selection_reason, selection_triggers = (
                active_sampler.consider(total_steps, signals))
            probe = {
                'selector_risk_min': float(np.min(selector_risks)),
                'selector_risk_max': float(np.max(selector_risks)),
                'selector_nominal_risk': float(selector_risks[0]),
                'validator_risk_min': float(np.min(validator_risks)),
                'validator_risk_max': float(np.max(validator_risks)),
                'validator_nominal_risk': float(validator_risks[0]),
                'max_disagreement': signals.max_disagreement,
                'support_coverage': float(np.mean(supported)),
                'min_risk_boundary_distance': (
                    signals.min_risk_boundary_distance),
                'min_support_boundary_distance': (
                    signals.min_support_boundary_distance),
                'would_replace': signals.would_replace,
                'would_abstain': signals.would_abstain,
                'stable_normal': signals.stable_normal,
            }
        elif total_steps % args.snapshot_interval == 0:
            selection_reason = 'interval'
        if selection_reason is not None:
            if candidates is None:
                candidates = candidate_actions(
                    observation, nominal, total_steps)
            snapshot = backend.snapshot(
                previous_action=previous,
                previous_executed_action=executed,
                command_speed=robot_cfg.move_speed,
                episode_id=episode_id,
                policy_step=episode_step)
            snapshots.append(snapshot)
            snapshot_diagnostics.append({
                'snapshot_index': len(snapshots) - 1,
                'selection_reason': selection_reason,
                'selection_triggers': selection_triggers,
                'episode_id': episode_id,
                'policy_step': episode_step,
                'command_speed': float(robot_cfg.move_speed),
                **(probe or {}),
            })
            branches.extend(evaluate_snapshot_candidates(
                backend, snapshot, candidates, policy,
                snapshot_index=len(snapshots) - 1,
                horizons=horizons))
            # Candidate branches leave the backend at a counterfactual state.
            backend.restore_state(snapshot.simulator_state)
        measurement = backend.step_action(nominal)
        episode_action_sum += nominal
        episode_action_square_sum += np.square(nominal)
        previous = nominal.copy()
        executed = nominal.copy()
        total_steps += 1
        episode_step += 1
        previous_near_failure = bool(measurement.near_failure)
        episode_limit_reached = bool(
            args.natural_episode_max_steps > 0
            and episode_step >= args.natural_episode_max_steps)
        if measurement.failure or episode_limit_reached:
            count = max(episode_step, 1)
            natural_episode_summaries.append({
                'episode_id': episode_id,
                'steps': episode_step,
                'failure': bool(measurement.failure),
                'truncated': bool(
                    episode_limit_reached and not measurement.failure),
                'action_mean': (
                    episode_action_sum / count).astype(float).tolist(),
                'action_rms': np.sqrt(
                    episode_action_square_sum / count
                ).astype(float).tolist(),
            })
            natural_failures += int(measurement.failure)
            episode_id += 1
            episode_step = 0
            episode_action_sum.fill(0.0)
            episode_action_square_sum.fill(0.0)
            if (total_steps < args.natural_steps
                    and episode_id < args.max_episodes):
                backend.reset_standing(settle_seconds=args.settle_seconds)
                previous = np.zeros(
                    robot_cfg.num_joints, dtype=np.float32)
                executed = previous.copy()
                previous_near_failure = False

    if episode_step > 0:
        natural_episode_summaries.append({
            'episode_id': episode_id,
            'steps': episode_step,
            'failure': False,
            'action_mean': (
                episode_action_sum / episode_step).astype(float).tolist(),
            'action_rms': np.sqrt(
                episode_action_square_sum / episode_step
            ).astype(float).tolist(),
        })
    output = save_counterfactual_artifact(
        args.output, snapshots=snapshots, branches=branches,
        metadata={
            'checkpoint': str(checkpoint),
            'model': str(Path(args.model).expanduser().resolve()),
            'seed': args.seed,
            'horizons': horizons,
            'natural_steps_requested': args.natural_steps,
            'natural_steps_completed': total_steps,
            'natural_action_noise_std': args.natural_action_noise_std,
            'natural_episode_max_steps': args.natural_episode_max_steps,
            'natural_failures': natural_failures,
            'episodes_started': episode_id + int(
                total_steps > 0 and episode_step > 0),
            'snapshot_interval': args.snapshot_interval,
            'perturbation_count': args.perturbation_count,
            'policy_sample_count': args.policy_sample_count,
            'perturbation_std': args.perturbation_std,
            'command_speed': robot_cfg.move_speed,
            'active_sampling': active_sampler is not None,
            'active_selector_checkpoint': (
                str(args.active_selector_checkpoint.expanduser().resolve())
                if args.active_selector_checkpoint is not None else None),
            'active_validator_checkpoint': (
                str(args.active_validator_checkpoint.expanduser().resolve())
                if args.active_validator_checkpoint is not None else None),
            'active_sampler': (
                active_sampler.summary()
                if active_sampler is not None else None),
            'snapshot_diagnostics': snapshot_diagnostics,
            'natural_episode_summaries': natural_episode_summaries,
        })
    summary = {
        'output': str(output),
        'snapshots': len(snapshots),
        'branches': len(branches),
        'natural_steps_completed': total_steps,
        'natural_failures': natural_failures,
        'active_sampler': (
            active_sampler.summary()
            if active_sampler is not None else None),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
