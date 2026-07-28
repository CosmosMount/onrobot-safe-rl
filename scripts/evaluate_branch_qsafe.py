#!/usr/bin/env python3
"""Evaluate Q_safe with natural calibration and branch-control metrics."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from jaxrl.agents import DroQLearner
from jaxrl.agents.safety_critic import (SafetyCritic,
                                        binary_prediction_metrics)
from jaxrl.data.safety_replay import SafetyReplayManager
from jaxrl.env.specs import BoxSpec
from learner.checkpoint import restore_training_snapshot
from learner.control_evaluation import evaluate_control_facing
from learner.counterfactual_dataset import load_counterfactual_artifact
from train.config import load_app_config


def _checkpoint_ensemble_size(path: str | Path) -> int:
    with Path(path).open('rb') as stream:
        payload = pickle.load(stream)
    try:
        bias = payload['safety_critic_state']['critic']['params'][
            'Dense_0']['bias']
        return int(np.asarray(bias).shape[-1])
    except (KeyError, TypeError, IndexError):
        return 1


def _templates(cfg, droq_cfg, obs_dim, action_dim, ensemble_size):
    observation_spec = BoxSpec(
        shape=(obs_dim,), low=-np.inf, high=np.inf)
    action_spec = BoxSpec(
        shape=(action_dim,), low=-np.ones(action_dim, np.float32),
        high=np.ones(action_dim, np.float32))
    agent = DroQLearner.create(
        cfg.seed, observation_spec, action_spec, **droq_cfg)
    safety = SafetyCritic.create(
        cfg.seed + 10_000, obs_dim, action_dim,
        hidden_dims=cfg.safety_critic_hidden_dims,
        learning_rate=cfg.safety_critic_learning_rate,
        discount=cfg.safety_discount, tau=cfg.safety_critic_tau,
        future_loss_weight=cfg.safety_future_loss_weight,
        ensemble_size=ensemble_size,
        conservative_weight=cfg.safety_conservative_weight,
        conservative_num_actions=cfg.safety_conservative_num_actions)
    replay = SafetyReplayManager(
        recent_capacity=cfg.safety_recent_capacity,
        failure_capacity=cfg.safety_failure_capacity,
        boundary_capacity=cfg.safety_boundary_capacity,
        recovery_capacity=cfg.safety_recovery_capacity,
        all_capacity=cfg.buffer_size,
        failure_history=cfg.safety_failure_history,
        n_step=cfg.safety_critic_n_step,
        failure_horizons=cfg.safety_failure_horizons,
        seed=cfg.seed)
    return agent, safety, replay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--branches', required=True)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--output',
                        default='saved/safety_evaluation/control_metrics.json')
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--k-values', default='4,8,16,32')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--natural-max-samples', type=int, default=4096)
    args = parser.parse_args()

    _, cfg, droq_cfg = load_app_config(args.config)
    artifact = load_counterfactual_artifact(args.branches)
    branches = artifact['branches']
    if not branches:
        raise RuntimeError('branch artifact contains no candidates')
    obs_dim = int(np.asarray(branches[0].observation).shape[-1])
    action_dim = int(np.asarray(branches[0].action).shape[-1])
    agent, safety, replay = _templates(
        cfg, droq_cfg, obs_dim, action_dim,
        _checkpoint_ensemble_size(args.checkpoint))
    snapshot = restore_training_snapshot(
        args.checkpoint, agent=agent, safety_replay=replay,
        safety_critic=safety)
    if 'safety_critic' not in snapshot:
        raise RuntimeError('checkpoint has no safety_critic_state')
    agent = snapshot['agent']
    safety = snapshot['safety_critic']

    observations = np.stack([item.observation for item in branches])
    actions = np.stack([item.action for item in branches])
    risks = safety.predict(observations, actions)
    behavior = agent.actor.apply_fn(
        {'params': agent.actor.params}, observations)
    log_prob = np.asarray(behavior.log_prob(actions))
    if log_prob.ndim > 1:
        log_prob = np.sum(log_prob, axis=-1)
    mode = np.asarray(behavior.mode())
    distance = np.sqrt(np.mean(np.square(actions - mode), axis=-1))
    support = (
        log_prob / action_dim >= cfg.sqrl_min_behavior_log_prob_per_dim
    ) & (distance <= cfg.sqrl_max_nominal_action_distance)

    report = evaluate_control_facing(
        branches, risks, horizon=args.horizon, epsilon=args.epsilon,
        support=support,
        k_values=tuple(int(value) for value in args.k_values.split(',')),
        seed=args.seed)
    natural = {}
    if 'safety_replay' in snapshot and len(replay.recent):
        batch = replay.sample_recent(min(
            args.natural_max_samples, len(replay.recent)))
        natural_risks = safety.predict(
            batch['observations'], batch['actions'])
        natural = binary_prediction_metrics(
            batch['future_failure_labels'], natural_risks)
    report['natural_metrics'] = natural
    report['metadata'] = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'branches': str(Path(args.branches).resolve()),
        'branch_metadata': artifact.get('metadata', {}),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
