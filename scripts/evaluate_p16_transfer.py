#!/usr/bin/env python3
"""Gate a transferred Q_safe on branches from the fresh target SAC actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jaxrl.agents.safety_critic import binary_prediction_metrics
from learner.checkpoint import restore_training_snapshot
from learner.control_evaluation import evaluate_control_facing
from learner.counterfactual_dataset import load_counterfactual_artifact
from learner.p15_protocol import evaluate_p15_gate
from scripts.evaluate_branch_qsafe import (
    _checkpoint_ensemble_size,
    _templates,
)
from train.config import load_app_config


def evaluate_transfer(
        checkpoint: str | Path,
        branches_path: str | Path,
        config: str | Path,
        *,
        horizon: int = 32,
        epsilon: float = 0.20,
        k_values: tuple[int, ...] = (4, 8, 16, 32),
        seed: int = 42,
) -> dict[str, object]:
    _, cfg, droq_cfg = load_app_config(str(config))
    cfg.seed = int(seed)
    artifact = load_counterfactual_artifact(branches_path)
    branches = artifact['branches']
    if not branches:
        raise RuntimeError('target branch artifact contains no candidates')
    observations = np.stack([item.observation for item in branches])
    actions = np.stack([item.action for item in branches])
    labels = np.asarray([
        item.outcomes[horizon].failure for item in branches
    ], dtype=np.float32)
    obs_dim = observations.shape[-1]
    action_dim = actions.shape[-1]
    agent, safety, replay = _templates(
        cfg, droq_cfg, obs_dim, action_dim,
        _checkpoint_ensemble_size(checkpoint))
    snapshot = restore_training_snapshot(
        checkpoint, agent=agent, safety_replay=replay,
        safety_critic=safety)
    agent = snapshot['agent']
    safety = snapshot['safety_critic']
    risks = np.asarray(safety.predict(observations, actions))

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

    point_metrics = binary_prediction_metrics(labels, risks)
    point_metrics['Q_safe_num_samples'] = float(len(labels))
    point_metrics['Q_safe_positive_rate'] = float(np.mean(labels))
    control_metrics = evaluate_control_facing(
        branches, risks, horizon=horizon, epsilon=epsilon,
        support=support, k_values=k_values, seed=seed,
        structured_fallback=True)
    p15_shape = evaluate_p15_gate(point_metrics, control_metrics)
    report: dict[str, object] = {
        'protocol': 'P16',
        'p16_gate_passed': bool(p15_shape['p15_gate_passed']),
        'p16_gate_failed_checks': p15_shape['p15_gate_failed_checks'],
        'p16_gate_checks': p15_shape['p15_gate_checks'],
        'p16_gate_thresholds': p15_shape['p15_gate_thresholds'],
        'natural_metrics': point_metrics,
        'control_metrics': control_metrics,
        # Flat keys are retained because the online gate and result table use
        # the existing control-facing schema.
        **control_metrics,
        'metadata': {
            'checkpoint': str(Path(checkpoint).resolve()),
            'branches': str(Path(branches_path).resolve()),
            'branch_metadata': artifact.get('metadata', {}),
            'gate_population': (
                'fresh-target-actor exact-state candidate branches'),
            'source_replay_used_for_gate': False,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--branches', required=True)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.20)
    parser.add_argument('--k-values', default='4,8,16,32')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    report = evaluate_transfer(
        args.checkpoint, args.branches, args.config,
        horizon=args.horizon, epsilon=args.epsilon,
        k_values=tuple(int(value) for value in args.k_values.split(',')),
        seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
