#!/usr/bin/env python3
"""Compare an ensemble with frozen critic A/B on identical branch episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jaxrl.agents.safety_critic import binary_prediction_metrics
from learner.checkpoint import restore_training_snapshot
from learner.control_evaluation import (
    evaluate_control_facing,
    evaluate_double_critic_control,
)
from learner.counterfactual_dataset import load_counterfactual_artifact
from learner.episode_bootstrap import filter_branches_by_episode
from scripts.train_branch_qsafe import (
    _checkpoint_ensemble_size,
    _support,
    _templates,
)
from train.config import load_app_config


def _load(path, cfg, droq_cfg, obs_dim, action_dim):
    ensemble_size = _checkpoint_ensemble_size(path)
    agent, safety, _, _ = _templates(
        cfg, droq_cfg, obs_dim, action_dim, ensemble_size)
    restored = restore_training_snapshot(
        path, agent=agent, safety_critic=safety)
    return restored['agent'], restored['safety_critic']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensemble-report', required=True, type=Path)
    parser.add_argument('--branches', required=True, type=Path)
    parser.add_argument('--critic-a', required=True, type=Path)
    parser.add_argument('--critic-b', required=True, type=Path)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    _, cfg, droq_cfg = load_app_config(args.config)
    report = json.loads(
        args.ensemble_report.read_text(encoding='utf-8'))
    artifact = load_counterfactual_artifact(args.branches)
    branches = list(artifact['branches'])
    snapshot_episode_ids = {
        index: int(snapshot.episode_id)
        for index, snapshot in enumerate(artifact['snapshots'])
    }
    validation = filter_branches_by_episode(
        branches, snapshot_episode_ids,
        report['episode_roles']['validation'])
    horizon = int(report['horizon'])
    epsilon = float(report['epsilon'])
    observations = np.stack([item.observation for item in validation])
    actions = np.stack([item.action for item in validation])
    labels = np.asarray([
        item.outcomes[horizon].failure for item in validation],
        dtype=np.float32)
    obs_dim = observations.shape[-1]
    action_dim = actions.shape[-1]

    member_risks = []
    support_agent = None
    for member_report in report['member_reports']:
        member_agent, member = _load(
            Path(member_report['checkpoint']),
            cfg, droq_cfg, obs_dim, action_dim)
        support_agent = member_agent
        member_risks.append(member.predict(observations, actions))
    ensemble_mean = np.mean(np.stack(member_risks), axis=0)
    support = _support(support_agent, validation, cfg)

    _, critic_a = _load(
        args.critic_a, cfg, droq_cfg, obs_dim, action_dim)
    _, critic_b = _load(
        args.critic_b, cfg, droq_cfg, obs_dim, action_dim)
    risks_a = critic_a.predict(observations, actions)
    risks_b = critic_b.predict(observations, actions)

    def single(risks):
        return {
            'point_metrics': binary_prediction_metrics(labels, risks),
            'control_metrics': evaluate_control_facing(
                validation, risks, horizon=horizon,
                epsilon=epsilon, support=support),
        }

    comparison = {
        'ensemble_report': str(args.ensemble_report.resolve()),
        'critic_a': str(args.critic_a.resolve()),
        'critic_b': str(args.critic_b.resolve()),
        'validation_episodes': report['episode_roles']['validation'],
        'validation_branches': len(validation),
        'critic_a_single': single(risks_a),
        'critic_b_single': single(risks_b),
        'independent_a_b': evaluate_double_critic_control(
            validation, risks_a, risks_b,
            horizon=horizon, epsilon=epsilon,
            improvement_margin=0.0, support=support),
        'ensemble_mean': single(ensemble_mean),
        'ensemble_disagreement': report['ensemble_disagreement'],
        'ensemble_uncertainty_gates': report['uncertainty_gates'],
        'ensemble_uncertainty_upper': report['uncertainty_upper'],
        'ensemble_conformal_upper': report['conformal_upper'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2), encoding='utf-8')
    print(f'[ensemble-compare] report={args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
