#!/usr/bin/env python3
"""Train episode-bootstrap Q_safe members and evaluate uncertainty controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jaxrl.agents.safety_critic import (
    SafetyCritic,
    binary_prediction_metrics,
)
from learner.branch_supervision import (
    BranchSupervisionDataset,
    conformal_upper_offset,
)
from learner.checkpoint import (
    restore_training_snapshot,
    save_training_snapshot,
)
from learner.control_evaluation import evaluate_control_facing
from learner.counterfactual_dataset import load_counterfactual_artifact
from learner.episode_bootstrap import (
    bootstrap_episode_branches,
    filter_branches_by_episode,
    split_episode_roles,
)
from scripts.train_branch_qsafe import (
    _checkpoint_ensemble_size,
    _support,
    _templates,
)
from train.config import load_app_config


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(',') if item.strip()]


def _arrays(branches, horizon):
    return (
        np.stack([item.observation for item in branches]),
        np.stack([item.action for item in branches]),
        np.asarray([
            item.outcomes[horizon].failure for item in branches
        ], dtype=np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', required=True, type=Path)
    parser.add_argument('--branches', required=True, type=Path)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--members', type=int, default=3)
    parser.add_argument('--natural-pretrain-steps', type=int, default=3000)
    parser.add_argument('--branch-steps', type=int, default=300)
    parser.add_argument('--ranking-weight', type=float, default=0.25)
    parser.add_argument('--point-batch-size', type=int, default=256)
    parser.add_argument('--pair-batch-size', type=int, default=256)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument('--temperature-fraction', type=float, default=0.1)
    parser.add_argument('--conformal-fraction', type=float, default=0.1)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--uncertainty-thresholds', type=_floats,
                        default=_floats('0.02,0.05,0.1,0.2'))
    parser.add_argument('--uncertainty-betas', type=_floats,
                        default=_floats('0,1,2'))
    parser.add_argument('--conformal-alphas', type=_floats,
                        default=_floats('0.05,0.1,0.2'))
    parser.add_argument('--seed', type=int, default=314)
    args = parser.parse_args()
    if args.members < 2:
        parser.error('--members must be at least two')

    _, cfg, droq_cfg = load_app_config(args.config)
    artifact = load_counterfactual_artifact(args.branches)
    branches = list(artifact['branches'])
    snapshots = list(artifact['snapshots'])
    if not branches:
        raise RuntimeError('branch artifact is empty')
    snapshot_episode_ids = {
        index: int(snapshot.episode_id)
        for index, snapshot in enumerate(snapshots)
    }
    episode_ids = sorted(set(snapshot_episode_ids.values()))
    roles = split_episode_roles(
        episode_ids,
        validation_fraction=args.validation_fraction,
        temperature_fraction=args.temperature_fraction,
        conformal_fraction=args.conformal_fraction,
        seed=args.seed)
    role_branches = {
        role: filter_branches_by_episode(
            branches, snapshot_episode_ids, ids)
        for role, ids in roles.items()
    }
    for role, selected in role_branches.items():
        if not selected:
            raise RuntimeError(f'{role} split has no branch snapshots')

    obs_dim = int(np.asarray(branches[0].observation).shape[-1])
    action_dim = int(np.asarray(branches[0].action).shape[-1])
    source_ensemble_size = _checkpoint_ensemble_size(
        args.source_checkpoint)
    agent_template, source_safety, source_replay, reward_replay = _templates(
        cfg, droq_cfg, obs_dim, action_dim, source_ensemble_size)
    source = restore_training_snapshot(
        args.source_checkpoint, agent=agent_template,
        safety_replay=source_replay, safety_critic=source_safety)
    agent = source['agent']
    if not len(source_replay.recent):
        raise RuntimeError('source checkpoint has no natural safety replay')

    temperature_observations, temperature_actions, temperature_labels = (
        _arrays(role_branches['temperature'], args.horizon))
    conformal_observations, conformal_actions, conformal_labels = _arrays(
        role_branches['conformal'], args.horizon)
    validation_observations, validation_actions, validation_labels = _arrays(
        role_branches['validation'], args.horizon)
    validation_support = _support(
        agent, role_branches['validation'], cfg)

    args.output_root.mkdir(parents=True, exist_ok=True)
    members = []
    member_reports = []
    for member_index in range(args.members):
        member_seed = args.seed + 100_000 + member_index * 10_000
        bootstrap_branches, bootstrap = bootstrap_episode_branches(
            branches, snapshot_episode_ids, roles['fit'],
            seed=member_seed + 1)
        dataset = BranchSupervisionDataset(
            bootstrap_branches, args.horizon,
            seed=member_seed + 2)

        # Restore a fresh replay manager so every member starts from the same
        # immutable natural dataset while sampling it with its own RNG stream.
        replay_agent_template, replay_safety_template, replay, member_reward = (
            _templates(
                cfg, droq_cfg, obs_dim, action_dim,
                source_ensemble_size))
        restored = restore_training_snapshot(
            args.source_checkpoint, agent=replay_agent_template,
            safety_replay=replay,
            safety_critic=replay_safety_template)
        member_agent = restored['agent']
        # Loading a checkpoint also restores its sampler RNG. Replace only
        # that RNG so members do not receive identical natural anchor batches.
        replay._rng = np.random.default_rng(member_seed + 3)
        member = SafetyCritic.create(
            seed=member_seed,
            observation_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=cfg.safety_critic_hidden_dims,
            learning_rate=cfg.safety_critic_learning_rate,
            discount=cfg.safety_discount,
            tau=cfg.safety_critic_tau,
            future_loss_weight=cfg.safety_future_loss_weight,
            ensemble_size=1,
            conservative_weight=cfg.safety_conservative_weight,
            conservative_num_actions=(
                cfg.safety_conservative_num_actions))

        natural_history = []
        for step in range(1, args.natural_pretrain_steps + 1):
            natural = replay.sample_mixed(
                cfg.safety_critic_batch_size)
            member, info = SafetyCritic.update(
                member, member_agent.actor, natural)
            if step % 1000 == 0 or step == args.natural_pretrain_steps:
                row = {
                    'step': step,
                    **{
                        key: float(np.asarray(value))
                        for key, value in info.items()
                    },
                }
                natural_history.append(row)
                print(
                    f'[ensemble] member={member_index} '
                    f'natural={step}/{args.natural_pretrain_steps} '
                    f'loss={row["safety_critic_loss"]:.4f}',
                    flush=True)

        branch_history = []
        for step in range(1, args.branch_steps + 1):
            natural = replay.sample_mixed(
                cfg.safety_critic_batch_size)
            member, natural_info = SafetyCritic.update(
                member, member_agent.actor, natural)
            branch_batch = dataset.sample(
                args.point_batch_size, args.pair_batch_size)
            member, branch_info = SafetyCritic.update_counterfactual(
                member, branch_batch,
                ranking_weight=args.ranking_weight)
            if step % 100 == 0 or step == args.branch_steps:
                row = {
                    'step': step,
                    'natural_loss': float(np.asarray(
                        natural_info['safety_critic_loss'])),
                    **{
                        key: float(np.asarray(value))
                        for key, value in branch_info.items()
                    },
                }
                branch_history.append(row)
                print(
                    f'[ensemble] member={member_index} '
                    f'branch={step}/{args.branch_steps} '
                    f'loss={row["branch_critic_loss"]:.4f} '
                    f'pair={row["branch_pair_accuracy"]:.3f}',
                    flush=True)

        member, temperature = member.calibrate(
            temperature_labels,
            member.predict_logits(
                temperature_observations, temperature_actions))
        validation_risks = member.predict(
            validation_observations, validation_actions)
        checkpoint = save_training_snapshot(
            args.output_root / f'member_{member_index}',
            agent=member_agent, replay_buffer=member_reward,
            safety_critic=member,
            step=(
                int(source['step'])
                + args.natural_pretrain_steps
                + args.branch_steps),
            metadata={
                'experiment_name': 'episode_bootstrap_qsafe',
                'member_index': member_index,
                'member_seed': member_seed,
                'episode_roles': roles,
                'bootstrap': bootstrap,
                'horizon': args.horizon,
            })
        members.append(member)
        member_reports.append({
            'member_index': member_index,
            'seed': member_seed,
            'checkpoint': str(checkpoint),
            'bootstrap': bootstrap,
            'temperature_calibration': temperature,
            'natural_history': natural_history,
            'branch_history': branch_history,
            'validation_point_metrics': binary_prediction_metrics(
                validation_labels, validation_risks),
            'validation_control_metrics': evaluate_control_facing(
                role_branches['validation'], validation_risks,
                horizon=args.horizon, epsilon=args.epsilon,
                support=validation_support),
        })

    validation_member_risks = np.stack([
        member.predict(validation_observations, validation_actions)
        for member in members
    ])
    conformal_member_risks = np.stack([
        member.predict(conformal_observations, conformal_actions)
        for member in members
    ])
    mean_risks = np.mean(validation_member_risks, axis=0)
    std_risks = np.std(validation_member_risks, axis=0)
    conformal_mean = np.mean(conformal_member_risks, axis=0)
    conformal_offsets = {
        str(alpha): conformal_upper_offset(
            conformal_labels, conformal_mean, alpha)
        for alpha in args.conformal_alphas
    }

    uncertainty_gates = {}
    for threshold in args.uncertainty_thresholds:
        gated_support = validation_support & (std_risks <= threshold)
        uncertainty_gates[str(threshold)] = evaluate_control_facing(
            role_branches['validation'], mean_risks,
            horizon=args.horizon, epsilon=args.epsilon,
            support=gated_support)
    uncertainty_upper = {}
    for beta in args.uncertainty_betas:
        upper = np.clip(mean_risks + beta * std_risks, 0.0, 1.0)
        uncertainty_upper[str(beta)] = {
            'point_metrics': binary_prediction_metrics(
                validation_labels, upper),
            'control_metrics': evaluate_control_facing(
                role_branches['validation'], upper,
                horizon=args.horizon, epsilon=args.epsilon,
                support=validation_support),
        }
    conformal_upper = {}
    for alpha, offset in conformal_offsets.items():
        upper = np.clip(mean_risks + offset, 0.0, 1.0)
        conformal_upper[alpha] = {
            'point_metrics': binary_prediction_metrics(
                validation_labels, upper),
            'control_metrics': evaluate_control_facing(
                role_branches['validation'], upper,
                horizon=args.horizon, epsilon=args.epsilon,
                support=validation_support),
        }

    report = {
        'source_checkpoint': str(args.source_checkpoint.resolve()),
        'branch_artifact': str(args.branches.resolve()),
        'horizon': args.horizon,
        'epsilon': args.epsilon,
        'members': args.members,
        'seed': args.seed,
        'episode_roles': roles,
        'role_branch_counts': {
            role: len(items) for role, items in role_branches.items()
        },
        'member_reports': member_reports,
        'ensemble_validation_point_metrics': binary_prediction_metrics(
            validation_labels, mean_risks),
        'ensemble_validation_control_metrics': evaluate_control_facing(
            role_branches['validation'], mean_risks,
            horizon=args.horizon, epsilon=args.epsilon,
            support=validation_support),
        'ensemble_disagreement': {
            'mean': float(np.mean(std_risks)),
            'median': float(np.median(std_risks)),
            'p90': float(np.quantile(std_risks, 0.9)),
            'p95': float(np.quantile(std_risks, 0.95)),
            'max': float(np.max(std_risks)),
        },
        'temperature_calibration_episodes': roles['temperature'],
        'conformal_calibration_episodes': roles['conformal'],
        'conformal_calibration_metrics': binary_prediction_metrics(
            conformal_labels, conformal_mean),
        'conformal_offsets': conformal_offsets,
        'uncertainty_gates': uncertainty_gates,
        'uncertainty_upper': uncertainty_upper,
        'conformal_upper': conformal_upper,
    }
    report_path = args.output_root / 'branch_ensemble_report.json'
    report_path.write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(f'[ensemble] report={report_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
