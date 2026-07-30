#!/usr/bin/env python3
"""Train an independent safety critic B and validate critic A's choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jaxrl.agents.safety_critic import (SafetyCritic,
                                        binary_prediction_metrics)
from learner.branch_supervision import (BranchSupervisionDataset,
                                        conformal_upper_offset,
                                        mine_selected_false_safe)
from learner.checkpoint import (restore_training_snapshot,
                                save_training_snapshot)
from learner.control_evaluation import (evaluate_control_facing,
                                        evaluate_double_critic_control)
from learner.counterfactual_dataset import load_counterfactual_artifact
from scripts.train_branch_qsafe import (_checkpoint_ensemble_size,
                                        _support, _templates)
from train.config import load_app_config


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split(',') if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint', required=True, type=Path)
    parser.add_argument('--selector-checkpoint', required=True, type=Path)
    parser.add_argument('--comparison-report', required=True, type=Path)
    parser.add_argument('--branches', required=True, type=Path)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument(
        '--validator-horizon', type=int, default=None,
        help='Optional longer horizon used only to train/calibrate critic B.')
    parser.add_argument('--natural-pretrain-steps', type=int, default=3000)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--ranking-weight', type=float, default=0.25)
    parser.add_argument('--point-batch-size', type=int, default=256)
    parser.add_argument('--pair-batch-size', type=int, default=256)
    parser.add_argument('--bootstrap-fraction', type=float, default=0.8)
    parser.add_argument('--calibration-fraction', type=float, default=0.2)
    parser.add_argument('--hard-negative-fraction', type=float, default=0.5)
    parser.add_argument('--hard-negative-weight', type=float, default=5.0)
    parser.add_argument('--mining-epsilon', type=float, default=0.2)
    parser.add_argument('--conformal-alphas', type=_floats,
                        default=_floats('0.05,0.1,0.2'))
    parser.add_argument('--epsilons', type=_floats,
                        default=_floats('0.1,0.15,0.2'))
    parser.add_argument('--margins', type=_floats,
                        default=_floats('0,0.02,0.05'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    validator_horizon = (
        args.horizon if args.validator_horizon is None
        else int(args.validator_horizon))
    if not 0.0 < args.bootstrap_fraction <= 1.0:
        parser.error('--bootstrap-fraction must be in (0, 1]')
    if not 0.0 < args.calibration_fraction < 1.0:
        parser.error('--calibration-fraction must be in (0, 1)')

    _, cfg, droq_cfg = load_app_config(args.config)
    artifact = load_counterfactual_artifact(args.branches)
    branches = list(artifact['branches'])
    split_report = json.loads(
        args.comparison_report.read_text(encoding='utf-8'))
    validation_ids = {
        int(value) for value in split_report['validation_snapshots']}
    train_branches = [
        item for item in branches
        if int(item.snapshot_index) not in validation_ids]
    validation_branches = [
        item for item in branches
        if int(item.snapshot_index) in validation_ids]
    snapshot_episode_ids = {
        index: int(snapshot.episode_id)
        for index, snapshot in enumerate(artifact['snapshots'])
    }
    train_episode_ids = sorted({
        snapshot_episode_ids[int(item.snapshot_index)]
        for item in train_branches
    })
    if len(train_episode_ids) < 3:
        raise RuntimeError(
            'independent validator needs at least three training episodes')
    split_rng = np.random.default_rng(args.seed + 70_001)
    shuffled_episodes = split_rng.permutation(train_episode_ids)
    calibration_count = min(
        len(train_episode_ids) - 2,
        max(1, int(round(
            len(train_episode_ids) * args.calibration_fraction))))
    calibration_episode_ids = {
        int(value) for value in shuffled_episodes[:calibration_count]}
    fit_pool_episode_ids = [
        int(value) for value in shuffled_episodes[calibration_count:]]
    fit_count = min(
        len(fit_pool_episode_ids),
        max(2, int(round(
            len(fit_pool_episode_ids) * args.bootstrap_fraction))))
    fit_episode_ids = set(
        int(value) for value in split_rng.choice(
            fit_pool_episode_ids, size=fit_count, replace=False))
    fit_branches = [
        item for item in train_branches
        if snapshot_episode_ids[int(item.snapshot_index)]
        in fit_episode_ids]
    calibration_branches = [
        item for item in train_branches
        if snapshot_episode_ids[int(item.snapshot_index)]
        in calibration_episode_ids]
    if not fit_branches or not calibration_branches:
        raise RuntimeError('validator fit/calibration episode split is empty')
    obs_dim = int(np.asarray(branches[0].observation).shape[-1])
    action_dim = int(np.asarray(branches[0].action).shape[-1])
    ensemble_size = _checkpoint_ensemble_size(args.source_checkpoint)

    # Actor and natural replay come from the frozen source checkpoint.
    agent_template, source_safety, replay, reward_replay = _templates(
        cfg, droq_cfg, obs_dim, action_dim, ensemble_size)
    source = restore_training_snapshot(
        args.source_checkpoint, agent=agent_template,
        safety_replay=replay, safety_critic=source_safety)
    agent = source['agent']
    if not len(replay.recent):
        raise RuntimeError('source checkpoint has no natural safety replay')

    # A is loaded exactly as produced by branch training.
    _, selector_template, _, _ = _templates(
        cfg, droq_cfg, obs_dim, action_dim, ensemble_size)
    selector_snapshot = restore_training_snapshot(
        args.selector_checkpoint, agent=agent_template,
        safety_critic=selector_template)
    selector = selector_snapshot['safety_critic']
    fit_observations = np.stack([
        item.observation for item in fit_branches])
    fit_actions = np.stack([item.action for item in fit_branches])
    fit_selector_risks = selector.predict(
        fit_observations, fit_actions)
    fit_supported = _support(agent, fit_branches, cfg)
    hard_negative_keys, hard_negative_stats = mine_selected_false_safe(
        fit_branches, fit_selector_risks,
        horizon=args.horizon, epsilon=args.mining_epsilon,
        support=fit_supported)
    dataset = BranchSupervisionDataset(
        fit_branches, validator_horizon, seed=args.seed + 9001,
        hard_negative_keys=hard_negative_keys)
    print(
        f'[validator-B] fit_episodes={len(fit_episode_ids)} '
        f'calibration_episodes={len(calibration_episode_ids)} '
        f'hard_negatives={len(hard_negative_keys)}',
        flush=True)

    # B is a genuinely independent parameter/optimizer/target/RNG tree.
    validator = SafetyCritic.create(
        seed=args.seed + 120_000,
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=cfg.safety_critic_hidden_dims,
        learning_rate=cfg.safety_critic_learning_rate,
        discount=cfg.safety_discount,
        tau=cfg.safety_critic_tau,
        future_loss_weight=cfg.safety_future_loss_weight,
        ensemble_size=ensemble_size,
        conservative_weight=cfg.safety_conservative_weight,
        conservative_num_actions=cfg.safety_conservative_num_actions)

    pretrain_history = []
    for step in range(1, args.natural_pretrain_steps + 1):
        natural = replay.sample_mixed(cfg.safety_critic_batch_size)
        validator, natural_info = SafetyCritic.update(
            validator, agent.actor, natural)
        if step % 500 == 0 or step == args.natural_pretrain_steps:
            row = {
                'step': step,
                **{
                    key: float(np.asarray(value))
                    for key, value in natural_info.items()
                },
            }
            pretrain_history.append(row)
            print(
                f'[validator-B] natural-pretrain='
                f'{step}/{args.natural_pretrain_steps} '
                f'loss={row["safety_critic_loss"]:.4f}',
                flush=True)

    history = []
    for step in range(1, args.steps + 1):
        natural = replay.sample_mixed(cfg.safety_critic_batch_size)
        validator, natural_info = SafetyCritic.update(
            validator, agent.actor, natural)
        branch_batch = dataset.sample(
            args.point_batch_size, args.pair_batch_size,
            hard_negative_fraction=args.hard_negative_fraction,
            hard_negative_weight=args.hard_negative_weight)
        validator, branch_info = SafetyCritic.update_counterfactual(
            validator, branch_batch,
            ranking_weight=float(args.ranking_weight))
        if step % 200 == 0 or step == args.steps:
            row = {
                'step': step,
                **{
                    f'natural_{key}': float(np.asarray(value))
                    for key, value in natural_info.items()
                },
                **{
                    key: float(np.asarray(value))
                    for key, value in branch_info.items()
                },
            }
            history.append(row)
            print(
                f'[validator-B] step={step}/{args.steps} '
                f'natural={row["natural_safety_critic_loss"]:.4f} '
                f'branch={row["branch_critic_loss"]:.4f} '
                f'pair={row["branch_pair_accuracy"]:.3f}',
                flush=True)

    fit_labels = np.asarray([
        item.outcomes[validator_horizon].failure
        for item in fit_branches], dtype=np.float32)
    validator, calibration = validator.calibrate(
        fit_labels,
        validator.predict_logits(fit_observations, fit_actions))

    conformal_observations = np.stack([
        item.observation for item in calibration_branches])
    conformal_actions = np.stack([
        item.action for item in calibration_branches])
    conformal_labels = np.asarray([
        item.outcomes[validator_horizon].failure
        for item in calibration_branches], dtype=np.float32)
    conformal_scores = validator.predict(
        conformal_observations, conformal_actions)
    conformal_offsets = {
        str(alpha): conformal_upper_offset(
            conformal_labels, conformal_scores, alpha)
        for alpha in args.conformal_alphas
    }

    validation_observations = np.stack([
        item.observation for item in validation_branches])
    validation_actions = np.stack([
        item.action for item in validation_branches])
    validation_labels = np.asarray([
        item.outcomes[args.horizon].failure
        for item in validation_branches], dtype=np.float32)
    validator_validation_labels = np.asarray([
        item.outcomes[validator_horizon].failure
        for item in validation_branches], dtype=np.float32)
    selector_risks = selector.predict(
        validation_observations, validation_actions)
    validator_risks = validator.predict(
        validation_observations, validation_actions)
    supported = _support(agent, validation_branches, cfg)
    selector_single = evaluate_control_facing(
        validation_branches, selector_risks,
        horizon=args.horizon, epsilon=0.2, support=supported)
    validator_single = evaluate_control_facing(
        validation_branches, validator_risks,
        horizon=args.horizon, epsilon=0.2, support=supported)
    grid = []
    for epsilon in args.epsilons:
        for margin in args.margins:
            grid.append(evaluate_double_critic_control(
                validation_branches, selector_risks, validator_risks,
                horizon=args.horizon, epsilon=epsilon,
                improvement_margin=margin, support=supported))
    conformal_grids = {}
    for alpha, offset in conformal_offsets.items():
        upper_risks = np.clip(validator_risks + offset, 0.0, 1.0)
        conformal_grids[alpha] = [
            evaluate_double_critic_control(
                validation_branches, selector_risks, upper_risks,
                horizon=args.horizon, epsilon=epsilon,
                improvement_margin=margin, support=supported)
            for epsilon in args.epsilons
            for margin in args.margins
        ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_checkpoint = save_training_snapshot(
        args.output_root, agent=agent, replay_buffer=reward_replay,
        safety_critic=validator,
        step=(
            int(source['step'])
            + args.natural_pretrain_steps
            + args.steps),
        metadata={
            'experiment_name': 'branch_safety_validator',
            'obs_dim': obs_dim,
            'source_checkpoint': str(args.source_checkpoint.resolve()),
            'selector_checkpoint': str(args.selector_checkpoint.resolve()),
            'independent_validator_seed': args.seed + 120_000,
            'selector_horizon': args.horizon,
            'validator_horizon': validator_horizon,
            'validation_snapshots': sorted(validation_ids),
            'fit_episodes': sorted(fit_episode_ids),
            'calibration_episodes': sorted(calibration_episode_ids),
        })
    natural_eval = replay.sample_recent(min(4096, len(replay.recent)))
    report = {
        'source_checkpoint': str(args.source_checkpoint.resolve()),
        'selector_checkpoint': str(args.selector_checkpoint.resolve()),
        'validator_checkpoint': str(output_checkpoint),
        'branch_artifact': str(args.branches.resolve()),
        'comparison_report': str(args.comparison_report.resolve()),
        'horizon': args.horizon,
        'validator_horizon': validator_horizon,
        'steps': args.steps,
        'natural_pretrain_steps': args.natural_pretrain_steps,
        'ranking_weight': args.ranking_weight,
        'bootstrap_fraction': args.bootstrap_fraction,
        'calibration_fraction': args.calibration_fraction,
        'hard_negative_fraction': args.hard_negative_fraction,
        'hard_negative_weight': args.hard_negative_weight,
        'mining_epsilon': args.mining_epsilon,
        'validator_seed': args.seed + 120_000,
        'validation_snapshots': sorted(validation_ids),
        'fit_episodes': sorted(fit_episode_ids),
        'calibration_episodes': sorted(calibration_episode_ids),
        'unused_train_episodes': sorted(
            set(train_episode_ids)
            - fit_episode_ids
            - calibration_episode_ids),
        'fit_branch_count': len(fit_branches),
        'calibration_branch_count': len(calibration_branches),
        'hard_negative_stats': hard_negative_stats,
        'natural_pretrain_history': pretrain_history,
        'history': history,
        'validator_calibration': calibration,
        'conformal_offsets': conformal_offsets,
        'conformal_calibration_metrics': binary_prediction_metrics(
            conformal_labels, conformal_scores),
        'selector_branch_point_metrics': binary_prediction_metrics(
            validation_labels, selector_risks),
        'validator_branch_point_metrics': binary_prediction_metrics(
            validator_validation_labels, validator_risks),
        'validator_on_selector_horizon_metrics': binary_prediction_metrics(
            validation_labels, validator_risks),
        'validator_natural_metrics': binary_prediction_metrics(
            natural_eval['future_failure_labels'],
            validator.predict(
                natural_eval['observations'], natural_eval['actions'])),
        'selector_single': selector_single,
        'validator_single': validator_single,
        'double_grid': grid,
        'double_conformal_grids': conformal_grids,
    }
    report_path = args.output_root / 'branch_validator_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'[validator-B] report={report_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
