#!/usr/bin/env python3
"""Train and compare Q_safe with exact-state branch supervision."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from jaxrl.agents import DroQLearner
from jaxrl.agents.safety_critic import (SafetyCritic,
                                        binary_prediction_metrics)
from jaxrl.data.replay_buffer import ReplayBuffer
from jaxrl.data.safety_replay import SafetyReplayManager
from jaxrl.env.specs import BoxSpec
from learner.branch_supervision import (BranchSupervisionDataset,
                                        split_branch_episodes,
                                        split_branch_snapshots)
from learner.checkpoint import (restore_training_snapshot,
                                save_training_snapshot)
from learner.control_evaluation import evaluate_control_facing
from learner.counterfactual_dataset import load_counterfactual_artifact
from train.config import load_app_config


def _weights(value: str) -> list[float]:
    result = [float(item) for item in value.split(',') if item.strip()]
    if not result or any(item < 0.0 for item in result):
        raise argparse.ArgumentTypeError(
            'ranking weights must be non-negative')
    return result


def _checkpoint_ensemble_size(path: Path) -> int:
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    return int(np.asarray(
        payload['safety_critic_state']['critic']['params'][
            'Dense_0']['bias']).shape[-1])


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
    reward_replay = ReplayBuffer(observation_spec, action_spec, 1)
    reward_replay.seed(cfg.seed)
    return agent, safety, replay, reward_replay


def _load_source(checkpoint, cfg, droq_cfg, obs_dim, action_dim,
                 ensemble_size):
    agent, safety, replay, reward_replay = _templates(
        cfg, droq_cfg, obs_dim, action_dim, ensemble_size)
    snapshot = restore_training_snapshot(
        checkpoint, agent=agent, safety_replay=replay,
        safety_critic=safety)
    if 'safety_critic' not in snapshot:
        raise RuntimeError('source checkpoint has no safety critic')
    return (
        snapshot['agent'], snapshot['safety_critic'], replay,
        reward_replay, int(snapshot['step']))


def _support(agent, branches, cfg):
    observations = np.stack([item.observation for item in branches])
    actions = np.stack([item.action for item in branches])
    distribution = agent.actor.apply_fn(
        {'params': agent.actor.params}, observations)
    log_probability = np.asarray(distribution.log_prob(actions))
    if log_probability.ndim > 1:
        log_probability = np.sum(log_probability, axis=-1)
    distance = np.sqrt(np.mean(np.square(
        actions - np.asarray(distribution.mode())), axis=-1))
    return (
        log_probability / actions.shape[-1]
        >= cfg.sqrl_min_behavior_log_prob_per_dim
    ) & (distance <= cfg.sqrl_max_nominal_action_distance)


def _natural_metrics(safety, natural_batch):
    scores = safety.predict(
        natural_batch['observations'], natural_batch['actions'])
    return binary_prediction_metrics(
        natural_batch['future_failure_labels'], scores)


def _evaluate_variant(
        safety, agent, train_branches, validation_branches,
        natural_batch, cfg, *, horizon, epsilon, k_values):
    # Temperature is fit only on train snapshots. Validation branches remain
    # untouched for both ranking and threshold-sensitive selector metrics.
    train_observations = np.stack([
        item.observation for item in train_branches])
    train_actions = np.stack([item.action for item in train_branches])
    train_labels = np.asarray([
        item.outcomes[horizon].failure for item in train_branches],
        dtype=np.float32)
    natural_before = _natural_metrics(safety, natural_batch)
    safety, calibration = safety.calibrate(
        train_labels,
        safety.predict_logits(train_observations, train_actions))
    natural_after = _natural_metrics(safety, natural_batch)

    observations = np.stack([
        item.observation for item in validation_branches])
    actions = np.stack([item.action for item in validation_branches])
    labels = np.asarray([
        item.outcomes[horizon].failure
        for item in validation_branches], dtype=np.float32)
    risks = safety.predict(observations, actions)
    point_metrics = binary_prediction_metrics(labels, risks)
    all_candidates = evaluate_control_facing(
        validation_branches, risks, horizon=horizon, epsilon=epsilon,
        support=np.ones(len(validation_branches), dtype=bool),
        k_values=k_values, seed=cfg.seed + 401)
    supported = evaluate_control_facing(
        validation_branches, risks, horizon=horizon, epsilon=epsilon,
        support=_support(agent, validation_branches, cfg),
        k_values=k_values, seed=cfg.seed + 401)
    return safety, {
        'temperature_calibration': calibration,
        'natural_before_branch_calibration': natural_before,
        'natural_after_branch_calibration': natural_after,
        'validation_branch_point_metrics': point_metrics,
        'validation_all_candidates': all_candidates,
        'validation_support_aware': supported,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--branches', required=True, type=Path)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--output-root',
                        default='saved/branch_qsafe_comparison')
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--point-batch-size', type=int, default=256)
    parser.add_argument('--pair-batch-size', type=int, default=256)
    parser.add_argument('--ranking-weights', type=_weights,
                        default=_weights('0,0.25,1.0'))
    parser.add_argument('--ranking-margin', type=float, default=0.0)
    parser.add_argument('--natural-update-interval', type=int, default=1)
    parser.add_argument('--validation-fraction', type=float, default=0.2)
    parser.add_argument(
        '--split-unit', choices=('episode', 'snapshot'), default='episode',
        help='Episode split is stricter and prevents adjacent-state leakage.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k-values', default='4,8,16,32')
    args = parser.parse_args()

    _, cfg, droq_cfg = load_app_config(args.config)
    artifact = load_counterfactual_artifact(args.branches)
    branches = list(artifact['branches'])
    validation_episodes = []
    if args.split_unit == 'episode':
        snapshot_episode_ids = {
            index: int(snapshot.episode_id)
            for index, snapshot in enumerate(artifact['snapshots'])
        }
        (
            train_branches, validation_branches, validation_ids,
            validation_episodes,
        ) = split_branch_episodes(
            branches, snapshot_episode_ids,
            validation_fraction=args.validation_fraction,
            seed=args.seed)
    else:
        train_branches, validation_branches, validation_ids = (
            split_branch_snapshots(
                branches, validation_fraction=args.validation_fraction,
                seed=args.seed))
    dataset = BranchSupervisionDataset(
        train_branches, args.horizon, seed=args.seed + 101)
    obs_dim = int(np.asarray(branches[0].observation).shape[-1])
    action_dim = int(np.asarray(branches[0].action).shape[-1])
    ensemble_size = _checkpoint_ensemble_size(args.checkpoint)
    k_values = tuple(int(value) for value in args.k_values.split(','))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_agent, base_safety, base_replay, _, source_step = _load_source(
        args.checkpoint, cfg, droq_cfg, obs_dim, action_dim, ensemble_size)
    if not len(base_replay.recent):
        raise RuntimeError(
            'source checkpoint needs recent safety replay for anchor metrics')
    natural_batch = base_replay.sample_recent(
        min(4096, len(base_replay.recent)))
    _, baseline_metrics = _evaluate_variant(
        base_safety, base_agent, train_branches, validation_branches,
        natural_batch, cfg, horizon=args.horizon, epsilon=args.epsilon,
        k_values=k_values)
    report = {
        'source_checkpoint': str(args.checkpoint.resolve()),
        'branch_artifact': str(args.branches.resolve()),
        'horizon': args.horizon,
        'epsilon': args.epsilon,
        'source_step': source_step,
        'train_snapshots': sorted({
            int(item.snapshot_index) for item in train_branches}),
        'validation_snapshots': validation_ids,
        'split_unit': args.split_unit,
        'validation_episodes': validation_episodes,
        'train_branches': len(train_branches),
        'validation_branches': len(validation_branches),
        'baseline': baseline_metrics,
        'variants': {},
    }

    for ranking_weight in args.ranking_weights:
        label = f'ranking_{ranking_weight:g}'.replace('.', 'p')
        print(
            f'[branch-q-safe] variant={label} steps={args.steps} '
            f'train={len(train_branches)} val={len(validation_branches)}',
            flush=True)
        agent, safety, replay, reward_replay, _ = _load_source(
            args.checkpoint, cfg, droq_cfg, obs_dim, action_dim,
            ensemble_size)
        history = []
        started = time.perf_counter()
        latest = {}
        for step in range(1, args.steps + 1):
            if (args.natural_update_interval > 0
                    and step % args.natural_update_interval == 0):
                natural = replay.sample_mixed(
                    cfg.safety_critic_batch_size)
                safety, _ = SafetyCritic.update(
                    safety, agent.actor, natural)
            batch = dataset.sample(
                args.point_batch_size, args.pair_batch_size)
            safety, info = SafetyCritic.update_counterfactual(
                safety, batch, float(ranking_weight),
                float(args.ranking_margin))
            latest = {
                key: float(np.asarray(value))
                for key, value in info.items()
            }
            if step % 100 == 0 or step == args.steps:
                history.append({'step': step, **latest})
                print(
                    f'[branch-q-safe] {label} step={step}/{args.steps} '
                    f'loss={latest["branch_critic_loss"]:.4f} '
                    f'pair={latest["branch_pair_accuracy"]:.3f}',
                    flush=True)

        safety, metrics = _evaluate_variant(
            safety, agent, train_branches, validation_branches,
            natural_batch, cfg, horizon=args.horizon,
            epsilon=args.epsilon, k_values=k_values)
        variant_dir = output_root / label
        checkpoint = save_training_snapshot(
            variant_dir, agent=agent, replay_buffer=reward_replay,
            safety_critic=safety, step=source_step + args.steps,
            metadata={
                'experiment_name': 'branch_qsafe',
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'source_checkpoint': str(args.checkpoint.resolve()),
                'branch_artifact': str(args.branches.resolve()),
                'ranking_weight': ranking_weight,
                'branch_steps': args.steps,
                'validation_snapshots': validation_ids,
                'split_unit': args.split_unit,
                'validation_episodes': validation_episodes,
            })
        report['variants'][label] = {
            'ranking_weight': ranking_weight,
            'ranking_margin': args.ranking_margin,
            'steps': args.steps,
            'natural_update_interval': args.natural_update_interval,
            'history': history,
            'metrics': metrics,
            'checkpoint': str(checkpoint),
            'elapsed_sec': time.perf_counter() - started,
        }

    report_path = output_root / 'branch_qsafe_comparison.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'[branch-q-safe] report={report_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
