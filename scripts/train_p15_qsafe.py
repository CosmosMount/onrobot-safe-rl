#!/usr/bin/env python3
"""Train P15 Q_safe beside a frozen common SAC actor/reward checkpoint."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from flax import serialization
import numpy as np

from jaxrl.agents import DroQLearner
from jaxrl.agents.safety_critic import (
    SafetyCritic,
    binary_prediction_metrics,
)
from jaxrl.data.replay_buffer import ReplayBuffer
from jaxrl.data.safety_replay import SafetyReplayManager
from jaxrl.env.specs import BoxSpec
from learner.branch_supervision import (
    BranchSupervisionDataset,
    split_branch_episodes_three_way,
)
from learner.checkpoint import (
    agent_state_hash,
    restore_training_snapshot,
    save_training_snapshot,
    snapshot_agent_hash,
)
from learner.control_evaluation import evaluate_control_facing
from learner.counterfactual_dataset import load_counterfactual_artifact
from learner.p15_protocol import (
    evaluate_p15_gate,
    split_safety_items_by_speed_episode,
)
from train.config import load_app_config


def _speed_bins(min_speed: float, max_speed: float,
                increment: float) -> list[float]:
    count = int(round((max_speed - min_speed) / increment)) + 1
    result = [
        round(min_speed + index * increment, 6)
        for index in range(count)
    ]
    if not result or abs(result[-1] - max_speed) > 1e-6:
        raise ValueError('speed range is not divisible by increment')
    return result


def _templates(cfg, droq_cfg, obs_dim, action_dim):
    observation_spec = BoxSpec(
        shape=(obs_dim,), low=-np.inf, high=np.inf)
    action_spec = BoxSpec(
        shape=(action_dim,),
        low=-np.ones(action_dim, np.float32),
        high=np.ones(action_dim, np.float32))
    agent = DroQLearner.create(
        cfg.seed, observation_spec, action_spec, **droq_cfg)
    safety = SafetyCritic.create(
        cfg.seed + 10_000,
        obs_dim,
        action_dim,
        hidden_dims=cfg.safety_critic_hidden_dims,
        learning_rate=cfg.safety_critic_learning_rate,
        discount=cfg.safety_discount,
        tau=cfg.safety_critic_tau,
        future_loss_weight=cfg.safety_future_loss_weight,
        ensemble_size=cfg.safety_critic_ensemble_size,
        conservative_weight=cfg.safety_conservative_weight,
        conservative_num_actions=cfg.safety_conservative_num_actions)
    safety_replay = SafetyReplayManager(
        recent_capacity=cfg.safety_recent_capacity,
        failure_capacity=cfg.safety_failure_capacity,
        boundary_capacity=cfg.safety_boundary_capacity,
        recovery_capacity=cfg.safety_recovery_capacity,
        all_capacity=cfg.buffer_size,
        failure_history=cfg.safety_failure_history,
        n_step=cfg.safety_critic_n_step,
        failure_horizons=cfg.safety_failure_horizons,
        seed=cfg.seed)
    reward_replay = ReplayBuffer(
        observation_spec, action_spec, cfg.buffer_size)
    reward_replay.seed(cfg.seed)
    return agent, safety, safety_replay, reward_replay


def _stack(items: list[dict[str, object]]) -> dict[str, np.ndarray]:
    if not items:
        raise ValueError('cannot stack an empty safety split')
    return {
        key: np.stack([item[key] for item in items])
        for key in items[0]
    }


def _speed_subset(items, speed_bins, speed):
    bins = np.asarray(speed_bins, dtype=np.float64)
    target = int(np.argmin(np.abs(bins - float(speed))))
    return [
        item for item in items
        if int(np.argmin(np.abs(
            bins - float(item.get('command_speeds', 0.0))))) == target
    ]


def _training_replay_from_split(
        source: SafetyReplayManager,
        cfg,
        speed_bins,
        split_manifest,
) -> SafetyReplayManager:
    training = SafetyReplayManager(
        recent_capacity=cfg.safety_recent_capacity,
        failure_capacity=cfg.safety_failure_capacity,
        boundary_capacity=cfg.safety_boundary_capacity,
        recovery_capacity=cfg.safety_recovery_capacity,
        all_capacity=cfg.buffer_size,
        failure_history=cfg.safety_failure_history,
        n_step=cfg.safety_critic_n_step,
        failure_horizons=cfg.safety_failure_horizons,
        seed=cfg.seed + 171)
    bins = np.asarray(speed_bins, dtype=np.float64)
    train_episodes = {
        (bin_id, int(episode_id))
        for bin_id, speed in enumerate(speed_bins)
        for episode_id in split_manifest['speed_episode_splits'][
            f'{speed:.2f}']['train']
    }

    def keep(item):
        bin_id = int(np.argmin(np.abs(
            bins - float(item.get('command_speeds', 0.0)))))
        return (bin_id, int(item.get('episode_ids', 0))) in train_episodes

    state = source.state_dict()
    for name in ('recent', 'failure', 'boundary', 'all'):
        state[name] = dict(state[name])
        state[name]['items'] = [
            item for item in state[name]['items'] if keep(item)]
    # Recovery is retained only as an isolated audit buffer. It is never one
    # of the sources sampled by sample_mixed_by_speed.
    state['history'] = []
    state['nstep_history'] = []
    training.load_state_dict(state)
    return training


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


def _per_speed_branch_split(branches, snapshots, speed_bins, seed):
    train = []
    calibration = []
    validation = []
    manifests = {}
    bins = np.asarray(speed_bins, dtype=np.float64)
    for bin_id, speed in enumerate(speed_bins):
        subset = [
            item for item in branches
            if int(np.argmin(np.abs(
                bins - float(item.command_speed)))) == bin_id
        ]
        snapshot_episode_ids = {
            int(item.snapshot_index):
                int(snapshots[int(item.snapshot_index)].episode_id)
            for item in subset
        }
        split = split_branch_episodes_three_way(
            subset, snapshot_episode_ids, seed=seed + bin_id * 1009)
        speed_train, speed_calibration, speed_validation, manifest = split
        train.extend(speed_train)
        calibration.extend(speed_calibration)
        validation.extend(speed_validation)
        manifests[f'{speed:.2f}'] = manifest
    return train, calibration, validation, manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--common-checkpoint', required=True, type=Path)
    parser.add_argument('--branches', required=True, type=Path)
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cmd-min', type=float, default=0.30)
    parser.add_argument('--cmd-max', type=float, default=1.00)
    parser.add_argument('--cmd-increment', type=float, default=0.05)
    parser.add_argument(
        '--gate-speeds', default='0.40,0.50,0.80,1.00')
    parser.add_argument('--min-natural-transitions', type=int, default=1600)
    parser.add_argument('--min-branch-snapshots', type=int, default=40)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--point-batch-size', type=int, default=256)
    parser.add_argument('--pair-batch-size', type=int, default=256)
    parser.add_argument('--ranking-weight', type=float, default=1.0)
    parser.add_argument('--ranking-margin', type=float, default=0.0)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--epsilon', type=float, default=0.20)
    args = parser.parse_args()

    _, cfg, droq_cfg = load_app_config(args.config)
    cfg.seed = args.seed
    bins = _speed_bins(args.cmd_min, args.cmd_max, args.cmd_increment)
    gate_speeds = [
        float(value) for value in args.gate_speeds.split(',')
        if value.strip()]
    artifact = load_counterfactual_artifact(args.branches)
    branches = list(artifact['branches'])
    snapshots = list(artifact['snapshots'])
    artifact_hash = (artifact.get('metadata') or {}).get(
        'common_actor_hash')
    common_hash = snapshot_agent_hash(args.common_checkpoint)
    if artifact_hash != common_hash:
        raise RuntimeError(
            'branch artifact was not collected by this common SAC actor: '
            f'artifact={artifact_hash} checkpoint={common_hash}')
    for speed in bins:
        count = len({
            int(item.snapshot_index) for item in branches
            if abs(float(item.command_speed) - speed) <= 1e-6
        })
        if count < args.min_branch_snapshots:
            raise RuntimeError(
                f'speed {speed:.2f} has only {count} branch snapshots')

    obs_dim = int(np.asarray(branches[0].observation).shape[-1])
    action_dim = int(np.asarray(branches[0].action).shape[-1])
    agent, safety, safety_replay, reward_replay = _templates(
        cfg, droq_cfg, obs_dim, action_dim)
    snapshot = restore_training_snapshot(
        args.common_checkpoint,
        agent=agent,
        replay_buffer=reward_replay,
        safety_replay=safety_replay,
        safety_critic=safety)
    agent = snapshot['agent']
    safety = snapshot.get('safety_critic', safety)
    source_step = int(snapshot['step'])
    restored_hash = agent_state_hash(serialization.to_state_dict(agent))
    if restored_hash != common_hash:
        raise RuntimeError('restored common agent hash changed unexpectedly')

    natural_items = list(safety_replay.all._items)
    natural_coverage = {}
    for speed in bins:
        count = len(_speed_subset(natural_items, bins, speed))
        natural_coverage[f'{speed:.2f}'] = count
        if count < args.min_natural_transitions:
            raise RuntimeError(
                f'speed {speed:.2f} has only {count} natural transitions')
    natural_split = split_safety_items_by_speed_episode(
        natural_items, bins, seed=args.seed)
    natural_train, natural_calibration, natural_validation, natural_manifest = (
        natural_split)
    training_replay = _training_replay_from_split(
        safety_replay, cfg, bins, natural_manifest)
    branch_split = _per_speed_branch_split(
        branches, snapshots, bins, args.seed)
    train_branches, calibration_branches, validation_branches, (
        branch_manifests) = branch_split
    branch_label_coverage = {}
    for split_name, split_branches in (
            ('train', train_branches),
            ('calibration', calibration_branches),
            ('validation', validation_branches)):
        labels = np.asarray([
            item.outcomes[args.horizon].failure
            for item in split_branches], dtype=np.float32)
        branch_label_coverage[split_name] = {
            'branches': int(len(labels)),
            'failures': int(np.sum(labels)),
            'non_failures': int(np.sum(labels < 0.5)),
        }
    train_labels = branch_label_coverage['train']
    if (train_labels['failures'] == 0
            or train_labels['non_failures'] == 0):
        args.output_root.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_root / 'p15_data_gate_failure.json'
        _failure = {
            'protocol': 'P15',
            'reason': 'branch-train-split-missing-class',
            'branch_label_coverage': branch_label_coverage,
            'natural_coverage': natural_coverage,
            'common_actor_hash': common_hash,
        }
        failure_path.write_text(
            json.dumps(_failure, indent=2), encoding='utf-8')
        raise RuntimeError(
            'P15 Q_safe data gate failed: branch train split needs both '
            f'failure and non-failure labels; report={failure_path}')
    dataset = BranchSupervisionDataset(
        train_branches, args.horizon, seed=args.seed + 101)

    history = []
    for step in range(1, args.steps + 1):
        natural_batch = training_replay.sample_mixed_by_speed(
            cfg.safety_critic_batch_size, bins)
        safety, natural_info = SafetyCritic.update(
            safety, agent.actor, natural_batch)
        branch_batch = dataset.sample(
            args.point_batch_size, args.pair_batch_size)
        safety, branch_info = SafetyCritic.update_counterfactual(
            safety, branch_batch, args.ranking_weight,
            args.ranking_margin)
        if step % 100 == 0 or step == args.steps:
            row = {
                'step': step,
                **{key: float(np.asarray(value))
                   for key, value in natural_info.items()},
                **{key: float(np.asarray(value))
                   for key, value in branch_info.items()},
            }
            history.append(row)
            print(
                f'[P15 Q_safe] {step}/{args.steps} '
                f'natural={row["safety_critic_loss"]:.4f} '
                f'branch={row["branch_critic_loss"]:.4f} '
                f'pair={row["branch_pair_accuracy"]:.3f}',
                flush=True)

    calibration_batch = _stack(natural_calibration)
    safety, calibration = safety.calibrate(
        calibration_batch['future_failure_labels'],
        safety.predict_logits(
            calibration_batch['observations'],
            calibration_batch['actions']))

    gates = {}
    gate_dir = args.output_root / 'gates'
    gate_dir.mkdir(parents=True, exist_ok=True)
    for speed in gate_speeds:
        natural_items_at_speed = _speed_subset(
            natural_validation, bins, speed)
        natural_batch = _stack(natural_items_at_speed)
        natural_scores = safety.predict(
            natural_batch['observations'], natural_batch['actions'])
        natural_metrics = binary_prediction_metrics(
            natural_batch['future_failure_labels'], natural_scores)
        branch_subset = [
            item for item in validation_branches
            if abs(float(item.command_speed) - speed)
            <= args.cmd_increment * 0.25
        ]
        observations = np.stack([
            item.observation for item in branch_subset])
        actions = np.stack([item.action for item in branch_subset])
        risks = safety.predict(observations, actions)
        control_metrics = evaluate_control_facing(
            branch_subset,
            risks,
            horizon=args.horizon,
            epsilon=args.epsilon,
            support=_support(agent, branch_subset, cfg),
            k_values=(4, 8, 16, 32),
            seed=args.seed + int(round(speed * 1000)),
            structured_fallback=True)
        gate = evaluate_p15_gate(natural_metrics, control_metrics)
        gate.update({
            'protocol': 'P15',
            'command_speed': speed,
            'common_checkpoint': str(
                args.common_checkpoint.expanduser().resolve()),
            'common_actor_hash': common_hash,
            'branch_artifact': str(args.branches.expanduser().resolve()),
        })
        gate_path = gate_dir / f'v{int(round(speed * 100)):03d}.json'
        gate_path.write_text(json.dumps(gate, indent=2), encoding='utf-8')
        gates[f'{speed:.2f}'] = {
            **gate,
            'path': str(gate_path.resolve()),
        }

    output_checkpoint = save_training_snapshot(
        args.output_root / 'checkpoint',
        agent=agent,
        replay_buffer=reward_replay,
        safety_replay=safety_replay,
        safety_critic=safety,
        step=source_step,
        metadata={
            'experiment_name': 'p15_qsafe_pretrain',
            'protocol': 'P15',
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'source_checkpoint': str(
                args.common_checkpoint.expanduser().resolve()),
            'common_actor_hash': common_hash,
            'speed_bins': bins,
            'natural_coverage': natural_coverage,
            'natural_split_fingerprint': natural_manifest['fingerprint'],
            'branch_split_fingerprints': {
                speed: value['fingerprint']
                for speed, value in branch_manifests.items()},
            'q_safe_steps': args.steps,
            'ranking_weight': args.ranking_weight,
        })
    output_hash = snapshot_agent_hash(output_checkpoint)
    if output_hash != common_hash:
        raise RuntimeError(
            'Q_safe training changed the common actor/reward state')
    report = {
        'protocol': 'P15',
        'common_checkpoint': str(
            args.common_checkpoint.expanduser().resolve()),
        'q_safe_checkpoint': str(output_checkpoint.resolve()),
        'common_actor_hash': common_hash,
        'output_actor_hash': output_hash,
        'speed_bins': bins,
        'natural_coverage': natural_coverage,
        'natural_split_manifest': natural_manifest,
        'branch_split_manifests': branch_manifests,
        'branch_label_coverage': branch_label_coverage,
        'train_branches': len(train_branches),
        'calibration_branches': len(calibration_branches),
        'validation_branches': len(validation_branches),
        'calibration': calibration,
        'history': history,
        'gates': gates,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / 'p15_qsafe_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({
        'report': str(report_path.resolve()),
        'checkpoint': str(output_checkpoint.resolve()),
        'actor_hash_match': output_hash == common_hash,
        'gates': {
            speed: value['p15_gate_passed']
            for speed, value in gates.items()},
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
