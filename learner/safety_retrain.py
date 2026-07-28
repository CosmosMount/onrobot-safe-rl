"""Offline / fixed-step Q_safe retraining from frozen episode artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import jax

from jaxrl.agents.safety_critic import (SafetyCritic,
                                        binary_prediction_metrics)
from jaxrl.data.replay_buffer import ReplayBuffer
from jaxrl.data.safety_replay import SafetyReplayManager
from jaxrl.env.specs import BoxSpec
from jaxrl.agents import DroQLearner
from learner.checkpoint import (restore_training_snapshot,
                                save_training_snapshot)
from learner.safety_dataset import (_json_safe, infer_obs_action_dims,
                                    list_safety_episode_artifacts,
                                    load_safety_episode_artifact,
                                    merge_episode_artifacts,
                                    split_episode_artifacts)


def _empty_safety_replay(train_cfg, *, seed: int) -> SafetyReplayManager:
    return SafetyReplayManager(
        recent_capacity=train_cfg.safety_recent_capacity,
        failure_capacity=train_cfg.safety_failure_capacity,
        boundary_capacity=train_cfg.safety_boundary_capacity,
        recovery_capacity=train_cfg.safety_recovery_capacity,
        all_capacity=train_cfg.buffer_size,
        failure_history=train_cfg.safety_failure_history,
        n_step=train_cfg.safety_critic_n_step,
        failure_horizons=train_cfg.safety_failure_horizons,
        seed=seed)


def _build_agent_templates(train_cfg, droq_cfg, obs_dim: int, action_dim: int):
    obs_spec = BoxSpec(shape=(obs_dim,), dtype=np.float32)
    action_spec = BoxSpec(
        shape=(action_dim,), dtype=np.float32,
        low=np.full(action_dim, -1.0, dtype=np.float32),
        high=np.full(action_dim, 1.0, dtype=np.float32))
    agent = DroQLearner.create(
        train_cfg.seed, obs_spec, action_spec, **droq_cfg)
    # SAC reward replay is unused offline; keep a tiny stub for the output
    # snapshot so we do not allocate a million-step empty buffer.
    replay_buffer = ReplayBuffer(obs_spec, action_spec, 1)
    replay_buffer.seed(train_cfg.seed)
    safety_critic = SafetyCritic.create(
        seed=train_cfg.seed + 10_000,
        observation_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=train_cfg.safety_critic_hidden_dims,
        learning_rate=train_cfg.safety_critic_learning_rate,
        discount=train_cfg.safety_discount,
        tau=train_cfg.safety_critic_tau,
        future_loss_weight=train_cfg.safety_future_loss_weight,
        ensemble_size=train_cfg.safety_critic_ensemble_size,
        conservative_weight=train_cfg.safety_conservative_weight,
        conservative_num_actions=train_cfg.safety_conservative_num_actions)
    return agent, replay_buffer, safety_critic


def _evaluate_replay(safety_critic: SafetyCritic,
                     replay: SafetyReplayManager,
                     *,
                     batch_size: int,
                     max_batches: int = 8) -> dict[str, float]:
    if len(replay) == 0:
        return {
            'Q_safe_AUROC': float('nan'),
            'Q_safe_average_precision': float('nan'),
            'Q_safe_calibration_ece': float('nan'),
            'mean_Q_safe': float('nan'),
            'future_positive_rate': float('nan'),
            'num_eval_samples': 0,
        }
    labels = []
    scores = []
    for _ in range(max_batches):
        batch = replay.sample_mixed(min(batch_size, len(replay)))
        pred = safety_critic.predict(batch['observations'], batch['actions'])
        labels.append(np.asarray(batch['future_failure_labels']))
        scores.append(np.asarray(pred))
    label_arr = np.concatenate(labels)
    score_arr = np.concatenate(scores)
    metrics = binary_prediction_metrics(label_arr, score_arr)
    metrics['mean_Q_safe'] = float(np.mean(score_arr))
    metrics['future_positive_rate'] = float(np.mean(label_arr))
    metrics['num_eval_samples'] = int(label_arr.size)
    return metrics


def run_safety_retrain(train_cfg,
                       droq_cfg,
                       *,
                       checkpoint: str,
                       dataset_dir: str | Path,
                       retrain_steps: int = 5000,
                       held_out_seeds: set[int] | None = None,
                       val_seed_fraction: float = 0.2,
                       include_checkpoint_replay: bool = False,
                       save_dir: str | None = None,
                       log_interval: int = 100) -> int:
    """Retrain Q_safe from episode artifacts without touching the robot."""
    dataset_path = Path(dataset_dir)
    artifacts = list_safety_episode_artifacts(dataset_path)
    if not artifacts:
        raise RuntimeError(f'No episode artifacts found in {dataset_path}')

    first = load_safety_episode_artifact(artifacts[0])
    obs_dim, action_dim = infer_obs_action_dims(first)
    agent, replay_buffer, safety_critic = _build_agent_templates(
        train_cfg, droq_cfg, obs_dim, action_dim)
    checkpoint_replay = _empty_safety_replay(train_cfg, seed=train_cfg.seed)
    path = Path(checkpoint)
    # Skip restoring the SAC reward replay: it can be huge and is unused for
    # offline Q_safe updates. Keep an empty buffer only for the output snapshot.
    snapshot = restore_training_snapshot(
        path, agent=agent, replay_buffer=None,
        safety_replay=checkpoint_replay if include_checkpoint_replay else None,
        safety_critic=safety_critic)
    if 'safety_critic_state' not in snapshot:
        print(
            '[safety_retrain] source snapshot has no Q_safe; '
            'training a fresh auxiliary critic while keeping SAC frozen',
            flush=True)
    agent = snapshot['agent']
    safety_critic = snapshot.get('safety_critic', safety_critic)
    source_step = int(snapshot['step'])

    train_paths, val_paths, resolved_held_out = split_episode_artifacts(
        artifacts,
        held_out_seeds=held_out_seeds,
        val_seed_fraction=val_seed_fraction,
        seed=train_cfg.seed + 17)
    if not train_paths:
        raise RuntimeError(
            'Train split is empty. Provide more episodes or fewer held-out '
            f'seeds (held_out={sorted(resolved_held_out)}).')

    train_replay = _empty_safety_replay(
        train_cfg, seed=train_cfg.seed + 101)
    val_replay = _empty_safety_replay(
        train_cfg, seed=train_cfg.seed + 202)
    train_stats = merge_episode_artifacts(train_paths, train_replay)
    val_stats = merge_episode_artifacts(val_paths, val_replay)
    if include_checkpoint_replay and len(checkpoint_replay) > 0:
        train_replay.extend_from_state(checkpoint_replay.state_dict())

    if train_stats['replay_sizes']['all'] == 0 and not include_checkpoint_replay:
        raise RuntimeError('Merged train replay is empty.')
    if (train_stats['outcome_counts'].get('failure', 0) == 0
            or train_stats['outcome_counts'].get('success', 0) == 0):
        print('[safety_retrain] WARNING: train set is not balanced across '
              f'failure/success: {train_stats["outcome_counts"]}', flush=True)

    output_dir = Path(save_dir) if save_dir else Path(
        'saved') / 'checkpoints_qsafe_retrain'
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(train_cfg.safety_critic_batch_size)
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    print(
        f'[safety_retrain] checkpoint={path} source_step={source_step} '
        f'dataset={dataset_path} train_episodes={train_stats["episodes_loaded"]} '
        f'val_episodes={val_stats["episodes_loaded"]} '
        f'held_out_seeds={sorted(resolved_held_out)} '
        f'train_all={train_stats["replay_sizes"]["all"]} '
        f'val_all={val_stats["replay_sizes"]["all"]} '
        f'steps={retrain_steps} include_checkpoint_replay='
        f'{include_checkpoint_replay}',
        flush=True)

    latest_info = {}
    for step in range(1, int(retrain_steps) + 1):
        batch = train_replay.sample_mixed(batch_size)
        safety_critic, latest_info = SafetyCritic.update(
            safety_critic, agent.actor, batch)
        if step % max(1, int(log_interval)) == 0 or step == retrain_steps:
            train_metrics = _evaluate_replay(
                safety_critic, train_replay, batch_size=batch_size)
            val_metrics = _evaluate_replay(
                safety_critic, val_replay, batch_size=batch_size)
            diagnostic_batch = train_replay.sample_mixed(batch_size)
            diagnostic_observations = diagnostic_batch['observations']
            diagnostic_data_risk = safety_critic.predict(
                diagnostic_observations, diagnostic_batch['actions'])
            diagnostic_dist = agent.actor.apply_fn(
                {'params': agent.actor.params}, diagnostic_observations)
            diagnostic_policy_actions = diagnostic_dist.sample(
                seed=jax.random.PRNGKey(train_cfg.seed + step))
            diagnostic_policy_risk = safety_critic.predict(
                diagnostic_observations, diagnostic_policy_actions)
            diagnostic_labels = np.asarray(
                diagnostic_batch['future_failure_labels']) >= 0.5

            def saturation(values):
                values = np.asarray(values)
                return float(np.mean(
                    (values <= 0.01) | (values >= 0.99)))

            row = {
                'step': step,
                'loss': float(latest_info['safety_critic_loss']),
                'td_loss': float(latest_info['safety_td_loss']),
                'future_bce': float(latest_info['safety_future_bce']),
                'conservative_loss': float(
                    latest_info['safety_conservative_loss']),
                'conservative_raw': float(
                    latest_info['safety_conservative_raw']),
                'data_risk': float(latest_info['safety_data_risk']),
                'policy_risk': float(latest_info['safety_policy_risk']),
                'risk_gap': float(
                    latest_info['safety_conservative_risk_gap']),
                'risk_saturation_rate': float(
                    latest_info['safety_risk_saturation_rate']),
                'eval_data_risk': float(np.mean(diagnostic_data_risk)),
                'eval_policy_risk': float(np.mean(diagnostic_policy_risk)),
                'eval_policy_minus_data_risk': float(np.mean(
                    diagnostic_policy_risk - diagnostic_data_risk)),
                'eval_policy_saturation_rate': saturation(
                    diagnostic_policy_risk),
                'eval_normal_saturation_rate': (
                    saturation(diagnostic_data_risk[~diagnostic_labels])
                    if np.any(~diagnostic_labels) else float('nan')),
                'eval_unsafe_saturation_rate': (
                    saturation(diagnostic_data_risk[diagnostic_labels])
                    if np.any(diagnostic_labels) else float('nan')),
                'train': train_metrics,
                'val': val_metrics,
            }
            history.append(row)
            print(
                f'[safety_retrain] step={step}/{retrain_steps} '
                f'loss={row["loss"]:.4f} '
                f'train_AUROC={train_metrics["Q_safe_AUROC"]:.3f} '
                f'val_AUROC={val_metrics["Q_safe_AUROC"]:.3f} '
                f'val_AP={val_metrics["Q_safe_average_precision"]:.3f}',
                flush=True)

    calibration_info: dict[str, float] = {}
    calibrated_metrics: dict[str, float] = {}
    validation_items = list(val_replay.all._items)
    if validation_items:
        validation_observations = np.stack([
            item['observations'] for item in validation_items])
        validation_actions = np.stack([
            item['actions'] for item in validation_items])
        validation_labels = np.asarray([
            item['future_failure_labels'] for item in validation_items])
        validation_logits = safety_critic.predict_logits(
            validation_observations, validation_actions)
        safety_critic, calibration_info = safety_critic.calibrate(
            validation_labels, validation_logits)
        calibrated_metrics = binary_prediction_metrics(
            validation_labels,
            safety_critic.predict(
                validation_observations, validation_actions))
        print(
            '[safety_retrain] calibrated on held-out episodes '
            f'T={calibration_info["Q_safe_calibration_temperature"]:.3f} '
            f'ECE={calibrated_metrics["Q_safe_calibration_ece"]:.3f} '
            f'Brier={calibrated_metrics["Q_safe_brier"]:.3f}',
            flush=True)

    output_step = source_step + int(retrain_steps)
    output = save_training_snapshot(
        output_dir,
        agent=agent,
        replay_buffer=replay_buffer,
        safety_replay=train_replay,
        safety_critic=safety_critic,
        step=output_step,
        metadata={
            'experiment_name': train_cfg.experiment_name,
            'seed': train_cfg.seed,
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'safety_retrain': True,
            'source_checkpoint': str(path.resolve()),
            'source_step': source_step,
            'dataset_dir': str(dataset_path.resolve()),
            'retrain_steps': int(retrain_steps),
            'held_out_seeds': sorted(resolved_held_out),
            'include_checkpoint_replay': bool(include_checkpoint_replay),
        })
    report = {
        'checkpoint': str(output),
        'source_checkpoint': str(path.resolve()),
        'source_step': source_step,
        'output_step': output_step,
        'dataset_dir': str(dataset_path.resolve()),
        'retrain_steps': int(retrain_steps),
        'held_out_seeds': sorted(resolved_held_out),
        'include_checkpoint_replay': bool(include_checkpoint_replay),
        'train': train_stats,
        'val': val_stats,
        'history': history,
        'heldout_calibration': {
            **calibration_info,
            **calibrated_metrics,
        },
        'elapsed_sec': time.perf_counter() - t0,
    }
    report_path = output_dir / f'safety_retrain_report_{output_step:012d}.json'
    report_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding='utf-8')
    print(f'[safety_retrain] saved={output}', flush=True)
    print(f'[safety_retrain] report={report_path}', flush=True)
    return 0
