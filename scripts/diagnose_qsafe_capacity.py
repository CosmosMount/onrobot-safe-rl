#!/usr/bin/env python3
"""Offline Q_safe calibration, action sensitivity, and capacity diagnostics.

The diagnostic intentionally never talks to the controller.  It reconstructs
episodes from the ordered ``all`` safety replay stored in the final SQRL
fine-tuning snapshots, holds out complete episodes at every command speed, and
then:

1. evaluates the currently trained Q_safe on the natural held-out prior;
2. measures within-state risk variation over policy/local/uniform actions;
3. trains several Q_safe MLP sizes from scratch on the same frozen split.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jaxrl.agents.safety_critic import SafetyCritic
from learner.checkpoint import restore_training_snapshot
from learner.safety_retrain import _build_agent_templates
from train.config import load_app_config


DEFAULT_SPEEDS = (0.40, 0.50, 0.60, 0.80, 1.00)
DEFAULT_ARCHITECTURES = (
    (256, 256),
    (512, 512),
    (512, 512, 256),
)
REQUIRED_BATCH_KEYS = (
    'observations',
    'actions',
    'n_step_next_observations',
    'n_step_unsafe_labels',
    'n_step_masks',
    'n_step_steps',
    'behavior_noise_std',
    'future_failure_labels',
)


@dataclass(frozen=True)
class Episode:
    speed: float
    index: int
    items: tuple[dict[str, Any], ...]

    @property
    def failed(self) -> bool:
        return any(
            float(item['future_failure_labels']) >= 0.5
            for item in self.items)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _architecture(text: str) -> tuple[int, ...]:
    dims = tuple(int(part) for part in text.lower().replace('x', ',').split(',')
                 if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError(f'invalid architecture: {text!r}')
    return dims


def _speed_tag(speed: float) -> str:
    return f'{int(round(speed * 100)):03d}'


def _load_tail_items(path: Path, *, tail_steps: int,
                     expected_speed: float) -> list[dict[str, Any]]:
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    state = payload.get('safety_replay_state')
    if not state:
        raise RuntimeError(f'{path} has no safety_replay_state')
    items = list(state['all']['items'])
    if len(items) < tail_steps:
        raise RuntimeError(
            f'{path} contains only {len(items)} safety transitions; '
            f'need {tail_steps}')
    items = items[-tail_steps:]
    commands = np.asarray(
        [float(item['observations'][-1]) for item in items],
        dtype=np.float32)
    match = np.isclose(commands, expected_speed, atol=1e-4)
    if float(np.mean(match)) < 0.99:
        raise RuntimeError(
            f'{path} tail is not fixed speed {expected_speed}: '
            f'match_rate={float(np.mean(match)):.3f}')
    return items


def reconstruct_episodes(items: Sequence[dict[str, Any]],
                         *, speed: float) -> list[Episode]:
    """Reconstruct ordered episodes without splitting terminal trajectories."""
    episodes: list[Episode] = []
    current: list[dict[str, Any]] = []
    for item in items:
        current.append(item)
        if bool(item['dones']):
            episodes.append(Episode(
                speed=float(speed), index=len(episodes),
                items=tuple(current)))
            current = []
    if current:
        episodes.append(Episode(
            speed=float(speed), index=len(episodes), items=tuple(current)))
    return episodes


def split_episodes(episodes: Sequence[Episode], *, val_fraction: float,
                   seed: int) -> tuple[list[Episode], list[Episode]]:
    """Stratified episode split, keeping success/failure trajectories whole."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError('val_fraction must be between zero and one')
    rng = np.random.default_rng(seed)
    train: list[Episode] = []
    val: list[Episode] = []
    by_outcome = {
        False: [episode for episode in episodes if not episode.failed],
        True: [episode for episode in episodes if episode.failed],
    }
    for group in by_outcome.values():
        if not group:
            continue
        order = rng.permutation(len(group))
        count = int(round(len(group) * val_fraction))
        if len(group) > 1:
            count = min(max(count, 1), len(group) - 1)
        else:
            count = 0
        val_ids = set(int(index) for index in order[:count])
        for index, episode in enumerate(group):
            (val if index in val_ids else train).append(episode)
    return sorted(train, key=lambda ep: ep.index), sorted(
        val, key=lambda ep: ep.index)


def flatten_episodes(episodes: Iterable[Episode]) -> list[dict[str, Any]]:
    return [item for episode in episodes for item in episode.items]


def _rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.clip(
        np.asarray(scores, dtype=np.float64).reshape(-1), 0.0, 1.0)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    auroc = float('nan')
    average_precision = float('nan')
    if positives and negatives:
        order = np.argsort(scores, kind='stable')
        sorted_scores = scores[order]
        ranks = np.empty(labels.size, dtype=np.float64)
        start = 0
        while start < labels.size:
            end = start + 1
            while (end < labels.size
                   and sorted_scores[end] == sorted_scores[start]):
                end += 1
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        auroc = float(
            (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
            / (positives * negatives))
        descending = np.argsort(-scores, kind='stable')
        sorted_labels = labels[descending]
        precision = np.cumsum(sorted_labels) / np.arange(
            1, labels.size + 1)
        average_precision = float(
            precision[sorted_labels == 1].mean())
    ece = 0.0
    for index in range(10):
        low = index / 10.0
        high = (index + 1) / 10.0
        mask = ((scores >= low) & (scores < high)
                if index < 9 else (scores >= low) & (scores <= high))
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(scores[mask])) - float(np.mean(labels[mask])))
    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
    pos = labels == 1
    neg = ~pos
    return {
        'num_samples': int(labels.size),
        'positive_rate': float(np.mean(labels)) if labels.size else float('nan'),
        'auroc': auroc,
        'average_precision': average_precision,
        'ece_10bin': float(ece),
        'brier': float(np.mean(np.square(scores - labels))),
        'log_loss': float(-np.mean(
            labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        'mean_score': float(np.mean(scores)),
        'mean_positive_score': (
            float(np.mean(scores[pos])) if np.any(pos) else float('nan')),
        'mean_negative_score': (
            float(np.mean(scores[neg])) if np.any(neg) else float('nan')),
        'safe_fraction_at_0p20': float(np.mean(scores <= 0.20)),
    }


def prior_correct(scores: np.ndarray, *, natural_prior: float,
                  sampled_prior: float = 0.5) -> np.ndarray:
    """Case-control prior correction for probability-like classifier scores."""
    natural = float(np.clip(natural_prior, 1e-6, 1.0 - 1e-6))
    sampled = float(np.clip(sampled_prior, 1e-6, 1.0 - 1e-6))
    score = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(score) - np.log1p(-score)
    offset = (
        math.log(natural / (1.0 - natural))
        - math.log(sampled / (1.0 - sampled)))
    return 1.0 / (1.0 + np.exp(-(logit + offset)))


def _predict_items(safety: SafetyCritic, items: Sequence[dict[str, Any]],
                   *, chunk_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for start in range(0, len(items), chunk_size):
        chunk = items[start:start + chunk_size]
        observations = np.stack([item['observations'] for item in chunk])
        actions = np.stack([item['actions'] for item in chunk])
        labels.append(np.asarray(
            [item['future_failure_labels'] for item in chunk],
            dtype=np.float32))
        scores.append(safety.predict(observations, actions))
    return np.concatenate(labels), np.concatenate(scores)


def _restore(path: Path, *, train_cfg, droq_cfg, obs_dim: int,
             action_dim: int):
    agent, _, safety = _build_agent_templates(
        train_cfg, droq_cfg, obs_dim, action_dim)
    payload = restore_training_snapshot(
        path, agent=agent, safety_critic=safety)
    if 'safety_critic' not in payload:
        raise RuntimeError(f'{path} has no trained Q_safe')
    return payload['agent'], payload['safety_critic']


def _candidate_risks(agent, safety: SafetyCritic, observations: np.ndarray,
                     recorded_actions: np.ndarray, *, candidates: int,
                     seed: int) -> dict[str, np.ndarray]:
    count, action_dim = recorded_actions.shape
    obs = jnp.asarray(observations, dtype=jnp.float32)
    repeated = jnp.repeat(obs, candidates, axis=0)
    dist = agent.actor.apply_fn({'params': agent.actor.params}, repeated)
    policy_actions = dist.sample(seed=jax.random.PRNGKey(seed))
    policy_actions = policy_actions.reshape(count, candidates, action_dim)
    rng = np.random.default_rng(seed + 1)
    local_actions = np.clip(
        recorded_actions[:, None, :]
        + rng.normal(0.0, 0.15, size=(count, candidates, action_dim)),
        -1.0, 1.0).astype(np.float32)
    uniform_actions = rng.uniform(
        -1.0, 1.0, size=(count, candidates, action_dim)).astype(np.float32)

    def risks(actions):
        flat_actions = jnp.asarray(actions.reshape(-1, action_dim))
        values = safety.critic.apply_fn(
            {'params': safety.critic.params}, repeated, flat_actions)
        return np.asarray(values).reshape(count, candidates)

    return {
        'policy': risks(np.asarray(policy_actions)),
        'local': risks(local_actions),
        'uniform': risks(uniform_actions),
    }


def _sensitivity_summary(risks: np.ndarray,
                         *, epsilon: float) -> dict[str, float]:
    minimum = np.min(risks, axis=1)
    maximum = np.max(risks, axis=1)
    standard_deviation = np.std(risks, axis=1)
    return {
        'states': int(risks.shape[0]),
        'mean_min': float(np.mean(minimum)),
        'mean_mean': float(np.mean(risks)),
        'mean_max': float(np.mean(maximum)),
        'mean_std': float(np.mean(standard_deviation)),
        'median_std': float(np.median(standard_deviation)),
        'mean_range': float(np.mean(maximum - minimum)),
        'no_safe_candidate_rate': float(np.mean(minimum > epsilon)),
        'all_saturated_high_rate': float(np.mean(minimum >= 0.90)),
        'at_least_one_low_risk_rate': float(np.mean(minimum <= epsilon)),
    }


def evaluate_current_models(checkpoints: dict[float, Path],
                            val_by_speed: dict[float, list[dict[str, Any]]],
                            *, train_cfg, droq_cfg, obs_dim: int,
                            action_dim: int, candidates: int,
                            max_states: int, epsilon: float,
                            seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    for speed, checkpoint in checkpoints.items():
        agent, safety = _restore(
            checkpoint, train_cfg=train_cfg, droq_cfg=droq_cfg,
            obs_dim=obs_dim, action_dim=action_dim)
        val_items = val_by_speed[speed]
        labels, scores = _predict_items(safety, val_items)
        natural_prior = float(np.mean(labels))
        calibration[f'{speed:.2f}'] = {
            'checkpoint': str(checkpoint),
            'raw': _rank_metrics(labels, scores),
            'prior_corrected_from_0p50': _rank_metrics(
                labels, prior_correct(
                    scores, natural_prior=natural_prior)),
        }

        rng = np.random.default_rng(seed + int(round(speed * 1000)))
        state_count = min(max_states, len(val_items))
        indices = rng.choice(len(val_items), size=state_count, replace=False)
        selected = [val_items[int(index)] for index in indices]
        observations = np.stack([item['observations'] for item in selected])
        actions = np.stack([item['actions'] for item in selected])
        selected_labels = np.asarray(
            [item['future_failure_labels'] for item in selected]) >= 0.5
        risk_sets = _candidate_risks(
            agent, safety, observations, actions, candidates=candidates,
            seed=seed + int(round(speed * 10_000)))
        speed_result: dict[str, Any] = {
            'states': state_count,
            'positive_states': int(np.sum(selected_labels)),
            'negative_states': int(np.sum(~selected_labels)),
        }
        for name, risks in risk_sets.items():
            entry: dict[str, Any] = {
                'all': _sensitivity_summary(risks, epsilon=epsilon)}
            if np.any(selected_labels):
                entry['future_failure'] = _sensitivity_summary(
                    risks[selected_labels], epsilon=epsilon)
            if np.any(~selected_labels):
                entry['normal'] = _sensitivity_summary(
                    risks[~selected_labels], epsilon=epsilon)
            speed_result[name] = entry
        sensitivity[f'{speed:.2f}'] = speed_result
    return calibration, sensitivity


def evaluate_calibration_by_speed(
        safety: SafetyCritic,
        val_by_speed: dict[float, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for speed, items in val_by_speed.items():
        labels, scores = _predict_items(safety, items)
        natural_prior = float(np.mean(labels))
        result[f'{speed:.2f}'] = {
            'raw': _rank_metrics(labels, scores),
            'prior_corrected_from_0p50': _rank_metrics(
                labels, prior_correct(scores, natural_prior=natural_prior)),
        }
    return result


def _stack_batch(items: Sequence[dict[str, Any]],
                 indices: np.ndarray) -> dict[str, np.ndarray]:
    selected = [items[int(index)] for index in indices]
    batch = {
        key: np.stack([item[key] for item in selected])
        for key in REQUIRED_BATCH_KEYS
    }
    batch['source_ids'] = np.zeros(len(selected), dtype=np.int8)
    return batch


def _balanced_index_schedule(items: Sequence[dict[str, Any]], *,
                             steps: int, batch_size: int,
                             seed: int) -> list[np.ndarray]:
    labels = np.asarray(
        [float(item['future_failure_labels']) >= 0.5 for item in items])
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    if not len(positive) or not len(negative):
        raise RuntimeError('capacity training requires positive and negative data')
    rng = np.random.default_rng(seed)
    n_positive = batch_size // 2
    n_negative = batch_size - n_positive
    schedule = []
    for _ in range(steps):
        indices = np.concatenate([
            rng.choice(positive, size=n_positive, replace=True),
            rng.choice(negative, size=n_negative, replace=True),
        ])
        rng.shuffle(indices)
        schedule.append(indices)
    return schedule


def train_capacity_ablation(train_items: list[dict[str, Any]],
                            val_by_speed: dict[float, list[dict[str, Any]]],
                            *, actor, architectures: Sequence[tuple[int, ...]],
                            train_cfg, obs_dim: int, action_dim: int,
                            steps: int, batch_size: int,
                            seed: int, repeats: int) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError('capacity repeats must be positive')
    schedule = _balanced_index_schedule(
        train_items, steps=steps, batch_size=batch_size, seed=seed)
    result: dict[str, Any] = {}
    for model_index, architecture in enumerate(architectures):
        name = 'x'.join(str(dim) for dim in architecture)
        runs: list[dict[str, Any]] = []
        parameter_count = 0
        for repeat in range(repeats):
            model_seed = seed + 1000 + model_index * 100 + repeat
            safety = SafetyCritic.create(
                seed=model_seed,
                observation_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=architecture,
                learning_rate=train_cfg.safety_critic_learning_rate,
                discount=train_cfg.safety_discount,
                tau=train_cfg.safety_critic_tau,
                future_loss_weight=train_cfg.safety_future_loss_weight)
            started = time.perf_counter()
            latest_info = {}
            for step, indices in enumerate(schedule, start=1):
                batch = _stack_batch(train_items, indices)
                safety, latest_info = SafetyCritic.update(
                    safety, actor.actor, batch)
                if step == 1 or step % 500 == 0 or step == steps:
                    print(
                        f'[capacity] architecture={name} repeat={repeat + 1}/'
                        f'{repeats} step={step}/{steps} '
                        f'loss={float(latest_info["safety_critic_loss"]):.5f}',
                        flush=True)
            combined_labels: list[np.ndarray] = []
            combined_scores: list[np.ndarray] = []
            by_speed: dict[str, Any] = {}
            for speed, items in val_by_speed.items():
                labels, scores = _predict_items(safety, items)
                combined_labels.append(labels)
                combined_scores.append(scores)
                prior = float(np.mean(labels))
                by_speed[f'{speed:.2f}'] = {
                    'raw': _rank_metrics(labels, scores),
                    'prior_corrected_from_0p50': _rank_metrics(
                        labels, prior_correct(scores, natural_prior=prior)),
                }
            labels = np.concatenate(combined_labels)
            scores = np.concatenate(combined_scores)
            prior = float(np.mean(labels))
            parameter_count = int(sum(
                np.asarray(leaf).size
                for leaf in jax.tree.leaves(safety.critic.params)))
            runs.append({
                'repeat': repeat,
                'seed': model_seed,
                'elapsed_sec': time.perf_counter() - started,
                'final_loss': float(latest_info['safety_critic_loss']),
                'combined_natural_validation': {
                    'raw': _rank_metrics(labels, scores),
                    'prior_corrected_from_0p50': _rank_metrics(
                        labels, prior_correct(scores, natural_prior=prior)),
                },
                'by_speed': by_speed,
            })
        metric_paths = {
            'final_loss': lambda run: run['final_loss'],
            'raw_auroc': lambda run: run[
                'combined_natural_validation']['raw']['auroc'],
            'raw_average_precision': lambda run: run[
                'combined_natural_validation']['raw']['average_precision'],
            'raw_ece_10bin': lambda run: run[
                'combined_natural_validation']['raw']['ece_10bin'],
            'raw_brier': lambda run: run[
                'combined_natural_validation']['raw']['brier'],
            'raw_log_loss': lambda run: run[
                'combined_natural_validation']['raw']['log_loss'],
            'corrected_ece_10bin': lambda run: run[
                'combined_natural_validation'][
                    'prior_corrected_from_0p50']['ece_10bin'],
            'corrected_brier': lambda run: run[
                'combined_natural_validation'][
                    'prior_corrected_from_0p50']['brier'],
        }
        aggregate = {}
        for metric_name, getter in metric_paths.items():
            values = np.asarray([getter(run) for run in runs], dtype=np.float64)
            aggregate[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'values': values,
            }
        result[name] = {
            'architecture': architecture,
            'parameter_count': parameter_count,
            'steps': steps,
            'repeats': repeats,
            'same_batch_schedule_across_repeats': True,
            'aggregate': aggregate,
            'runs': runs,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/go2.yaml')
    parser.add_argument(
        '--checkpoint-root', default='saved/checkpoints_multispeed')
    parser.add_argument(
        '--output',
        default='saved/safety_evaluation/qsafe_diagnostics/report.json')
    parser.add_argument('--speeds', default='0.40,0.50,0.60,0.80,1.00')
    parser.add_argument('--tail-steps', type=int, default=4000)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--candidate-count', type=int, default=64)
    parser.add_argument('--candidate-states', type=int, default=512)
    parser.add_argument('--epsilon', type=float, default=0.20)
    parser.add_argument('--capacity-steps', type=int, default=2000)
    parser.add_argument('--capacity-repeats', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--architectures', type=_architecture, nargs='+',
                        default=list(DEFAULT_ARCHITECTURES))
    parser.add_argument('--seed', type=int, default=20260728)
    args = parser.parse_args()

    speeds = tuple(float(value) for value in args.speeds.split(','))
    root = Path(args.checkpoint_root)
    checkpoints = {
        speed: root / 'ft' / f'sqrl_ft_v{_speed_tag(speed)}'
        / 'training_snapshot_000000016000.pkl'
        for speed in speeds
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f'missing checkpoints: {missing}')

    robot_cfg, train_cfg, droq_cfg = load_app_config(path=args.config)
    obs_dim = int(robot_cfg.obs_dim)
    action_dim = int(robot_cfg.num_joints)
    train_episodes: list[Episode] = []
    val_by_speed: dict[float, list[dict[str, Any]]] = {}
    split_report: dict[str, Any] = {}
    for speed, checkpoint in checkpoints.items():
        items = _load_tail_items(
            checkpoint, tail_steps=args.tail_steps,
            expected_speed=speed / float(robot_cfg.cmd_speed_obs_scale))
        episodes = reconstruct_episodes(items, speed=speed)
        train_eps, val_eps = split_episodes(
            episodes, val_fraction=args.val_fraction,
            seed=args.seed + int(round(speed * 1000)))
        train_episodes.extend(train_eps)
        val_by_speed[speed] = flatten_episodes(val_eps)
        split_report[f'{speed:.2f}'] = {
            'episodes_total': len(episodes),
            'episodes_train': len(train_eps),
            'episodes_validation': len(val_eps),
            'failed_episodes_total': sum(ep.failed for ep in episodes),
            'failed_episodes_train': sum(ep.failed for ep in train_eps),
            'failed_episodes_validation': sum(ep.failed for ep in val_eps),
            'transitions_train': sum(len(ep.items) for ep in train_eps),
            'transitions_validation': sum(len(ep.items) for ep in val_eps),
        }
    train_items = flatten_episodes(train_episodes)
    print(
        f'[diagnostic] train_transitions={len(train_items)} '
        f'validation_transitions={sum(map(len, val_by_speed.values()))}',
        flush=True)

    current_calibration, current_sensitivity = evaluate_current_models(
        checkpoints, val_by_speed, train_cfg=train_cfg, droq_cfg=droq_cfg,
        obs_dim=obs_dim, action_dim=action_dim,
        candidates=args.candidate_count, max_states=args.candidate_states,
        epsilon=args.epsilon, seed=args.seed)

    pre_checkpoint = (
        root / 'sqrl_pre' / 'training_snapshot_000000012000.pkl')
    actor, pre_safety = _restore(
        pre_checkpoint, train_cfg=train_cfg, droq_cfg=droq_cfg,
        obs_dim=obs_dim, action_dim=action_dim)
    pre_calibration = evaluate_calibration_by_speed(
        pre_safety, val_by_speed)
    capacity = train_capacity_ablation(
        train_items, val_by_speed, actor=actor,
        architectures=args.architectures, train_cfg=train_cfg,
        obs_dim=obs_dim, action_dim=action_dim,
        steps=args.capacity_steps, batch_size=args.batch_size,
        seed=args.seed, repeats=args.capacity_repeats)

    report = {
        'protocol': 'qsafe_offline_diagnostics_v1',
        'config': str(Path(args.config).resolve()),
        'checkpoint_root': str(root.resolve()),
        'speeds': speeds,
        'tail_steps_per_speed': args.tail_steps,
        'episode_validation_fraction': args.val_fraction,
        'epsilon_safe': args.epsilon,
        'candidate_count': args.candidate_count,
        'candidate_states_per_speed': args.candidate_states,
        'split': split_report,
        'current_checkpoint_replay_slice_calibration': {
            'note': (
                'Diagnostic only: these final online critics previously saw '
                'their own replay. This is not an independent estimate.'),
            'by_speed': current_calibration,
        },
        'pre_checkpoint_independent_target_calibration': {
            'checkpoint': str(pre_checkpoint),
            'note': (
                'The 12k pretraining critic did not train on these later '
                'fine-tuning episodes; split remains grouped by episode.'),
            'by_speed': pre_calibration,
        },
        'current_checkpoint_action_sensitivity': current_sensitivity,
        'capacity_ablation': capacity,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding='utf-8')
    temporary.replace(output)
    print(f'[diagnostic] report={output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
