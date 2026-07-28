"""Episode-level safety dataset artifacts for offline Q_safe retraining.

Collection should freeze the reference critic and persist raw masked/unmasked
rollouts without writing mutated training snapshots back over the baseline.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

EPISODE_PREFIX = 'episode_'
EPISODE_SUFFIX = '.pkl'
MANIFEST_NAME = 'manifest.jsonl'
FORMAT_VERSION = 'safety_episode_v1'


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def episode_artifact_path(dataset_dir: str | Path, episode_index: int,
                          *, outcome: str, rollout_seed: int) -> Path:
    name = (
        f'{EPISODE_PREFIX}{episode_index:04d}'
        f'_seed{int(rollout_seed)}'
        f'_{outcome}{EPISODE_SUFFIX}')
    return Path(dataset_dir) / name


def save_safety_episode_artifact(
        dataset_dir: str | Path,
        *,
        episode_index: int,
        safety_replay_state: dict[str, Any],
        metadata: dict[str, Any]) -> Path:
    """Persist one completed episode's safety replay and metadata."""
    root = Path(dataset_dir)
    root.mkdir(parents=True, exist_ok=True)
    outcome = str(metadata.get('outcome', 'unknown'))
    rollout_seed = int(metadata.get('rollout_seed', 0))
    path = episode_artifact_path(
        root, episode_index, outcome=outcome, rollout_seed=rollout_seed)
    payload = {
        'format': FORMAT_VERSION,
        'metadata': dict(metadata),
        'safety_replay_state': safety_replay_state,
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        with temporary.open('wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest_path = root / MANIFEST_NAME
    record = {
        'path': path.name,
        **{key: metadata[key] for key in (
            'episode_index', 'rollout_seed', 'action_noise_std',
            'safety_mask', 'safety_mask_epsilon', 'outcome', 'steps',
            'unsafe_steps', 'episode_return', 'source_checkpoint')
           if key in metadata},
    }
    with manifest_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(_json_safe(record), allow_nan=False) + '\n')
    return path


def load_safety_episode_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open('rb') as f:
        payload = pickle.load(f)
    if payload.get('format') != FORMAT_VERSION:
        raise ValueError(
            f'Unsupported safety episode format in {path}: '
            f'{payload.get("format")}')
    if 'safety_replay_state' not in payload or 'metadata' not in payload:
        raise ValueError(f'Incomplete safety episode artifact: {path}')
    return payload


def list_safety_episode_artifacts(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        return []
    return sorted(root.glob(f'{EPISODE_PREFIX}*{EPISODE_SUFFIX}'))


def load_manifest(dataset_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(dataset_dir) / MANIFEST_NAME
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def infer_obs_action_dims(payload: dict[str, Any]) -> tuple[int, int]:
    state = payload['safety_replay_state']
    for name in ('all', 'recent', 'failure', 'boundary', 'recovery'):
        items = state[name]['items']
        if items:
            obs = np.asarray(items[0]['observations'])
            act = np.asarray(items[0]['actions'])
            return int(obs.shape[0]), int(act.shape[0])
    raise ValueError('Cannot infer dims from an empty safety episode artifact')


def split_episode_artifacts(
        artifacts: list[Path],
        *,
        held_out_seeds: set[int] | None = None,
        val_seed_fraction: float = 0.2,
        seed: int = 0,
) -> tuple[list[Path], list[Path], set[int]]:
    """Split episode files into train/val by rollout_seed.

    If ``held_out_seeds`` is provided it is used directly. Otherwise a fraction
    of the unique rollout seeds is held out for validation.
    """
    if not artifacts:
        return [], [], set()
    by_seed: dict[int, list[Path]] = {}
    for path in artifacts:
        payload = load_safety_episode_artifact(path)
        rollout_seed = int(payload['metadata'].get('rollout_seed', 0))
        by_seed.setdefault(rollout_seed, []).append(path)
    unique_seeds = sorted(by_seed)
    if held_out_seeds is None:
        if len(unique_seeds) <= 1 or val_seed_fraction <= 0.0:
            held_out_seeds = set()
        else:
            rng = np.random.default_rng(seed)
            hold_count = max(1, int(round(len(unique_seeds) * val_seed_fraction)))
            hold_count = min(hold_count, len(unique_seeds) - 1)
            held_out_seeds = set(
                int(s) for s in rng.choice(
                    unique_seeds, size=hold_count, replace=False))
    else:
        held_out_seeds = {int(s) for s in held_out_seeds}
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for rollout_seed, paths in by_seed.items():
        if rollout_seed in held_out_seeds:
            val_paths.extend(paths)
        else:
            train_paths.extend(paths)
    return sorted(train_paths), sorted(val_paths), held_out_seeds


def merge_episode_artifacts(
        artifacts: list[Path],
        replay,
        *,
        outcomes: set[str] | None = None,
) -> dict[str, Any]:
    """Load episode artifacts into an existing SafetyReplayManager."""
    loaded = 0
    skipped = 0
    outcome_counts: dict[str, int] = {}
    seeds: set[int] = set()
    for path in artifacts:
        payload = load_safety_episode_artifact(path)
        metadata = payload['metadata']
        outcome = str(metadata.get('outcome', 'unknown'))
        if outcomes is not None and outcome not in outcomes:
            skipped += 1
            continue
        added = replay.extend_from_state(payload['safety_replay_state'])
        loaded += 1
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        seeds.add(int(metadata.get('rollout_seed', 0)))
        if added == 0 and len(payload['safety_replay_state']['recovery']['items']):
            # Recovery-only episode still counts as loaded.
            pass
    return {
        'episodes_loaded': loaded,
        'episodes_skipped': skipped,
        'outcome_counts': outcome_counts,
        'rollout_seeds': sorted(seeds),
        'replay_sizes': {
            'recent': len(replay.recent),
            'boundary': len(replay.boundary),
            'failure': len(replay.failure),
            'recovery': len(replay.recovery),
            'all': len(replay.all),
        },
    }
