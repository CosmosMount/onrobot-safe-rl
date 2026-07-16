"""Complete online-training checkpoints.

The legacy Flax-only checkpoint restored the agent without replay data.  That is
unsafe for online training because the critic, actor, and replay distribution no
longer describe the same run.  This module stores the agent and replay buffer as
one snapshot.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


SNAPSHOT_PREFIX = 'training_snapshot_'
SNAPSHOT_SUFFIX = '.pkl'


def _snapshot_path(save_dir: str | Path, step: int) -> Path:
    return Path(save_dir) / f'{SNAPSHOT_PREFIX}{step:012d}{SNAPSHOT_SUFFIX}'


def latest_snapshot(save_dir: str | Path) -> Path | None:
    root = Path(save_dir)
    if not root.exists():
        return None
    snapshots = sorted(root.glob(f'{SNAPSHOT_PREFIX}*{SNAPSHOT_SUFFIX}'))
    return snapshots[-1] if snapshots else None


def has_legacy_agent_checkpoint(save_dir: str | Path) -> bool:
    root = Path(save_dir)
    if not root.exists():
        return False
    return any(path.name.startswith('checkpoint_') for path in root.iterdir())


def save_training_snapshot(save_dir: str | Path,
                           *,
                           agent: Any,
                           replay_buffer: Any,
                           step: int,
                           metadata: dict[str, Any] | None = None) -> Path:
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(root, step)
    payload = {
        'agent': agent,
        'replay_buffer': replay_buffer,
        'step': step,
        'metadata': metadata or {},
    }
    with path.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def restore_training_snapshot(path: str | Path) -> dict[str, Any]:
    with Path(path).open('rb') as f:
        payload = pickle.load(f)
    required = {'agent', 'replay_buffer', 'step'}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f'Incomplete training snapshot {path}: missing {missing}')
    return payload
