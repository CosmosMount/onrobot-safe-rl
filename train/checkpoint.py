"""Complete online-training checkpoints.

The legacy Flax-only checkpoint restored the agent without replay data.  That is
unsafe for online training because the critic, actor, and replay distribution no
longer describe the same run.  This module stores the agent and replay buffer as
one snapshot.
"""

from __future__ import annotations

import pickle
import os
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
                           replay_buffer: Any | None,
                           step: int,
                           metadata: dict[str, Any] | None = None) -> Path:
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(root, step)
    payload = {
        'agent_type': getattr(agent, 'agent_type', None),
        'agent_state': agent.state_dict(),
        'step': step,
        'metadata': metadata or {},
    }
    if replay_buffer is not None:
        payload['replay_buffer_state'] = replay_buffer.state_dict()
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
    return path


def load_training_snapshot_metadata(path: str | Path) -> dict[str, Any]:
    with Path(path).open('rb') as f:
        payload = pickle.load(f)
    return dict(payload.get('metadata') or {})


def restore_training_snapshot(path: str | Path,
                              agent: Any | None = None,
                              replay_buffer: Any | None = None) -> dict[str, Any]:
    with Path(path).open('rb') as f:
        payload = pickle.load(f)
    required = {'step'}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f'Incomplete training snapshot {path}: missing {missing}')
    if 'agent_state' in payload:
        if agent is None:
            raise ValueError(
                'An agent template is required to restore state snapshots')
        snapshot_agent_type = payload.get('agent_type')
        current_agent_type = getattr(agent, 'agent_type', None)
        if (snapshot_agent_type is not None
                and current_agent_type != snapshot_agent_type):
            raise ValueError(
                'Agent checkpoint type mismatch: '
                f'snapshot={snapshot_agent_type!r} '
                f'current={current_agent_type!r}')
        payload['agent'] = agent.load_state_dict(payload['agent_state'])
    elif 'agent' not in payload:
        raise ValueError(
            f'Incomplete training snapshot {path}: missing agent state')
    owns_replay_buffer = bool(getattr(payload.get('agent'), 'owns_replay_buffer',
                                      False))
    if 'replay_buffer_state' in payload:
        if replay_buffer is not None:
            replay_buffer.load_state_dict(payload['replay_buffer_state'])
            payload['replay_buffer'] = replay_buffer
    elif 'replay_buffer' not in payload and not owns_replay_buffer:
        raise ValueError(
            f'Incomplete training snapshot {path}: missing replay buffer state')
    return payload
