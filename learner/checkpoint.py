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

from flax import serialization


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


def experiments_compatible(snapshot_name: str | None,
                           current_name: str | None) -> bool:
    """True if a snapshot may be resumed under the current experiment name.

    SQRL Route A intentionally continues from ``sqrl_pretrain`` into
    ``sqrl_finetune`` in the same ``save_dir``.
    """
    if snapshot_name in (None, current_name):
        return True
    return {snapshot_name, current_name} == {'sqrl_pretrain', 'sqrl_finetune'}


def has_legacy_agent_checkpoint(save_dir: str | Path) -> bool:
    root = Path(save_dir)
    if not root.exists():
        return False
    return any(path.name.startswith('checkpoint_') for path in root.iterdir())


def save_training_snapshot(save_dir: str | Path,
                           *,
                           agent: Any,
                           replay_buffer: Any,
                           safety_replay: Any | None = None,
                           safety_critic: Any | None = None,
                           safety_validator: Any | None = None,
                           step: int,
                           metadata: dict[str, Any] | None = None) -> Path:
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(root, step)
    payload = {
        # Flax TrainState contains static apply/optimizer callables which are
        # intentionally not picklable. Persist only the registered PyTree
        # state and restore it into a freshly constructed agent template.
        'agent_state': serialization.to_state_dict(agent),
        'replay_buffer_state': replay_buffer.state_dict(),
        'step': step,
        'metadata': metadata or {},
    }
    if safety_replay is not None:
        payload['safety_replay_state'] = safety_replay.state_dict()
    if safety_critic is not None:
        payload['safety_critic_state'] = serialization.to_state_dict(
            safety_critic)
    if safety_validator is not None:
        payload['safety_validator_state'] = serialization.to_state_dict(
            safety_validator)
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
                              replay_buffer: Any | None = None,
                              safety_replay: Any | None = None,
                              safety_critic: Any | None = None,
                              safety_validator: Any | None = None) -> dict[str, Any]:
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
        agent_state = payload['agent_state']
        # Older SAC snapshots predate the SQRL Lagrange multiplier field.
        if (isinstance(agent_state, dict)
                and 'safety_lagrange' not in agent_state
                and hasattr(agent, 'safety_lagrange')):
            template_state = serialization.to_state_dict(agent)
            agent_state = dict(agent_state)
            agent_state['safety_lagrange'] = template_state['safety_lagrange']
        payload['agent'] = serialization.from_state_dict(agent, agent_state)
    elif 'agent' not in payload:
        raise ValueError(
            f'Incomplete training snapshot {path}: missing agent state')
    if 'replay_buffer_state' in payload:
        if replay_buffer is not None:
            replay_buffer.load_state_dict(payload['replay_buffer_state'])
            payload['replay_buffer'] = replay_buffer
    elif 'replay_buffer' not in payload:
        raise ValueError(
            f'Incomplete training snapshot {path}: missing replay buffer state')
    if 'safety_replay_state' in payload and safety_replay is not None:
        safety_replay.load_state_dict(payload['safety_replay_state'])
        payload['safety_replay'] = safety_replay
    if 'safety_critic_state' in payload and safety_critic is not None:
        safety_state = payload['safety_critic_state']
        # Safety calibration was added after the original Stage-1 snapshots.
        # Preserve those checkpoints by taking the neutral T=1 value from the
        # current template when the serialized field is absent.
        if (isinstance(safety_state, dict)
                and 'calibration_temperature' not in safety_state
                and hasattr(safety_critic, 'calibration_temperature')):
            template_state = serialization.to_state_dict(safety_critic)
            safety_state = dict(safety_state)
            safety_state['calibration_temperature'] = (
                template_state['calibration_temperature'])
        payload['safety_critic'] = serialization.from_state_dict(
            safety_critic, safety_state)
    if ('safety_validator_state' in payload
            and safety_validator is not None):
        validator_state = payload['safety_validator_state']
        if (isinstance(validator_state, dict)
                and 'calibration_temperature' not in validator_state):
            template_state = serialization.to_state_dict(safety_validator)
            validator_state = dict(validator_state)
            validator_state['calibration_temperature'] = (
                template_state['calibration_temperature'])
        payload['safety_validator'] = serialization.from_state_dict(
            safety_validator, validator_state)
    return payload
