#!/usr/bin/env python3
"""Compose a deployable checkpoint from a complete base and trained Q_safe."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

from learner.checkpoint import agent_state_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', required=True, type=Path)
    parser.add_argument('--safety-checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    with args.base_checkpoint.open('rb') as stream:
        base = pickle.load(stream)
    with args.safety_checkpoint.open('rb') as stream:
        safety = pickle.load(stream)
    required_base = {
        'agent_state', 'replay_buffer_state', 'safety_replay_state', 'step',
    }
    missing = required_base.difference(base)
    if missing:
        raise RuntimeError(
            f'base checkpoint is not deployable; missing {sorted(missing)}')
    if 'safety_critic_state' not in safety:
        raise RuntimeError('safety checkpoint has no safety_critic_state')
    if 'agent_state' not in safety:
        raise RuntimeError(
            'safety checkpoint has no agent_state; cannot prove that Q_safe '
            'was trained beside the common SAC actor')
    base_agent_hash = agent_state_hash(base['agent_state'])
    safety_agent_hash = agent_state_hash(safety['agent_state'])
    if base_agent_hash != safety_agent_hash:
        raise RuntimeError(
            'refusing to compose checkpoints with different actor/reward '
            f'states: base={base_agent_hash} safety={safety_agent_hash}')

    payload = dict(base)
    payload['safety_critic_state'] = safety['safety_critic_state']
    metadata = dict(base.get('metadata') or {})
    metadata.update({
        'experiment_name': 'sqrl_pretrain',
        'composed_base_checkpoint': str(args.base_checkpoint.resolve()),
        'composed_safety_checkpoint': str(args.safety_checkpoint.resolve()),
        'agent_state_hash': base_agent_hash,
        'common_actor_hash': base_agent_hash,
    })
    payload['metadata'] = metadata

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    try:
        with temporary.open('wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
