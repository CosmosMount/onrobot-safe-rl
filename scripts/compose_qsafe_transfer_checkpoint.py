#!/usr/bin/env python3
"""Attach a source Q_safe to an unrelated, fresh target SAC checkpoint.

This is intentionally separate from ``compose_control_checkpoint``.  P15 must
reject different actor hashes; P16 is specifically a cross-actor Q_safe reuse
experiment and therefore records, rather than hides, that mismatch.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

from learner.checkpoint import agent_state_hash


def compose_qsafe_transfer_payload(
        target: dict[str, Any],
        source: dict[str, Any],
        *,
        target_checkpoint: str,
        source_checkpoint: str,
) -> dict[str, Any]:
    required_target = {
        'agent_state', 'replay_buffer_state', 'safety_replay_state', 'step',
    }
    missing = required_target.difference(target)
    if missing:
        raise RuntimeError(
            f'target checkpoint is incomplete; missing {sorted(missing)}')
    if int(target['step']) != 0:
        raise RuntimeError(
            f'P16 target checkpoint must be step 0, got {target["step"]}')
    if 'agent_state' not in source:
        raise RuntimeError('source checkpoint has no agent_state')
    if 'safety_critic_state' not in source:
        raise RuntimeError('source checkpoint has no safety_critic_state')

    target_hash = agent_state_hash(target['agent_state'])
    source_hash = agent_state_hash(source['agent_state'])
    payload = dict(target)
    payload['safety_critic_state'] = source['safety_critic_state']
    metadata = dict(target.get('metadata') or {})
    metadata.update({
        'protocol': 'P16',
        'experiment_name': 'sqrl_pretrain',
        'qsafe_transfer_only': True,
        'target_base_checkpoint': str(Path(target_checkpoint).resolve()),
        'source_safety_checkpoint': str(Path(source_checkpoint).resolve()),
        'agent_state_hash': target_hash,
        'target_initial_agent_hash': target_hash,
        'source_agent_hash': source_hash,
        'actor_transferred': False,
        'reward_critic_transferred': False,
        'reward_replay_transferred': False,
        'safety_critic_transferred': True,
    })
    payload['metadata'] = metadata

    # The output must remain byte-for-byte the target SAC state/replay.
    if agent_state_hash(payload['agent_state']) != target_hash:
        raise AssertionError('target actor/reward state changed during transfer')
    if payload['replay_buffer_state'] is not target['replay_buffer_state']:
        raise AssertionError('target reward replay was replaced')
    return payload


def compose_qsafe_transfer_checkpoint(
        target_checkpoint: Path,
        source_checkpoint: Path,
        output: Path,
) -> dict[str, Any]:
    with target_checkpoint.open('rb') as stream:
        target = pickle.load(stream)
    with source_checkpoint.open('rb') as stream:
        source = pickle.load(stream)
    payload = compose_qsafe_transfer_payload(
        target, source,
        target_checkpoint=str(target_checkpoint),
        source_checkpoint=str(source_checkpoint))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        with temporary.open('wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(payload['metadata'])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-checkpoint', required=True, type=Path)
    parser.add_argument('--source-checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    metadata = compose_qsafe_transfer_checkpoint(
        args.target_checkpoint, args.source_checkpoint, args.output)
    print(args.output)
    print(
        f'target_agent={metadata["target_initial_agent_hash"]} '
        f'source_agent={metadata["source_agent_hash"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
