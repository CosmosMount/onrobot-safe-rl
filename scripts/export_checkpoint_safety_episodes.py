#!/usr/bin/env python3
"""Export ordered checkpoint safety replay into episode-level artifacts."""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

from learner.safety_dataset import save_safety_episode_artifact


def export_checkpoint(checkpoint: Path, output: Path) -> list[Path]:
    with checkpoint.open('rb') as stream:
        payload = pickle.load(stream)
    state = payload.get('safety_replay_state')
    if not state:
        raise RuntimeError(f'{checkpoint} has no safety replay')

    episodes: dict[int, list[dict]] = defaultdict(list)
    for item in state['all']['items']:
        episodes[int(item.get('episode_ids', 0))].append(item)

    paths = []
    for output_index, (episode_id, all_items) in enumerate(
            sorted(episodes.items())):
        episode_state = dict(state)
        for name in ('recent', 'failure', 'boundary', 'recovery', 'all'):
            buffer_state = dict(state[name])
            buffer_state['items'] = [
                item for item in state[name]['items']
                if int(item.get('episode_ids', -1)) == episode_id
            ]
            episode_state[name] = buffer_state
        episode_state['history'] = []
        episode_state['nstep_history'] = []
        failed = any(
            float(item.get('unsafe_labels', 0.0)) >= 0.5
            for item in all_items)
        command_speeds = [
            float(item.get('command_speeds', 0.0)) for item in all_items]
        paths.append(save_safety_episode_artifact(
            output,
            episode_index=output_index,
            safety_replay_state=episode_state,
            metadata={
                'source_checkpoint': str(checkpoint.resolve()),
                'episode_index': output_index,
                'source_episode_id': episode_id,
                # Episode ID is unique and gives leakage-free train/val split.
                'rollout_seed': episode_id,
                'action_noise_std': 0.0,
                'safety_mask': False,
                'outcome': 'failure' if failed else 'success',
                'steps': len(all_items),
                'unsafe_steps': sum(
                    float(item.get('unsafe_labels', 0.0)) >= 0.5
                    for item in all_items),
                'episode_return': sum(
                    float(item.get('rewards', 0.0)) for item in all_items),
                'command_speed': (
                    sum(command_speeds) / len(command_speeds)),
            }))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    paths = export_checkpoint(args.checkpoint, args.output)
    print(f'exported {len(paths)} episodes to {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
