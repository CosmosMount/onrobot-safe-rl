from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.transition import Transition, zero_costs
from jaxrl.data.safety_replay import SafetyReplayManager
from learner.safety_dataset import (list_safety_episode_artifacts,
                                    load_safety_episode_artifact,
                                    save_safety_episode_artifact)


def _transition(value: float, *, unsafe: bool = False,
                done: bool = False) -> Transition:
    return Transition(
        observation=np.full(3, value, dtype=np.float32),
        requested_action=np.full(2, value, dtype=np.float32),
        projected_action=np.full(2, value, dtype=np.float32),
        executed_q_target=np.full(2, value, dtype=np.float32),
        reward=value,
        costs=zero_costs(),
        next_observation=np.full(3, value + 1, dtype=np.float32),
        terminated=done and unsafe,
        truncated=done and not unsafe,
        unsafe_label=unsafe,
        near_failure_label=unsafe,
    )


class SafetyDatasetTest(unittest.TestCase):

    def test_episode_artifact_round_trip_and_manifest(self):
        replay = SafetyReplayManager(
            recent_capacity=20, failure_capacity=20, boundary_capacity=20,
            recovery_capacity=10, all_capacity=50, failure_history=3,
            n_step=3, seed=11)
        replay.insert(_transition(0.0), behavior_noise_std=0.45)
        replay.insert(_transition(1.0, unsafe=True, done=True),
                      behavior_noise_std=0.45)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_safety_episode_artifact(
                tmp,
                episode_index=2,
                safety_replay_state=replay.state_dict(),
                metadata={
                    'source_checkpoint': '/tmp/training_snapshot_15327.pkl',
                    'episode_index': 2,
                    'rollout_seed': 1002,
                    'action_noise_std': 0.45,
                    'safety_mask': True,
                    'safety_mask_epsilon': 0.15,
                    'outcome': 'failure',
                    'steps': 2,
                    'unsafe_steps': 1,
                    'episode_return': 1.0,
                })
            self.assertTrue(path.exists())
            self.assertIn('seed1002', path.name)
            self.assertIn('failure', path.name)

            payload = load_safety_episode_artifact(path)
            self.assertEqual(payload['metadata']['outcome'], 'failure')
            restored = SafetyReplayManager(
                recent_capacity=20, failure_capacity=20, boundary_capacity=20,
                recovery_capacity=10, all_capacity=50, failure_history=3,
                n_step=3, seed=0)
            restored.load_state_dict(payload['safety_replay_state'])
            self.assertEqual(restored.sizes, replay.sizes)

            artifacts = list_safety_episode_artifacts(tmp)
            self.assertEqual(artifacts, [path])
            manifest = Path(tmp) / 'manifest.jsonl'
            rows = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['outcome'], 'failure')
            self.assertEqual(rows[0]['rollout_seed'], 1002)


if __name__ == '__main__':
    unittest.main()
