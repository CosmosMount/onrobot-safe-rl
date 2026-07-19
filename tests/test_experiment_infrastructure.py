from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from jaxrl.data.replay_buffer import ReplayBuffer
from jaxrl.env.specs import BoxSpec
from learner.checkpoint import (restore_training_snapshot,
                                save_training_snapshot)
from train.rolling_metrics import RollingTrainingSummary


def _replay(capacity: int = 100) -> ReplayBuffer:
    obs = BoxSpec(shape=(3,), dtype=np.float32)
    action = BoxSpec(shape=(2,), dtype=np.float32)
    replay = ReplayBuffer(obs, action, capacity)
    replay.seed(7)
    return replay


def _transition(value: float) -> dict:
    return {
        'observations': np.full(3, value, dtype=np.float32),
        'actions': np.full(2, value, dtype=np.float32),
        'rewards': value,
        'masks': 1.0,
        'dones': False,
        'next_observations': np.full(3, value + 1, dtype=np.float32),
    }


class ExperimentInfrastructureTest(unittest.TestCase):

    def test_compact_checkpoint_restores_replay_and_metadata(self):
        replay = _replay()
        replay.insert(_transition(1.0))
        replay.insert(_transition(2.0))

        with tempfile.TemporaryDirectory() as tmp:
            path = save_training_snapshot(
                tmp,
                agent={'weight': np.asarray([3.0])},
                replay_buffer=replay,
                step=17,
                metadata={'experiment_name': 'e00'},
            )
            restored_replay = _replay()
            payload = restore_training_snapshot(
                path,
                agent={'weight': np.asarray([0.0])},
                replay_buffer=restored_replay,
            )

            self.assertEqual(payload['step'], 17)
            np.testing.assert_allclose(payload['agent']['weight'], [3.0])
            self.assertEqual(payload['metadata']['experiment_name'], 'e00')
            self.assertEqual(len(payload['replay_buffer']), 2)
            np.testing.assert_allclose(
                payload['replay_buffer'].dataset_dict['observations'][:2],
                np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            )
            self.assertFalse(Path(str(path) + '.tmp').exists())
            # Compact state should be much smaller than the 100-row allocation.
            self.assertLess(path.stat().st_size, 10_000)

    def test_replay_capacity_mismatch_is_rejected(self):
        replay = _replay(capacity=10)
        state = replay.state_dict()
        with self.assertRaisesRegex(ValueError, 'capacity mismatch'):
            _replay(capacity=11).load_state_dict(state)

    def test_rolling_summary_uses_only_policy_steps(self):
        summary = RollingTrainingSummary(window=2, action_dim=2)
        timing = {
            'timing/step_ms': 50.0,
            'timing/update_ms': 20.0,
            'timing/loop_ms': 70.0,
            'timing/effective_hz': 14.2857,
        }
        summary.record_step(
            action=np.asarray([1.0, 0.0]),
            info={
                'policy_step': True,
                'forward_velocity': 0.2,
                'world_x': 1.0,
                'upright_gate': 1.0,
            },
            timing=timing,
            update_info={'q': 2.0, 'entropy': 3.0},
        )
        summary.record_step(
            action=np.asarray([0.5, 0.5]),
            info={
                'policy_step': False,
                'forward_velocity': 99.0,
                'world_x': 99.0,
                'upright_gate': 0.0,
            },
            timing=timing,
            update_info=None,
        )
        summary.record_step(
            action=np.asarray([-1.0, 0.0]),
            info={
                'policy_step': True,
                'forward_velocity': 0.4,
                'world_x': 1.5,
                'upright_gate': 1.0,
                'terminated': True,
                'is_recovering': True,
            },
            timing=timing,
            update_info={'q': 4.0, 'entropy': 5.0},
        )

        metrics = summary.metrics(replay_size=2)
        self.assertEqual(metrics['rolling/window_steps'], 2.0)
        self.assertEqual(metrics['rolling/total_policy_steps'], 2.0)
        self.assertAlmostEqual(metrics['rolling/forward_velocity_mean'], 0.3)
        self.assertAlmostEqual(metrics['rolling/world_x_delta'], 0.5)
        self.assertEqual(metrics['rolling/falls_total'], 1.0)
        self.assertEqual(metrics['rolling/recoveries_total'], 1.0)
        self.assertAlmostEqual(metrics['rolling/action_saturation_rate'], 0.5)
        self.assertAlmostEqual(metrics['rolling/q_mean'], 3.0)

    def test_truncated_policy_step_counts_as_timeout(self):
        summary = RollingTrainingSummary(window=2, action_dim=2)
        summary.record_step(
            action=np.zeros(2, dtype=np.float32),
            info={
                'policy_step': True,
                'truncated': True,
            },
            timing={},
            update_info=None,
        )

        metrics = summary.metrics(replay_size=1)
        self.assertEqual(metrics['rolling/timeouts_total'], 1.0)


if __name__ == '__main__':
    unittest.main()
