from __future__ import annotations

import unittest
import tempfile

import numpy as np

from common.transition import Transition, zero_costs
from jaxrl.data.safety_replay import SafetyReplayManager
from jaxrl.data.replay_buffer import ReplayBuffer
from jaxrl.env.specs import BoxSpec
from learner.checkpoint import (
    agent_state_hash,
    load_training_snapshot_metadata,
    restore_training_snapshot,
    save_training_snapshot,
    snapshot_agent_hash,
)


def _transition(value: float, *, unsafe: bool = False,
                boundary: bool = False, done: bool = False,
                command_speed: float = 0.6) -> Transition:
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
        near_failure_label=boundary or unsafe,
        policy_version=int(value),
        episode_id=17,
        command_speed=command_speed,
    )


def _manager(history: int = 3) -> SafetyReplayManager:
    return SafetyReplayManager(
        recent_capacity=20, failure_capacity=20, boundary_capacity=20,
        recovery_capacity=10, all_capacity=50, failure_history=history,
        seed=7)


class SafetyReplayTest(unittest.TestCase):

    def test_sample_recent_balanced_half_positives(self):
        replay = _manager(history=3)
        for value in range(20):
            replay.insert(_transition(float(value)))
        replay.insert(_transition(20.0, unsafe=True, done=True))
        batch = replay.sample_recent_balanced(32)
        rate = float(np.mean(batch['future_failure_labels']))
        self.assertGreaterEqual(rate, 0.4)
        self.assertLessEqual(rate, 0.6)

    def test_failure_buffer_backfills_h_preceding_steps(self):
        replay = _manager(history=3)
        for value in range(5):
            replay.insert(_transition(float(value)))
        replay.insert(_transition(5.0, unsafe=True, done=True))

        self.assertEqual(replay.sizes.recent, 6)
        self.assertEqual(replay.sizes.all, 6)
        self.assertEqual(replay.sizes.failure, 4)
        failure_values = [
            float(item['observations'][0]) for item in replay.failure._items
        ]
        self.assertEqual(failure_values, [2.0, 3.0, 4.0, 5.0])
        self.assertTrue(all(
            item['future_failure_labels'] == 1.0
            for item in replay.failure._items))
        recent_labels = [
            item['future_failure_labels'] for item in replay.recent._items
        ]
        self.assertEqual(recent_labels, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    def test_multi_horizon_labels_and_metadata(self):
        replay = _manager(history=20)
        for value in range(12):
            replay.insert(_transition(float(value)))
        replay.insert(_transition(12.0, unsafe=True, done=True))

        items = list(replay.recent._items)
        oldest = items[0]
        eight_steps_before = items[4]
        failure = items[-1]
        self.assertEqual(int(oldest['time_to_failure_steps']), 12)
        self.assertEqual(oldest['future_failure_h8'], 0.0)
        self.assertEqual(oldest['future_failure_h16'], 1.0)
        self.assertEqual(eight_steps_before['future_failure_h8'], 1.0)
        self.assertEqual(int(failure['time_to_failure_steps']), 0)
        self.assertEqual(failure['future_failure_h8'], 1.0)
        self.assertEqual(int(failure['episode_ids']), 17)
        self.assertEqual(int(failure['policy_versions']), 12)
        self.assertAlmostEqual(float(failure['command_speeds']), 0.6)

    def test_boundary_and_recovery_are_isolated(self):
        replay = _manager()
        replay.insert(_transition(1.0, boundary=True))
        replay.insert(_transition(2.0), policy_step=False)

        self.assertEqual(replay.sizes.boundary, 1)
        self.assertEqual(replay.sizes.recovery, 1)
        self.assertEqual(replay.sizes.all, 1)
        self.assertEqual(replay.sizes.recent, 1)

    def test_n_step_target_stops_at_failure(self):
        replay = SafetyReplayManager(
            recent_capacity=20, failure_capacity=20, boundary_capacity=20,
            recovery_capacity=10, all_capacity=50, failure_history=3,
            n_step=3, seed=7)
        replay.insert(_transition(0.0), behavior_noise_std=0.4)
        replay.insert(_transition(1.0), behavior_noise_std=0.4)
        replay.insert(
            _transition(2.0, unsafe=True, done=True),
            behavior_noise_std=0.4)
        first = replay.recent._items[0]
        self.assertEqual(int(first['n_step_steps']), 3)
        self.assertEqual(first['n_step_unsafe_labels'], 1.0)
        self.assertEqual(first['n_step_masks'], 0.0)
        self.assertEqual(first['behavior_noise_std'], 0.4)
        np.testing.assert_array_equal(
            first['n_step_next_observations'],
            np.full(3, 3.0, dtype=np.float32))

    def test_mixed_batch_uses_requested_40_30_20_10_split(self):
        replay = _manager()
        replay.insert(_transition(0.0))
        replay.insert(_transition(1.0, boundary=True))
        replay.insert(_transition(2.0, unsafe=True, done=True))

        batch = replay.sample_mixed(100)
        counts = np.bincount(batch['source_ids'], minlength=4)
        np.testing.assert_array_equal(counts, [40, 30, 20, 10])
        self.assertEqual(batch['observations'].shape, (100, 3))
        self.assertEqual(batch['actions'].shape, (100, 2))
        self.assertEqual(batch['costs'].shape, (100, 9))
        self.assertEqual(batch['unsafe_labels'].shape, (100,))
        self.assertEqual(batch['future_failure_labels'].shape, (100,))

    def test_sample_recent_only_uses_recent_buffer(self):
        replay = _manager()
        replay.insert(_transition(0.0))
        replay.insert(_transition(1.0, boundary=True))
        replay.insert(_transition(2.0, unsafe=True, done=True))
        batch = replay.sample_recent(16)
        self.assertEqual(batch['observations'].shape, (16, 3))
        self.assertTrue(np.all(batch['source_ids'] == 0))
        with self.assertRaises(ValueError):
            _manager().sample_recent(4)

    def test_sparse_sources_fall_back_without_reducing_batch(self):
        replay = _manager()
        replay.insert(_transition(0.0))
        batch = replay.sample_mixed(17)
        self.assertEqual(batch['observations'].shape[0], 17)
        self.assertTrue(np.all(np.isin(batch['source_ids'], [0, 3])))

    def test_mixed_speed_stratification_balances_available_speed_bins(self):
        replay = _manager()
        for index in range(30):
            replay.insert(_transition(
                float(index), command_speed=0.30))
        for index in range(2):
            replay.insert(_transition(
                float(100 + index), command_speed=0.35))
        batch = replay.sample_mixed_by_speed(200, [0.30, 0.35])
        counts = np.bincount(batch['speed_bin_ids'], minlength=2)
        self.assertEqual(int(np.sum(counts)), 200)
        self.assertGreaterEqual(int(counts[0]), 90)
        self.assertGreaterEqual(int(counts[1]), 90)

    def test_compact_state_round_trip(self):
        replay = _manager()
        replay.insert(_transition(1.0, boundary=True))
        replay.insert(_transition(2.0, unsafe=True, done=True))
        restored = _manager()
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.sizes, replay.sizes)
        self.assertEqual(restored.failure_history, replay.failure_history)

    def test_legacy_state_adds_metadata_and_horizon_defaults(self):
        replay = _manager(history=3)
        replay.insert(_transition(1.0))
        state = replay.state_dict()
        state.pop('failure_horizons')
        for name in ('recent', 'failure', 'boundary', 'recovery', 'all'):
            for item in state[name]['items']:
                for key in (
                        'episode_ids', 'policy_versions', 'command_speeds',
                        'time_to_failure_steps', 'future_failure_h8',
                        'future_failure_h16', 'future_failure_h32'):
                    item.pop(key, None)
        restored = _manager(history=3)
        restored.load_state_dict(state)
        item = restored.recent._items[0]
        self.assertIn('episode_ids', item)
        self.assertIn('time_to_failure_steps', item)
        self.assertIn('future_failure_h8', item)

    def test_training_checkpoint_restores_safety_replay(self):
        replay = _manager()
        replay.insert(_transition(1.0, boundary=True))
        rl = ReplayBuffer(BoxSpec(shape=(3,), dtype=np.float32),
                          BoxSpec(shape=(2,), dtype=np.float32), 10)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_training_snapshot(
                tmp, agent={'weight': np.asarray([1.0])}, replay_buffer=rl,
                safety_replay=replay, step=3)
            restored_safety = _manager()
            restored_rl = ReplayBuffer(
                BoxSpec(shape=(3,), dtype=np.float32),
                BoxSpec(shape=(2,), dtype=np.float32), 10)
            restore_training_snapshot(
                path, agent={'weight': np.asarray([0.0])},
                replay_buffer=restored_rl, safety_replay=restored_safety)
            self.assertEqual(restored_safety.sizes, replay.sizes)
            metadata = load_training_snapshot_metadata(path)
            self.assertEqual(
                metadata['agent_state_hash'], snapshot_agent_hash(path))

    def test_agent_state_hash_is_stable_and_detects_parameter_changes(self):
        left = {
            'critic': {'b': np.asarray([2.0]), 'a': np.asarray([1.0])},
            'step': np.int32(4),
        }
        reordered = {
            'step': np.int32(4),
            'critic': {'a': np.asarray([1.0]), 'b': np.asarray([2.0])},
        }
        changed = {
            'critic': {'a': np.asarray([1.0]), 'b': np.asarray([2.1])},
            'step': np.int32(4),
        }
        self.assertEqual(agent_state_hash(left), agent_state_hash(reordered))
        self.assertNotEqual(agent_state_hash(left), agent_state_hash(changed))


if __name__ == '__main__':
    unittest.main()
