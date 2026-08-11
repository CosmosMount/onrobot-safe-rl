from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from safety_data.natural_ppo_falls import (
    DeterministicNormalReservoir,
    NaturalFallRecorder,
    NaturalFallShardWriter,
    NaturalPpoFrame,
    PREFALL_OFFSETS,
)


def frame(step: int, *, environment: int = 0, episode: int = 1,
          training: int = 1000, force: float = 0.0) -> NaturalPpoFrame:
    return NaturalPpoFrame(
        environment_id=environment,
        episode_id=episode,
        episode_step=step,
        global_policy_step=10000 + step,
        ppo_training_step=training,
        integration_state=np.asarray([step, 1.0], dtype=np.float64),
        qpos=np.full(19, step, dtype=np.float64),
        qvel=np.full(18, step, dtype=np.float64),
        act=np.empty(0, dtype=np.float64),
        ctrl=np.full(12, step, dtype=np.float64),
        observation_history=np.full((5, 46), step, dtype=np.float32),
        previous_action_requested=np.zeros(12, dtype=np.float32),
        previous_action_executed=np.zeros(12, dtype=np.float32),
        previous_action_q_target=np.zeros(12, dtype=np.float32),
        randomization={"friction": 0.8},
        rng_identity=20000 + step,
        external_force=np.full((1, 6), force, dtype=np.float64),
    )


class NaturalFallRecorderTest(unittest.TestCase):
    def test_first_terminal_is_one_fall_and_requires_immediate_reset(self):
        recorder = NaturalFallRecorder([0])
        for step in range(10):
            recorder.append(frame(step))
        event = recorder.finish_episode(0, fell=True)
        self.assertIsNotNone(event)
        self.assertEqual(recorder.recorded_falls, 1)
        self.assertEqual(recorder.independent_fall_episodes, 1)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            recorder.append(frame(10))
        with self.assertRaisesRegex(RuntimeError, "already"):
            recorder.finish_episode(0, fell=True)
        recorder.reset(0)
        recorder.append(frame(0, episode=2))

    def test_short_episode_is_retained_with_availability_mask(self):
        recorder = NaturalFallRecorder([0])
        for step in range(5):
            recorder.append(frame(step))
        event = recorder.finish_episode(0, fell=True)
        assert event is not None
        self.assertEqual(event.availability.tolist(),
                         [True, True, True, False, False, False, False])
        self.assertEqual(len(event.prefall), len(PREFALL_OFFSETS))

    def test_nonzero_external_force_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "external force"):
            frame(0, force=1.0)

    def test_duplicate_snapshot_identity_fails_closed(self):
        recorder = NaturalFallRecorder([0])
        value = frame(0)
        recorder.append(value)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            recorder.append(value)


class NormalReservoirTest(unittest.TestCase):
    def test_fall_nearby_frames_are_not_normal_candidates(self):
        reservoir = DeterministicNormalReservoir()
        values = [frame(step) for step in range(120)]
        reservoir.add_episode(values, fell=True)
        # Only steps at least 97 before the terminal can enter the reservoir.
        matched = reservoir.match(frame(0))
        self.assertIsNotNone(matched)
        assert matched is not None
        self.assertLessEqual(matched.episode_step, 22)

    def test_matching_is_deterministic_and_without_replacement(self):
        values = [frame(step) for step in range(5)]
        first = DeterministicNormalReservoir()
        second = DeterministicNormalReservoir()
        first.add_episode(values, fell=False)
        second.add_episode(reversed(values), fell=False)
        a = first.match(values[0])
        b = second.match(values[0])
        self.assertEqual(a.identity, b.identity)
        self.assertNotEqual(first.match(values[0]).identity, a.identity)


class NaturalFallShardWriterTest(unittest.TestCase):
    def test_manifest_is_published_last_and_forbids_direct_labels(self):
        recorder = NaturalFallRecorder([0])
        for step in range(4):
            recorder.append(frame(step))
        event = recorder.finish_episode(0, fell=True)
        assert event is not None
        with tempfile.TemporaryDirectory() as temporary:
            writer = NaturalFallShardWriter(temporary, events_per_shard=1)
            writer.add(event)
            manifest_path = writer.close(provenance={"commit": "abc"})
            manifest = json.loads(manifest_path.read_text())
            self.assertFalse(manifest["ppo_outcomes_are_qsafe_labels"])
            self.assertEqual(manifest["event_count"], 1)
            shard = Path(temporary) / manifest["shards"][0]["path"]
            with np.load(shard, allow_pickle=False) as arrays:
                self.assertEqual(arrays["availability"].shape, (1, 7))
                self.assertEqual(
                    arrays["trajectory_observation_history"].shape,
                    (1, 65, 5, 46))
                self.assertEqual(arrays["prefall_qpos"].shape, (1, 7, 19))
                self.assertEqual(
                    arrays["prefall_previous_action_q_target"].shape,
                    (1, 7, 12))
                self.assertNotIn("fall_label", arrays.files)
            with self.assertRaisesRegex(RuntimeError, "already"):
                writer.close(provenance={})


if __name__ == "__main__":
    unittest.main()
