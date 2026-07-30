from __future__ import annotations

import unittest

import numpy as np

from common.transition import Transition, zero_costs
from jaxrl.data.safety_replay import SafetyReplayManager
from learner.p15_protocol import split_safety_items_by_speed_episode
from train.speed_curriculum import PerformanceSpeedCurriculum


def _transition(speed, episode, value):
    return Transition(
        observation=np.asarray([value, speed], np.float32),
        requested_action=np.asarray([value], np.float32),
        projected_action=np.asarray([value], np.float32),
        executed_q_target=np.asarray([value], np.float32),
        reward=1.0,
        costs=zero_costs(),
        next_observation=np.asarray([value + 1, speed], np.float32),
        terminated=False,
        truncated=False,
        unsafe_label=False,
        near_failure_label=False,
        episode_id=episode,
        command_speed=speed,
    )


class P15TwoSpeedIntegrationTest(unittest.TestCase):

    def test_030_035_curriculum_replay_and_split(self):
        curriculum = PerformanceSpeedCurriculum(
            min_speed=0.30,
            max_speed=0.35,
            increment=0.05,
            window=1,
            min_episode_length=2,
            min_velocity_ratio=0.5,
            max_fall_rate=0.0,
            mode='performance_then_balanced',
            balance_min_transitions=6,
            balance_min_episodes=3,
        )
        curriculum.record_episode(
            command_speed=0.30, mean_forward_velocity=0.30,
            episode_length=2, fell=False)
        update = curriculum.record_episode(
            command_speed=0.35, mean_forward_velocity=0.35,
            episode_length=2, fell=False)
        self.assertEqual(update.phase, 'balanced')

        replay = SafetyReplayManager(
            recent_capacity=100,
            failure_capacity=100,
            boundary_capacity=100,
            recovery_capacity=100,
            all_capacity=100,
            failure_history=3,
            seed=42)
        episode = 0
        for speed in (0.30, 0.35):
            for _ in range(3):
                for step in range(2):
                    curriculum.record_transition(speed)
                    replay.insert(_transition(
                        speed, episode, float(step)))
                curriculum.record_episode(
                    command_speed=speed,
                    mean_forward_velocity=speed,
                    episode_length=2,
                    fell=False)
                episode += 1
        self.assertTrue(curriculum.coverage_complete)
        batch = replay.sample_mixed_by_speed(20, [0.30, 0.35])
        speed_counts = np.bincount(
            batch['speed_bin_ids'], minlength=2)
        self.assertLessEqual(abs(int(speed_counts[0] - speed_counts[1])), 2)
        train, calibration, validation, manifest = (
            split_safety_items_by_speed_episode(
                list(replay.all._items), [0.30, 0.35], seed=42))
        self.assertEqual(len(train) + len(calibration) + len(validation), 12)
        self.assertTrue(manifest['fingerprint'])


if __name__ == '__main__':
    unittest.main()
