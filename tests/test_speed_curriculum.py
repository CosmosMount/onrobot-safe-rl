import unittest

from train.speed_curriculum import PerformanceSpeedCurriculum


class PerformanceSpeedCurriculumTest(unittest.TestCase):

    def make_curriculum(self):
        return PerformanceSpeedCurriculum(
            min_speed=0.50,
            max_speed=0.65,
            increment=0.05,
            window=3,
            min_episode_length=100,
            min_velocity_ratio=0.70,
            max_fall_rate=0.0,
            new_stage_exploration_scale=0.4,
            exploration_recovery_episodes=2,
        )

    def test_promotes_only_after_all_frontier_conditions_pass(self):
        curriculum = self.make_curriculum()
        curriculum.record_episode(
            command_speed=0.50, mean_forward_velocity=0.40,
            episode_length=100, fell=False)
        curriculum.record_episode(
            command_speed=0.50, mean_forward_velocity=0.40,
            episode_length=99, fell=False)
        update = curriculum.record_episode(
            command_speed=0.50, mean_forward_velocity=0.40,
            episode_length=100, fell=False)
        self.assertFalse(update.promoted)
        self.assertEqual(update.upper_speed, 0.50)

        promoted = False
        for _ in range(3):
            update = curriculum.record_episode(
                command_speed=0.50, mean_forward_velocity=0.40,
                episode_length=120, fell=False)
            promoted = promoted or update.promoted
        self.assertTrue(promoted)
        self.assertAlmostEqual(curriculum.upper_speed, 0.55)

    def test_lower_speed_episodes_cannot_promote_frontier(self):
        curriculum = self.make_curriculum()
        for _ in range(10):
            update = curriculum.record_episode(
                command_speed=0.30, mean_forward_velocity=0.30,
                episode_length=400, fell=False)
        self.assertFalse(update.promoted)
        self.assertEqual(update.frontier_episodes, 0)

    def test_exploration_recovers_only_on_stable_frontier_episodes(self):
        curriculum = self.make_curriculum()
        for _ in range(3):
            update = curriculum.record_episode(
                command_speed=0.50, mean_forward_velocity=0.40,
                episode_length=120, fell=False)
        self.assertTrue(update.promoted)
        self.assertAlmostEqual(curriculum.exploration_multiplier, 0.4)

        curriculum.record_episode(
            command_speed=0.55, mean_forward_velocity=0.10,
            episode_length=20, fell=True)
        self.assertAlmostEqual(curriculum.exploration_multiplier, 0.4)
        curriculum.record_episode(
            command_speed=0.55, mean_forward_velocity=0.44,
            episode_length=120, fell=False)
        self.assertGreater(curriculum.exploration_multiplier, 0.4)
        self.assertLess(curriculum.exploration_multiplier, 1.0)
        curriculum.record_episode(
            command_speed=0.55, mean_forward_velocity=0.44,
            episode_length=120, fell=False)
        self.assertAlmostEqual(curriculum.exploration_multiplier, 1.0)

    def test_rejects_increment_above_five_centimeters_per_second(self):
        with self.assertRaises(ValueError):
            PerformanceSpeedCurriculum(
                min_speed=0.3, max_speed=1.0, increment=0.051)

    def test_resume_restores_clipped_frontier(self):
        curriculum = PerformanceSpeedCurriculum(
            min_speed=0.3, max_speed=0.8, increment=0.05,
            initial_upper_speed=0.4)
        self.assertAlmostEqual(curriculum.upper_speed, 0.4)


if __name__ == '__main__':
    unittest.main()
