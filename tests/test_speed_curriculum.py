import unittest

from train.env import Go2Env
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

    def test_balanced_phase_requires_max_frontier_to_pass(self):
        curriculum = PerformanceSpeedCurriculum(
            min_speed=0.30,
            max_speed=0.35,
            increment=0.05,
            window=2,
            min_episode_length=10,
            min_velocity_ratio=0.70,
            max_fall_rate=0.0,
            mode='performance_then_balanced',
            balance_min_transitions=3,
            balance_min_episodes=1,
        )
        for _ in range(2):
            update = curriculum.record_episode(
                command_speed=0.30,
                mean_forward_velocity=0.30,
                episode_length=10,
                fell=False,
            )
        self.assertTrue(update.promoted)
        self.assertFalse(update.frontier_complete)
        self.assertEqual(update.phase, 'performance')

        # Reaching max_speed is not sufficient: the 0.35 frontier itself must
        # collect a complete passing window.
        curriculum.record_episode(
            command_speed=0.35,
            mean_forward_velocity=0.35,
            episode_length=10,
            fell=False,
        )
        self.assertEqual(curriculum.phase, 'performance')
        update = curriculum.record_episode(
            command_speed=0.35,
            mean_forward_velocity=0.35,
            episode_length=10,
            fell=False,
        )
        self.assertTrue(update.frontier_complete)
        self.assertEqual(update.phase, 'balanced')

    def test_balanced_coverage_requires_transitions_and_episodes_per_speed(self):
        curriculum = PerformanceSpeedCurriculum(
            min_speed=0.30,
            max_speed=0.35,
            increment=0.05,
            window=1,
            min_episode_length=1,
            min_velocity_ratio=0.0,
            max_fall_rate=1.0,
            mode='performance_then_balanced',
            balance_min_transitions=2,
            balance_min_episodes=1,
        )
        curriculum.record_episode(
            command_speed=0.30,
            mean_forward_velocity=0.30,
            episode_length=1,
            fell=False,
        )
        curriculum.record_episode(
            command_speed=0.35,
            mean_forward_velocity=0.35,
            episode_length=1,
            fell=False,
        )
        self.assertEqual(curriculum.phase, 'balanced')

        for speed in (0.30, 0.35):
            curriculum.record_transition(speed)
            curriculum.record_transition(speed)
        self.assertFalse(curriculum.coverage_complete)
        curriculum.record_episode(
            command_speed=0.30,
            mean_forward_velocity=0.30,
            episode_length=2,
            fell=False,
        )
        self.assertFalse(curriculum.coverage_complete)
        update = curriculum.record_episode(
            command_speed=0.35,
            mean_forward_velocity=0.35,
            episode_length=2,
            fell=True,
        )
        self.assertTrue(update.coverage_complete)
        manifest = curriculum.manifest()
        self.assertEqual(manifest['balanced_transitions']['0.30'], 2)
        self.assertEqual(manifest['balanced_episodes']['0.35'], 1)
        self.assertEqual(manifest['balanced_falls']['0.35'], 1)

    def test_balanced_environment_sampler_round_robins_all_bins(self):
        # The sampler does not need a controller connection, so construct only
        # the state used by _sample_episode_cmd_speed.
        env = Go2Env.__new__(Go2Env)
        env.cmd_speed_curriculum = True
        env.cmd_speed_curriculum_mode = 'performance_then_balanced'
        env._curriculum_phase = 'balanced'
        env._balanced_speed_cursor = 0
        env.cmd_speed_min = 0.30
        env.cmd_speed_max = 0.40
        env.cmd_speed_increment = 0.05
        speeds = [env._sample_episode_cmd_speed() for _ in range(5)]
        self.assertEqual(speeds, [0.30, 0.35, 0.40, 0.30, 0.35])


if __name__ == '__main__':
    unittest.main()
