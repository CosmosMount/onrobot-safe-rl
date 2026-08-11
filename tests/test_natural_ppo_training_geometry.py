from __future__ import annotations

import unittest

import yaml

from scripts.train_mjlab_go2_natural_ppo import (
    CHECKPOINT_EXPOSURES,
    require_clean_production_worktree,
)


class NaturalPpoTrainingGeometryTest(unittest.TestCase):
    def test_only_fixed_30m_production_requires_clean_worktree(self):
        require_clean_production_worktree(1_000_000, b"dirty")
        require_clean_production_worktree(30_000_000, b"")
        with self.assertRaisesRegex(RuntimeError, "clean generator"):
            require_clean_production_worktree(30_000_000, b"dirty")

    def test_production_geometry_hits_every_checkpoint_exactly(self):
        steps_per_iteration = 2000 * 125
        for exposure in CHECKPOINT_EXPOSURES:
            self.assertEqual(exposure % steps_per_iteration, 0)

    def test_ppo_command_is_constant_forward_and_pushes_are_forbidden(self):
        with open("config/qsafe_natural_ppo_falls_v2.yaml", encoding="utf-8") as stream:
            protocol = yaml.safe_load(stream)
        command = protocol["environment"]["ppo_command"]
        self.assertEqual(command["distribution"], "constant")
        self.assertEqual(command["vx_mps"], 0.3)
        self.assertEqual(command["vy_mps"], 0.0)
        self.assertEqual(command["yaw_rate_rps"], 0.0)
        self.assertEqual(command["standing_environment_fraction"], 0.0)
        force = protocol["environment"]["external_force"]
        self.assertEqual(force["push_event"], "disabled")
        self.assertEqual(force["impulse"], "forbidden")

    def test_qsafe_is_state_trigger_for_original_nonpolicy_recovery(self):
        with open("config/qsafe_natural_ppo_falls_v2.yaml", encoding="utf-8") as stream:
            protocol = yaml.safe_load(stream)
        supervision = protocol["direct_ppo_supervision"]
        self.assertEqual(supervision["supervised_heads"], ["state_risk"])
        runtime = protocol["qsafe_runtime"]
        self.assertEqual(runtime["critic_role"], "state_risk_trigger")
        self.assertFalse(runtime["learned_candidate_selector"])
        recovery = runtime["recovery"]
        self.assertFalse(recovery["learned_policy_used"])
        self.assertEqual(recovery["stages"], [
            "fold", "above", "swing_down", "push"])
        self.assertEqual(recovery["gains"], {"kp": 100.0, "kd": 8.0})
        self.assertTrue(recovery["fall_predicate_remains_active_during_recovery"])
        self.assertEqual(recovery["recovery_threshold_crossing_exemption"], "forbidden")
        self.assertTrue(protocol["qsafe_validation"]["fixed_recovery_preflight"][
            "any_recovery_threshold_crossing_counts_as_fall"])
        self.assertEqual(protocol["qsafe_validation"]["paired_arms"], [
            "nominal_sac", "qsafe_fixed_recovery",
            "matched_random_fixed_recovery",
        ])


if __name__ == "__main__":
    unittest.main()
