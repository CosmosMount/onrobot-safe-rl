from __future__ import annotations

import unittest
from pathlib import Path
import json
import tempfile

import yaml

from scripts.train_mjlab_go2_natural_ppo import (
    CHECKPOINT_EXPOSURES,
    require_clean_production_worktree,
    validate_preflight_authorizations,
)
from safety_data.mjlab_target_alignment import target_alignment_manifest


class NaturalPpoTrainingGeometryTest(unittest.TestCase):
    def test_training_requires_capacity_model_and_parity_authorization(self):
        contract = target_alignment_manifest()["contract_sha256"]
        versions = {"mujoco": "3.5.0", "mujoco_warp": "3.5.0", "warp": "1.12.0"}
        capacity = {
            "schema_version": "qsafe.mjlab_capacity_authorization.v1",
            "authorized": True,
            "production_envs": 2000,
            "selected_capacity_envs": 2048,
            "target_alignment_contract_sha256": contract,
            "backend_versions": versions,
        }
        model = {
            "schema_version": "qsafe.mjlab_target_model_contract.v1",
            "pass": True,
            "external_force_nonzero": False,
            "target_alignment": {"contract_sha256": contract},
            "versions": versions,
        }
        parity = {
            "schema_version": "qsafe.mjlab_native_parity.v1",
            "pass": True,
            "states": 100,
            "policy_steps_per_state": 100,
            "fall_predicate_agreement": 1.0,
            "external_force_nonzero": False,
            "target_alignment": {"contract_sha256": contract},
            "versions": versions,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name, value in (("capacity", capacity), ("model", model), ("parity", parity)):
                path = root / f"{name}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                paths.append(path)
            result = validate_preflight_authorizations(
                capacity_path=paths[0], model_contract_path=paths[1],
                parity_path=paths[2], production_envs=2000)
            self.assertEqual(result["target_alignment_contract_sha256"], contract)
            parity["fall_predicate_agreement"] = 0.99
            paths[2].write_text(json.dumps(parity), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees on falls"):
                validate_preflight_authorizations(
                    capacity_path=paths[0], model_contract_path=paths[1],
                    parity_path=paths[2], production_envs=2000)

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
        randomization = protocol["environment"]["domain_randomization"]
        self.assertEqual(randomization["enabled"], [
            "foot_friction", "base_com", "encoder_bias", "reset_base_pose"])
        self.assertIn("body_mass", randomization["disabled_or_zero_range"])

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
