from __future__ import annotations

from types import SimpleNamespace
import unittest

from safety_data.mjlab_target_alignment import (
    TARGET_ACTION_SCALE,
    TARGET_INIT_JOINT,
    target_alignment_manifest,
    validate_target_aligned_go2,
)


class MjlabTargetAlignmentTest(unittest.TestCase):
    def _cfg(self):
        twist = SimpleNamespace(
            rel_standing_envs=0.0,
            ranges=SimpleNamespace(
                lin_vel_x=(0.3, 0.3),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
            ),
        )
        robot = SimpleNamespace(
            init_state=SimpleNamespace(
                pos=(0.0, 0.0, 0.445),
                joint_pos=dict(TARGET_INIT_JOINT),
            ),
        )
        return SimpleNamespace(
            sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=0.002)),
            decimation=10,
            episode_length_s=10.0,
            events={},
            terminations={"time_out": object(), "target_fall": object()},
            commands={"twist": twist},
            scene=SimpleNamespace(entities={"robot": robot}),
            actions={"joint_pos": SimpleNamespace(
                scale=dict(TARGET_ACTION_SCALE), use_default_offset=True)},
        )

    def test_manifest_is_stable_and_self_hashed(self):
        first = target_alignment_manifest()
        second = target_alignment_manifest()
        self.assertEqual(first, second)
        self.assertRegex(first["contract_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(first["physics_substeps_per_policy_step"], 10)
        self.assertEqual(first["initial_joint_position"], TARGET_INIT_JOINT)
        self.assertEqual(first["normalized_action_scale"], TARGET_ACTION_SCALE)

    def test_validator_accepts_exact_contract(self):
        validate_target_aligned_go2(self._cfg())

    def test_validator_rejects_upstream_timing_and_pose(self):
        cfg = self._cfg()
        cfg.sim.mujoco.timestep = 0.005
        with self.assertRaisesRegex(ValueError, "timing"):
            validate_target_aligned_go2(cfg)
        cfg = self._cfg()
        cfg.scene.entities["robot"].init_state.joint_pos[".*calf_joint"] = -1.8
        with self.assertRaisesRegex(ValueError, "initial pose"):
            validate_target_aligned_go2(cfg)

    def test_validator_rejects_push_and_command_drift(self):
        cfg = self._cfg()
        cfg.events["push_robot"] = object()
        with self.assertRaisesRegex(ValueError, "force or termination"):
            validate_target_aligned_go2(cfg)
        cfg = self._cfg()
        cfg.commands["twist"].ranges.lin_vel_x = (-0.4, 0.4)
        with self.assertRaisesRegex(ValueError, "command"):
            validate_target_aligned_go2(cfg)


if __name__ == "__main__":
    unittest.main()
