from __future__ import annotations

import unittest

import yaml

from scripts.train_mjlab_go2_natural_ppo import CHECKPOINT_EXPOSURES


class NaturalPpoTrainingGeometryTest(unittest.TestCase):
    def test_production_geometry_hits_every_checkpoint_exactly(self):
        steps_per_iteration = 2000 * 125
        for exposure in CHECKPOINT_EXPOSURES:
            self.assertEqual(exposure % steps_per_iteration, 0)

    def test_ppo_command_is_constant_forward_and_pushes_are_forbidden(self):
        with open("config/qsafe_natural_ppo_falls_v1.yaml", encoding="utf-8") as stream:
            protocol = yaml.safe_load(stream)
        command = protocol["environment"]["ppo_command"]
        self.assertEqual(command["distribution"], "constant")
        self.assertEqual(command["vx_mps"], 0.4)
        self.assertEqual(command["vy_mps"], 0.0)
        self.assertEqual(command["yaw_rate_rps"], 0.0)
        force = protocol["environment"]["external_force"]
        self.assertEqual(force["push_event"], "disabled")
        self.assertEqual(force["impulse"], "forbidden")


if __name__ == "__main__":
    unittest.main()
