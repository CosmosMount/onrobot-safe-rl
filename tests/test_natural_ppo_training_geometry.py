from __future__ import annotations

import unittest

from scripts.train_mjlab_go2_natural_ppo import CHECKPOINT_EXPOSURES


class NaturalPpoTrainingGeometryTest(unittest.TestCase):
    def test_production_geometry_hits_every_checkpoint_exactly(self):
        steps_per_iteration = 2000 * 125
        for exposure in CHECKPOINT_EXPOSURES:
            self.assertEqual(exposure % steps_per_iteration, 0)


if __name__ == "__main__":
    unittest.main()
