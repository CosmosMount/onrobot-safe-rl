from __future__ import annotations

import unittest

from safety_data.parallel_capacity import CapacityResult, select_capacity


class CapacitySelectionTest(unittest.TestCase):
    def test_selects_largest_rung_with_fifteen_percent_gain(self):
        selected = select_capacity([
            CapacityResult(256, 1000.0, 4000, True),
            CapacityResult(512, 1300.0, 7000, True),
            CapacityResult(1024, 1510.0, 12000, True),
            CapacityResult(2048, 1600.0, 19000, True),
        ])
        self.assertEqual(selected.environments, 1024)

    def test_rejects_vram_force_nan_and_unstable_runs(self):
        selected = select_capacity([
            CapacityResult(256, 1000.0, 4000, True),
            CapacityResult(512, 2000.0, 21000, True),
            CapacityResult(1024, 4000.0, 15000, True,
                           external_force_nonzero=True),
            CapacityResult(2048, 8000.0, 19000, False),
        ])
        self.assertEqual(selected.environments, 256)

    def test_empty_or_all_failed_has_no_selection(self):
        self.assertIsNone(select_capacity([]))
        self.assertIsNone(select_capacity([
            CapacityResult(512, 1000.0, 1000, True, nonfinite=True),
        ]))


if __name__ == "__main__":
    unittest.main()
