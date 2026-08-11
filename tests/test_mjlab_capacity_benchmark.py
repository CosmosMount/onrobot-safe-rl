from __future__ import annotations

import unittest

from scripts.benchmark_mjlab_go2_natural_ppo import capacity_run_passes


class MjlabCapacityBenchmarkTest(unittest.TestCase):
    def test_capacity_gate_checks_duration_memory_and_runtime_integrity(self):
        valid = dict(
            elapsed_seconds=301.0,
            minimum_seconds=300.0,
            peak_vram_mib=19000,
            memory_growth_mib=0.0,
            nonfinite=False,
            external_force_nonzero=False,
            push_event_present=False,
        )
        self.assertTrue(capacity_run_passes(**valid))
        for field, value in (
            ("elapsed_seconds", 299.0),
            ("peak_vram_mib", 20481),
            ("memory_growth_mib", 129.0),
            ("nonfinite", True),
            ("external_force_nonzero", True),
            ("push_event_present", True),
        ):
            with self.subTest(field=field):
                invalid = {**valid, field: value}
                self.assertFalse(capacity_run_passes(**invalid))


if __name__ == "__main__":
    unittest.main()
