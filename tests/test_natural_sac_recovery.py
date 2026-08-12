import unittest
from unittest.mock import patch

import numpy as np

from safety_data.natural_sac_recovery import (
    _bootstrap_paired_lcb,
    FixedNonpolicyRecoveryView,
)
from safety_data.recovery_behaviors import (
    RECOVERY_BEHAVIOR_STEPS,
    RecoveryBehaviorLibrary,
)


class NaturalSacRecoveryTest(unittest.TestCase):
    def test_fixed_view_makes_policy_options_unreachable(self):
        library = object.__new__(RecoveryBehaviorLibrary)
        library.__dict__["_fingerprint"] = "a" * 64
        view = FixedNonpolicyRecoveryView(library)
        self.assertEqual(view.behavior_steps.tolist(), [
            0, RECOVERY_BEHAVIOR_STEPS[4], RECOVERY_BEHAVIOR_STEPS[5],
            RECOVERY_BEHAVIOR_STEPS[6], RECOVERY_BEHAVIOR_STEPS[7],
            RECOVERY_BEHAVIOR_STEPS[8]])
        self.assertNotIn(1, view.manifest()["original_k9_indices"])
        history = np.zeros((5, 46), dtype=np.float32)
        nominal = np.zeros(12, dtype=np.float32)
        with patch.object(
                RecoveryBehaviorLibrary, "__call__",
                return_value=np.ones(12, dtype=np.float32)) as called:
            result = view(1, history, 0, nominal)
        np.testing.assert_array_equal(result, 1.0)
        self.assertEqual(called.call_args.args[0], 4)

    def test_paired_bootstrap_detects_consistent_improvement(self):
        low, median, high = _bootstrap_paired_lcb(
            np.asarray([1.0] * 80 + [-1.0] * 20), seed=7)
        self.assertGreater(low, 0.0)
        self.assertGreaterEqual(high, median)


if __name__ == "__main__":
    unittest.main()
