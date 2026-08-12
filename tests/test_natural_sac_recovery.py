import unittest
from unittest.mock import Mock

import numpy as np

from safety_data.natural_sac_recovery import FixedNonpolicyRecoveryView
from safety_data.recovery_behaviors import RECOVERY_BEHAVIOR_STEPS


class NaturalSacRecoveryTest(unittest.TestCase):
    def test_fixed_view_makes_policy_options_unreachable(self):
        library = Mock()
        library.__class__ = __import__(
            "safety_data.recovery_behaviors", fromlist=["RecoveryBehaviorLibrary"]
        ).RecoveryBehaviorLibrary
        # Avoid faking the attested concrete type; construct without __init__
        library = object.__new__(library.__class__)
        calls = []
        library.__dict__["_fingerprint"] = "a" * 64
        library.__dict__["fingerprint"] = lambda: "a" * 64
        library.__dict__["__call__"] = None
        view = FixedNonpolicyRecoveryView(library)
        self.assertEqual(view.behavior_steps.tolist(), [
            0, RECOVERY_BEHAVIOR_STEPS[4], RECOVERY_BEHAVIOR_STEPS[5],
            RECOVERY_BEHAVIOR_STEPS[6], RECOVERY_BEHAVIOR_STEPS[7],
            RECOVERY_BEHAVIOR_STEPS[8]])
        self.assertNotIn(1, view.manifest()["original_k9_indices"])


if __name__ == "__main__":
    unittest.main()
