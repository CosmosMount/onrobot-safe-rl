from __future__ import annotations

import unittest

from safety_data.action_qsafe_protocol import (
    action_qsafe_protocol_sha256,
    load_action_qsafe_protocol,
)


class ActionQsafeProtocolTest(unittest.TestCase):
    def test_active_protocol_locks_action_conditioning_and_phase_order(self):
        protocol = load_action_qsafe_protocol()
        self.assertEqual(
            protocol["target"]["critic_estimand"],
            "P(fall within H96 | deployable state s, execute candidate a now, paired SAC continuation)")
        self.assertTrue(
            protocol["candidate_oracle_gate"]["required_before_model_training"])
        self.assertFalse(
            protocol["state_only_model"]["final_selector"])
        self.assertFalse(
            protocol["protected_evidence"][
                "old_state_only_recovery_results_objective1_eligible"])
        self.assertFalse(
            protocol["protected_evidence"][
                "objective2_authorized_before_objective1_pass"])
        digest = action_qsafe_protocol_sha256()
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
