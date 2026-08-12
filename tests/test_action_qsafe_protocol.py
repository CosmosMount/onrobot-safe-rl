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
        cohort = protocol["candidate_oracle_gate"]["protected_cohort"]
        self.assertEqual(cohort["actor_training"]["seeds"], [57, 58])
        self.assertEqual(cohort["groups_per_source"], 30)
        self.assertEqual(cohort["replicas_per_action"], 32)
        self.assertEqual(
            [(item["actor_seed"], item["source_seed"])
             for item in cohort["sources"]],
            [(57, 9701), (57, 9702), (58, 9703), (58, 9704)])
        self.assertEqual(
            protocol["protected_evidence"]["ppo_unexecuted_action_labels"],
            "forbidden")
        digest = action_qsafe_protocol_sha256()
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
