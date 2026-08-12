from __future__ import annotations

import unittest

from safety_data.ppo_sqrl_protocol import (
    load_ppo_sqrl_protocol,
    ppo_sqrl_protocol_sha256,
)


class PpoSqrlProtocolTest(unittest.TestCase):
    def test_locks_first_round_and_action_semantics(self):
        protocol = load_ppo_sqrl_protocol()
        self.assertEqual(protocol["critic"]["gamma_safe"], 0.70)
        self.assertEqual(
            protocol["critic"]["critic_action"]["semantic"],
            "absolute_12d_joint_target_applied_to_pd_for_current_20ms_interval",
        )
        self.assertEqual(
            protocol["ppo_master_dataset"]["nested_aggregate_transition_counts"],
            [1_000_000, 3_000_000, 5_000_000],
        )
        self.assertFalse(
            protocol["first_round"]["allow_new_sac_50k_training"])
        self.assertEqual(len(ppo_sqrl_protocol_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
