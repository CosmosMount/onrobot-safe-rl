from __future__ import annotations

import unittest

from learner.active_branch_sampling import (
    ActiveBranchSampler,
    ActiveSnapshotSignals,
    build_active_snapshot_signals,
)


class ActiveBranchSamplingTest(unittest.TestCase):
    def test_probe_detects_validated_replacement(self):
        signals = build_active_snapshot_signals(
            [0.7, 0.1, 0.4],
            validator_risks=[0.8, 0.15, 0.5],
            supported=[True, True, True],
            behavior_log_prob_per_dim=[-1.0, -2.0, -3.9],
            action_distances=[0.0, 0.2, 0.95],
            epsilon=0.2,
            improvement_margin=0.02)
        self.assertTrue(signals.would_replace)
        self.assertFalse(signals.would_abstain)
        self.assertAlmostEqual(signals.max_disagreement, 0.1)
        self.assertLessEqual(
            signals.min_support_boundary_distance, 0.05)

    def test_probe_abstains_when_validator_rejects(self):
        signals = build_active_snapshot_signals(
            [0.7, 0.1],
            validator_risks=[0.8, 0.6],
            supported=[True, True],
            epsilon=0.2)
        self.assertFalse(signals.would_replace)
        self.assertTrue(signals.would_abstain)

    def test_sampler_enforces_priority_gap_and_quotas(self):
        sampler = ActiveBranchSampler(
            quota_per_reason=1, normal_quota=1,
            min_snapshot_gap=2, normal_interval=1)
        signals = ActiveSnapshotSignals(
            near_failure=True,
            max_disagreement=0.5,
            stable_normal=True)
        reason, triggered = sampler.consider(0, signals)
        self.assertEqual(reason, 'near_failure')
        self.assertIn('disagreement', triggered)
        reason, _ = sampler.consider(1, signals)
        self.assertIsNone(reason)
        reason, _ = sampler.consider(2, signals)
        self.assertEqual(reason, 'disagreement')
        reason, _ = sampler.consider(
            4, ActiveSnapshotSignals(stable_normal=True))
        self.assertEqual(reason, 'normal')
        self.assertEqual(sampler.counts['normal'], 1)


if __name__ == '__main__':
    unittest.main()
