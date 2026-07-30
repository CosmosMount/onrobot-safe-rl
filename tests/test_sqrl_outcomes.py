from __future__ import annotations

import unittest

from train.sqrl_outcomes import ReplacementOutcomeTracker


class ReplacementOutcomeTrackerTest(unittest.TestCase):

    def test_replacement_gets_horizon_outcomes(self):
        tracker = ReplacementOutcomeTracker((2, 4))
        tracker.record_replacement(10, {
            'nominal_Q_safe_A': 0.8,
            'selected_Q_safe': 0.1,
            'selected_nominal_action_distance': 0.2,
            'sqrl_selected_group': 1,
        })
        tracker.record_step(
            10, unsafe=False, near_failure=False, done=False)
        tracker.record_step(
            11, unsafe=False, near_failure=True, done=False)
        tracker.record_step(
            12, unsafe=True, near_failure=True, done=True)
        self.assertEqual(len(tracker.completed), 1)
        event = tracker.completed[0]
        self.assertFalse(event['outcomes']['2']['failure'])
        self.assertTrue(event['outcomes']['4']['failure'])
        self.assertEqual(event['outcomes']['4']['time_to_failure'], 3)

    def test_fall_without_recent_replacement_is_false_negative(self):
        tracker = ReplacementOutcomeTracker((2, 4))
        tracker.record_step(
            10, unsafe=True, near_failure=True, done=True)
        metrics = tracker.metrics()
        self.assertEqual(metrics['sqrl/false_negative_falls_h2'], 1.0)
        self.assertEqual(metrics['sqrl/false_negative_falls_h4'], 1.0)

    def test_fallback_does_not_hide_false_negative_fall(self):
        tracker = ReplacementOutcomeTracker((2,))
        tracker.record_replacement(10, {'_selection_kind': 'fallback'})
        tracker.record_step(
            10, unsafe=False, near_failure=True, done=False)
        tracker.record_step(
            11, unsafe=True, near_failure=True, done=True)
        metrics = tracker.metrics()
        self.assertEqual(metrics['sqrl/false_negative_falls_h2'], 1.0)
        self.assertEqual(metrics['sqrl/fallback_outcomes_h2'], 1.0)
        self.assertEqual(metrics['sqrl/replacement_outcomes_h2'], 0.0)


if __name__ == '__main__':
    unittest.main()
