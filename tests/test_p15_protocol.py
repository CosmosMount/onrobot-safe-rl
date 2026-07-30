from __future__ import annotations

import unittest

from learner.p15_protocol import (
    evaluate_p15_gate,
    split_safety_items_by_speed_episode,
)


class P15ProtocolTest(unittest.TestCase):

    def test_gate_requires_every_metric(self):
        natural = {
            'Q_safe_AUROC': 0.85,
            'Q_safe_calibration_ece': 0.05,
            'Q_safe_brier': 0.10,
        }
        control = {
            'control_pairwise_risk_ranking_accuracy': 0.70,
            'control_selected_false_safe_rate': 0.04,
            'control_coverage': 0.40,
            'control_replacement_rate': 0.20,
            'control_replacement_failure_contribution': 0.02,
            'control_fallback_reduction_fraction': 0.40,
        }
        self.assertTrue(
            evaluate_p15_gate(natural, control)['p15_gate_passed'])
        control['control_replacement_rate'] = 0.14
        failed = evaluate_p15_gate(natural, control)
        self.assertFalse(failed['p15_gate_passed'])
        self.assertIn('replacement_rate', failed['p15_gate_failed_checks'])

    def test_speed_episode_split_has_no_leakage(self):
        items = []
        for speed in (0.30, 0.35):
            for episode in range(10):
                for step in range(2):
                    items.append({
                        'command_speeds': speed,
                        'episode_ids': episode + int(speed * 1000),
                        'step': step,
                    })
        train, calibration, validation, manifest = (
            split_safety_items_by_speed_episode(
                items, [0.30, 0.35], seed=7))
        episode_sets = [
            {(item['command_speeds'], item['episode_ids'])
             for item in split}
            for split in (train, calibration, validation)
        ]
        self.assertFalse(episode_sets[0].intersection(episode_sets[1]))
        self.assertFalse(episode_sets[0].intersection(episode_sets[2]))
        self.assertFalse(episode_sets[1].intersection(episode_sets[2]))
        self.assertEqual(sum(map(len, (train, calibration, validation))), 40)
        self.assertEqual(len(manifest['fingerprint']), 64)


if __name__ == '__main__':
    unittest.main()
