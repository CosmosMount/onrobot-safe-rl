from __future__ import annotations

import unittest

import numpy as np

from learner.control_evaluation import (evaluate_control_facing,
                                        evaluate_double_critic_control)
from learner.counterfactual_dataset import CandidateBranch, HorizonOutcome


def _branch(snapshot, index, family, failure, near=False):
    outcome = HorizonOutcome(
        horizon=32, failure=failure, near_failure=near,
        time_to_failure=8 if failure else -1,
        max_tilt_rad=0.2, min_base_height_m=0.3,
        max_contact_count=4, max_undesired_contact_count=0,
        max_contact_force=1.0)
    action = np.asarray([float(index)], dtype=np.float32)
    return CandidateBranch(
        snapshot_index=snapshot, candidate_index=index,
        candidate_family=family, observation=np.zeros(2, np.float32),
        action=action, nominal_action=np.zeros(1, np.float32),
        previous_action=np.zeros(1, np.float32), command_speed=0.5,
        action_distance=float(index), outcomes={32: outcome})


class ControlEvaluationTest(unittest.TestCase):
    def test_perfect_ranking_and_replacement_reduce_failure(self):
        branches = [
            _branch(0, 0, 'nominal', True),
            _branch(0, 1, 'nominal_delta', False),
            _branch(1, 0, 'nominal', False),
            _branch(1, 1, 'nominal_delta', True),
        ]
        metrics = evaluate_control_facing(
            branches, [0.9, 0.1, 0.1, 0.9],
            epsilon=0.2, k_values=(1, 2))
        self.assertEqual(
            metrics['control_pairwise_risk_ranking_accuracy'], 1.0)
        self.assertEqual(
            metrics['control_selected_false_safe_rate'], 0.0)
        self.assertEqual(
            metrics['control_nominal_relative_failure_reduction'], 0.5)
        self.assertEqual(metrics['control_top1_safety_regret'], 0.0)
        self.assertEqual(len(metrics['control_k_curve']), 2)

    def test_reversed_candidate_ranking_is_exposed(self):
        branches = [
            _branch(0, 0, 'nominal', True),
            _branch(0, 1, 'nominal_delta', False),
        ]
        metrics = evaluate_control_facing(
            branches, [0.1, 0.9], epsilon=0.2)
        self.assertEqual(
            metrics['control_pairwise_risk_ranking_accuracy'], 0.0)
        self.assertEqual(
            metrics['control_selected_false_safe_rate'], 1.0)
        self.assertEqual(
            metrics['control_nominal_relative_failure_reduction'], 0.0)

    def test_structured_fallback_is_reported_separately(self):
        branches = [
            _branch(0, 0, 'nominal', True),
            _branch(0, 1, 'contracted_previous', False),
        ]
        metrics = evaluate_control_facing(
            branches, [0.9, 0.8], epsilon=0.2,
            structured_fallback=True)
        self.assertEqual(metrics['control_coverage'], 0.0)
        self.assertEqual(metrics['control_replacement_rate'], 0.0)
        self.assertEqual(metrics['control_fallback_rate'], 1.0)
        self.assertEqual(
            metrics['control_replacement_failure_contribution'], 0.0)
        self.assertEqual(
            metrics['control_fallback_failure_contribution'], 1.0)
        self.assertEqual(
            metrics['control_fallback_reduction_fraction'], 1.0)

    def test_validator_rejects_false_safe_without_searching(self):
        branches = [
            _branch(0, 0, 'nominal', True),
            _branch(0, 1, 'nominal_delta', True),
            _branch(0, 2, 'contracted_previous', False),
        ]
        metrics = evaluate_double_critic_control(
            branches,
            selector_risks=[0.9, 0.05, 0.4],
            validator_risks=[0.9, 0.95, 0.1],
            horizon=32, epsilon=0.2, improvement_margin=0.05)
        # B rejects A's candidate. It does not search for its own q=0.1
        # candidate; the predefined contracted action is used as abstention.
        self.assertEqual(metrics['double_validation_reject_rate'], 1.0)
        self.assertEqual(metrics['double_abstention_rate'], 1.0)
        self.assertEqual(metrics['double_replacement_rate'], 0.0)
        self.assertEqual(metrics['double_selected_failure_rate'], 0.0)
        self.assertEqual(metrics['double_failure_reduction'], 1.0)
        self.assertEqual(metrics['double_abstention_failure_reduction'], 1.0)


if __name__ == '__main__':
    unittest.main()
