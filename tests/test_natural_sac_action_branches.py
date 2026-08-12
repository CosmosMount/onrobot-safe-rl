from __future__ import annotations

import unittest

import numpy as np

from safety_data.natural_sac_action_branches import (
    NaturalActionBranchPlan,
    build_risk_stratified_plan,
)


class NaturalSacActionBranchPlanTest(unittest.TestCase):
    def test_plan_is_deterministic_unique_and_risk_stratified(self):
        identities = np.asarray(
            [f"identity-{index:04d}" for index in range(400)], dtype="S64")
        risk = np.linspace(0.0, 1.0, 400, dtype=np.float32)
        first = build_risk_stratified_plan(
            identities=identities, state_risk=risk, groups=101)
        second = build_risk_stratified_plan(
            identities=identities, state_risk=risk, groups=101)
        np.testing.assert_array_equal(first.row_index, second.row_index)
        self.assertEqual(len(first.row_index), 101)
        self.assertEqual(len(np.unique(first.row_index)), 101)
        self.assertEqual(np.bincount(first.risk_band, minlength=4).tolist(),
                         [26, 25, 25, 25])
        self.assertTrue(np.all(first.acceptance_probability > 0.0))
        self.assertTrue(np.all(first.acceptance_probability <= 1.0))

    def test_plan_does_not_accept_or_read_branch_outcomes(self):
        identities = np.asarray([f"id-{index}" for index in range(20)], dtype="S64")
        risk = np.linspace(0.01, 0.99, 20)
        plan = build_risk_stratified_plan(
            identities=identities, state_risk=risk, groups=8)
        self.assertIsInstance(plan, NaturalActionBranchPlan)
        self.assertEqual(
            set(plan.__dataclass_fields__),
            {"row_index", "identity", "state_risk", "state_uncertainty", "risk_band",
             "acceptance_probability"})

    def test_tied_scores_reallocate_empty_quantile_bands(self):
        identities = np.asarray([f"id-{index}" for index in range(12)], dtype="S64")
        risk = np.full(12, 0.5, dtype=np.float32)
        plan = build_risk_stratified_plan(
            identities=identities, state_risk=risk, groups=7)
        self.assertEqual(len(plan.row_index), 7)
        self.assertEqual(len(np.unique(plan.row_index)), 7)

    def test_admission_excludes_safe_and_likely_unrecoverable_states(self):
        identities = np.asarray([f"id-{index}" for index in range(100)], dtype="S64")
        risk = np.linspace(0.0, 1.0, 100)
        uncertainty = np.zeros(100)
        uncertainty[40] = 0.3
        plan = build_risk_stratified_plan(
            identities=identities, state_risk=risk,
            state_uncertainty=uncertainty, groups=20)
        self.assertTrue(np.all(plan.state_risk >= 0.08))
        self.assertTrue(np.all(plan.state_risk <= 0.60))
        self.assertTrue(np.all(plan.state_uncertainty <= 0.20))
        self.assertNotIn(40, plan.row_index.tolist())


if __name__ == "__main__":
    unittest.main()
