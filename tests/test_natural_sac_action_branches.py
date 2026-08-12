from __future__ import annotations

import unittest

import numpy as np

from safety_data.natural_sac_action_branches import (
    NaturalActionBranchPlan,
    build_early_prefall_plan,
)


class NaturalSacActionBranchPlanTest(unittest.TestCase):
    def test_plan_is_deterministic_unique_and_risk_stratified(self):
        identities = np.asarray(
            [f"identity-{index:04d}" for index in range(400)], dtype="S64")
        risk = np.linspace(0.0, 1.0, 400, dtype=np.float32)
        label = np.ones(400, dtype=bool)
        steps = np.tile(np.arange(48, 97), 9)[:400]
        uncertainty = np.zeros(400)
        first = build_early_prefall_plan(
            identities=identities, state_risk=risk,
            state_uncertainty=uncertainty, natural_fall_label=label,
            natural_steps_to_outcome=steps, groups=101)
        second = build_early_prefall_plan(
            identities=identities, state_risk=risk,
            state_uncertainty=uncertainty, natural_fall_label=label,
            natural_steps_to_outcome=steps, groups=101)
        np.testing.assert_array_equal(first.row_index, second.row_index)
        self.assertEqual(len(first.row_index), 101)
        self.assertEqual(len(np.unique(first.row_index)), 101)
        self.assertEqual(np.bincount(first.admission_band, minlength=4).tolist(),
                         [26, 25, 25, 25])
        self.assertTrue(np.all(first.acceptance_probability > 0.0))
        self.assertTrue(np.all(first.acceptance_probability <= 1.0))

    def test_plan_does_not_accept_or_read_branch_outcomes(self):
        identities = np.asarray([f"id-{index}" for index in range(20)], dtype="S64")
        risk = np.linspace(0.01, 0.99, 20)
        plan = build_early_prefall_plan(
            identities=identities, state_risk=risk,
            state_uncertainty=np.zeros(20), natural_fall_label=np.ones(20),
            natural_steps_to_outcome=np.full(20, 64), groups=8)
        self.assertIsInstance(plan, NaturalActionBranchPlan)
        self.assertEqual(
            set(plan.__dataclass_fields__),
            {"row_index", "identity", "state_risk", "state_uncertainty",
             "natural_steps_to_fall", "admission_band",
             "acceptance_probability"})

    def test_tied_scores_reallocate_empty_quantile_bands(self):
        identities = np.asarray([f"id-{index}" for index in range(12)], dtype="S64")
        risk = np.full(12, 0.5, dtype=np.float32)
        plan = build_early_prefall_plan(
            identities=identities, state_risk=risk,
            state_uncertainty=np.zeros(12), natural_fall_label=np.ones(12),
            natural_steps_to_outcome=np.full(12, 64), groups=7)
        self.assertEqual(len(plan.row_index), 7)
        self.assertEqual(len(np.unique(plan.row_index)), 7)

    def test_admission_uses_natural_early_prefall_window_only(self):
        identities = np.asarray([f"id-{index}" for index in range(100)], dtype="S64")
        risk = np.linspace(0.0, 1.0, 100)
        uncertainty = np.linspace(0.0, 0.5, 100)
        label = np.ones(100, dtype=bool)
        label[:10] = False
        steps = np.arange(100)
        plan = build_early_prefall_plan(
            identities=identities, state_risk=risk,
            state_uncertainty=uncertainty, natural_fall_label=label,
            natural_steps_to_outcome=steps, groups=20)
        self.assertTrue(np.all(plan.natural_steps_to_fall >= 48))
        self.assertTrue(np.all(plan.natural_steps_to_fall <= 96))
        self.assertTrue(np.all(label[plan.row_index]))


if __name__ == "__main__":
    unittest.main()
