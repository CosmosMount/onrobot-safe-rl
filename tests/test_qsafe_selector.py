from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from rl.qsafe.selector import (
    CandidateBatch,
    SelectorConfig,
    select_candidate,
)


def _fixture() -> tuple[np.ndarray, CandidateBatch, SelectorConfig]:
    risk = np.asarray([
        [0.60, 0.30, 0.34, 0.18],
        [0.62, 0.31, 0.35, 0.19],
        [0.58, 0.29, 0.33, 0.17],
        [0.61, 0.30, 0.34, 0.18],
        [0.59, 0.30, 0.34, 0.18],
    ], dtype=np.float64)
    requested = np.asarray([
        [0.00, 0.00, 0.00],
        [0.10, 0.10, 0.10],
        [0.12, 0.12, 0.12],
        [0.15, 0.15, 0.15],
    ], dtype=np.float64)
    executed = requested.copy()
    q_target = np.asarray([
        [0.00, 0.70, -1.40],
        [0.04, 0.74, -1.36],
        [0.05, 0.75, -1.35],
        [0.06, 0.76, -1.34],
    ], dtype=np.float64)
    candidates = CandidateBatch(
        requested=requested,
        executed=executed,
        q_target=q_target,
        reward_q=np.asarray([10.0, 9.8, 9.7, 9.8]),
        mask=np.asarray([True, True, True, False]),
    )
    config = SelectorConfig(
        nominal_risk_lcb_trigger=0.55,
        min_benefit_lcb=0.05,
        max_risk_ucb=0.45,
        max_epistemic_std=0.08,
        max_action_delta_rms=0.20,
        max_q_target_delta_rms=0.10,
        reward_q_margin=0.50,
        uncertainty_beta=1.0,
    )
    return risk, candidates, config


class QSafeSelectorTest(unittest.TestCase):
    def test_selects_only_fully_eligible_candidate(self):
        risk, candidates, config = _fixture()
        result = select_candidate(risk, candidates, config)
        self.assertTrue(result.intervened)
        self.assertEqual(result.reason, "selected")
        self.assertEqual(result.selected_index, 1)
        np.testing.assert_array_equal(
            result.eligible, [False, True, True, False])
        self.assertGreater(result.nominal_risk_lcb, 0.55)
        self.assertGreater(result.benefit_lcb[1], 0.05)

    def test_state_lcb_must_trigger_before_any_intervention(self):
        risk, candidates, config = _fixture()
        risk[:, 0] = [0.49, 0.51, 0.50, 0.50, 0.50]
        risk[:, 1] = 0.10
        result = select_candidate(risk, candidates, config)
        self.assertFalse(result.intervened)
        self.assertEqual(result.reason, "state_below_trigger")
        self.assertEqual(result.selected_index, 0)

    def test_every_candidate_gate_is_fail_closed(self):
        risk, candidates, config = _fixture()
        cases = {}

        mask = candidates.mask.copy()
        mask[1:] = False
        cases["support"] = (risk, replace(candidates, mask=mask), config)

        weak_benefit = risk.copy()
        weak_benefit[:, 1:] = risk[:, :1] - 0.01
        cases["benefit"] = (weak_benefit, candidates, config)

        high_risk = risk.copy()
        high_risk[:, 1:] = 0.50
        cases["risk_ucb"] = (high_risk, candidates, config)

        uncertain = risk.copy()
        uncertain[:, 1:] = np.asarray(
            [0.10, 0.42, 0.10, 0.42, 0.10])[:, None]
        cases["epistemic"] = (uncertain, candidates, config)

        far_requested = candidates.requested.copy()
        far_requested[1:] = 0.40
        cases["requested_delta"] = (
            risk, replace(candidates, requested=far_requested), config)

        far_executed = candidates.executed.copy()
        far_executed[1:] = 0.40
        cases["executed_delta"] = (
            risk, replace(candidates, executed=far_executed), config)

        far_target = candidates.q_target.copy()
        far_target[1:] += 0.30
        cases["q_target_delta"] = (
            risk, replace(candidates, q_target=far_target), config)

        low_reward = candidates.reward_q.copy()
        low_reward[1:] = 9.0
        cases["reward_q"] = (
            risk, replace(candidates, reward_q=low_reward), config)

        for name, (case_risk, case_candidates, case_config) in cases.items():
            with self.subTest(gate=name):
                result = select_candidate(
                    case_risk, case_candidates, case_config)
                self.assertFalse(result.intervened)
                self.assertEqual(result.reason, "no_eligible")
                self.assertEqual(result.selected_index, 0)

    def test_no_eligible_never_falls_back_to_minimum_risk(self):
        risk, candidates, config = _fixture()
        # Candidate three has the lowest risk but is outside measured support.
        candidates = replace(
            candidates,
            reward_q=np.asarray([10.0, 9.0, 9.0, 10.0]),
        )
        result = select_candidate(risk, candidates, config)
        self.assertEqual(int(np.argmin(result.risk_mean)), 3)
        self.assertEqual(result.reason, "no_eligible")
        self.assertEqual(result.selected_index, 0)

    def test_any_nonfinite_numeric_input_abstains(self):
        risk, candidates, config = _fixture()
        mutations = []
        bad_risk = risk.copy()
        bad_risk[0, 3] = np.nan  # Even a masked candidate invalidates the batch.
        mutations.append((bad_risk, candidates))
        for field in ("requested", "executed", "q_target", "reward_q"):
            value = getattr(candidates, field).copy()
            value.reshape(-1)[-1] = np.inf
            mutations.append((risk, replace(candidates, **{field: value})))
        mutations.append((
            risk,
            replace(candidates, mask=np.asarray([1.0, 1.0, 1.0, np.nan])),
        ))

        for index, (case_risk, case_candidates) in enumerate(mutations):
            with self.subTest(input=index):
                result = select_candidate(
                    case_risk, case_candidates, config)
                self.assertEqual(result.reason, "nonfinite_input")
                self.assertEqual(result.selected_index, 0)
                self.assertFalse(result.intervened)
                self.assertFalse(np.any(result.eligible))

    def test_ties_resolve_to_smallest_candidate_index(self):
        risk, candidates, config = _fixture()
        risk[:, 2] = risk[:, 1]
        requested = candidates.requested.copy()
        executed = candidates.executed.copy()
        q_target = candidates.q_target.copy()
        requested[2] = requested[1]
        executed[2] = executed[1]
        q_target[2] = q_target[1]
        reward_q = candidates.reward_q.copy()
        reward_q[2] = reward_q[1]
        candidates = replace(
            candidates,
            requested=requested,
            executed=executed,
            q_target=q_target,
            reward_q=reward_q,
        )
        selections = [
            select_candidate(risk, candidates, config).selected_index
            for _ in range(20)
        ]
        self.assertEqual(selections, [1] * 20)

    def test_non_tied_selection_is_invariant_to_candidate_order(self):
        risk, candidates, config = _fixture()
        original = select_candidate(risk, candidates, config)
        permutation = np.asarray([0, 2, 1, 3])
        permuted = CandidateBatch(
            requested=candidates.requested[permutation],
            executed=candidates.executed[permutation],
            q_target=candidates.q_target[permutation],
            reward_q=candidates.reward_q[permutation],
            mask=candidates.mask[permutation],
        )
        reordered = select_candidate(
            risk[:, permutation], permuted, config)
        self.assertEqual(original.selected_index, 1)
        self.assertEqual(int(permutation[reordered.selected_index]), 1)

    def test_benefit_uncertainty_is_paired_member_by_member(self):
        risk, candidates, config = _fixture()
        nominal = np.asarray([0.80, 0.60, 0.75, 0.55, 0.70])
        risk[:, 0] = nominal
        risk[:, 1] = nominal - 0.25
        # Each absolute prediction varies substantially across members, while
        # their same-member counterfactual difference is exactly identified.
        config = replace(
            config,
            nominal_risk_lcb_trigger=0.40,
            min_benefit_lcb=0.20,
            max_risk_ucb=0.70,
            max_epistemic_std=0.20,
            uncertainty_beta=2.0,
        )
        result = select_candidate(risk, candidates, config)
        self.assertAlmostEqual(result.benefit_mean[1], 0.25)
        self.assertAlmostEqual(result.benefit_std[1], 0.0)
        self.assertAlmostEqual(result.benefit_lcb[1], 0.25)
        self.assertTrue(result.benefit_gate[1])

    def test_masked_nominal_abstains_before_selection(self):
        risk, candidates, config = _fixture()
        mask = candidates.mask.copy()
        mask[0] = False
        result = select_candidate(
            risk, replace(candidates, mask=mask), config)
        self.assertEqual(result.reason, "nominal_masked")
        self.assertEqual(result.selected_index, 0)
        self.assertFalse(result.intervened)
        self.assertFalse(np.any(result.eligible))

    def test_invalid_probability_and_single_member_abstain(self):
        risk, candidates, config = _fixture()
        invalid = risk.copy()
        invalid[0, 1] = 1.1
        self.assertEqual(
            select_candidate(invalid, candidates, config).reason,
            "invalid_risk",
        )
        self.assertEqual(
            select_candidate(risk[:1], candidates, config).reason,
            "insufficient_members",
        )

    def test_shape_mismatch_is_a_programming_error(self):
        risk, candidates, config = _fixture()
        with self.assertRaisesRegex(ValueError, "candidate dimension"):
            select_candidate(
                risk, replace(candidates, requested=candidates.requested[:2]),
                config)


if __name__ == "__main__":
    unittest.main()
