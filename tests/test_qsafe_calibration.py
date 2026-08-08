from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import yaml

from rl.qsafe.calibration import (
    SelectorCalibrationInputs,
    SelectorCalibrationSpec,
    calibrate_selector,
)


def _inputs() -> SelectorCalibrationInputs:
    groups, candidates = 20, 16
    member_risk = np.full((2, groups, candidates), 0.90, dtype=np.float64)
    member_risk[:, :, 0] = 0.10
    member_risk[:, :6, 0] = np.asarray([[0.80], [0.82]])
    member_risk[:, :6, 1] = np.asarray([[0.10], [0.12]])
    empirical = np.full((groups, candidates), 0.9, dtype=np.float64)
    empirical[:, 0] = 0.1
    empirical[:6, 0] = 0.8
    empirical[:6, 1] = 0.1
    requested = np.zeros((groups, candidates, 12), dtype=np.float64)
    requested[:, 1, 0] = 0.10
    executed = requested.copy()
    q_target = requested * 0.5
    return SelectorCalibrationInputs(
        member_risk=member_risk,
        empirical_risk=empirical,
        requested=requested,
        executed=executed,
        q_target=q_target,
        reward_q=np.zeros((groups, candidates), dtype=np.float64),
        candidate_mask=np.ones((groups, candidates), dtype=bool),
        acceptance_probability=np.ones(groups, dtype=np.float64),
        trajectory_id=np.asarray([f"trajectory-{index}" for index in range(groups)]),
        source_seed=np.resize(np.asarray([1, 2, 3], dtype=np.int64), groups),
    )


def _spec() -> SelectorCalibrationSpec:
    return SelectorCalibrationSpec(
        min_independent_groups=20,
        min_trajectory_clusters=20,
        min_source_seeds=3,
        require_calibration_absolute_reduction=0.10,
        require_calibration_reduction_ci_low=0.0,
        max_intervention_rate=0.35,
        uncertainty_beta=1.0,
        max_epistemic_std=0.20,
        max_action_delta_rms=0.50,
        max_q_target_delta_rms=0.25,
        nominal_risk_lcb_trigger=(0.5,),
        min_benefit_lcb=(0.1,),
        max_risk_ucb=(0.5,),
        reward_q_margin=(0.0, 0.5),
    )


class SelectorCalibrationTest(unittest.TestCase):
    def test_group_macro_grid_selects_locked_lowest_reward_margin(self):
        result = calibrate_selector(
            _inputs(), _spec(), bootstrap_replicates=2000, bootstrap_seed=19)

        self.assertTrue(result.feasible)
        self.assertEqual(result.grid_configurations, 2)
        self.assertEqual(result.source_seeds, (1, 2, 3))
        self.assertIsNotNone(result.selector_config)
        assert result.selector_config is not None
        self.assertEqual(result.selector_config.reward_q_margin, 0.0)
        selected = result.rows[result.selected_row_index]
        self.assertAlmostEqual(selected.absolute_fall_reduction, 0.21)
        self.assertAlmostEqual(selected.intervention_rate, 0.30)
        self.assertGreater(selected.reduction_ci95_low, 0.0)
        self.assertEqual(selected.selection_reason_counts["selected"], 6)
        self.assertEqual(
            selected.selection_reason_counts["state_below_trigger"], 14)

    def test_no_feasible_grid_returns_explicit_abstain_result(self):
        result = calibrate_selector(
            _inputs(),
            replace(_spec(), max_intervention_rate=0.10),
            bootstrap_replicates=100,
            bootstrap_seed=2,
        )
        self.assertFalse(result.feasible)
        self.assertIsNone(result.selector_config)
        self.assertIsNone(result.selected_row_index)
        self.assertFalse(any(row.feasible for row in result.rows))

    def test_protocol_grid_and_data_minimums_fail_closed(self):
        protocol = yaml.safe_load(Path(
            "config/qsafe_evidence_protocol.yaml").read_text(encoding="utf-8"))
        spec = SelectorCalibrationSpec.from_protocol(
            protocol["phase1"]["selector_calibration"])
        self.assertEqual(len(spec.configurations()), 700)
        with self.assertRaisesRegex(ValueError, "too few independent groups"):
            calibrate_selector(
                _inputs(), spec, bootstrap_replicates=10)

        invalid = _inputs()
        invalid.member_risk[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            calibrate_selector(
                invalid, _spec(), bootstrap_replicates=10)


if __name__ == "__main__":
    unittest.main()
