from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rl.qsafe.network import QSafeEnsemble, QSafeNetworkConfig, SelectiveAdvantageQSafe
from rl.qsafe.recovery_calibration import (
    finite_sample_upper_order_statistic,
    fit_signed_recovery_conformal,
    hierarchical_bootstrap_means,
    predict_recovery_member_risk,
    recovery_selector_grid,
    search_recovery_selector_grid,
    simultaneous_one_sided_lower_band,
)
from rl.qsafe.recovery_program import RECOVERY_PROGRAM_VIEW
from rl.qsafe.recovery_selector import (
    RecoveryConformalOffsets,
    select_recovery_program,
)


class _Normalization:
    def equivalent_to(self, other):
        return self is other


class _PredictionView:
    def __init__(self, normalization):
        self.normalization = normalization
        self.action_view = RECOVERY_PROGRAM_VIEW
        self.action_dim = 82
        self.privileged_dim = 0
        self.command_vx = 0.5
        self.group_count = 3
        self.mask = np.ones((3, 9), dtype=bool)
        self.recovery_program_binding = {"binding": "same"}
        self.recovery_program_feature_manifest = {"feature": "same"}
        self.recovery_program_feature_contract_sha256 = "a" * 64
        self.recovery_library_fingerprint_sha256 = "b" * 64
        generator = torch.Generator().manual_seed(17)
        self.observation = torch.randn(3, 5, 46, generator=generator)
        self.nominal = torch.randn(3, 82, generator=generator)
        self.candidate = torch.randn(3, 9, 82, generator=generator)

    def all_indices(self):
        return np.arange(self.group_count, dtype=np.int64)

    def batch(self, indices, device):
        selected = np.asarray(indices, dtype=np.int64)
        return SimpleNamespace(
            observation_history=self.observation[selected].to(device),
            nominal_action=self.nominal[selected].to(device),
            candidate_action=self.candidate[selected].to(device),
            privileged_state=None,
        )


class RecoveryCalibrationTest(unittest.TestCase):
    def test_finite_sample_ranks_and_signed_offsets_are_exact(self):
        scores = np.arange(384, dtype=np.float64)
        option = finite_sample_upper_order_statistic(scores, alpha=0.00625)
        nominal = finite_sample_upper_order_statistic(scores, alpha=0.05)
        self.assertEqual(option.one_based_rank, 383)
        self.assertEqual(option.value, 382.0)
        self.assertEqual(nominal.one_based_rank, 366)
        self.assertEqual(nominal.value, 365.0)

        member = np.full((384, 5, 9), 0.40, dtype=np.float64)
        member[:, :, 0] = 0.60
        empirical = np.full((384, 9), 0.35, dtype=np.float64)
        empirical[:, 0] = 0.65
        result = fit_signed_recovery_conformal(
            member,
            empirical,
            candidate_mask=np.ones((384, 9), dtype=bool),
            execution_lock={"stage": "B", "attempt": 1},
            expected_group_count=384,
        )
        self.assertEqual(result.option_rank, 383)
        self.assertEqual(result.nominal_rank, 366)
        self.assertLess(result.nominal_lower, 0.0)
        self.assertTrue(np.all(result.risk_upper[1:] < 0.0))
        self.assertTrue(np.all(result.benefit_lower[1:] < 0.0))
        self.assertTrue(result.report_sha256)
        self.assertEqual(
            result.offsets.calibration_report_sha256,
            result.report_sha256,
        )

        tied = finite_sample_upper_order_statistic(
            np.asarray([-0.2] * 383 + [0.7]), alpha=0.00625)
        self.assertEqual(tied.one_based_rank, 383)
        self.assertEqual(tied.value, -0.2)

    def test_member_helper_retains_calibrated_five_member_axis(self):
        config = QSafeNetworkConfig(
            observation_dim=46,
            history_frames=5,
            action_dim=82,
            frame_hidden_dim=8,
            state_hidden_dim=8,
            action_hidden_dim=8,
            privileged_dim=0,
            action_mode="selective_advantage",
        )
        torch.manual_seed(5)
        ensemble = QSafeEnsemble(
            [SelectiveAdvantageQSafe(config) for _ in range(5)],
            temperatures=[0.8, 0.9, 1.0, 1.1, 1.2],
        )
        normalization = _Normalization()
        view = _PredictionView(normalization)
        trained = SimpleNamespace(
            ensemble=ensemble,
            normalization=normalization,
            command_vx=0.5,
            privileged_dim=0,
            action_view=RECOVERY_PROGRAM_VIEW,
            action_dim=82,
            recovery_program_binding={"binding": "same"},
            recovery_program_feature_manifest={"feature": "same"},
            recovery_program_feature_contract_sha256="a" * 64,
            recovery_library_fingerprint_sha256="b" * 64,
        )
        result = predict_recovery_member_risk(
            trained, view, device="cpu", batch_size=2)
        expected = ensemble.predict(
            view.observation,
            view.nominal,
            view.candidate,
        ).member_risk.detach().numpy().transpose(1, 0, 2)
        self.assertEqual(result.shape, (3, 5, 9))
        np.testing.assert_allclose(result, expected, rtol=2e-7, atol=1e-7)
        self.assertFalse(result.flags.writeable)

    def test_exact_grid_and_comparison_boundaries(self):
        grid = recovery_selector_grid()
        self.assertEqual(len(grid), 100)
        self.assertEqual(grid[0].nominal_risk_lcb_trigger, 0.10)
        self.assertEqual(grid[0].min_benefit_lcb, 0.00)
        self.assertEqual(grid[0].max_risk_ucb, 0.25)
        self.assertEqual(grid[-1].nominal_risk_lcb_trigger, 0.50)
        self.assertEqual(grid[-1].min_benefit_lcb, 0.12)
        self.assertEqual(grid[-1].max_risk_ucb, 0.70)

        risk = np.full((5, 9), 0.90, dtype=np.float64)
        risk[:, 0] = 0.30
        risk[:, 1] = 0.25
        actions = np.zeros((9, 12), dtype=np.float64)
        actions[1] = 0.10
        offsets = RecoveryConformalOffsets(
            nominal_lower=0.0,
            risk_upper=np.zeros(9),
            benefit_lower=np.zeros(9),
            calibration_report_sha256="c" * 64,
        )
        strict_config = next(
            config for config in grid
            if config.nominal_risk_lcb_trigger == 0.30
            and config.min_benefit_lcb == 0.05
            and config.max_risk_ucb == 0.25
        )
        decision = select_recovery_program(
            risk,
            candidate_requested=actions,
            candidate_executed=actions,
            candidate_q_target=actions,
            candidate_mask=np.ones(9, dtype=bool),
            offsets=offsets,
            config=strict_config,
        )
        self.assertEqual(decision.nominal_risk_lcb, 0.30)
        self.assertFalse(decision.intervened)
        self.assertEqual(decision.risk_ucb[1], 0.25)

        permissive = next(
            config for config in grid
            if config.nominal_risk_lcb_trigger == 0.30
            and config.min_benefit_lcb == 0.02
            and config.max_risk_ucb == 0.25
        )
        decision = select_recovery_program(
            risk,
            candidate_requested=actions,
            candidate_executed=actions,
            candidate_q_target=actions,
            candidate_mask=np.ones(9, dtype=bool),
            offsets=offsets,
            config=permissive,
        )
        self.assertEqual(decision.selected_index, 1)

    def test_actor_outer_inner_bootstrap_and_max_stat_are_reproducible(self):
        values = np.asarray([
            [1.0, 0.0],
            [3.0, 2.0],
            [5.0, 4.0],
            [7.0, 6.0],
        ])
        actor = np.asarray([51, 51, 52, 52])
        trajectory = np.asarray(["a", "b", "c", "d"])
        first = hierarchical_bootstrap_means(
            values,
            actor_training_seed=actor,
            source_seed=np.asarray([8501, 8501, 8502, 8502]),
            inner_cluster_id=trajectory,
            replicates=500,
            seed=20260811,
            inner_unit="trajectory",
        )
        second = hierarchical_bootstrap_means(
            values,
            actor_training_seed=actor,
            source_seed=np.asarray([8501, 8501, 8502, 8502]),
            inner_cluster_id=trajectory,
            replicates=500,
            seed=20260811,
            inner_unit="trajectory",
        )
        np.testing.assert_array_equal(first.replicates, second.replicates)
        np.testing.assert_array_equal(first.point_estimate, [4.0, 3.0])
        self.assertEqual(first.actor_count, 2)
        self.assertEqual(first.source_counts, (1, 1))
        self.assertEqual(first.inner_cluster_counts, ((2,), (2,)))

        band = simultaneous_one_sided_lower_band(
            values,
            actor_training_seed=actor,
            source_seed=np.asarray([8501, 8501, 8502, 8502]),
            inner_cluster_id=trajectory,
            replicates=500,
            seed=20260811,
            inner_unit="trajectory",
        )
        expected_critical = np.quantile(
            np.max(first.point_estimate[None] - first.replicates, axis=1),
            0.95,
            method="linear",
        )
        self.assertEqual(band.common_critical_value, expected_critical)
        np.testing.assert_array_equal(
            band.lower_bound,
            band.point_estimate - expected_critical,
        )

        # Source strata, rather than raw group count, receive equal mass.  The
        # first source has one value 0 and the second has three values 6.
        weighted = hierarchical_bootstrap_means(
            np.asarray([0.0, 6.0, 6.0, 6.0]),
            actor_training_seed=np.asarray([51, 51, 51, 51]),
            source_seed=np.asarray([8661, 8671, 8671, 8671]),
            inner_cluster_id=np.asarray(["a", "b", "c", "d"]),
            replicates=10,
            seed=20260811,
            inner_unit="trajectory",
        )
        self.assertEqual(weighted.point_estimate[0], 3.0)
        self.assertEqual(weighted.source_counts, (2,))
        self.assertEqual(weighted.inner_cluster_counts, ((1, 3),))

    def test_selector_search_evaluates_complete_grid_before_choice(self):
        groups = 12
        member = np.full((groups, 5, 9), 0.95, dtype=np.float64)
        empirical = np.full((groups, 9), 0.95, dtype=np.float64)
        high = np.asarray([0, 1, 6, 7])
        member[:, :, 0] = 0.05
        member[:, :, 1] = 0.05
        empirical[:, 0] = 0.05
        empirical[:, 1] = 0.05
        member[high, :, 0] = 0.80
        member[high, :, 1] = 0.10
        empirical[high, 0] = 0.80
        empirical[high, 1] = 0.10
        requested = np.zeros((groups, 9, 12), dtype=np.float64)
        requested[:, 1] = 0.10
        offsets = RecoveryConformalOffsets(
            nominal_lower=0.0,
            risk_upper=np.zeros(9),
            benefit_lower=np.zeros(9),
            calibration_report_sha256="d" * 64,
        )
        result = search_recovery_selector_grid(
            member,
            empirical,
            candidate_requested=requested,
            candidate_executed=requested,
            candidate_q_target=requested,
            candidate_mask=np.ones((groups, 9), dtype=bool),
            offsets=offsets,
            actor_training_seed=np.asarray([51] * 6 + [52] * 6),
            source_seed=np.asarray([
                8661, 8661, 8671, 8671, 8681, 8681,
                8662, 8662, 8672, 8672, 8682, 8682,
            ]),
            inner_cluster_id=np.asarray([f"trajectory-{i}" for i in range(groups)]),
            execution_lock={"stage": "B", "role": "selector_calibration"},
            bootstrap_replicates=1_000,
            bootstrap_seed=20260811,
            bootstrap_inner_unit="trajectory",
            expected_group_count=groups,
        )
        self.assertEqual(len(result.rows), 100)
        self.assertEqual(result.selected_grid_index, 0)
        self.assertEqual(result.selected_config, recovery_selector_grid()[0])
        self.assertAlmostEqual(result.rows[0].intervention_rate, 1.0 / 3.0)
        self.assertGreater(result.rows[0].simultaneous_lcb, 0.0)
        self.assertEqual(result.to_report()["report_sha256"], result.report_sha256)


if __name__ == "__main__":
    unittest.main()
