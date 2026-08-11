from __future__ import annotations

import unittest

import numpy as np

from rl.qsafe.recovery_model_test import (
    evaluate_stage_b_model_test,
    hierarchical_model_test_bootstrap,
    stable_equal_mass_ece,
)
from rl.qsafe.recovery_placebo import (
    MatchedRandomPlaceboBundle,
    PlaceboFitMetrics,
)
from rl.qsafe.recovery_selector import (
    RecoveryConformalOffsets,
    RecoverySelectorBundle,
    RecoverySelectorConfig,
)


def _selector_bundle() -> RecoverySelectorBundle:
    uncertainty = "b" * 64
    return RecoverySelectorBundle.create(
        offsets=RecoveryConformalOffsets(
            nominal_lower=0.0,
            risk_upper=np.zeros(9),
            benefit_lower=np.zeros(9),
            calibration_report_sha256=uncertainty,
        ),
        selector_config=RecoverySelectorConfig(
            nominal_risk_lcb_trigger=0.10,
            min_benefit_lcb=0.00,
            max_risk_ucb=0.70,
            max_epistemic_std=0.20,
            max_action_delta_rms=0.50,
            max_q_target_delta_rms=0.25,
        ),
        probability_calibration_report_sha256="a" * 64,
        uncertainty_calibration_report_sha256=uncertainty,
        selector_search_report_sha256="c" * 64,
    )


def _placebo_bundle(selector: RecoverySelectorBundle) -> MatchedRandomPlaceboBundle:
    return MatchedRandomPlaceboBundle(
        selector_bundle_sha256=selector.bundle_sha256,
        execution_lock_sha256="d" * 64,
        fit_rng_assignment_count=12,
        fit_rng_assignment_sha256="e" * 64,
        selector_config=selector.selector_config,
        nominal_risk_bin_edges=np.linspace(0.0, 1.0, 11),
        first_action_distance_edges=np.linspace(0.0, 0.4, 5),
        intervention_probability=np.zeros(10),
        conditional_cell_probability=np.zeros((10, 3, 4)),
        fit_metrics=PlaceboFitMetrics(
            target_intervention_rate=0.0,
            realized_intervention_rate=0.0,
            absolute_intervention_rate_error=0.0,
            duration_histogram_total_variation=0.0,
            first_action_distance_ecdf_distance=0.0,
            eligible=True,
        ),
    )


def _passing_inputs() -> dict[str, object]:
    # Two actors x two source/age strata x three complete trajectories.  Exactly
    # one state in every source is above the selector trigger, so intervention
    # is 1/3 while every actor/source/age effect remains strictly positive.
    actors = np.repeat([53, 53, 54, 54], 3)
    sources = np.repeat([8701, 8711, 8702, 8712], 3)
    checkpoints = np.repeat([25_000, 50_000, 25_000, 50_000], 3)
    groups = len(actors)
    empirical = np.zeros((groups, 9), dtype=np.float64)
    low = np.asarray([0.05, 0.00, 0.10, 0.15, 0.20,
                      0.25, 0.30, 0.35, 0.40])
    high = np.asarray([0.80, 0.10, 0.20, 0.30, 0.40,
                       0.50, 0.60, 0.70, 0.75])
    for group in range(groups):
        empirical[group] = high if group % 3 == 0 else low
    replicas = 20
    fall = np.zeros((groups, 9, replicas), dtype=bool)
    for group in range(groups):
        for candidate in range(9):
            fall[group, candidate, :int(round(
                empirical[group, candidate] * replicas))] = True
    member = np.repeat(empirical[:, None, :], 5, axis=1)
    actions = np.zeros((groups, 9, 12), dtype=np.float64)
    actions[:, 1:] = 0.05
    selector = _selector_bundle()
    return {
        "member_risk": member,
        "fall": fall,
        "candidate_requested": actions,
        "candidate_executed": actions,
        "candidate_q_target": actions,
        "candidate_mask": np.ones((groups, 9), dtype=bool),
        "candidate_behavior_steps": np.asarray(
            [0, 10, 25, 50, 10, 10, 25, 25, 25]),
        "actor_training_seed": actors,
        "source_seed": sources,
        "checkpoint_step": checkpoints,
        "trajectory_fingerprint_sha256": np.asarray([
            f"{index + 101:064x}" for index in range(groups)]),
        "group_id": np.asarray([
            f"group-{index:02d}" for index in range(groups)]),
        "group_fingerprint_sha256": np.asarray([
            f"{index + 1:064x}" for index in range(groups)]),
        "selector_bundle": selector,
        "placebo_bundle": _placebo_bundle(selector),
    }


class RecoveryModelTestTest(unittest.TestCase):
    def test_hierarchical_bootstrap_is_reproducible_and_retains_sources(self):
        actors = np.repeat([1, 1, 2, 2], 2)
        sources = np.repeat([11, 12, 21, 22], 2)
        trajectory = np.asarray([f"t-{index}" for index in range(8)])
        values = np.arange(8, dtype=np.float64)
        pair = np.ones(8, dtype=np.float64)
        first = hierarchical_model_test_bootstrap(
            values,
            pair,
            actor_training_seed=actors,
            source_seed=sources,
            trajectory_id=trajectory,
            replicates=17,
            seed=20260812,
        )
        second = hierarchical_model_test_bootstrap(
            values,
            pair,
            actor_training_seed=actors,
            source_seed=sources,
            trajectory_id=trajectory,
            replicates=17,
            seed=20260812,
        )
        self.assertEqual(first.seed, 20260812)
        self.assertAlmostEqual(first.point[0], 3.5)
        self.assertAlmostEqual(first.pair_point, 1.0)
        np.testing.assert_array_equal(first.replicates, second.replicates)
        np.testing.assert_array_equal(
            first.pair_replicates, second.pair_replicates)

    def test_stable_equal_mass_ece_is_tie_order_invariant(self):
        prediction = np.asarray([0.1, 0.1, 0.8, 0.8])
        target = np.asarray([0.0, 0.2, 1.0, 0.6])
        weight = np.asarray([0.1, 0.4, 0.2, 0.3])
        first = stable_equal_mass_ece(
            prediction, target, weight, bins=2)
        permutation = np.asarray([1, 0, 3, 2])
        second = stable_equal_mass_ece(
            prediction[permutation], target[permutation],
            weight[permutation], bins=2)
        self.assertAlmostEqual(first["ece"], second["ece"])
        np.testing.assert_allclose(first["mass_fraction"], [0.5, 0.5])

    def test_passing_model_test_uses_frozen_thresholds_and_quantiles(self):
        report = evaluate_stage_b_model_test(
            **_passing_inputs(),
            bootstrap_replicates=200,
            bootstrap_seed=20260812,
            production_contract=False,
        )
        self.assertTrue(report["pass"])
        self.assertTrue(all(report["gates"].values()))
        self.assertEqual(report["bootstrap"]["rng_bit_generator"], "numpy_PCG64")
        self.assertEqual(report["bootstrap"]["quantile_method"], "linear")
        self.assertGreaterEqual(
            report["metrics"]["pair_accuracy_q025_lcb"], 0.55)
        self.assertGreaterEqual(
            report["metrics"]["top1_reduction_q05_lcb"], 0.03)
        self.assertLessEqual(
            report["metrics"]["frozen_selector_intervention_rate"], 0.35)
        for metric in (
            "top1_absolute_fall_reduction",
            "frozen_selector_absolute_fall_reduction",
        ):
            for values in report["directional_subgroups"][metric].values():
                self.assertTrue(all(value > 0.0 for value in values.values()))
        self.assertFalse(report["model_or_threshold_updates_from_model_test"])

    def test_pair_target_ties_are_excluded_and_malformed_clusters_fail(self):
        inputs = _passing_inputs()
        fall = np.asarray(inputs["fall"]).copy()
        # Tie two candidates everywhere; the pair count must decrease without
        # counting the tied comparison as correct, wrong, or one half.
        fall[:, 2] = fall[:, 3]
        inputs["fall"] = fall
        tied = evaluate_stage_b_model_test(
            **inputs, bootstrap_replicates=20, bootstrap_seed=4,
            production_contract=False)
        self.assertLess(tied["metrics"]["pair_comparisons"], 12 * 36)

        malformed = _passing_inputs()
        trajectory = np.asarray(
            malformed["trajectory_fingerprint_sha256"]).copy()
        trajectory[1] = trajectory[0]
        malformed["trajectory_fingerprint_sha256"] = trajectory
        with self.assertRaisesRegex(ValueError, "one state per complete trajectory"):
            evaluate_stage_b_model_test(
                **malformed, bootstrap_replicates=10, bootstrap_seed=5,
                production_contract=False)

    def test_production_contract_rejects_bootstrap_overrides(self):
        with self.assertRaisesRegex(ValueError, "bootstrap contract drifted"):
            evaluate_stage_b_model_test(
                **_passing_inputs(),
                bootstrap_replicates=10,
                bootstrap_seed=20260812,
                production_contract=True,
            )
        with self.assertRaisesRegex(ValueError, "bootstrap contract drifted"):
            evaluate_stage_b_model_test(
                **_passing_inputs(),
                bootstrap_replicates=50_000,
                bootstrap_seed=7,
                production_contract=True,
            )

    def test_full_evaluation_is_invariant_to_input_row_order(self):
        inputs = _passing_inputs()
        first = evaluate_stage_b_model_test(
            **inputs,
            bootstrap_replicates=40,
            bootstrap_seed=20260812,
            production_contract=False,
        )
        permuted: dict[str, object] = {}
        for name, value in inputs.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and (
                    value.shape[0] == 12) and name != "candidate_behavior_steps":
                permuted[name] = value[::-1].copy()
            else:
                permuted[name] = value
        second = evaluate_stage_b_model_test(
            **permuted,
            bootstrap_replicates=40,
            bootstrap_seed=20260812,
            production_contract=False,
        )
        self.assertEqual(first["statistics_sha256"], second["statistics_sha256"])


if __name__ == "__main__":
    unittest.main()
