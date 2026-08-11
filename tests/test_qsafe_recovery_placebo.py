from __future__ import annotations

import copy
import inspect
import unittest

import numpy as np

from rl.qsafe.recovery_placebo import (
    MATCHED_RANDOM_PLACEBO_DURATIONS,
    MatchedRandomPlaceboBundle,
    PlaceboFitMetrics,
    derive_matched_random_placebo_seed,
    fit_matched_random_placebo,
    select_matched_random_placebo,
)
from rl.qsafe.recovery_selector import RecoverySelectorConfig


def _selector_config():
    return RecoverySelectorConfig(
        nominal_risk_lcb_trigger=0.10,
        min_benefit_lcb=0.00,
        max_risk_ucb=0.70,
        max_epistemic_std=0.20,
        max_action_delta_rms=0.50,
        max_q_target_delta_rms=0.25,
    )


def _fitted_bundle(execution_lock="c" * 64):
    groups = 20
    risk = np.linspace(0.10, 0.90, groups)
    selected = np.zeros(groups, dtype=np.int64)
    selected[::5] = np.asarray([1, 2, 3, 4])
    durations = np.asarray([0, 10, 25, 50, 10, 10, 25, 25, 25])
    distance = np.tile(np.linspace(0.0, 0.8, 9), (groups, 1))
    return fit_matched_random_placebo(
        nominal_risk_lcb=risk,
        qsafe_selected_index=selected,
        candidate_support_mask=np.ones((groups, 9), dtype=bool),
        candidate_duration_steps=durations,
        first_action_distance=distance,
        placebo_source_seed=np.asarray([8661] * 10 + [8662] * 10),
        group_fingerprint_sha256=np.asarray([
            f"{index:064x}" for index in range(groups)
        ]),
        selector_config=_selector_config(),
        selector_bundle_sha256="b" * 64,
        execution_lock=execution_lock,
    )


class RecoveryPlaceboTest(unittest.TestCase):
    def test_counter_seed_derivation_has_frozen_uint256_vector(self):
        seed = derive_matched_random_placebo_seed(
            source_seed=8661,
            group_fingerprint_sha256="01" * 32,
            draw_index=7,
        )
        self.assertEqual(
            seed.to_bytes(32, "little").hex(),
            "62f2e806c3ae27908740714c43c6da1e"
            "2eaa6f6ccc3ca6bcc7888cfa36e91b06",
        )
        self.assertGreater(seed, np.iinfo(np.uint64).max)

    def test_fit_is_outcome_free_matches_decisions_and_hashes_canonically(self):
        parameters = inspect.signature(
            fit_matched_random_placebo).parameters
        for forbidden in ("fall", "outcome", "empirical_risk", "reward"):
            self.assertNotIn(forbidden, parameters)

        bundle = _fitted_bundle()
        self.assertTrue(bundle.fit_metrics.eligible)
        self.assertEqual(bundle.fit_metrics.absolute_intervention_rate_error, 0.0)
        self.assertEqual(
            bundle.fit_metrics.duration_histogram_total_variation, 0.0)
        self.assertEqual(
            bundle.fit_metrics.first_action_distance_ecdf_distance, 0.0)
        self.assertEqual(bundle.nominal_risk_bin_edges.shape, (11,))
        self.assertEqual(bundle.first_action_distance_edges.shape, (5,))
        self.assertEqual(bundle.conditional_cell_probability.shape, (10, 3, 4))

        serialized = bundle.to_dict()
        restored = MatchedRandomPlaceboBundle.from_dict(serialized)
        self.assertEqual(restored.to_dict(), serialized)
        self.assertEqual(restored.bundle_sha256, bundle.bundle_sha256)
        self.assertFalse(restored.intervention_probability.flags.writeable)

        changed_lock = _fitted_bundle(execution_lock="d" * 64)
        self.assertNotEqual(changed_lock.bundle_sha256, bundle.bundle_sha256)
        tampered = copy.deepcopy(serialized)
        tampered["table"]["intervention_probability"][0] = 0.123
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            MatchedRandomPlaceboBundle.from_dict(tampered)
        extra = copy.deepcopy(serialized)
        extra["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "fields are not exact"):
            MatchedRandomPlaceboBundle.from_dict(extra)

    def test_sampling_is_seed_reproducible_and_uniform_only_within_cell(self):
        bundle = _fitted_bundle()
        durations = np.asarray([0, 10, 25, 50, 10, 10, 25, 25, 25])
        distance = np.linspace(0.0, 0.8, 9)
        first = select_matched_random_placebo(
            bundle,
            nominal_risk_lcb=0.90,
            candidate_support_mask=np.ones(9, dtype=bool),
            candidate_duration_steps=durations,
            first_action_distance=distance,
            source_seed=8661,
            group_fingerprint_sha256="1" * 64,
            draw_index=0,
        )
        second = select_matched_random_placebo(
            bundle,
            nominal_risk_lcb=0.90,
            candidate_support_mask=np.ones(9, dtype=bool),
            candidate_duration_steps=durations,
            first_action_distance=distance,
            source_seed=8661,
            group_fingerprint_sha256="1" * 64,
            draw_index=0,
        )
        self.assertEqual(first, second)
        if first.intervened:
            self.assertIn(first.duration_steps, MATCHED_RANDOM_PLACEBO_DURATIONS)
            self.assertEqual(durations[first.selected_index], first.duration_steps)

    def test_below_trigger_and_empty_cell_abstain_without_fallback(self):
        probability = np.zeros(10, dtype=np.float64)
        probability[9] = 1.0
        cells = np.zeros((10, 3, 4), dtype=np.float64)
        cells[9, 2, 3] = 1.0
        bundle = MatchedRandomPlaceboBundle(
            selector_bundle_sha256="e" * 64,
            execution_lock_sha256="f" * 64,
            fit_rng_assignment_count=1,
            fit_rng_assignment_sha256="a" * 64,
            selector_config=_selector_config(),
            nominal_risk_bin_edges=np.linspace(0.0, 1.0, 11),
            first_action_distance_edges=np.asarray([0.0, 0.2, 0.4, 0.6, 0.8]),
            intervention_probability=probability,
            conditional_cell_probability=cells,
            fit_metrics=PlaceboFitMetrics(
                target_intervention_rate=0.0,
                realized_intervention_rate=0.0,
                absolute_intervention_rate_error=0.0,
                duration_histogram_total_variation=0.0,
                first_action_distance_ecdf_distance=0.0,
                eligible=True,
            ),
        )
        duration = np.asarray([0, 10, 25, 50, 10, 10, 25, 25, 25])
        distance = np.linspace(0.0, 0.8, 9)
        below = select_matched_random_placebo(
            bundle,
            nominal_risk_lcb=0.05,
            candidate_support_mask=np.ones(9, dtype=bool),
            candidate_duration_steps=duration,
            first_action_distance=distance,
            source_seed=8661,
            group_fingerprint_sha256="2" * 64,
        )
        self.assertEqual(below.selected_index, 0)
        self.assertEqual(below.reason, "state_below_trigger")

        # The table requests duration-50 / distance-quartile-3, while the only
        # supported nonnominal option is duration-10 / quartile-0.  A fallback
        # to that option would violate the placebo contract.
        support = np.zeros(9, dtype=bool)
        support[[0, 1]] = True
        empty = select_matched_random_placebo(
            bundle,
            nominal_risk_lcb=0.95,
            candidate_support_mask=support,
            candidate_duration_steps=duration,
            first_action_distance=distance,
            source_seed=8661,
            group_fingerprint_sha256="2" * 64,
        )
        self.assertEqual(empty.selected_index, 0)
        self.assertFalse(empty.intervened)
        self.assertEqual(empty.reason, "empty_cell")


if __name__ == "__main__":
    unittest.main()
