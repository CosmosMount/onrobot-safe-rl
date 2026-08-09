from __future__ import annotations

import copy
from dataclasses import replace
import unittest

import numpy as np

from rl.qsafe.recovery_selector import (
    RECOVERY_SELECTOR_ENSEMBLE_STD_DDOF,
    RecoveryConformalOffsets,
    RecoverySelectorBundle,
    RecoverySelectorConfig,
    select_recovery_program,
)


_PROBABILITY_REPORT = "b" * 64
_UNCERTAINTY_REPORT = "a" * 64
_SEARCH_REPORT = "c" * 64


def _bundle(offsets=None, config=None):
    if offsets is None:
        offsets = RecoveryConformalOffsets(
            nominal_lower=0.03,
            risk_upper=np.asarray([0.0] + [0.04] * 8),
            benefit_lower=np.asarray([0.0] + [0.05] * 8),
            calibration_report_sha256=_UNCERTAINTY_REPORT,
        )
    if config is None:
        config = RecoverySelectorConfig(
            nominal_risk_lcb_trigger=0.50,
            min_benefit_lcb=0.08,
            max_risk_ucb=0.55,
            max_epistemic_std=0.20,
            max_action_delta_rms=0.50,
            max_q_target_delta_rms=0.25,
        )
    return RecoverySelectorBundle.create(
        offsets=offsets,
        selector_config=config,
        probability_calibration_report_sha256=_PROBABILITY_REPORT,
        uncertainty_calibration_report_sha256=_UNCERTAINTY_REPORT,
        selector_search_report_sha256=_SEARCH_REPORT,
    )


def _case():
    member_risk = np.repeat(
        np.asarray([[0.65, 0.20, 0.25, 0.30, 0.50, 0.55, 0.60, 0.45, 0.40]]),
        5,
        axis=0,
    )
    member_risk += np.asarray([-0.02, -0.01, 0.0, 0.01, 0.02])[:, None]
    requested = np.zeros((9, 12), dtype=np.float64)
    requested[1:] = 0.10
    executed = requested.copy()
    target = requested.copy()
    return member_risk, requested, executed, target, _bundle()


def _select(risk, requested, executed, target, bundle, mask=None):
    return select_recovery_program(
        risk,
        candidate_requested=requested,
        candidate_executed=executed,
        candidate_q_target=target,
        candidate_mask=(np.ones(9, dtype=bool) if mask is None else mask),
        offsets=bundle.offsets,
        config=bundle.selector_config,
    )


class RecoverySelectorTest(unittest.TestCase):
    def test_selects_lowest_conformal_risk_without_reward_q(self):
        risk, requested, executed, target, bundle = _case()
        result = _select(risk, requested, executed, target, bundle)
        self.assertTrue(result.intervened)
        self.assertEqual(result.selected_index, 1)
        self.assertGreater(result.nominal_risk_lcb, 0.5)
        self.assertAlmostEqual(result.risk_ucb[1], 0.24)
        self.assertAlmostEqual(result.benefit_lcb[1], 0.40)
        self.assertAlmostEqual(
            result.risk_std[1], np.std(risk[:, 1], ddof=0))
        self.assertEqual(bundle.ensemble_std_ddof,
                         RECOVERY_SELECTOR_ENSEMBLE_STD_DDOF)

    def test_signed_conformal_offsets_are_valid_and_nominal_vectors_stay_zero(self):
        risk, requested, executed, target, old_bundle = _case()
        offsets = RecoveryConformalOffsets(
            nominal_lower=-0.04,
            risk_upper=np.asarray([0.0] + [-0.02] * 8),
            benefit_lower=np.asarray([0.0] + [-0.03] * 8),
            calibration_report_sha256=_UNCERTAINTY_REPORT,
        )
        bundle = _bundle(offsets=offsets,
                         config=old_bundle.selector_config)
        result = _select(risk, requested, executed, target, bundle)
        self.assertAlmostEqual(result.nominal_risk_lcb, 0.69)
        self.assertAlmostEqual(result.risk_ucb[1], 0.18)
        self.assertAlmostEqual(result.benefit_lcb[1], 0.48)
        with self.assertRaisesRegex(ValueError, "nominal.*zero"):
            _bundle(offsets=replace(
                offsets,
                benefit_lower=np.asarray([0.1] + [-0.03] * 8),
            ))

    def test_frozen_bundle_rejects_off_grid_or_changed_support(self):
        _, _, _, _, bundle = _case()
        with self.assertRaisesRegex(ValueError, "selector-grid"):
            _bundle(config=replace(
                bundle.selector_config, min_benefit_lcb=0.10))
        with self.assertRaisesRegex(ValueError, "selector support"):
            _bundle(config=replace(
                bundle.selector_config, max_epistemic_std=0.19))

    def test_slew_uses_requested_and_qtarget_but_never_executed(self):
        risk, requested, executed, target, bundle = _case()
        far_executed = executed.copy()
        far_executed[1:] = 100.0
        result = _select(risk, requested, far_executed, target, bundle)
        self.assertEqual(result.selected_index, 1)
        np.testing.assert_allclose(
            result.action_delta_rms,
            np.sqrt(np.mean((requested - requested[:1]) ** 2, axis=1)),
        )

        far_requested = requested.copy()
        far_requested[1:] = 0.9
        self.assertFalse(
            _select(risk, far_requested, executed, target, bundle).intervened)
        far_target = target.copy()
        far_target[1:] = 0.9
        self.assertFalse(
            _select(risk, requested, executed, far_target, bundle).intervened)

    def test_every_gate_abstains_to_nominal(self):
        risk, requested, executed, target, bundle = _case()
        cases = []
        cases.append((risk - np.asarray([0.30] + [0.0] * 8), requested,
                      executed, target, bundle, np.ones(9, bool)))
        cases.append((risk, requested, executed, target, _bundle(
            offsets=replace(
                bundle.offsets,
                benefit_lower=np.asarray([0.0] + [1.0] * 8))),
                      np.ones(9, bool)))
        cases.append((risk, requested, executed, target, _bundle(
            offsets=replace(
                bundle.offsets,
                risk_upper=np.asarray([0.0] + [0.9] * 8))),
                      np.ones(9, bool)))
        epistemically_wide = risk.copy()
        epistemically_wide[:, 1:] = np.asarray(
            [0.0, 0.0, 0.2, 0.6, 1.0])[:, None]
        cases.append((epistemically_wide, requested, executed, target, bundle,
                      np.ones(9, dtype=bool)))
        for index, values in enumerate(cases):
            with self.subTest(index=index):
                result = _select(*values)
                self.assertFalse(result.intervened)
                self.assertEqual(result.selected_index, 0)

    def test_bundle_serialization_is_exact_hashed_and_immutable(self):
        bundle = _bundle()
        serialized = bundle.to_dict()
        restored = RecoverySelectorBundle.from_dict(serialized)
        self.assertEqual(restored.to_dict(), serialized)
        self.assertEqual(restored.bundle_sha256, bundle.bundle_sha256)
        self.assertFalse(restored.offsets.risk_upper.flags.writeable)
        with self.assertRaises(ValueError):
            restored.offsets.risk_upper[1] = 1.0

        extra = copy.deepcopy(serialized)
        extra["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "fields are not exact"):
            RecoverySelectorBundle.from_dict(extra)
        drift = copy.deepcopy(serialized)
        drift["candidate_choice_semantics"][
            "executed_action_slew_gate"] = "allowed"
        with self.assertRaisesRegex(ValueError, "semantics"):
            RecoverySelectorBundle.from_dict(drift)
        bad_hash = copy.deepcopy(serialized)
        bad_hash["bundle_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            RecoverySelectorBundle.from_dict(bad_hash)
        bad_ddof = copy.deepcopy(serialized)
        bad_ddof["ensemble_std_ddof"] = 1
        with self.assertRaisesRegex(ValueError, "ddof must be zero"):
            RecoverySelectorBundle.from_dict(bad_ddof)

    def test_malformed_inputs_fail_closed(self):
        risk, requested, executed, target, bundle = _case()
        bad = risk.copy()
        bad[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            _select(bad, requested, executed, target, bundle)
        with self.assertRaisesRegex(ValueError, "uncertainty calibration"):
            RecoverySelectorBundle.create(
                offsets=bundle.offsets,
                selector_config=bundle.selector_config,
                probability_calibration_report_sha256=_PROBABILITY_REPORT,
                uncertainty_calibration_report_sha256="f" * 64,
                selector_search_report_sha256=_SEARCH_REPORT,
            )
        partial_mask = np.ones(9, dtype=bool)
        partial_mask[-1] = False
        with self.assertRaisesRegex(ValueError, "every locked K9"):
            _select(
                risk, requested, executed, target, bundle, partial_mask)


if __name__ == "__main__":
    unittest.main()
