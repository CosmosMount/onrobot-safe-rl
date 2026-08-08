from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from safety_data.phase1_stats import (
    ARMS,
    CONFIRMATION_SEEDS,
    CommonGateStatus,
    OnlineRun,
    Phase1EvidenceError,
    RouteSpec,
    compile_phase1_evidence,
    evaluate_online_route,
    exact_paired_label_swap_p_value,
)
from scripts import evaluate_phase1_online


def online_records(route: str = "fresh_030") -> list[OnlineRun]:
    speed = {"fresh_030": 0.30, "shift_027": 0.27, "shift_033": 0.33}[route]
    records: list[OnlineRun] = []
    for offset, seed in enumerate(CONFIRMATION_SEEDS):
        baseline_falls = 1_000 + 25 * offset
        values = {
            "baseline": {
                "falls": baseline_falls,
                "mean_task_return": 100.0 + offset,
                "forward_velocity_error_mps": 0.040,
                "deadline_misses": 150,
            },
            "treatment": {
                "falls": baseline_falls - 500,
                "mean_task_return": 0.98 * (100.0 + offset),
                "forward_velocity_error_mps": 0.050,
                "deadline_misses": 100,
            },
            "placebo": {
                "falls": baseline_falls - 100,
                "mean_task_return": 0.99 * (100.0 + offset),
                "forward_velocity_error_mps": 0.047,
                "deadline_misses": 120,
            },
        }
        for arm in ARMS:
            records.append(OnlineRun(
                route=route,
                training_seed=seed,
                arm=arm,
                target_speed_mps=speed,
                exposure_policy_steps=500_000,
                **values[arm],
            ))
    return records


def route_spec(route: str) -> RouteSpec:
    return RouteSpec(
        route=route,
        starts_from_zero=route == "fresh_030",
        independently_finetuned_target_actor=route.startswith("shift_"),
        placebo_matching_verified=True,
    )


class Phase1RouteStatisticsTest(unittest.TestCase):
    def test_pooled_rates_paired_bootstrap_and_exact_test_pass(self):
        report = evaluate_online_route(
            online_records(), route_spec("fresh_030"),
            bootstrap_replicates=2_000, bootstrap_seed=19)

        self.assertAlmostEqual(report.fall_rate_per_1000["baseline"], 2.225)
        self.assertAlmostEqual(report.fall_rate_per_1000["treatment"], 1.225)
        self.assertAlmostEqual(report.fall_rate_per_1000["placebo"], 2.025)
        self.assertEqual(report.fall_count["baseline"], 11_125)
        self.assertEqual(
            report.total_exposure_policy_steps["baseline"], 5_000_000)
        self.assertAlmostEqual(report.mean_task_return["treatment"], 102.41)
        self.assertAlmostEqual(
            report.deadline_miss_rate["treatment"], 0.0002)
        self.assertAlmostEqual(
            report.absolute_fall_reduction_per_1000.estimate, 1.0)
        # Every paired seed has exactly the same difference, so a paired
        # cluster bootstrap has a degenerate interval at the correct effect.
        self.assertAlmostEqual(report.absolute_fall_reduction_per_1000.low, 1.0)
        self.assertAlmostEqual(report.absolute_fall_reduction_per_1000.high, 1.0)
        self.assertAlmostEqual(
            report.treatment_vs_placebo_reduction_per_1000.estimate, 0.8)
        self.assertEqual(report.exact_label_swap_permutations, 1024)
        self.assertAlmostEqual(
            report.exact_paired_label_swap_p_value, 1.0 / 1024.0)
        self.assertAlmostEqual(report.return_ratio, 0.98)
        self.assertAlmostEqual(
            report.forward_velocity_error_increase_mps, 0.01)
        self.assertAlmostEqual(report.treatment_deadline_miss_rate, 0.0002)
        self.assertTrue(report.route_pass)
        self.assertTrue(all(report.gate_checks.values()))

    def test_exact_label_swap_is_one_sided_and_includes_all_assignments(self):
        baseline = np.full(10, 8)
        treatment = np.full(10, 7)
        exposure = np.full(10, 500_000)
        p_value, permutations = exact_paired_label_swap_p_value(
            baseline, treatment, exposure)
        self.assertEqual(permutations, 2 ** 10)
        self.assertEqual(p_value, 1.0 / (2 ** 10))

        tied_p, tied_permutations = exact_paired_label_swap_p_value(
            baseline, baseline, exposure)
        self.assertEqual(tied_permutations, 2 ** 10)
        self.assertEqual(tied_p, 1.0)

    def test_actor_provenance_and_placebo_matching_fail_closed(self):
        no_provenance = evaluate_online_route(
            online_records(),
            RouteSpec(route="fresh_030", placebo_matching_verified=True),
            bootstrap_replicates=300)
        self.assertFalse(no_provenance.gate_checks["actor_provenance_verified"])
        self.assertFalse(no_provenance.route_pass)

        no_placebo_match = evaluate_online_route(
            online_records(),
            RouteSpec(route="fresh_030", starts_from_zero=True),
            bootstrap_replicates=300)
        self.assertFalse(no_placebo_match.gate_checks["placebo_matching_verified"])
        self.assertFalse(no_placebo_match.route_pass)

    def test_nonconfirmation_seed_or_exposure_design_cannot_claim_phase1(self):
        records = online_records()
        pilot_seeds = CONFIRMATION_SEEDS[:5]
        pilot = [record for record in records if record.training_seed in pilot_seeds]
        pilot_report = evaluate_online_route(
            pilot,
            RouteSpec(
                route="fresh_030",
                expected_seeds=pilot_seeds,
                starts_from_zero=True,
                placebo_matching_verified=True,
            ),
            bootstrap_replicates=300,
        )
        self.assertFalse(pilot_report.gate_checks["confirmation_seed_set"])
        self.assertFalse(pilot_report.route_pass)

        short_records = [
            replace(record, exposure_policy_steps=100_000)
            for record in records
        ]
        short_report = evaluate_online_route(
            short_records,
            RouteSpec(
                route="fresh_030",
                expected_exposure_policy_steps=100_000,
                starts_from_zero=True,
                placebo_matching_verified=True,
            ),
            bootstrap_replicates=300,
        )
        self.assertFalse(short_report.gate_checks["confirmation_fixed_exposure"])
        self.assertFalse(short_report.route_pass)

    def test_noninferiority_checks_can_veto_fall_effect(self):
        records = online_records()
        records = [
            replace(
                record,
                mean_task_return=80.0,
                forward_velocity_error_mps=0.09,
                deadline_misses=1_000,
            ) if record.arm == "treatment" else record
            for record in records
        ]
        report = evaluate_online_route(
            records, route_spec("fresh_030"), bootstrap_replicates=300)
        self.assertTrue(report.gate_checks["relative_fall_reduction"])
        self.assertFalse(report.gate_checks["task_return_ratio"])
        self.assertFalse(report.gate_checks["forward_velocity_error_increase"])
        self.assertFalse(report.gate_checks["runtime_deadline_miss_rate"])
        self.assertFalse(report.route_pass)


class Phase1IntegrityTest(unittest.TestCase):
    def assert_invalid(self, records, pattern: str):
        with self.assertRaisesRegex(Phase1EvidenceError, pattern):
            evaluate_online_route(
                records, route_spec("fresh_030"), bootstrap_replicates=50)

    def test_rejects_missing_duplicate_and_wrong_exposure_rows(self):
        records = online_records()
        self.assert_invalid(records[:-1], "three-arm table is incomplete")
        self.assert_invalid(records + [records[0]], "duplicate online arm")
        wrong_exposure = records.copy()
        wrong_exposure[0] = replace(
            wrong_exposure[0], exposure_policy_steps=499_999)
        self.assert_invalid(wrong_exposure, "has exposure")

    def test_rejects_seed_route_speed_and_nonfinite_metric_mismatch(self):
        records = online_records()
        wrong_seed = records.copy()
        wrong_seed[0] = replace(wrong_seed[0], training_seed=999)
        self.assert_invalid(wrong_seed, "training seed set mismatch")

        wrong_route = records.copy()
        wrong_route[0] = replace(
            wrong_route[0], route="shift_027", target_speed_mps=0.27)
        self.assert_invalid(wrong_route, "route mismatch")

        wrong_speed = records.copy()
        wrong_speed[0] = replace(wrong_speed[0], target_speed_mps=0.31)
        self.assert_invalid(wrong_speed, "requires target speed")

        nonfinite = records.copy()
        nonfinite[0] = replace(nonfinite[0], mean_task_return=float("nan"))
        self.assert_invalid(nonfinite, "mean_task_return must be a finite")

    def test_mappings_require_every_field(self):
        mapping = online_records()[0].__dict__.copy()
        mapping.pop("falls")
        with self.assertRaisesRegex(Phase1EvidenceError, "missing required field 'falls'"):
            evaluate_online_route(
                [mapping], route_spec("fresh_030"), bootstrap_replicates=50)


class Phase1CompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reports = {
            route: evaluate_online_route(
                online_records(route), route_spec(route),
                bootstrap_replicates=500, bootstrap_seed=7)
            for route in ("fresh_030", "shift_027", "shift_033")
        }
        cls.all_common = CommonGateStatus(
            data_gate=True,
            mechanics_gate=True,
            model_gate=True,
            paired_closed_loop_gate=True,
        )

    def test_fresh_route_passes_only_with_all_common_gates(self):
        decision = compile_phase1_evidence(
            self.all_common, {"fresh_030": self.reports["fresh_030"]})
        self.assertTrue(decision.fresh_030_online)
        self.assertFalse(decision.small_shift_online)
        self.assertTrue(decision.online_route_expression)
        self.assertTrue(decision.phase1_pass)
        self.assertIn("phase1_pass", decision.to_dict())

        failed_common = replace(self.all_common, mechanics_gate=False)
        decision = compile_phase1_evidence(
            failed_common, {"fresh_030": self.reports["fresh_030"]})
        self.assertTrue(decision.online_route_expression)
        self.assertFalse(decision.common_mechanism_gates)
        self.assertFalse(decision.phase1_pass)

    def test_small_shift_requires_both_targets(self):
        one_shift = compile_phase1_evidence(
            self.all_common, {"shift_027": self.reports["shift_027"]})
        self.assertFalse(one_shift.small_shift_online)
        self.assertFalse(one_shift.phase1_pass)

        both_shifts = compile_phase1_evidence(
            self.all_common,
            {
                "shift_027": self.reports["shift_027"],
                "shift_033": self.reports["shift_033"],
            },
        )
        self.assertTrue(both_shifts.small_shift_online)
        self.assertTrue(both_shifts.phase1_pass)

    def test_compiler_rejects_mislabeled_route(self):
        with self.assertRaisesRegex(Phase1EvidenceError, "does not match"):
            compile_phase1_evidence(
                self.all_common,
                {"shift_027": self.reports["fresh_030"]},
            )

    def test_development_cli_locks_protocol_and_never_authorizes_phase2(self):
        payload = {
            "common_gates": {
                "data_gate": True,
                "mechanics_gate": True,
                "model_gate": True,
                "paired_closed_loop_gate": True,
            },
            "routes": {
                "fresh_030": {
                    "starts_from_zero": True,
                    "placebo_matching_verified": True,
                    "runs": [record.__dict__ for record in online_records()],
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "phase1-input.json"
            output_path = Path(directory) / "phase1-report.json"
            input_path.write_text(json.dumps(payload), encoding="utf-8")
            argv = [
                "evaluate_phase1_online",
                "--input", str(input_path),
                "--output", str(output_path),
                "--bootstrap-replicates", "100",
            ]
            with mock.patch("sys.argv", argv):
                self.assertEqual(evaluate_phase1_online.main(), 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertTrue(report["development_only"])
        self.assertTrue(report["phase1_pass"])
        self.assertFalse(report["phase2_authorized"])
        self.assertEqual(
            report["route_reports"]["fresh_030"]
            ["exact_label_swap_permutations"],
            1024,
        )


if __name__ == "__main__":
    unittest.main()
