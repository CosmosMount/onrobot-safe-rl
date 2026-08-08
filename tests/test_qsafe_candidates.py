from __future__ import annotations

import unittest

import numpy as np

from runtime.inference.actions import ActionApplier, ActionFilterButter
from safety_data.candidates import (
    CANDIDATE_COUNT,
    CANDIDATE_KINDS,
    CandidateProtocolError,
    EvidenceCandidateConfig,
    InsufficientCandidateSupportError,
    build_evidence_candidates,
)


def _applier(
    *,
    filtered: bool = False,
    max_joint_delta: float | np.ndarray | None = None,
) -> ActionApplier:
    init_qpos = np.asarray([0.0, 0.8, -1.5] * 4, dtype=np.float32)
    result = ActionApplier(
        init_qpos=init_qpos,
        action_offset=np.asarray([0.4, 0.5, 0.6] * 4, dtype=np.float32),
        joint_min=np.asarray([-0.8, -0.2, -2.7] * 4, dtype=np.float32),
        joint_max=np.asarray([0.8, 1.8, -0.4] * 4, dtype=np.float32),
        max_joint_delta=max_joint_delta,
        action_filter=(
            ActionFilterButter(12, sampling_rate=50.0, highcut=4.0, order=2)
            if filtered
            else None
        ),
    )
    result.reset_filter()
    return result


def _inputs() -> dict[str, np.ndarray]:
    nominal = np.asarray(
        [0.10, -0.05, 0.15, -0.10, 0.05, -0.15] * 2,
        dtype=np.float32,
    )
    mean = nominal + np.asarray([0.02, -0.01, 0.01] * 4, dtype=np.float32)
    previous = nominal + np.asarray([-0.12, 0.10, -0.08] * 4, dtype=np.float32)
    offsets = np.asarray([
        [0.08, 0.02, -0.03] * 4,
        [-0.04, 0.09, 0.02] * 4,
        [0.03, -0.06, 0.10] * 4,
        [-0.09, -0.02, -0.04] * 4,
    ], dtype=np.float32)
    return {
        "nominal": nominal,
        "deterministic_mean": mean,
        "previous_requested": previous,
        "actor_samples": mean[None, :] + offsets,
    }


def _build(
    *,
    action_applier: ActionApplier | None = None,
    candidate_seed: int = 1234,
    config: EvidenceCandidateConfig | None = None,
    inputs: dict[str, np.ndarray] | None = None,
):
    applier = action_applier or _applier()
    return build_evidence_candidates(
        **(inputs or _inputs()),
        action_applier=applier,
        current_qpos=applier.init_qpos.copy(),
        candidate_seed=candidate_seed,
        config=config,
    )


class EvidenceCandidateTest(unittest.TestCase):
    def test_native_poc_defaults_are_locked_in_manifest(self):
        config = EvidenceCandidateConfig()
        manifest = config.manifest_protocol()
        self.assertEqual(config.actor_sample_max_delta_rms, 0.50)
        self.assertEqual(config.perturbation_radius_rms, 0.25)
        self.assertEqual(
            manifest["local_actor_samples"]["max_delta_rms"], 0.50)
        self.assertEqual(
            manifest["symmetric_perturbations"]["radius_rms"], 0.25)

    def test_fixed_order_and_candidate_families(self):
        inputs = _inputs()
        config = EvidenceCandidateConfig(perturbation_radius_rms=0.12)
        candidates = _build(inputs=inputs, config=config)

        self.assertEqual(candidates.requested.shape, (CANDIDATE_COUNT, 12))
        self.assertEqual(tuple(candidates.kind), CANDIDATE_KINDS)
        np.testing.assert_array_equal(candidates.requested[0], inputs["nominal"])
        np.testing.assert_array_equal(
            candidates.requested[1], inputs["deterministic_mean"])
        np.testing.assert_array_equal(
            candidates.requested[2], inputs["previous_requested"])
        for index, factor in zip((3, 4, 5), (0.25, 0.50, 0.75), strict=True):
            expected = inputs["previous_requested"] + factor * (
                inputs["nominal"] - inputs["previous_requested"])
            np.testing.assert_allclose(
                candidates.requested[index], expected, rtol=0.0, atol=1e-7)
        np.testing.assert_array_equal(
            candidates.requested[6:10], inputs["actor_samples"])

        for plus_index, minus_index in ((10, 11), (12, 13), (14, 15)):
            plus_delta = candidates.requested[plus_index] - inputs["nominal"]
            minus_delta = candidates.requested[minus_index] - inputs["nominal"]
            np.testing.assert_allclose(
                plus_delta, -minus_delta, rtol=0.0, atol=2e-7)
            self.assertAlmostEqual(
                float(np.sqrt(np.mean(np.square(plus_delta)))),
                config.perturbation_radius_rms,
                places=6,
            )
        self.assertTrue(np.all(candidates.mask))
        self.assertEqual(candidates.valid_count, CANDIDATE_COUNT)

    def test_seeded_generation_is_deterministic_and_seed_is_explicit(self):
        first = _build(candidate_seed=99)
        second = _build(candidate_seed=99)
        different = _build(candidate_seed=100)
        np.testing.assert_array_equal(first.requested, second.requested)
        np.testing.assert_array_equal(first.executed, second.executed)
        np.testing.assert_array_equal(first.q_target, second.q_target)
        np.testing.assert_array_equal(first.mask, second.mask)
        np.testing.assert_array_equal(first.requested[:10], different.requested[:10])
        self.assertFalse(np.array_equal(first.requested[10:], different.requested[10:]))
        self.assertEqual(first.candidate_seed, 99)
        self.assertEqual(
            first.manifest_protocol["symmetric_perturbations"]["seed_argument"],
            "candidate_seed",
        )
        self.assertEqual(first.manifest_protocol["count"], 16)
        self.assertEqual(first.manifest_protocol["nominal_index"], 0)

    def test_projection_is_bound_safe_and_does_not_mutate_filter_baseline(self):
        applier = _applier(
            filtered=True,
            max_joint_delta=np.asarray([0.08, 0.10, 0.12] * 4, dtype=np.float32),
        )
        assert applier.action_filter is not None
        applier.project(
            np.asarray([0.2, -0.1, 0.05] * 4, dtype=np.float32),
            applier.init_qpos,
        )
        baseline = applier.action_filter.capture_state()
        candidates = _build(action_applier=applier)
        after = applier.action_filter.capture_state()
        np.testing.assert_array_equal(after.x_history, baseline.x_history)
        np.testing.assert_array_equal(after.y_history, baseline.y_history)

        self.assertTrue(np.all(candidates.requested >= -1.0))
        self.assertTrue(np.all(candidates.requested <= 1.0))
        self.assertTrue(np.all(candidates.executed >= -1.0))
        self.assertTrue(np.all(candidates.executed <= 1.0))
        self.assertTrue(np.all(candidates.q_target >= applier.joint_min - 1e-6))
        self.assertTrue(np.all(candidates.q_target <= applier.joint_max + 1e-6))
        repeated = applier.preview_many(candidates.requested, applier.init_qpos)
        np.testing.assert_array_equal(
            candidates.q_target,
            np.stack([item.action_q_target for item in repeated]),
        )

    def test_q_target_duplicates_are_masked_in_first_occurrence_order(self):
        inputs = _inputs()
        inputs["deterministic_mean"] = inputs["nominal"].copy()
        inputs["actor_samples"][0] = inputs["nominal"].copy()
        candidates = _build(inputs=inputs)
        self.assertTrue(candidates.mask[0])
        self.assertFalse(candidates.mask[1])
        self.assertFalse(candidates.mask[6])
        retained = candidates.q_target[candidates.mask]
        for left in range(len(retained)):
            for right in range(left + 1, len(retained)):
                self.assertFalse(np.allclose(
                    retained[left], retained[right], rtol=0.0, atol=1e-6))
        self.assertGreaterEqual(candidates.valid_count, 8)

    def test_fails_closed_when_projection_has_fewer_than_eight_unique_targets(self):
        applier = _applier(max_joint_delta=0.0)
        with self.assertRaisesRegex(
            InsufficientCandidateSupportError,
            "only 1 unique q_target values; at least 8",
        ) as caught:
            _build(action_applier=applier)
        self.assertEqual(caught.exception.valid_count, 1)
        self.assertEqual(caught.exception.minimum_required, 8)

    def test_actor_samples_are_radially_local_to_deterministic_mean(self):
        inputs = _inputs()
        inputs["actor_samples"] = np.asarray([
            np.full(12, 0.9),
            np.full(12, -0.9),
            np.asarray([0.9, -0.9] * 6),
            np.asarray([-0.9, 0.9] * 6),
        ], dtype=np.float32)
        config = EvidenceCandidateConfig(actor_sample_max_delta_rms=0.20)
        candidates = _build(inputs=inputs, config=config)
        delta = candidates.requested[6:10] - inputs["deterministic_mean"]
        rms = np.sqrt(np.mean(np.square(delta), axis=1))
        self.assertTrue(np.all(rms <= 0.20 + 1e-6))
        self.assertTrue(np.all(rms >= 0.20 - 1e-6))

    def test_bound_limited_perturbations_remain_symmetric_and_normalized(self):
        inputs = _inputs()
        inputs["nominal"] = np.asarray(
            [0.95, -0.92, 0.90] * 4, dtype=np.float32)
        inputs["deterministic_mean"] = np.asarray(
            [0.75, -0.70, 0.72] * 4, dtype=np.float32)
        inputs["previous_requested"] = np.asarray(
            [0.50, -0.45, 0.48] * 4, dtype=np.float32)
        inputs["actor_samples"] = np.stack([
            inputs["deterministic_mean"] + offset
            for offset in (
                np.asarray([0.08, 0.02, -0.03] * 4, dtype=np.float32),
                np.asarray([-0.04, 0.09, 0.02] * 4, dtype=np.float32),
                np.asarray([0.03, -0.06, 0.10] * 4, dtype=np.float32),
                np.asarray([-0.09, -0.02, -0.04] * 4, dtype=np.float32),
            )
        ])
        candidates = _build(
            inputs=inputs,
            config=EvidenceCandidateConfig(perturbation_radius_rms=0.40),
        )

        self.assertTrue(np.all(candidates.requested >= -1.0))
        self.assertTrue(np.all(candidates.requested <= 1.0))
        for plus_index, minus_index in ((10, 11), (12, 13), (14, 15)):
            midpoint = 0.5 * (
                candidates.requested[plus_index]
                + candidates.requested[minus_index])
            np.testing.assert_allclose(
                midpoint, inputs["nominal"], rtol=0.0, atol=6e-8)

    def test_invalid_seed_and_actor_shape_fail_before_projection(self):
        for seed in (True, -1, 1.5):
            with self.subTest(seed=seed), self.assertRaisesRegex(
                CandidateProtocolError, "candidate_seed"
            ):
                _build(candidate_seed=seed)  # type: ignore[arg-type]
        inputs = _inputs()
        inputs["actor_samples"] = inputs["actor_samples"][:3]
        with self.assertRaisesRegex(CandidateProtocolError, "exactly four"):
            _build(inputs=inputs)

    def test_config_accepts_numpy_integer_and_rejects_non_numeric_radii(self):
        config = EvidenceCandidateConfig(min_unique_candidates=np.int64(9))
        self.assertEqual(config.min_unique_candidates, 9)
        for value in (True, "wide", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "finite number"):
                EvidenceCandidateConfig(
                    actor_sample_max_delta_rms=value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
