from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from runtime.inference.actions import ActionApplier
from safety_data.candidates import (
    CandidateSet,
    EvidenceCandidateConfig,
    InsufficientCandidateSupportError,
    build_evidence_candidates,
)
from safety_data.recovery_options import (
    RECOVERY_OPTION_BASE_INDICES,
    RECOVERY_OPTION_COUNT,
    RECOVERY_OPTION_DURATIONS,
    RECOVERY_OPTION_KINDS,
    RECOVERY_OPTION_PROTOCOL_VERSION,
    RECOVERY_OPTION_STEPS,
    RECOVERY_OPTION_TEMPLATE_IDS,
    RecoveryOptionCandidateConfig,
    build_recovery_option_candidates,
)


def _applier() -> ActionApplier:
    return ActionApplier(
        init_qpos=np.asarray([0.0, 0.8, -1.5] * 4, dtype=np.float32),
        action_offset=np.asarray([0.4, 0.5, 0.6] * 4, dtype=np.float32),
        joint_min=np.asarray([-0.8, -0.2, -2.7] * 4, dtype=np.float32),
        joint_max=np.asarray([0.8, 1.8, -0.4] * 4, dtype=np.float32),
    )


def _raw_args(*, candidate_seed: int = 1234) -> dict[str, object]:
    nominal = np.asarray(
        [0.10, -0.05, 0.15, -0.10, 0.05, -0.15] * 2,
        dtype=np.float32,
    )
    deterministic = nominal + np.asarray(
        [0.02, -0.01, 0.01] * 4, dtype=np.float32)
    previous = nominal + np.asarray(
        [-0.12, 0.10, -0.08] * 4, dtype=np.float32)
    actor_offsets = np.asarray([
        [0.08, 0.02, -0.03] * 4,
        [-0.04, 0.09, 0.02] * 4,
        [0.03, -0.06, 0.10] * 4,
        [-0.09, -0.02, -0.04] * 4,
    ], dtype=np.float32)
    applier = _applier()
    return {
        "nominal": nominal,
        "deterministic_mean": deterministic,
        "previous_requested": previous,
        "actor_samples": deterministic[None, :] + actor_offsets,
        "action_applier": applier,
        "current_qpos": applier.init_qpos.copy(),
        "candidate_seed": candidate_seed,
    }


def _base_candidates(raw_args: dict[str, object]) -> CandidateSet:
    return build_evidence_candidates(
        **raw_args, config=RecoveryOptionCandidateConfig().base_config())


def _copy_base(
    base: CandidateSet,
    *,
    mask: np.ndarray | None = None,
    q_target: np.ndarray | None = None,
) -> CandidateSet:
    return CandidateSet(
        requested=base.requested,
        executed=base.executed,
        q_target=base.q_target if q_target is None else q_target,
        kind=base.kind,
        mask=base.mask if mask is None else mask,
        candidate_seed=base.candidate_seed,
        manifest_protocol=base.manifest_protocol,
    )


class RecoveryOptionProtocolTest(unittest.TestCase):
    def test_manifest_exactly_matches_preregistered_candidate_contract(self):
        config = RecoveryOptionCandidateConfig()
        expected = {
            "protocol_version": RECOVERY_OPTION_PROTOCOL_VERSION,
            "count": 29,
            "nominal_index": 0,
            "ordered_kinds": list(RECOVERY_OPTION_KINDS),
            "base_generator": "qsafe.evidence_candidates.v1",
            "base_generator_parameters": {
                "actor_sample_max_delta_rms": 0.50,
                "perturbation_radius_rms": 0.25,
                "q_target_dedup_atol": 0.000001,
                "min_unique_base_candidates": 8,
            },
            "residual_templates": [
                {
                    "template_id": template_id,
                    "base_candidate_index": base_index,
                }
                for template_id, base_index in zip(
                    RECOVERY_OPTION_TEMPLATE_IDS,
                    RECOVERY_OPTION_BASE_INDICES,
                    strict=True,
                )
            ],
            "option_steps": [1, 2, 3, 4],
            "ordering": "nominal_then_template_major_duration_minor",
            "option_count_formula": "1_plus_7_templates_times_4_durations",
            "duplicate_rule": (
                "duration_distinguishes_options_with_the_same_first_q_target"
            ),
            "option_steps_array": "candidate_option_steps",
            "option_semantics": "linear_decay_actor_residual_v1",
        }
        self.assertEqual(config.manifest_protocol(), expected)
        self.assertEqual(config.actor_sample_max_delta_rms, 0.50)
        self.assertEqual(config.perturbation_radius_rms, 0.25)

    def test_preregistered_geometry_cannot_be_tuned(self):
        changes = (
            {"actor_sample_max_delta_rms": 0.51},
            {"perturbation_radius_rms": 0.24},
            {"q_target_dedup_atol": 0.0},
            {"min_unique_base_candidates": 9},
        )
        for kwargs in changes:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "preregistered v2 triage protocol"):
                RecoveryOptionCandidateConfig(**kwargs)


class RecoveryOptionExpansionTest(unittest.TestCase):
    def test_k29_is_nominal_then_template_major_duration_minor(self):
        raw_args = _raw_args(candidate_seed=991)
        base = _base_candidates(raw_args)
        candidates = build_recovery_option_candidates(**raw_args)
        expansion_indices = np.asarray([
            0,
            *(
                base_index
                for base_index in RECOVERY_OPTION_BASE_INDICES
                for _ in RECOVERY_OPTION_DURATIONS
            ),
        ])

        self.assertEqual(candidates.requested.shape, (RECOVERY_OPTION_COUNT, 12))
        np.testing.assert_array_equal(
            candidates.requested, base.requested[expansion_indices])
        np.testing.assert_array_equal(
            candidates.executed, base.executed[expansion_indices])
        np.testing.assert_array_equal(
            candidates.q_target, base.q_target[expansion_indices])
        np.testing.assert_array_equal(
            candidates.option_steps, np.asarray(RECOVERY_OPTION_STEPS))
        self.assertEqual(tuple(candidates.kind), RECOVERY_OPTION_KINDS)
        self.assertEqual(len(set(candidates.kind.tolist())), 29)
        self.assertTrue(np.all(candidates.mask))
        self.assertEqual(candidates.valid_count, 29)
        self.assertEqual(candidates.candidate_seed, 991)

        for template_index in range(len(RECOVERY_OPTION_TEMPLATE_IDS)):
            start = 1 + template_index * len(RECOVERY_OPTION_DURATIONS)
            stop = start + len(RECOVERY_OPTION_DURATIONS)
            for name in ("requested", "executed", "q_target"):
                values = getattr(candidates, name)[start:stop]
                np.testing.assert_array_equal(
                    values, np.repeat(values[:1], 4, axis=0))
            np.testing.assert_array_equal(
                candidates.option_steps[start:stop], [1, 2, 3, 4])

    def test_config_build_candidates_accepts_raw_k16_arguments(self):
        raw_args = _raw_args(candidate_seed=992)
        direct = build_recovery_option_candidates(**raw_args)
        through_config = RecoveryOptionCandidateConfig().build_candidates(
            **raw_args)
        for name in (
                "requested", "executed", "q_target", "kind", "mask",
                "option_steps"):
            np.testing.assert_array_equal(
                getattr(direct, name), getattr(through_config, name))
        self.assertEqual(direct.candidate_seed, through_config.candidate_seed)
        with self.assertRaisesRegex(TypeError, "owns config"):
            RecoveryOptionCandidateConfig().build_candidates(
                **raw_args, config=RecoveryOptionCandidateConfig())

    def test_all_public_arrays_are_immutable_copies(self):
        candidates = build_recovery_option_candidates(**_raw_args())
        for name in (
                "requested", "executed", "q_target", "kind", "mask",
                "option_steps"):
            value = getattr(candidates, name)
            with self.subTest(name=name):
                self.assertFalse(value.flags.writeable)
                with self.assertRaises(ValueError):
                    value.flat[0] = value.flat[0]


class RecoveryOptionSupportTest(unittest.TestCase):
    def test_masked_selected_base_action_fails_before_expansion(self):
        raw_args = _raw_args()
        base = _base_candidates(raw_args)
        mask = base.mask.copy()
        mask[RECOVERY_OPTION_BASE_INDICES[2]] = False
        unsupported = _copy_base(base, mask=mask)

        with patch(
            "safety_data.recovery_options.build_evidence_candidates",
            return_value=unsupported,
        ), self.assertRaisesRegex(
            InsufficientCandidateSupportError,
            "only 7 unique q_target values; at least 8",
        ) as caught:
            build_recovery_option_candidates(**raw_args)
        self.assertEqual(caught.exception.valid_count, 7)
        self.assertEqual(caught.exception.minimum_required, 8)

    def test_duplicate_selected_base_target_fails_even_if_mask_claims_valid(self):
        raw_args = _raw_args()
        base = _base_candidates(raw_args)
        q_target = base.q_target.copy()
        q_target[RECOVERY_OPTION_BASE_INDICES[4]] = q_target[
            RECOVERY_OPTION_BASE_INDICES[3]]
        unsupported = _copy_base(
            base, mask=np.ones_like(base.mask), q_target=q_target)

        with patch(
            "safety_data.recovery_options.build_evidence_candidates",
            return_value=unsupported,
        ), self.assertRaises(InsufficientCandidateSupportError) as caught:
            build_recovery_option_candidates(**raw_args)
        self.assertEqual(caught.exception.valid_count, 7)

    def test_unselected_base_mask_does_not_remove_a_duration_option(self):
        raw_args = _raw_args()
        base = _base_candidates(raw_args)
        mask = base.mask.copy()
        mask[3] = False  # contraction_0.25 is not in the v2 template set.
        supported = _copy_base(base, mask=mask)

        with patch(
            "safety_data.recovery_options.build_evidence_candidates",
            return_value=supported,
        ):
            candidates = build_recovery_option_candidates(**raw_args)
        self.assertEqual(candidates.valid_count, 29)


if __name__ == "__main__":
    unittest.main()
