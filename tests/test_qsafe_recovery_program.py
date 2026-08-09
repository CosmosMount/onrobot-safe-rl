from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from rl.qsafe.network import QSafeNetworkConfig, SelectiveAdvantageQSafe
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_APPLICATION_DIM,
    RECOVERY_PROGRAM_BEHAVIOR_STEPS,
    RECOVERY_PROGRAM_CANDIDATE_DIM,
    RECOVERY_PROGRAM_LIBRARY_FINGERPRINT_SHA256,
    RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
    RECOVERY_PROGRAM_NAMES,
    bind_recovery_program_manifest,
    build_recovery_program_features,
    make_recovery_program_feature_manifest,
    validate_recovery_program_binding,
)
from safety_data.recovery_behaviors import RecoveryBehaviorLibrary
from tests.test_qsafe_recovery_behaviors import _MaturePolicy, _applier


_LIBRARY_FINGERPRINT = RECOVERY_PROGRAM_LIBRARY_FINGERPRINT_SHA256


def _inputs(*, batch: int | None = None) -> dict[str, object]:
    requested = np.zeros((9, 12), dtype=np.float32)
    requested[:, 0] = np.arange(9, dtype=np.float32) / np.float32(10.0)
    # The mature-actor programs deliberately have identical step-zero tuples.
    requested[1:4] = np.float32(0.25)
    executed = requested * np.float32(0.8)
    q_target = requested * np.float32(0.5)
    names = np.asarray(RECOVERY_PROGRAM_NAMES, dtype=str)
    steps = np.asarray(RECOVERY_PROGRAM_BEHAVIOR_STEPS, dtype=np.int16)
    mask = np.ones(9, dtype=bool)
    if batch is not None:
        requested = np.repeat(requested[None, ...], batch, axis=0)
        executed = np.repeat(executed[None, ...], batch, axis=0)
        q_target = np.repeat(q_target[None, ...], batch, axis=0)
        names = np.repeat(names[None, ...], batch, axis=0)
        steps = np.repeat(steps[None, ...], batch, axis=0)
        mask = np.repeat(mask[None, ...], batch, axis=0)
        for index in range(batch):
            requested[index, 0, 1] = np.float32(index / 20.0)
            executed[index, 0, 1] = np.float32(index / 30.0)
            q_target[index, 0, 1] = np.float32(index / 40.0)
    manifest = make_recovery_program_feature_manifest(_LIBRARY_FINGERPRINT)
    return {
        "candidate_requested": requested,
        "candidate_executed": executed,
        "candidate_q_target": q_target,
        "candidate_names": names,
        "candidate_behavior_steps": steps,
        "candidate_mask": mask,
        "nominal_index": 0,
        "feature_manifest": manifest,
        "feature_manifest_fingerprint_sha256": manifest[
            "feature_contract_sha256"],
        "recovery_library_fingerprint_sha256": _LIBRARY_FINGERPRINT,
    }


class RecoveryProgramFeatureTest(unittest.TestCase):
    def test_full_recovery_program_manifest_is_hash_bound(self):
        manifest = RecoveryBehaviorLibrary(
            _MaturePolicy(), _applier()).manifest()
        binding = bind_recovery_program_manifest(manifest)
        self.assertEqual(
            validate_recovery_program_binding(binding),
            binding["fingerprint_sha256"],
        )
        drifted = copy.deepcopy(binding)
        drifted["manifest"]["action_projection"]["joint_min"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "preregistered"):
            validate_recovery_program_binding(drifted)

    def test_identical_first_action_programs_remain_distinct_by_identity_and_duration(self):
        features = build_recovery_program_features(**_inputs())

        self.assertEqual(features.candidate_program.shape, (9, 46))
        self.assertEqual(features.candidate_descriptor.shape, (9, 82))
        np.testing.assert_array_equal(
            features.candidate_program[1, :RECOVERY_PROGRAM_APPLICATION_DIM],
            features.candidate_program[2, :RECOVERY_PROGRAM_APPLICATION_DIM],
        )
        np.testing.assert_array_equal(
            features.candidate_program[2, :RECOVERY_PROGRAM_APPLICATION_DIM],
            features.candidate_program[3, :RECOVERY_PROGRAM_APPLICATION_DIM],
        )
        self.assertFalse(np.array_equal(
            features.candidate_program[1], features.candidate_program[2]))
        self.assertFalse(np.array_equal(
            features.candidate_program[2], features.candidate_program[3]))
        self.assertEqual(
            features.candidate_program[1, -1], np.float32(10.0 / 96.0))
        self.assertEqual(
            features.candidate_program[2, -1], np.float32(25.0 / 96.0))
        self.assertEqual(
            features.candidate_program[3, -1], np.float32(50.0 / 96.0))

    def test_candidate_zero_is_exactly_centered_and_widths_are_locked(self):
        values = _inputs()
        features = build_recovery_program_features(**values)
        application = np.concatenate([
            values["candidate_requested"],
            values["candidate_executed"],
            values["candidate_q_target"],
        ], axis=-1)

        self.assertEqual(RECOVERY_PROGRAM_CANDIDATE_DIM, 46)
        self.assertEqual(RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM, 82)
        np.testing.assert_array_equal(
            features.nominal_descriptor, features.candidate_descriptor[0])
        np.testing.assert_array_equal(
            features.nominal_descriptor[:36], application[0])
        np.testing.assert_array_equal(
            features.nominal_descriptor[36:72], application[0])
        np.testing.assert_array_equal(
            features.nominal_descriptor[72:81],
            np.eye(9, dtype=np.float32)[0],
        )
        self.assertEqual(features.nominal_descriptor[81], np.float32(0.0))
        self.assertEqual(features.nominal_descriptor.dtype, np.float32)
        self.assertFalse(features.nominal_descriptor.flags.writeable)
        self.assertFalse(features.candidate_descriptor.flags.writeable)

    def test_batch_and_single_builder_calls_are_bit_identical(self):
        batched_values = _inputs(batch=2)
        batched = build_recovery_program_features(**batched_values)
        for batch_index in range(2):
            single_values = {
                key: value[batch_index]
                if isinstance(value, np.ndarray) else value
                for key, value in batched_values.items()
            }
            single = build_recovery_program_features(**single_values)
            np.testing.assert_array_equal(
                batched.candidate_program[batch_index],
                single.candidate_program,
            )
            np.testing.assert_array_equal(
                batched.nominal_descriptor[batch_index],
                single.nominal_descriptor,
            )
            np.testing.assert_array_equal(
                batched.candidate_descriptor[batch_index],
                single.candidate_descriptor,
            )
            np.testing.assert_array_equal(
                batched.candidate_mask[batch_index], single.candidate_mask)

    def test_any_invalid_candidate_mask_is_rejected(self):
        values = _inputs()
        values["candidate_mask"][7] = False
        with self.assertRaisesRegex(ValueError, "every locked K9"):
            build_recovery_program_features(**values)

    def test_joint_permutation_is_rejected_by_locked_k9_semantics(self):
        values = _inputs()
        permutation = np.asarray([0, 2, 1, 3, 4, 5, 6, 7, 8])
        for name in (
            "candidate_requested",
            "candidate_executed",
            "candidate_q_target",
            "candidate_names",
            "candidate_behavior_steps",
            "candidate_mask",
        ):
            values[name] = values[name][permutation]

        with self.assertRaisesRegex(ValueError, "locked K9 order"):
            build_recovery_program_features(**values)

    def test_post_canonical_candidate_axis_is_network_equivariant(self):
        features = build_recovery_program_features(**_inputs())
        torch.manual_seed(17)
        model = SelectiveAdvantageQSafe(QSafeNetworkConfig(
            action_dim=RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
            frame_hidden_dim=8,
            state_hidden_dim=8,
            action_hidden_dim=8,
        )).eval()
        history = torch.zeros(1, 5, 46)
        nominal = torch.from_numpy(
            features.nominal_descriptor.copy())[None]
        candidate = torch.from_numpy(
            features.candidate_descriptor.copy())[None]
        permutation = torch.tensor([0, 2, 1, 8, 4, 5, 6, 7, 3])
        with torch.inference_mode():
            original = model(history, nominal, candidate).risk
            reordered = model(
                history, nominal, candidate[:, permutation]).risk
        torch.testing.assert_close(reordered, original[:, permutation])

    def test_single_field_permutation_is_rejected(self):
        values = _inputs()
        values["candidate_behavior_steps"] = values[
            "candidate_behavior_steps"].copy()
        values["candidate_behavior_steps"][[1, 2]] = values[
            "candidate_behavior_steps"][[2, 1]]
        with self.assertRaisesRegex(ValueError, "locked K9 durations"):
            build_recovery_program_features(**values)

    def test_missing_wrong_shape_and_wrong_dtype_steps_are_rejected(self):
        values = _inputs()
        values["candidate_behavior_steps"] = None
        with self.assertRaisesRegex(TypeError, "numpy.ndarray"):
            build_recovery_program_features(**values)

        values = _inputs()
        values["candidate_behavior_steps"] = values[
            "candidate_behavior_steps"][:-1]
        with self.assertRaisesRegex(ValueError, "integer array.*shape"):
            build_recovery_program_features(**values)

        values = _inputs()
        values["candidate_behavior_steps"] = values[
            "candidate_behavior_steps"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "integer array"):
            build_recovery_program_features(**values)

    def test_wrong_order_nominal_index_and_fingerprints_are_rejected(self):
        values = _inputs()
        values["candidate_names"] = values["candidate_names"].copy()
        values["candidate_names"][[1, 2]] = values[
            "candidate_names"][[2, 1]]
        with self.assertRaisesRegex(ValueError, "locked K9 order"):
            build_recovery_program_features(**values)

        values = _inputs()
        values["nominal_index"] = 1
        with self.assertRaisesRegex(ValueError, "index 0"):
            build_recovery_program_features(**values)

        values = _inputs()
        values["feature_manifest_fingerprint_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            build_recovery_program_features(**values)

        values = _inputs()
        values["feature_manifest"] = copy.deepcopy(values["feature_manifest"])
        values["feature_manifest"]["feature_contract_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            build_recovery_program_features(**values)

        values = _inputs()
        values["recovery_library_fingerprint_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "locked mature recovery library"):
            build_recovery_program_features(**values)

    def test_missing_or_mutated_manifest_schema_is_rejected(self):
        values = _inputs()
        manifest = copy.deepcopy(values["feature_manifest"])
        del manifest["behavior_steps"]
        values["feature_manifest"] = manifest
        with self.assertRaisesRegex(ValueError, "locked V4 schema"):
            build_recovery_program_features(**values)

        values = _inputs()
        manifest = copy.deepcopy(values["feature_manifest"])
        manifest["schema_version"] = "qsafe.recovery_program_features.v2"
        values["feature_manifest"] = manifest
        with self.assertRaisesRegex(ValueError, "locked V4 schema"):
            build_recovery_program_features(**values)

    def test_shape_dtype_finiteness_and_mask_are_strict(self):
        values = _inputs()
        values["candidate_requested"] = values[
            "candidate_requested"].astype(np.float64)
        with self.assertRaisesRegex(ValueError, "dtype float32"):
            build_recovery_program_features(**values)

        values = _inputs()
        values["candidate_mask"] = values["candidate_mask"].astype(np.int8)
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            build_recovery_program_features(**values)

        values = _inputs()
        values["candidate_q_target"][2, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "must contain only finite"):
            build_recovery_program_features(**values)

        values = _inputs()
        values["candidate_mask"][0] = False
        with self.assertRaisesRegex(ValueError, "every locked K9"):
            build_recovery_program_features(**values)


if __name__ == "__main__":
    unittest.main()
