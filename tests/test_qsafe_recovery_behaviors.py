from __future__ import annotations

import copy
import hashlib
import json
import unittest

import numpy as np

from runtime.inference.actions import (
    ActionApplier,
    ActionFilterButter,
    qpos_to_action,
)
from safety_data.recovery_behaviors import (
    MATURE_CHECKPOINT_FINGERPRINT_SHA256,
    MATURE_POLICY_ACTOR_SHA256,
    MATURE_POLICY_CONFIG_SHA256,
    MATURE_POLICY_FINGERPRINT_SHA256,
    MATURE_POLICY_STATE_DICT_SHA256,
    MATURE_POLICY_TRAINING_STEP,
    Q_CROUCH_PER_LEG_RAD,
    Q_NEUTRAL_PER_LEG_RAD,
    RAMP_MAX_DELTA_RAD,
    RECOVERY_BEHAVIOR_COUNT,
    RECOVERY_BEHAVIOR_KINDS,
    RECOVERY_BEHAVIOR_PROTOCOL_VERSION,
    RECOVERY_BEHAVIOR_STEPS,
    RecoveryBehaviorConfig,
    RecoveryBehaviorLibrary,
    RecoveryBehaviorPreview,
    build_recovery_behavior_library,
)


def _policy_manifest() -> dict[str, object]:
    return {
        "training_step": MATURE_POLICY_TRAINING_STEP,
        "config_sha256": MATURE_POLICY_CONFIG_SHA256,
        "actor_sha256": MATURE_POLICY_ACTOR_SHA256,
        "actor_state_dict_sha256": MATURE_POLICY_STATE_DICT_SHA256,
        "policy_fingerprint_sha256": MATURE_POLICY_FINGERPRINT_SHA256,
        "checkpoint_fingerprint_sha256": (
            MATURE_CHECKPOINT_FINGERPRINT_SHA256),
        "observation_dim": 46,
        "actor_observation_dim": 46,
        "action_dim": 12,
        "irrelevant_device_field": "cpu",
    }


class _MaturePolicy:
    def __init__(self, action: np.ndarray | None = None) -> None:
        self.action = np.asarray(
            [-0.25, 0.15, 0.40] * 4 if action is None else action,
            dtype=np.float32,
        )
        self.observations: list[np.ndarray] = []
        self.source_manifest = _policy_manifest()

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        self.observations.append(np.asarray(observation).copy())
        return self.action.copy()

    def manifest(self) -> dict[str, object]:
        return copy.deepcopy(self.source_manifest)


def _applier(
    *,
    max_joint_delta: float | None = None,
    filtered: bool = False,
    init_qpos: np.ndarray | None = None,
    action_offset: np.ndarray | None = None,
) -> ActionApplier:
    return ActionApplier(
        init_qpos=np.asarray(
            Q_NEUTRAL_PER_LEG_RAD * 4
            if init_qpos is None else init_qpos,
            dtype=np.float32,
        ),
        action_offset=np.asarray(
            [0.2, 0.4, 0.4] * 4
            if action_offset is None else action_offset,
            dtype=np.float32,
        ),
        joint_min=np.asarray(
            [-1.05, -1.57, -2.72, -1.05, -1.57, -2.72,
             -1.05, -0.52, -2.72, -1.05, -0.52, -2.72],
            dtype=np.float32,
        ),
        joint_max=np.asarray(
            [1.05, 3.49, -0.84, 1.05, 3.49, -0.84,
             1.05, 4.54, -0.84, 1.05, 4.54, -0.84],
            dtype=np.float32,
        ),
        max_joint_delta=max_joint_delta,
        action_filter=(
            ActionFilterButter(
                12, sampling_rate=50.0, highcut=4.0, order=2)
            if filtered else None
        ),
    )


def _history() -> np.ndarray:
    history = np.zeros((5, 46), dtype=np.float32)
    for frame in range(5):
        history[frame] = np.linspace(
            -0.2 + frame * 0.01,
            0.2 + frame * 0.01,
            46,
            dtype=np.float32,
        )
    history[-1, 0:12] = np.asarray(
        [0.15, 0.50, -1.10] * 4, dtype=np.float32)
    history[-1, 34:46] = np.asarray(
        [0.17, 0.78, -1.52] * 4, dtype=np.float32)
    return history


def _nominal() -> np.ndarray:
    return np.asarray([0.11, -0.22, 0.33] * 4, dtype=np.float32)


def _library(
    policy: _MaturePolicy | None = None,
    applier: ActionApplier | None = None,
) -> RecoveryBehaviorLibrary:
    return build_recovery_behavior_library(
        policy or _MaturePolicy(), applier or _applier())


class RecoveryBehaviorProtocolTest(unittest.TestCase):
    def test_manifest_matches_locked_yaml_contract(self):
        config = RecoveryBehaviorConfig()
        manifest = config.manifest_protocol()

        self.assertEqual(
            manifest["protocol_version"],
            RECOVERY_BEHAVIOR_PROTOCOL_VERSION,
        )
        self.assertEqual(manifest["count"], 9)
        self.assertEqual(manifest["nominal_index"], 0)
        self.assertEqual(
            manifest["ordered_names"], list(RECOVERY_BEHAVIOR_KINDS))
        self.assertEqual(
            manifest["behavior_steps_array"], "candidate_behavior_steps")
        self.assertEqual(
            manifest["behavior_override_steps"],
            [0, 10, 25, 50, 10, 10, 25, 25, 25],
        )
        self.assertEqual(
            manifest["observation_history_shape"], [5, 46])
        self.assertEqual(
            manifest["observation_joint_q_slice"], [0, 12])
        self.assertEqual(
            manifest["observation_previous_q_target_slice"], [34, 46])
        self.assertEqual(
            manifest["q_neutral_per_leg_rad"], [0.05, 0.70, -1.40])
        self.assertEqual(
            manifest["q_crouch_per_leg_rad"], [0.05, 0.90, -1.60])
        self.assertEqual(
            manifest["ramp_max_delta_rad_per_joint_per_policy_step"], 0.04)
        self.assertEqual(manifest["max_joint_delta"], None)
        self.assertFalse(manifest["use_action_filter"])
        expected_hash = hashlib.sha256(json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")).hexdigest()
        self.assertEqual(config.protocol_sha256(), expected_hash)

    def test_preregistered_parameters_cannot_be_tuned(self):
        changes = (
            {"q_neutral_per_leg_rad": (0.05, 0.71, -1.40)},
            {"q_crouch_per_leg_rad": (0.05, 0.90, -1.59)},
            {"ramp_max_delta_rad": 0.041},
            {"observation_history_shape": (4, 46)},
            {"observation_joint_q_slice": (1, 13)},
            {"observation_previous_q_target_slice": (33, 45)},
        )
        for kwargs in changes:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                    ValueError, "preregistered v3 protocol"):
                RecoveryBehaviorConfig(**kwargs)

    def test_protocol_collections_return_isolated_values(self):
        config = RecoveryBehaviorConfig()
        first = config.manifest_protocol()
        first["ordered_names"][0] = "changed"
        first["q_neutral_per_leg_rad"][0] = 99.0
        self.assertEqual(
            config.manifest_protocol()["ordered_names"][0], "nominal")
        self.assertEqual(
            config.manifest_protocol()["q_neutral_per_leg_rad"][0], 0.05)


class RecoveryBehaviorBindingTest(unittest.TestCase):
    def test_library_binds_policy_projection_and_stable_hash(self):
        first = _library()
        second = _library()
        manifest = first.manifest()

        self.assertEqual(first.candidate_count, RECOVERY_BEHAVIOR_COUNT)
        np.testing.assert_array_equal(
            first.behavior_steps, RECOVERY_BEHAVIOR_STEPS)
        np.testing.assert_array_equal(first.durations, RECOVERY_BEHAVIOR_STEPS)
        self.assertFalse(first.behavior_steps.flags.writeable)
        self.assertEqual(
            manifest["mature_policy_identity"],
            {
                key: value for key, value in _policy_manifest().items()
                if key != "irrelevant_device_field"
            },
        )
        self.assertEqual(
            manifest["candidate_protocol"],
            RecoveryBehaviorConfig().manifest_protocol(),
        )
        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(len(first.fingerprint()), 64)

        manifest["mature_policy_identity"]["training_step"] = 1
        self.assertEqual(
            first.manifest()["mature_policy_identity"]["training_step"],
            MATURE_POLICY_TRAINING_STEP,
        )

    def test_duration_arrays_cannot_mutate_the_locked_library_state(self):
        library = _library()

        exposed_steps = library.behavior_steps
        exposed_durations = library.durations
        for value in (exposed_steps, exposed_durations):
            self.assertFalse(value.flags.writeable)
            value.setflags(write=True)
            value[:] = 999

        np.testing.assert_array_equal(
            library.behavior_steps, RECOVERY_BEHAVIOR_STEPS)
        np.testing.assert_array_equal(
            library.durations, RECOVERY_BEHAVIOR_STEPS)
        self.assertEqual(
            library.manifest_protocol()["behavior_override_steps"],
            list(RECOVERY_BEHAVIOR_STEPS),
        )

    def test_wrong_mature_policy_identity_is_rejected_field_by_field(self):
        for key in (
            "training_step",
            "config_sha256",
            "actor_sha256",
            "actor_state_dict_sha256",
            "policy_fingerprint_sha256",
            "checkpoint_fingerprint_sha256",
            "observation_dim",
            "actor_observation_dim",
            "action_dim",
        ):
            policy = _MaturePolicy()
            policy.source_manifest[key] = "wrong"
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, repr(key)):
                _library(policy=policy)

    def test_policy_interface_and_projection_contract_are_strict(self):
        with self.assertRaisesRegex(TypeError, "deterministic_action"):
            RecoveryBehaviorLibrary(object(), _applier())  # type: ignore[arg-type]

        policy_without_manifest = type(
            "Policy", (), {"deterministic_action": lambda self, obs: obs[:12]})()
        with self.assertRaisesRegex(TypeError, "evidence manifest"):
            RecoveryBehaviorLibrary(
                policy_without_manifest, _applier())  # type: ignore[arg-type]

        with self.assertRaisesRegex(ValueError, "max_joint_delta"):
            _library(applier=_applier(max_joint_delta=0.04))
        with self.assertRaisesRegex(ValueError, "action_filter"):
            _library(applier=_applier(filtered=True))
        with self.assertRaisesRegex(ValueError, "init_qpos"):
            _library(applier=_applier(init_qpos=np.zeros(12)))
        with self.assertRaisesRegex(ValueError, "action_offset"):
            _library(applier=_applier(action_offset=np.ones(12)))


class RecoveryBehaviorLawTest(unittest.TestCase):
    def test_step_zero_candidates_follow_all_five_locked_laws(self):
        policy = _MaturePolicy()
        applier = _applier()
        library = _library(policy=policy, applier=applier)
        history = _history()
        before = history.copy()
        nominal = _nominal()

        actions = library.preview_candidates(history, nominal)

        self.assertEqual(actions.shape, (RECOVERY_BEHAVIOR_COUNT, 12))
        self.assertFalse(actions.flags.writeable)
        np.testing.assert_array_equal(actions[0], nominal)
        np.testing.assert_array_equal(
            actions[1:4], np.repeat(policy.action[None, :], 3, axis=0))
        q_measured = history[-1, 0:12]
        q_previous = history[-1, 34:46]
        q_neutral = np.asarray(Q_NEUTRAL_PER_LEG_RAD * 4, np.float32)
        q_crouch = np.asarray(Q_CROUCH_PER_LEG_RAD * 4, np.float32)
        expected_targets = {
            4: q_measured,
            5: 0.5 * q_measured + 0.5 * q_neutral,
            6: 0.5 * q_measured + 0.5 * q_neutral,
            7: q_previous + np.clip(
                q_neutral - q_previous,
                -RAMP_MAX_DELTA_RAD,
                RAMP_MAX_DELTA_RAD,
            ),
            8: q_previous + np.clip(
                q_crouch - q_previous,
                -RAMP_MAX_DELTA_RAD,
                RAMP_MAX_DELTA_RAD,
            ),
        }
        for index, q_target in expected_targets.items():
            with self.subTest(index=index):
                np.testing.assert_allclose(
                    actions[index],
                    qpos_to_action(
                        q_target,
                        init_qpos=applier.init_qpos,
                        action_offset=applier.action_offset,
                    ),
                    rtol=0.0,
                    atol=1e-7,
                )
        np.testing.assert_array_equal(history, before)
        self.assertEqual(len(policy.observations), 1)
        np.testing.assert_array_equal(policy.observations[0], history[-1])

    def test_only_newest_deployable_frame_can_affect_actions(self):
        policy = _MaturePolicy()
        library = _library(policy=policy)
        first_history = _history()
        second_history = first_history.copy()
        second_history[:-1] = np.linspace(
            -100.0, 100.0, 4 * 46, dtype=np.float32).reshape(4, 46)

        first = library.preview_candidates(first_history, _nominal())
        second = library.preview_candidates(second_history, _nominal())

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(policy.observations[-1], second_history[-1])

    def test_feedback_laws_recompute_from_each_latest_history(self):
        library = _library()
        first = _history()
        second = first.copy()
        second[-1, 0:12] += np.asarray(
            [0.02, -0.03, 0.01] * 4, dtype=np.float32)
        second[-1, 34:46] += np.asarray(
            [-0.08, 0.08, -0.08] * 4, dtype=np.float32)

        for index in (4, 5, 6, 7, 8):
            with self.subTest(index=index):
                first_action = library(index, first, 1, _nominal())
                second_action = library(index, second, 1, _nominal())
                self.assertFalse(np.array_equal(first_action, second_action))

    def test_candidate_duration_and_argument_validation(self):
        library = _library()
        history = _history()
        nominal = _nominal()

        np.testing.assert_array_equal(library(0, history, 0, nominal), nominal)
        for index, duration in enumerate(RECOVERY_BEHAVIOR_STEPS[1:], start=1):
            with self.subTest(index=index):
                action = library(index, history, duration - 1, nominal)
                self.assertEqual(action.shape, (12,))
                with self.assertRaisesRegex(ValueError, "inactive"):
                    library(index, history, duration, nominal)

        invalid_calls = (
            (False, history, 0, nominal),
            (-1, history, 0, nominal),
            (RECOVERY_BEHAVIOR_COUNT, history, 0, nominal),
        )
        for index, value, step, action in invalid_calls:
            with self.subTest(index=index), self.assertRaises(ValueError):
                library(index, value, step, action)
        with self.assertRaisesRegex(ValueError, "step"):
            library(1, history, True, nominal)
        with self.assertRaisesRegex(ValueError, "step"):
            library(1, history, -1, nominal)
        with self.assertRaisesRegex(ValueError, "step zero"):
            library(0, history, 1, nominal)

    def test_deployable_input_and_actor_output_validation(self):
        library = _library()
        nominal = _nominal()
        for history in (
            np.zeros((4, 46), dtype=np.float32),
            np.zeros((5, 45), dtype=np.float32),
            np.full((5, 46), np.nan, dtype=np.float32),
        ):
            with self.subTest(shape=history.shape), self.assertRaisesRegex(
                    ValueError, "observation_history"):
                library.preview_candidates(history, nominal)
        with self.assertRaisesRegex(ValueError, "nominal_action"):
            library.preview_candidates(_history(), np.zeros(11))
        with self.assertRaisesRegex(ValueError, "nominal_action"):
            library.preview_candidates(_history(), np.full(12, 1.1))
        with self.assertRaisesRegex(ValueError, "mature_policy action"):
            _library(_MaturePolicy(np.full(12, 1.2))).preview_candidates(
                _history(), nominal)

    def test_branch_state_contract_is_explicitly_stateless(self):
        library = _library()
        self.assertIsNone(library.capture_branch_state())
        self.assertIsNone(library.restore_branch_state(None))
        with self.assertRaisesRegex(ValueError, "state must be None"):
            library.restore_branch_state({})  # type: ignore[arg-type]


class RecoveryBehaviorPreviewTest(unittest.TestCase):
    def test_projection_preview_records_exact_first_actions(self):
        applier = _applier()
        library = _library(applier=applier)
        history = _history()
        actions = library.preview_candidates(history, _nominal())

        preview = library.preview_projected(history, _nominal())
        expected = applier.preview_many(actions, history[-1, 0:12])

        self.assertEqual(preview.valid_count, RECOVERY_BEHAVIOR_COUNT)
        self.assertEqual(tuple(preview.kind), RECOVERY_BEHAVIOR_KINDS)
        np.testing.assert_array_equal(
            preview.behavior_steps, RECOVERY_BEHAVIOR_STEPS)
        np.testing.assert_array_equal(preview.requested, actions)
        np.testing.assert_array_equal(
            preview.executed,
            np.stack([item.action_executed for item in expected]),
        )
        np.testing.assert_array_equal(
            preview.q_target,
            np.stack([item.action_q_target for item in expected]),
        )
        self.assertEqual(
            preview.library_fingerprint_sha256, library.fingerprint())
        self.assertEqual(
            preview.manifest_protocol, library.manifest_protocol())
        for name in (
                "requested", "executed", "q_target", "kind", "mask",
                "behavior_steps"):
            value = getattr(preview, name)
            with self.subTest(name=name):
                self.assertFalse(value.flags.writeable)
                with self.assertRaises(ValueError):
                    value.flat[0] = value.flat[0]

    def test_preview_validation_rejects_protocol_or_shape_corruption(self):
        valid = _library().preview_projected(_history(), _nominal())
        with self.assertRaisesRegex(ValueError, "requested must have shape"):
            RecoveryBehaviorPreview(
                requested=np.zeros((8, 12)),
                executed=valid.executed,
                q_target=valid.q_target,
                kind=valid.kind,
                mask=valid.mask,
                behavior_steps=valid.behavior_steps,
                manifest_protocol=valid.manifest_protocol,
                library_fingerprint_sha256=valid.library_fingerprint_sha256,
            )
        changed = copy.deepcopy(valid.manifest_protocol)
        changed["behavior_override_steps"][1] = 9
        with self.assertRaisesRegex(ValueError, "manifest_protocol"):
            RecoveryBehaviorPreview(
                requested=valid.requested,
                executed=valid.executed,
                q_target=valid.q_target,
                kind=valid.kind,
                mask=valid.mask,
                behavior_steps=valid.behavior_steps,
                manifest_protocol=changed,
                library_fingerprint_sha256=valid.library_fingerprint_sha256,
            )


if __name__ == "__main__":
    unittest.main()
