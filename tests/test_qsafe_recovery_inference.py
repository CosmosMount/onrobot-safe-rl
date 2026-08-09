from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rl.qsafe.artifact import load_qsafe_artifact, save_qsafe_artifact
from rl.qsafe.data import NormalizationStats
from rl.qsafe.network import QSafeEnsemble, SelectiveAdvantageQSafe
from rl.qsafe.recovery_inference import run_recovery_qsafe_inference
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_BEHAVIOR_STEPS,
    RECOVERY_PROGRAM_CANDIDATE_COUNT,
    RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
    RECOVERY_PROGRAM_NAMES,
    bind_recovery_program_manifest,
    build_recovery_program_features,
    make_recovery_program_feature_manifest,
)
from rl.qsafe.recovery_selector import (
    RecoveryConformalOffsets,
    RecoverySelectorBundle,
    RecoverySelectorConfig,
)
from rl.qsafe.training import (
    RECOVERY_PROGRAM_V4_LOSS_CONFIG,
    RECOVERY_PROGRAM_V4_MEMBER_SEED_STRIDE,
    RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
    RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
    TrainedQSafeEnsemble,
    TrainedQSafeMember,
)
from runtime.inference.actions import ActionApplier
from safety_data.recovery_behaviors import (
    MATURE_CHECKPOINT_FINGERPRINT_SHA256,
    MATURE_POLICY_ACTOR_SHA256,
    MATURE_POLICY_CONFIG_SHA256,
    MATURE_POLICY_FINGERPRINT_SHA256,
    MATURE_POLICY_STATE_DICT_SHA256,
    MATURE_POLICY_TRAINING_STEP,
    Q_NEUTRAL_PER_LEG_RAD,
    RecoveryBehaviorLibrary,
)


_COMMAND_SPEED_MPS = 0.30


def _canonical_manifest_sha256(path: Path) -> str:
    manifest = json.loads(
        (path / "manifest.json").read_text(encoding="utf-8"))
    payload = json.dumps(
        manifest,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _MaturePolicy:
    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        del observation
        return np.asarray([-0.25, 0.15, 0.40] * 4, dtype=np.float32)

    def manifest(self) -> dict[str, object]:
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
        }


def _applier() -> ActionApplier:
    return ActionApplier(
        init_qpos=np.asarray(Q_NEUTRAL_PER_LEG_RAD * 4, dtype=np.float32),
        action_offset=np.asarray([0.2, 0.4, 0.4] * 4, dtype=np.float32),
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
        max_joint_delta=None,
        action_filter=None,
    )


def _selector_bundle() -> RecoverySelectorBundle:
    return RecoverySelectorBundle.create(
        offsets=RecoveryConformalOffsets(
            nominal_lower=0.0,
            risk_upper=np.zeros(RECOVERY_PROGRAM_CANDIDATE_COUNT),
            benefit_lower=np.zeros(RECOVERY_PROGRAM_CANDIDATE_COUNT),
            calibration_report_sha256="d" * 64,
        ),
        selector_config=RecoverySelectorConfig(
            nominal_risk_lcb_trigger=0.50,
            min_benefit_lcb=0.08,
            max_risk_ucb=0.55,
            max_epistemic_std=0.20,
            max_action_delta_rms=0.50,
            max_q_target_delta_rms=0.25,
        ),
        probability_calibration_report_sha256="e" * 64,
        uncertainty_calibration_report_sha256="d" * 64,
        selector_search_report_sha256="f" * 64,
    )


def _configure_claim_weights(
    model: SelectiveAdvantageQSafe,
    selected_index: int | None,
) -> None:
    """Make one real K9 one-hot drive the lowest risk in every member."""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.action_head[0].weight.fill_(1.0)
        state_probability = 0.20 if selected_index is None else 0.80
        model.state_risk_head[-1].bias.fill_(
            math.log(state_probability / (1.0 - state_probability)))
        if selected_index is None:
            return
        if not 1 <= selected_index < RECOVERY_PROGRAM_CANDIDATE_COUNT:
            raise ValueError("selected_index must be one non-nominal K9 slot")
        # action_head input = state[128] || nominal[82] || candidate[82]
        #                     || (candidate - nominal)[82].  Candidate-program
        # one-hot starts at descriptor column 72.
        one_hot_column = (
            RECOVERY_PROGRAM_V4_NETWORK_CONFIG.state_hidden_dim
            + RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM
            + 72
            + selected_index
        )
        model.action_head[1].weight[0, one_hot_column] = 10.0
        model.action_head[1].bias[0] = 2.0
        model.action_head[3].weight[0, 0] = 1.0
        model.action_head[5].weight[0, 0] = -0.25


class ClaimArtifactFactory:
    """Save and reload exact V4 claim artifacts; never fabricate load claims."""

    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.selector_bundle = _selector_bundle()
        self.library = RecoveryBehaviorLibrary(_MaturePolicy(), _applier())
        self.recovery_binding = bind_recovery_program_manifest(
            self.library.manifest())
        self.feature_manifest = make_recovery_program_feature_manifest(
            self.library.fingerprint())
        self._paths: dict[int | None, Path] = {}

    def close(self) -> None:
        self._temporary.cleanup()

    def _save(self, selected_index: int | None) -> Path:
        key = selected_index
        if key in self._paths:
            return self._paths[key]
        models: list[SelectiveAdvantageQSafe] = []
        members: list[TrainedQSafeMember] = []
        for member_index in range(
                RECOVERY_PROGRAM_V4_TRAINING_CONFIG.ensemble_members):
            model = SelectiveAdvantageQSafe(
                RECOVERY_PROGRAM_V4_NETWORK_CONFIG)
            _configure_claim_weights(model, selected_index)
            models.append(model)
            members.append(TrainedQSafeMember(
                model=model,
                seed=(
                    RECOVERY_PROGRAM_V4_TRAINING_CONFIG.seed
                    + RECOVERY_PROGRAM_V4_MEMBER_SEED_STRIDE * member_index
                ),
                bootstrap_trajectories=[f"synthetic-{member_index}"],
                epoch_loss=[0.0] * RECOVERY_PROGRAM_V4_TRAINING_CONFIG.epochs,
                temperature=1.0,
            ))
        normalization = NormalizationStats(
            np.zeros(46, dtype=np.float32),
            np.ones(46, dtype=np.float32),
            fit_content_sha256="9" * 64,
            fit_split="synthetic_claim_contract_only",
        )
        trained = TrainedQSafeEnsemble(
            ensemble=QSafeEnsemble(models),
            members=members,
            normalization=normalization,
            command_vx=_COMMAND_SPEED_MPS,
            privileged_dim=0,
            train_split="synthetic_claim_contract_only",
            action_view="recovery_program_v1",
            action_dim=RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
            recovery_program_binding=copy.deepcopy(self.recovery_binding),
            recovery_program_feature_manifest=copy.deepcopy(
                self.feature_manifest),
            recovery_program_feature_contract_sha256=self.feature_manifest[
                "feature_contract_sha256"],
            recovery_library_fingerprint_sha256=self.library.fingerprint(),
            network_config=RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
            training_config=RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
            loss_config=RECOVERY_PROGRAM_V4_LOSS_CONFIG,
        )
        label = "nominal" if selected_index is None else str(selected_index)
        output = self.root / f"claim-artifact-target-{label}"
        provenance = {
            "command_vx": _COMMAND_SPEED_MPS,
            "recovery_program": copy.deepcopy(self.recovery_binding),
            "recovery_program_feature_contract": copy.deepcopy(
                self.feature_manifest),
            "recovery_selector_bundle": self.selector_bundle.to_dict(),
            "recovery_selector_bundle_sha256": (
                self.selector_bundle.bundle_sha256),
            "synthetic_fixture": "no_simulator_outcomes",
        }
        save_qsafe_artifact(
            output,
            trained,
            normalization,
            RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
            RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
            RECOVERY_PROGRAM_V4_LOSS_CONFIG,
            provenance=provenance,
            recovery_selector_bundle=self.selector_bundle,
        )
        self._paths[key] = output
        return output

    def load(self, selected_index: int | None = 1):
        path = self._save(selected_index)
        return load_qsafe_artifact(
            path,
            expected_manifest_sha256=_canonical_manifest_sha256(path),
        )


def _inputs() -> dict[str, np.ndarray]:
    requested = np.zeros((RECOVERY_PROGRAM_CANDIDATE_COUNT, 12), np.float32)
    requested[1:] = np.arange(1, 9, dtype=np.float32)[:, None] * 0.01
    return {
        "candidate_requested": requested,
        "candidate_executed": requested.copy(),
        "candidate_q_target": requested.copy(),
        "candidate_names": np.asarray(RECOVERY_PROGRAM_NAMES, dtype=str),
        "candidate_behavior_steps": np.asarray(
            RECOVERY_PROGRAM_BEHAVIOR_STEPS, dtype=np.int16),
        "candidate_mask": np.ones(
            RECOVERY_PROGRAM_CANDIDATE_COUNT, dtype=bool),
    }


class RecoveryInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.factory = ClaimArtifactFactory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.factory.close()

    def _infer(self, artifact=None, **overrides):
        if artifact is None:
            artifact = self.factory.load(1)
        arguments = _inputs()
        arguments.update(overrides)
        return run_recovery_qsafe_inference(
            artifact,
            np.zeros((5, 46), dtype=np.float32),
            **arguments,
            recovery_library_fingerprint_sha256=(
                self.factory.library.fingerprint()),
            selector_bundle=self.factory.selector_bundle,
            expected_command_speed_mps=_COMMAND_SPEED_MPS,
        )

    def test_loaded_exact_v4_ensemble_emits_bound_immutable_proof(self):
        artifact = self.factory.load(1)
        self.assertEqual(len(artifact.ensemble.members), 5)
        self.assertEqual(
            artifact.network_config, RECOVERY_PROGRAM_V4_NETWORK_CONFIG)
        self.assertEqual(
            artifact.manifest["training_config"],
            {
                name: getattr(RECOVERY_PROGRAM_V4_TRAINING_CONFIG, name)
                for name in RECOVERY_PROGRAM_V4_TRAINING_CONFIG.__dataclass_fields__
            },
        )

        result = self._infer(artifact)
        expected = build_recovery_program_features(
            **_inputs(),
            nominal_index=0,
            feature_manifest=self.factory.feature_manifest,
            feature_manifest_fingerprint_sha256=self.factory.feature_manifest[
                "feature_contract_sha256"],
            recovery_library_fingerprint_sha256=(
                self.factory.library.fingerprint()),
        )
        self.assertEqual(result.selected_index, 1)
        self.assertLess(
            result.member_risk[:, 1].max(),
            result.member_risk[:, 0].min(),
        )
        np.testing.assert_array_equal(
            result.nominal_action_features, expected.nominal_descriptor)
        np.testing.assert_array_equal(
            result.candidate_action_features, expected.candidate_descriptor)
        np.testing.assert_array_equal(
            result.raw_candidate_requested, _inputs()["candidate_requested"])
        self.assertEqual(
            result.selector_bundle_sha256,
            self.factory.selector_bundle.bundle_sha256,
        )
        self.assertEqual(
            result.artifact_manifest_sha256,
            artifact.claim_identity_sha256,
        )
        self.assertFalse(result.raw_candidate_requested.flags.writeable)
        self.assertFalse(result.selection.risk_mean.flags.writeable)

    def test_wrong_library_speed_and_incomplete_k9_fail_closed(self):
        artifact = self.factory.load(1)
        common = _inputs()
        with self.assertRaisesRegex(ValueError, "runtime recovery library"):
            run_recovery_qsafe_inference(
                artifact,
                np.zeros((5, 46), dtype=np.float32),
                **common,
                recovery_library_fingerprint_sha256="e" * 64,
                selector_bundle=self.factory.selector_bundle,
                expected_command_speed_mps=_COMMAND_SPEED_MPS,
            )
        with self.assertRaisesRegex(ValueError, "command speed mismatch"):
            run_recovery_qsafe_inference(
                artifact,
                np.zeros((5, 46), dtype=np.float32),
                **common,
                recovery_library_fingerprint_sha256=(
                    self.factory.library.fingerprint()),
                selector_bundle=self.factory.selector_bundle,
                expected_command_speed_mps=0.31,
            )
        incomplete = _inputs()
        incomplete["candidate_mask"][-1] = False
        with self.assertRaisesRegex(ValueError, "complete K9 support"):
            run_recovery_qsafe_inference(
                artifact,
                np.zeros((5, 46), dtype=np.float32),
                **incomplete,
                recovery_library_fingerprint_sha256=(
                    self.factory.library.fingerprint()),
                selector_bundle=self.factory.selector_bundle,
                expected_command_speed_mps=_COMMAND_SPEED_MPS,
            )

    def test_frozen_bundle_rejects_substitution_and_off_grid_config(self):
        changed = RecoverySelectorBundle.create(
            offsets=self.factory.selector_bundle.offsets,
            selector_config=replace(
                self.factory.selector_bundle.selector_config,
                nominal_risk_lcb_trigger=0.40,
            ),
            probability_calibration_report_sha256="e" * 64,
            uncertainty_calibration_report_sha256="d" * 64,
            selector_search_report_sha256="f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "selector bundle differs"):
            run_recovery_qsafe_inference(
                self.factory.load(1),
                np.zeros((5, 46), dtype=np.float32),
                **_inputs(),
                recovery_library_fingerprint_sha256=(
                    self.factory.library.fingerprint()),
                selector_bundle=changed,
                expected_command_speed_mps=_COMMAND_SPEED_MPS,
            )
        with self.assertRaisesRegex(ValueError, "not one of the preregistered"):
            RecoverySelectorBundle.create(
                offsets=self.factory.selector_bundle.offsets,
                selector_config=replace(
                    self.factory.selector_bundle.selector_config,
                    nominal_risk_lcb_trigger=0.41,
                ),
                probability_calibration_report_sha256="e" * 64,
                uncertainty_calibration_report_sha256="d" * 64,
                selector_search_report_sha256="f" * 64,
            )

    def test_loaded_artifact_manifest_weights_and_hooks_are_live_claims(self):
        cases = ("manifest", "authorization", "weights", "hooks")
        for case in cases:
            with self.subTest(case=case):
                artifact = self.factory.load(1)
                if case == "manifest":
                    artifact.manifest["provenance"]["command_vx"] = 0.31
                    message = "manifest mutated"
                elif case == "authorization":
                    object.__setattr__(
                        artifact, "authorized_manifest_sha256", "0" * 64)
                    message = "authorization mutated"
                elif case == "weights":
                    with torch.no_grad():
                        next(artifact.ensemble.parameters()).add_(1.0)
                    message = "structure or tensors mutated"
                else:
                    artifact.ensemble.register_forward_hook(
                        lambda module, inputs, output: output)
                    message = "runtime hooks"
                with self.assertRaisesRegex(ValueError, message):
                    self._infer(artifact)


if __name__ == "__main__":
    unittest.main()
