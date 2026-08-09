from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import unittest

import numpy as np
import torch

from rl.qsafe.recovery_inference import run_recovery_qsafe_inference
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_BEHAVIOR_STEPS,
    RECOVERY_PROGRAM_NAMES,
)
from rl.qsafe.recovery_runtime import (
    BoundActionProjectionProvider,
    CounterBasedActorShadow,
    PersistentRecoveryController,
    RecoveryActionOwner,
    RecoveryRuntimeState,
    StageCActorCounterDomain,
    StageCActorCounterKey,
    StageDActorCounterDomain,
    StageDActorCounterKey,
)
from rl.qsafe.recovery_selector import RecoverySelectorBundle
from runtime.inference.actions import ActionProjection
from safety_data.recovery_behaviors import RecoveryBehaviorLibrary
from tests.test_qsafe_recovery_inference import (
    ClaimArtifactFactory,
    _COMMAND_SPEED_MPS,
    _MaturePolicy,
    _applier,
)


_STAGE_C_STATE_SHA256 = "b" * 64
_FACTORY: ClaimArtifactFactory | None = None


def setUpModule() -> None:
    global _FACTORY
    _FACTORY = ClaimArtifactFactory()


def tearDownModule() -> None:
    global _FACTORY
    if _FACTORY is not None:
        _FACTORY.close()
    _FACTORY = None


def _factory() -> ClaimArtifactFactory:
    if _FACTORY is None:
        raise RuntimeError("runtime claim fixture is not initialized")
    return _FACTORY


def _canonical_sha256(value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _observation(offset: float = 0.0) -> np.ndarray:
    result = np.linspace(
        -0.4 + offset, 0.4 + offset, 46, dtype=np.float32)
    result[:12] = np.asarray(
        [0.15, 0.50, -1.10] * 4, dtype=np.float32) + offset
    result[34:46] = np.asarray(
        [0.17, 0.78, -1.52] * 4, dtype=np.float32) + offset
    return result


def _history(observation: np.ndarray) -> np.ndarray:
    result = np.repeat(observation[None, :], 5, axis=0)
    result[:-1] -= np.arange(4, 0, -1, dtype=np.float32)[:, None] * 0.01
    return result.astype(np.float32)


class _ExternalNoiseActor:
    def __init__(self) -> None:
        self.inputs: list[tuple[np.ndarray, np.ndarray]] = []
        self.actor_state_sha256 = "1" * 64
        self.actor_weight_version = 0
        self.actor_update_hash_chain_sha256 = "2" * 64

    def external_noise_contract(self) -> dict[str, object]:
        return CounterBasedActorShadow.external_noise_contract()

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": "qsafe.synthetic_external_noise_actor.v1",
            "external_noise_contract": self.external_noise_contract(),
            "observation_dim": 46,
            "action_dim": 12,
            "deterministic_formula": (
                "tanh(0.05*observation_first_12+0.20*standard_normal)"),
        }

    def fingerprint(self) -> str:
        return _canonical_sha256(self.manifest())

    def actor_snapshot_manifest(self, absolute_step: int) -> dict[str, object]:
        if absolute_step < 0:
            raise ValueError("absolute_step must be nonnegative")
        return {
            "schema_version": "qsafe.actor_snapshot.v1",
            "actor_state_sha256": self.actor_state_sha256,
            "actor_weight_version": self.actor_weight_version,
            "actor_update_hash_chain_sha256": (
                self.actor_update_hash_chain_sha256),
        }

    def actor_snapshot_fingerprint(self, absolute_step: int) -> str:
        return _canonical_sha256(self.actor_snapshot_manifest(absolute_step))

    def action_from_external_noise(
        self,
        observation: np.ndarray,
        standard_normal_noise: np.ndarray,
    ) -> np.ndarray:
        self.inputs.append((observation.copy(), standard_normal_noise.copy()))
        return np.tanh(
            0.05 * observation[:12] + 0.20 * standard_normal_noise
        ).astype(np.float32)


class _LegacyStatefulActor:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(4)

    def __call__(self, observation, absolute_step):
        del observation, absolute_step
        return self.rng.standard_normal(12).astype(np.float32)


class _HiddenStateActor(_ExternalNoiseActor):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    def action_from_external_noise(self, observation, standard_normal_noise):
        del observation, standard_normal_noise
        self.counter += 1
        return np.full(12, self.counter * 0.01, dtype=np.float32)


def _components(
    selected_index: int | None = 1,
    *,
    domain=None,
    actor: _ExternalNoiseActor | None = None,
):
    applier = _applier()
    library = RecoveryBehaviorLibrary(_MaturePolicy(), applier)
    projection = BoundActionProjectionProvider(applier)
    actor = _ExternalNoiseActor() if actor is None else actor
    if domain is None:
        domain = StageDActorCounterDomain(
            training_seed=201,
            stream_kind="nominal_actor",
        )
    shadow = CounterBasedActorShadow(actor, domain)
    artifact = _factory().load(selected_index)
    controller = PersistentRecoveryController(
        library,
        projection,
        shadow,
        qsafe_artifact=artifact,
        selector_bundle=_factory().selector_bundle,
    )
    return controller, library, projection, shadow, actor, artifact


def _proof(
    artifact,
    library: RecoveryBehaviorLibrary,
    history: np.ndarray,
    nominal_action: np.ndarray,
    *,
    requested_override: np.ndarray | None = None,
    executed_override: np.ndarray | None = None,
    q_target_override: np.ndarray | None = None,
):
    preview = library.preview_projected(history, nominal_action)
    requested = (
        preview.requested if requested_override is None
        else requested_override)
    executed = (
        preview.executed if executed_override is None
        else executed_override)
    q_target = (
        preview.q_target if q_target_override is None
        else q_target_override)
    return run_recovery_qsafe_inference(
        artifact,
        history,
        candidate_requested=np.asarray(requested, dtype=np.float32),
        candidate_executed=np.asarray(executed, dtype=np.float32),
        candidate_q_target=np.asarray(q_target, dtype=np.float32),
        candidate_names=np.asarray(RECOVERY_PROGRAM_NAMES, dtype=str),
        candidate_behavior_steps=np.asarray(
            RECOVERY_PROGRAM_BEHAVIOR_STEPS, dtype=np.int16),
        candidate_mask=np.ones(9, dtype=bool),
        recovery_library_fingerprint_sha256=library.fingerprint(),
        selector_bundle=_factory().selector_bundle,
        expected_command_speed_mps=_COMMAND_SPEED_MPS,
    )


class PersistentRecoveryStateMachineTest(unittest.TestCase):
    def test_exact_l10_then_spent_and_only_terminal_reset_rearms(self):
        controller, library, _, shadow, _, artifact = _components(1)
        observation = _observation()
        history = _history(observation)

        for absolute_step in range(10):
            proposal = shadow.consume(
                absolute_step=absolute_step,
                current_observation=observation,
            )
            proof = (
                _proof(artifact, library, history, proposal.action)
                if absolute_step == 0 else None)
            result = controller.step(
                absolute_step=absolute_step,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=proposal,
                decision_proof=proof,
            )
            if proof is not None:
                self.assertEqual(proof.selected_index, 1)
            self.assertEqual(result.owner, RecoveryActionOwner.RECOVERY_BEHAVIOR)
            self.assertEqual(result.behavior_index, 1)
            self.assertEqual(result.behavior_name, "mature_actor_L10")
            self.assertEqual(result.behavior_step, absolute_step)
            self.assertEqual(result.behavior_duration, 10)
            self.assertTrue(result.nominal_rejected)
            self.assertEqual(
                result.state_after_action,
                RecoveryRuntimeState.SPENT_UNTIL_RESET
                if absolute_step == 9 else RecoveryRuntimeState.OPTION,
            )
            controller.observe_outcome(
                absolute_step=absolute_step, fell=False, terminated=False)

        with self.assertRaisesRegex(RuntimeError, "terminal"):
            controller.reset()
        proposal = shadow.consume(
            absolute_step=10, current_observation=observation)
        with self.assertRaisesRegex(RuntimeError, "outside idle"):
            controller.step(
                absolute_step=10,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=proposal,
                decision_proof=_proof(
                    artifact, library, history, proposal.action),
            )
        nominal = controller.step(
            absolute_step=10,
            current_observation=observation,
            observation_history=history,
            nominal_proposal=proposal,
        )
        self.assertEqual(nominal.owner, RecoveryActionOwner.NOMINAL_ACTOR)
        controller.observe_outcome(
            absolute_step=10, fell=False, terminated=True)
        self.assertEqual(controller.state, RecoveryRuntimeState.TERMINAL)
        controller.reset()

        proposal = shadow.consume(
            absolute_step=11, current_observation=observation)
        restarted = controller.step(
            absolute_step=11,
            current_observation=observation,
            observation_history=history,
            nominal_proposal=proposal,
            decision_proof=_proof(
                artifact, library, history, proposal.action),
        )
        self.assertEqual(restarted.behavior_name, "mature_actor_L10")
        self.assertEqual(restarted.episode_index, 1)

    def test_idle_requires_real_proof_and_bare_index_api_is_disabled(self):
        controller, library, _, shadow, _, artifact = _components(None)
        observation = _observation()
        history = _history(observation)
        proposal = shadow.consume(
            absolute_step=0, current_observation=observation)
        with self.assertRaisesRegex(TypeError, "RecoveryQSafeInference"):
            controller.step(
                absolute_step=0,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=proposal,
            )
        with self.assertRaisesRegex(TypeError, "selected_behavior_index"):
            controller.step(
                absolute_step=0,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=proposal,
                selected_behavior_index=1,  # type: ignore[call-arg]
            )
        proof = _proof(artifact, library, history, proposal.action)
        self.assertEqual(proof.selected_index, 0)
        nominal = controller.step(
            absolute_step=0,
            current_observation=observation,
            observation_history=history,
            nominal_proposal=proposal,
            decision_proof=proof,
        )
        self.assertEqual(nominal.owner, RecoveryActionOwner.NOMINAL_ACTOR)
        self.assertEqual(nominal.state_after_action, RecoveryRuntimeState.IDLE)

    def test_active_option_rejects_reselection_and_early_reset(self):
        controller, library, _, shadow, _, artifact = _components(2)
        observation = _observation()
        history = _history(observation)
        first = shadow.consume(
            absolute_step=0, current_observation=observation)
        proof = _proof(artifact, library, history, first.action)
        self.assertEqual(proof.selected_index, 2)
        controller.step(
            absolute_step=0,
            current_observation=observation,
            observation_history=history,
            nominal_proposal=first,
            decision_proof=proof,
        )
        with self.assertRaisesRegex(RuntimeError, "acknowledging"):
            controller.reset()
        controller.observe_outcome(
            absolute_step=0, fell=False, terminated=False)
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            controller.reset()

        second = shadow.consume(
            absolute_step=1, current_observation=observation)
        with self.assertRaisesRegex(RuntimeError, "outside idle"):
            controller.step(
                absolute_step=1,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=second,
                decision_proof=_proof(
                    artifact, library, history, second.action),
            )
        continued = controller.step(
            absolute_step=1,
            current_observation=observation,
            observation_history=history,
            nominal_proposal=second,
        )
        self.assertEqual(continued.behavior_index, 2)
        self.assertEqual(continued.behavior_step, 1)
        self.assertEqual(continued.behavior_duration, 25)

    def test_fall_is_terminal_and_cannot_resume_option(self):
        controller, library, _, shadow, _, artifact = _components(3)
        observation = _observation()
        history = _history(observation)
        proposal = shadow.consume(
            absolute_step=0, current_observation=observation)
        controller.step(
            absolute_step=0,
            current_observation=observation,
            observation_history=history,
            nominal_proposal=proposal,
            decision_proof=_proof(
                artifact, library, history, proposal.action),
        )
        with self.assertRaisesRegex(ValueError, "fall must be terminal"):
            controller.observe_outcome(
                absolute_step=0, fell=True, terminated=False)
        outcome = controller.observe_outcome(
            absolute_step=0, fell=True, terminated=True)
        self.assertEqual(
            outcome.state_after_outcome, RecoveryRuntimeState.TERMINAL)
        with self.assertRaisesRegex(RuntimeError, "must be reset"):
            controller.step(
                absolute_step=1,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=proposal,
            )


class RuntimeProofAndProjectionTest(unittest.TestCase):
    def test_constructor_locks_library_selector_projection_and_actor_manifests(self):
        controller, library, projection, shadow, _, artifact = _components(1)
        del controller
        changed_bundle = RecoverySelectorBundle.create(
            offsets=_factory().selector_bundle.offsets,
            selector_config=replace(
                _factory().selector_bundle.selector_config,
                nominal_risk_lcb_trigger=0.40,
            ),
            probability_calibration_report_sha256="e" * 64,
            uncertainty_calibration_report_sha256="d" * 64,
            selector_search_report_sha256="f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "selector bundle differs"):
            PersistentRecoveryController(
                library,
                projection,
                shadow,
                qsafe_artifact=artifact,
                selector_bundle=changed_bundle,
            )

        class _ManifestDrift(BoundActionProjectionProvider):
            def manifest(self):
                value = super().manifest()
                value["use_action_filter"] = True
                return value

        with self.assertRaisesRegex(TypeError, "must be BoundActionProjectionProvider"):
            PersistentRecoveryController(
                library,
                _ManifestDrift(_applier()),
                shadow,
                qsafe_artifact=artifact,
                selector_bundle=_factory().selector_bundle,
            )

        class _FingerprintDrift(BoundActionProjectionProvider):
            def fingerprint(self):
                return "0" * 64

        with self.assertRaisesRegex(TypeError, "must be BoundActionProjectionProvider"):
            PersistentRecoveryController(
                library,
                _FingerprintDrift(_applier()),
                shadow,
                qsafe_artifact=artifact,
                selector_bundle=_factory().selector_bundle,
            )

        class _ActorManifestDrift(_ExternalNoiseActor):
            def manifest(self):
                value = super().manifest()
                value["external_noise_contract"] = {"drifted": True}
                return value

        with self.assertRaisesRegex(ValueError, "does not bind"):
            CounterBasedActorShadow(
                _ActorManifestDrift(), StageDActorCounterDomain(201))

    def test_wrong_history_artifact_identity_and_k9_preview_fail_closed(self):
        observation = _observation()
        history = _history(observation)

        controller, library, _, shadow, _, artifact = _components(1)
        proposal = shadow.consume(
            absolute_step=0, current_observation=observation)
        changed_history = history.copy()
        changed_history[0, 0] = np.nextafter(
            changed_history[0, 0], np.float32(1.0), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "another observation history"):
            controller.step(
                absolute_step=0,
                current_observation=observation,
                observation_history=changed_history,
                nominal_proposal=proposal,
                decision_proof=_proof(
                    artifact, library, history, proposal.action),
            )

        controller, library, _, shadow, _, _ = _components(1)
        proposal = shadow.consume(
            absolute_step=0, current_observation=observation)
        other_artifact = _factory().load(2)
        with self.assertRaisesRegex(ValueError, "artifact identity mismatch"):
            controller.step(
                absolute_step=0,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=proposal,
                decision_proof=_proof(
                    other_artifact, library, history, proposal.action),
            )

        controller, library, _, shadow, _, artifact = _components(1)
        proposal = shadow.consume(
            absolute_step=0, current_observation=observation)
        preview = library.preview_projected(history, proposal.action)
        changed = preview.requested.copy()
        changed[4, 0] = np.nextafter(
            changed[4, 0], np.float32(1.0), dtype=np.float32)
        proof = _proof(
            artifact,
            library,
            history,
            proposal.action,
            requested_override=changed,
        )
        with self.assertRaisesRegex(ValueError, "differs from library"):
            controller.step(
                absolute_step=0,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=proposal,
                decision_proof=proof,
            )

    def test_forged_or_mutated_proof_is_rejected(self):
        observation = _observation()
        history = _history(observation)
        for case in ("forged", "mutated"):
            with self.subTest(case=case):
                controller, library, _, shadow, _, artifact = _components(1)
                proposal = shadow.consume(
                    absolute_step=0, current_observation=observation)
                proof = _proof(
                    artifact, library, history, proposal.action)
                if case == "forged":
                    proof = copy.copy(proof)
                    object.__setattr__(proof, "_inference_token", object())
                    message = "must come from run_recovery_qsafe_inference"
                else:
                    proof.raw_candidate_requested.setflags(write=True)
                    proof.raw_candidate_requested[1, 0] = np.nextafter(
                        proof.raw_candidate_requested[1, 0],
                        np.float32(1.0),
                        dtype=np.float32,
                    )
                    message = "mutated after inference"
                with self.assertRaisesRegex(ValueError, message):
                    controller.step(
                        absolute_step=0,
                        current_observation=observation,
                        observation_history=history,
                        nominal_proposal=proposal,
                        decision_proof=proof,
                    )

    def test_projection_subclass_cannot_override_applied_action(self):
        applier = _applier()
        library = RecoveryBehaviorLibrary(_MaturePolicy(), applier)

        class _ActualDrift(BoundActionProjectionProvider):
            def __call__(self, requested_action, current_observation):
                value = super().__call__(requested_action, current_observation)
                changed = value.action_executed.copy()
                changed[0] = np.nextafter(
                    changed[0], np.float32(1.0), dtype=np.float32)
                return ActionProjection(
                    action_requested=value.action_requested,
                    action_executed=changed,
                    action_q_target=value.action_q_target,
                )

        actor = _ExternalNoiseActor()
        shadow = CounterBasedActorShadow(
            actor, StageDActorCounterDomain(training_seed=201))
        artifact = _factory().load(1)
        with self.assertRaisesRegex(
                TypeError, "must be BoundActionProjectionProvider"):
            PersistentRecoveryController(
                library,
                _ActualDrift(applier),
                shadow,
                qsafe_artifact=artifact,
                selector_bundle=_factory().selector_bundle,
            )

    def test_controller_rechecks_artifact_weights_and_hooks(self):
        observation = _observation()
        history = _history(observation)
        for case in ("weights", "hooks"):
            with self.subTest(case=case):
                controller, library, _, shadow, _, artifact = _components(1)
                proposal = shadow.consume(
                    absolute_step=0, current_observation=observation)
                proof = _proof(
                    artifact, library, history, proposal.action)
                if case == "weights":
                    with torch.no_grad():
                        next(artifact.ensemble.parameters()).add_(1.0)
                    message = "structure or tensors mutated"
                else:
                    artifact.ensemble.register_forward_hook(
                        lambda module, inputs, output: output)
                    message = "runtime hooks"
                with self.assertRaisesRegex(ValueError, message):
                    controller.step(
                        absolute_step=0,
                        current_observation=observation,
                        observation_history=history,
                        nominal_proposal=proposal,
                        decision_proof=proof,
                    )


class RecoveryReplaySemanticsTest(unittest.TestCase):
    def test_actual_recovery_request_is_training_action(self):
        controller, library, _, shadow, _, artifact = _components(1)
        observation = _observation()
        history = _history(observation)
        proposal = shadow.consume(
            absolute_step=0, current_observation=observation)
        result = controller.step(
            absolute_step=0,
            current_observation=observation,
            observation_history=history,
            nominal_proposal=proposal,
            decision_proof=_proof(
                artifact, library, history, proposal.action),
        )

        self.assertFalse(np.array_equal(
            result.actual_requested, result.rejected_nominal))
        self.assertTrue(result.nominal_rejected)
        self.assertTrue(result.replay.nominal_is_log_only)
        self.assertEqual(
            result.replay.training_action_semantic, "actual_requested_action")
        np.testing.assert_array_equal(
            result.replay.action, result.actual_requested)
        np.testing.assert_array_equal(
            result.replay.action_nominal, result.rejected_nominal)

        fields = result.replay.transition_fields()
        result.replay.validate_transition_fields(fields)
        wrong = copy.deepcopy(fields)
        wrong["action"] = result.rejected_nominal.copy()
        with self.assertRaisesRegex(ValueError, "never the rejected nominal"):
            result.replay.validate_transition_fields(wrong)

        transition = result.transition_fields()
        result.replay.validate_transition_fields(transition)
        self.assertEqual(
            transition["nominal_actor_proposal_sha256"],
            result.nominal_proposal.proposal_sha256,
        )
        self.assertEqual(
            transition["recovery_runtime_step_sha256"], result._live_sha256)

        with self.assertRaisesRegex(ValueError, "issued by"):
            replace(result.replay, _issue_token=object())
        result.actual_requested.setflags(write=True)
        result.actual_requested[0] = np.nextafter(
            result.actual_requested[0], np.float32(1.0), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "mutable or malformed"):
            result.require_live_integrity()


class ActorShadowConsumptionTest(unittest.TestCase):
    def test_stage_c_short_and_long_options_leave_all_draws_bit_aligned(self):
        domain = StageCActorCounterDomain(
            state_hash_sha256=_STAGE_C_STATE_SHA256,
            replica=7,
            stream_kind="nominal_actor",
        )
        short = _components(1, domain=domain)
        long = _components(3, domain=domain)
        controller_short, library_short, _, shadow_short, actor_short, art_short = (
            short)
        controller_long, library_long, _, shadow_long, actor_long, art_long = (
            long)
        observation = _observation()
        history = _history(observation)

        for absolute_step in range(64):
            proposal_short = shadow_short.consume(
                absolute_step=absolute_step,
                current_observation=observation,
            )
            proposal_long = shadow_long.consume(
                absolute_step=absolute_step,
                current_observation=observation,
            )
            self.assertEqual(
                proposal_short.counter_key, proposal_long.counter_key)
            self.assertEqual(
                proposal_short.counter_seed_sha256,
                proposal_long.counter_seed_sha256,
            )
            np.testing.assert_array_equal(
                proposal_short.external_noise, proposal_long.external_noise)
            np.testing.assert_array_equal(
                proposal_short.action, proposal_long.action)
            result_short = controller_short.step(
                absolute_step=absolute_step,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=proposal_short,
                decision_proof=(
                    _proof(
                        art_short, library_short, history,
                        proposal_short.action)
                    if absolute_step == 0 else None),
            )
            result_long = controller_long.step(
                absolute_step=absolute_step,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=proposal_long,
                decision_proof=(
                    _proof(
                        art_long, library_long, history,
                        proposal_long.action)
                    if absolute_step == 0 else None),
            )
            np.testing.assert_array_equal(
                result_short.rejected_nominal, result_long.rejected_nominal)
            controller_short.observe_outcome(
                absolute_step=absolute_step, fell=False, terminated=False)
            controller_long.observe_outcome(
                absolute_step=absolute_step, fell=False, terminated=False)

        self.assertEqual(shadow_short.consumed_count, 64)
        self.assertEqual(shadow_long.consumed_count, 64)
        self.assertEqual(len(actor_short.inputs), 128)
        self.assertEqual(len(actor_long.inputs), 128)
        self.assertEqual(
            controller_short.state, RecoveryRuntimeState.SPENT_UNTIL_RESET)
        self.assertEqual(
            controller_long.state, RecoveryRuntimeState.SPENT_UNTIL_RESET)

    def test_stage_c_and_stage_d_key_bytes_are_exact_and_distinct(self):
        stage_c = StageCActorCounterKey(
            state_hash_sha256="12" * 32,
            replica=5,
            absolute_step=17,
            stream_kind="nominal_actor",
        )
        stream = b"nominal_actor"
        expected_c = b"".join((
            b"qsafe.recovery_actor_shadow.v1\0stage_c\0",
            bytes.fromhex("12" * 32),
            (5).to_bytes(8, "little"),
            (17).to_bytes(8, "little"),
            len(stream).to_bytes(2, "little"),
            stream,
            (0).to_bytes(8, "little"),
        ))
        self.assertEqual(stage_c.seed_payload(), expected_c)
        self.assertEqual(
            hashlib.sha256(expected_c).hexdigest(),
            "125aa845f4dfcadb71bd0ebb16fc23a77a81909db3977268c1aef6c09dd87e34",
        )

        stage_d = StageDActorCounterKey(
            training_seed=201,
            absolute_exposure_step=17,
            stream_kind="nominal_actor",
        )
        expected_d = b"".join((
            b"qsafe.recovery_actor_shadow.v1\0stage_d\0",
            (201).to_bytes(8, "little"),
            (17).to_bytes(8, "little"),
            len(stream).to_bytes(2, "little"),
            stream,
            (0).to_bytes(8, "little"),
        ))
        self.assertEqual(stage_d.seed_payload(), expected_d)
        self.assertEqual(
            hashlib.sha256(expected_d).hexdigest(),
            "06b437c25e80d0193c61fff7c3c03b29b8886e563519ef414b1de89b948f5f26",
        )
        shadow = CounterBasedActorShadow(
            _ExternalNoiseActor(),
            StageDActorCounterDomain(201, "nominal_actor"),
            first_absolute_step=17,
        )
        proposal = shadow.consume(
            absolute_step=17, current_observation=_observation())
        self.assertEqual(
            hashlib.sha256(
                np.ascontiguousarray(
                    proposal.external_noise, dtype="<f4").tobytes(order="C")
            ).hexdigest(),
            "5b1492577f16cf89821842b250d420101c30d21870daa376b90f93cb4cc55703",
        )
        audit = proposal.audit_fields()
        self.assertEqual(
            audit["nominal_actor_counter_key"]["stream_kind"],
            "nominal_actor",
        )
        self.assertEqual(
            audit["nominal_actor_snapshot_fingerprint_sha256"],
            proposal.actor_snapshot_fingerprint_sha256,
        )
        self.assertEqual(
            audit["nominal_actor_proposal_sha256"], proposal.proposal_sha256)

    def test_actor_static_and_weight_snapshots_are_revalidated_each_step(self):
        observation = _observation()
        stage_c_actor = _ExternalNoiseActor()
        stage_c = CounterBasedActorShadow(
            stage_c_actor,
            StageCActorCounterDomain(_STAGE_C_STATE_SHA256, 0),
        )
        stage_c.consume(absolute_step=0, current_observation=observation)
        stage_c_actor.actor_weight_version = 1
        stage_c_actor.actor_state_sha256 = "3" * 64
        stage_c_actor.actor_update_hash_chain_sha256 = "4" * 64
        with self.assertRaisesRegex(ValueError, "Stage-C actor snapshot changed"):
            stage_c.consume(absolute_step=1, current_observation=observation)

        stage_d_actor = _ExternalNoiseActor()
        stage_d = CounterBasedActorShadow(
            stage_d_actor, StageDActorCounterDomain(201))
        stage_d.consume(absolute_step=0, current_observation=observation)
        stage_d_actor.actor_state_sha256 = "3" * 64
        with self.assertRaisesRegex(
                ValueError, "changed without a version advance"):
            stage_d.consume(absolute_step=1, current_observation=observation)

        static_actor = _ExternalNoiseActor()
        static_shadow = CounterBasedActorShadow(
            static_actor, StageDActorCounterDomain(201))
        static_actor.manifest = lambda: {"changed": True}  # type: ignore[method-assign]
        with self.assertRaisesRegex(ValueError, "static manifest mutated"):
            static_shadow.consume(
                absolute_step=0, current_observation=observation)

    def test_legacy_hidden_state_and_counter_gaps_fail_closed(self):
        domain = StageDActorCounterDomain(training_seed=201)
        with self.assertRaisesRegex(TypeError, "external-noise API"):
            CounterBasedActorShadow(_LegacyStatefulActor(), domain)

        hidden = CounterBasedActorShadow(_HiddenStateActor(), domain)
        with self.assertRaisesRegex(RuntimeError, "stateful or nondeterministic"):
            hidden.consume(
                absolute_step=0, current_observation=_observation())
        self.assertEqual(hidden.consumed_count, 0)

        shadow = CounterBasedActorShadow(_ExternalNoiseActor(), domain)
        with self.assertRaisesRegex(RuntimeError, "expected absolute step 0"):
            shadow.consume(
                absolute_step=1, current_observation=_observation())
        proposal = shadow.consume(
            absolute_step=0, current_observation=_observation())
        self.assertEqual(proposal.counter_key.draw_index, 0)
        with self.assertRaisesRegex(ValueError, "draw_index zero"):
            StageDActorCounterKey(201, 1, "nominal_actor", draw_index=1)

    def test_cross_shadow_and_mutated_proposals_are_rejected(self):
        observation = _observation()
        history = _history(observation)

        controller, _, _, _, _, _ = _components(1)
        other_shadow = CounterBasedActorShadow(
            _ExternalNoiseActor(), StageDActorCounterDomain(202))
        cross = other_shadow.consume(
            absolute_step=0, current_observation=observation)
        with self.assertRaisesRegex(ValueError, "another actor shadow/domain"):
            controller.step(
                absolute_step=0,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=cross,
            )

        controller, _, _, shadow, _, _ = _components(1)
        mutated = shadow.consume(
            absolute_step=0, current_observation=observation)
        mutated.action.setflags(write=True)
        mutated.action[0] = np.nextafter(
            mutated.action[0], np.float32(1.0), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "forged or mutated"):
            controller.step(
                absolute_step=0,
                current_observation=observation,
                observation_history=history,
                nominal_proposal=mutated,
            )


if __name__ == "__main__":
    unittest.main()
