"""Deployable K9 recovery-program Q_safe inference for V4."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from math import isclose
from typing import Any, Mapping

import numpy as np
import torch

from rl.qsafe.artifact import LoadedQSafeArtifact
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_CANDIDATE_COUNT,
    RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
    RECOVERY_PROGRAM_VIEW,
    build_recovery_program_features,
    validate_recovery_program_binding,
)
from rl.qsafe.recovery_selector import (
    RecoverySelection,
    RecoverySelectorBundle,
    select_recovery_program,
)


_HISTORY_SHAPE = (5, 46)
_SPEED_ATOL = 1e-6
_ACTION_COMPONENTS = (
    "common_current_nominal_application_tuple",
    "candidate_recovery_program_v1",
)
_RECOVERY_INFERENCE_TOKEN = object()


@dataclass(frozen=True)
class RecoveryQSafeInference:
    selection: RecoverySelection
    member_risk: np.ndarray
    member_state_risk: np.ndarray
    normalized_observation_history: np.ndarray
    nominal_action_features: np.ndarray
    candidate_action_features: np.ndarray
    candidate_mask: np.ndarray
    feature_contract_sha256: str
    recovery_library_fingerprint_sha256: str
    selector_bundle_sha256: str
    raw_candidate_requested: np.ndarray
    raw_candidate_executed: np.ndarray
    raw_candidate_q_target: np.ndarray
    observation_history_sha256: str
    artifact_manifest_sha256: str
    _inference_token: object | None = field(
        default=None, repr=False, compare=False)
    _proof_live_sha256: str | None = field(
        default=None, repr=False, compare=False)

    @property
    def selected_index(self) -> int:
        return self.selection.selected_index

    @property
    def intervened(self) -> bool:
        return self.selection.intervened

    @property
    def history_sha256(self) -> str:
        return self.observation_history_sha256

    @property
    def candidate_requested(self) -> np.ndarray:
        return self.raw_candidate_requested

    @property
    def candidate_executed(self) -> np.ndarray:
        return self.raw_candidate_executed

    @property
    def candidate_q_target(self) -> np.ndarray:
        return self.raw_candidate_q_target

    def require_live_integrity(
        self,
        selector_bundle: RecoverySelectorBundle,
    ) -> None:
        """Prove this decision was emitted intact by the inference boundary."""
        if self._inference_token is not _RECOVERY_INFERENCE_TOKEN:
            raise ValueError(
                "recovery decision proof must come from "
                "run_recovery_qsafe_inference")
        if self._proof_live_sha256 != _proof_live_sha256(self):
            raise ValueError("recovery decision proof mutated after inference")
        if not isinstance(selector_bundle, RecoverySelectorBundle):
            raise TypeError("selector_bundle must be RecoverySelectorBundle")
        checked_bundle = selector_bundle.validated()
        if self.selector_bundle_sha256 != checked_bundle.bundle_sha256:
            raise ValueError("decision proof selector bundle differs")
        recomputed = select_recovery_program(
            self.member_risk,
            candidate_requested=self.raw_candidate_requested,
            candidate_executed=self.raw_candidate_executed,
            candidate_q_target=self.raw_candidate_q_target,
            candidate_mask=self.candidate_mask,
            offsets=checked_bundle.offsets,
            config=checked_bundle.selector_config,
        )
        if _selection_live_sha256(self.selection) != (
                _selection_live_sha256(recomputed)):
            raise ValueError(
                "decision proof selection does not follow the frozen bundle")


def _readonly(value: np.ndarray, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _digest_array(
    digest: Any,
    name: str,
    value: Any,
) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"decision proof {name} must be a numpy array")
    array = np.ascontiguousarray(value)
    digest.update(name.encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes(order="C"))


def _selection_live_sha256(selection: RecoverySelection) -> str:
    if type(selection) is not RecoverySelection:
        raise TypeError("decision proof selection must be RecoverySelection")
    digest = hashlib.sha256(b"qsafe.recovery_selection.live.v1\0")
    scalars = (
        int(selection.selected_index),
        bool(selection.intervened),
        str(selection.reason),
        float(selection.nominal_risk_lcb),
    )
    digest.update(repr(scalars).encode("ascii"))
    for name in (
        "risk_mean", "risk_std", "risk_ucb", "benefit_mean",
        "benefit_lcb", "action_delta_rms", "q_target_delta_rms",
        "eligible",
    ):
        _digest_array(digest, name, getattr(selection, name))
    return digest.hexdigest()


def _proof_live_sha256(proof: RecoveryQSafeInference) -> str:
    digest = hashlib.sha256(b"qsafe.recovery_inference.live.v1\0")
    for name in (
        "feature_contract_sha256",
        "recovery_library_fingerprint_sha256",
        "selector_bundle_sha256",
        "observation_history_sha256",
        "artifact_manifest_sha256",
    ):
        value = getattr(proof, name)
        if not isinstance(value, str):
            raise ValueError(f"decision proof {name} must be a string")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(value.encode("ascii") + b"\0")
    digest.update(_selection_live_sha256(proof.selection).encode("ascii"))
    for name in (
        "member_risk", "member_state_risk",
        "normalized_observation_history", "nominal_action_features",
        "candidate_action_features", "candidate_mask",
        "raw_candidate_requested", "raw_candidate_executed",
        "raw_candidate_q_target",
    ):
        _digest_array(digest, name, getattr(proof, name))
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _artifact_contract(
    artifact: LoadedQSafeArtifact,
    expected_command_speed_mps: Any,
    selector_bundle: RecoverySelectorBundle,
) -> tuple[dict[str, Any], str, float]:
    if not isinstance(artifact, LoadedQSafeArtifact):
        raise TypeError("artifact must be a LoadedQSafeArtifact")
    artifact.require_live_integrity()
    manifest = artifact.manifest
    if not isinstance(manifest, Mapping):
        raise ValueError("artifact manifest must be a mapping")
    config = artifact.network_config
    if manifest.get("feature_view") != "deployable" or config.privileged_dim != 0 or (
            artifact.normalization.privileged_mean is not None):
        raise ValueError("recovery Q_safe runtime requires a deployable artifact")
    if config.observation_dim != 46 or config.history_frames != 5 or (
            config.action_mode != "selective_advantage"):
        raise ValueError("artifact network contract is not V4 deployable Q_safe")
    if artifact.action_view != RECOVERY_PROGRAM_VIEW or tuple(
            artifact.action_components) != _ACTION_COMPONENTS or (
            config.action_dim != RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM):
        raise ValueError("artifact action contract is not recovery_program_v1")

    provenance = manifest.get("provenance")
    contract = manifest.get("action_feature_contract")
    if not isinstance(provenance, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("artifact recovery provenance is incomplete")
    feature_manifest = provenance.get("recovery_program_feature_contract")
    recovery_binding = provenance.get("recovery_program")
    if not isinstance(feature_manifest, Mapping) or not isinstance(
            recovery_binding, Mapping):
        raise ValueError("artifact recovery feature/library binding is incomplete")
    library_fingerprint = validate_recovery_program_binding(recovery_binding)
    feature_fingerprint = feature_manifest.get("feature_contract_sha256")
    serialized_bundle = provenance.get("recovery_selector_bundle")
    artifact_bundle_sha256 = provenance.get(
        "recovery_selector_bundle_sha256")
    if not isinstance(serialized_bundle, Mapping):
        raise ValueError("artifact recovery selector bundle is absent")
    try:
        artifact_bundle = RecoverySelectorBundle.from_dict(serialized_bundle)
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact recovery selector bundle is invalid") from exc
    if artifact_bundle_sha256 != artifact_bundle.bundle_sha256:
        raise ValueError("artifact recovery selector bundle hash disagrees")
    expected_action_contract = {
        "view": RECOVERY_PROGRAM_VIEW,
        "components_in_order": list(_ACTION_COMPONENTS),
        "total_width": RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
        "feature_contract_sha256": feature_fingerprint,
        "recovery_library_fingerprint_sha256": library_fingerprint,
        "recovery_selector_bundle_sha256": artifact_bundle.bundle_sha256,
    }
    if dict(contract) != expected_action_contract:
        raise ValueError("artifact action feature/library fingerprints disagree")

    checked_bundle = selector_bundle.validated()
    if artifact_bundle.to_dict() != checked_bundle.to_dict() or (
            artifact_bundle_sha256 != checked_bundle.bundle_sha256):
        raise ValueError(
            "runtime selector bundle differs from Q_safe artifact provenance")
    artifact_speed = _finite_float(
        provenance.get("command_vx"), "artifact command_vx")
    expected_speed = _finite_float(
        expected_command_speed_mps, "expected_command_speed_mps")
    if not isclose(
            artifact_speed, expected_speed, rel_tol=0.0, abs_tol=_SPEED_ATOL):
        raise ValueError("recovery Q_safe command speed mismatch")
    if artifact.normalization.observation_mean.shape != (46,) or (
            artifact.normalization.observation_std.shape != (46,)):
        raise ValueError("artifact normalization does not match 46D observation")
    return dict(feature_manifest), str(library_fingerprint), expected_speed


def _history_sha256(history: np.ndarray) -> str:
    canonical = np.ascontiguousarray(history, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    tensors = list(module.parameters()) + list(module.buffers())
    if not tensors:
        return torch.device("cpu"), torch.float32
    devices = {value.device for value in tensors}
    if len(devices) != 1 or next(iter(devices)).type == "meta":
        raise ValueError("Q_safe ensemble must occupy one real device")
    floating = {value.dtype for value in tensors if value.is_floating_point()}
    if len(floating) > 1:
        raise ValueError("Q_safe ensemble uses mixed floating dtypes")
    return next(iter(devices)), next(iter(floating), torch.float32)


def run_recovery_qsafe_inference(
    artifact: LoadedQSafeArtifact,
    observation_history: np.ndarray,
    *,
    candidate_requested: np.ndarray,
    candidate_executed: np.ndarray,
    candidate_q_target: np.ndarray,
    candidate_names: np.ndarray,
    candidate_behavior_steps: np.ndarray,
    candidate_mask: np.ndarray,
    recovery_library_fingerprint_sha256: str,
    selector_bundle: RecoverySelectorBundle,
    expected_command_speed_mps: float,
) -> RecoveryQSafeInference:
    """Evaluate the frozen ensemble and select one persistent K9 program."""
    if not isinstance(selector_bundle, RecoverySelectorBundle):
        raise TypeError("selector_bundle must be a RecoverySelectorBundle")
    checked_bundle = selector_bundle.validated()
    feature_manifest, artifact_library_fingerprint, _ = _artifact_contract(
        artifact, expected_command_speed_mps, checked_bundle)
    artifact_identity = artifact.claim_identity_sha256
    if recovery_library_fingerprint_sha256 != artifact_library_fingerprint:
        raise ValueError("runtime recovery library differs from Q_safe artifact")
    history = np.asarray(observation_history)
    if history.dtype != np.dtype(np.float32) or history.shape != _HISTORY_SHAPE or (
            not np.all(np.isfinite(history))):
        raise ValueError("observation_history must be finite float32 [5,46]")
    runtime_mask = np.asarray(candidate_mask)
    if runtime_mask.dtype != np.dtype(np.bool_) or runtime_mask.shape != (
            RECOVERY_PROGRAM_CANDIDATE_COUNT,) or not bool(
            np.all(runtime_mask)):
        raise ValueError("recovery Q_safe inference requires complete K9 support")
    features = build_recovery_program_features(
        candidate_requested=candidate_requested,
        candidate_executed=candidate_executed,
        candidate_q_target=candidate_q_target,
        candidate_names=candidate_names,
        candidate_behavior_steps=candidate_behavior_steps,
        candidate_mask=candidate_mask,
        nominal_index=0,
        feature_manifest=feature_manifest,
        feature_manifest_fingerprint_sha256=feature_manifest.get(
            "feature_contract_sha256"),
        recovery_library_fingerprint_sha256=(
            recovery_library_fingerprint_sha256),
    )
    normalized = (
        history - artifact.normalization.observation_mean[None, :]
    ) / artifact.normalization.observation_std[None, :]
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized observation history is non-finite")

    device, dtype = _device_dtype(artifact.ensemble)
    with torch.inference_mode():
        prediction = artifact.ensemble.predict(
            torch.as_tensor(normalized[None], device=device, dtype=dtype),
            torch.as_tensor(
                features.nominal_descriptor[None].copy(),
                device=device, dtype=dtype),
            torch.as_tensor(
                features.candidate_descriptor[None].copy(),
                device=device, dtype=dtype),
        )
    if not isinstance(prediction.member_risk, torch.Tensor) or not isinstance(
            prediction.member_state_risk, torch.Tensor):
        raise ValueError("Q_safe ensemble returned malformed predictions")
    member_risk = prediction.member_risk.detach().to(
        "cpu", torch.float64).numpy()
    member_state = prediction.member_state_risk.detach().to(
        "cpu", torch.float64).numpy()
    if member_risk.ndim != 3 or member_risk.shape[1:] != (
            1, RECOVERY_PROGRAM_CANDIDATE_COUNT) or member_risk.shape[0] < 2:
        raise ValueError("Q_safe member_risk must have shape [M>=2,1,9]")
    if member_state.shape != (member_risk.shape[0], 1):
        raise ValueError("Q_safe member_state_risk must have shape [M,1]")
    member_risk = member_risk[:, 0]
    member_state = member_state[:, 0]
    if not np.allclose(
            member_risk[:, 0], member_state, rtol=1e-5, atol=1e-6):
        raise ValueError("candidate zero is not centered on state risk")
    selection = select_recovery_program(
        member_risk,
        candidate_requested=candidate_requested,
        candidate_executed=candidate_executed,
        candidate_q_target=candidate_q_target,
        candidate_mask=features.candidate_mask,
        offsets=checked_bundle.offsets,
        config=checked_bundle.selector_config,
    )
    artifact.require_live_integrity()
    if artifact.claim_identity_sha256 != artifact_identity:
        raise ValueError("Q_safe artifact identity changed during inference")
    proof = RecoveryQSafeInference(
        selection=selection,
        member_risk=_readonly(member_risk, np.dtype(np.float64)),
        member_state_risk=_readonly(member_state, np.dtype(np.float64)),
        normalized_observation_history=_readonly(
            normalized, np.dtype(np.float32)),
        nominal_action_features=_readonly(
            features.nominal_descriptor, np.dtype(np.float32)),
        candidate_action_features=_readonly(
            features.candidate_descriptor, np.dtype(np.float32)),
        candidate_mask=_readonly(features.candidate_mask, np.dtype(np.bool_)),
        feature_contract_sha256=features.feature_contract_sha256,
        recovery_library_fingerprint_sha256=(
            features.recovery_library_fingerprint_sha256),
        selector_bundle_sha256=checked_bundle.bundle_sha256,
        raw_candidate_requested=_readonly(
            candidate_requested, np.dtype(np.float32)),
        raw_candidate_executed=_readonly(
            candidate_executed, np.dtype(np.float32)),
        raw_candidate_q_target=_readonly(
            candidate_q_target, np.dtype(np.float32)),
        observation_history_sha256=_history_sha256(history),
        artifact_manifest_sha256=artifact_identity,
        _inference_token=_RECOVERY_INFERENCE_TOKEN,
    )
    return replace(proof, _proof_live_sha256=_proof_live_sha256(proof))


__all__ = ["RecoveryQSafeInference", "run_recovery_qsafe_inference"]
