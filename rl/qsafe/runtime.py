"""Deployable, single-step Selective Advantage Q_safe inference.

This module is deliberately a narrow adapter between a hash-verified
``LoadedQSafeArtifact``, the fixed evidence ``CandidateSet``, and the
fail-closed selector.  It reproduces the training feature construction:

* normalize the five 46-D observation frames with train-fitted statistics;
* concatenate requested, executed, and absolute q-target actions in that
  exact order; and
* use candidate zero's complete application tuple as the nominal action.

No random sampling is performed here.  Candidate and reward-Q generation are
owned by their callers so repeated closed-loop comparisons can make every RNG
stream explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from math import isclose
from typing import Any, Mapping

import numpy as np
import torch

from rl.qsafe.artifact import LoadedQSafeArtifact
from rl.qsafe.selector import (
    CandidateBatch,
    SelectionResult,
    SelectorConfig,
    select_candidate,
)
from safety_data.candidates import CANDIDATE_COUNT, CandidateSet


_DEPLOYABLE_FEATURE_VIEW = "deployable"
_ACTION_VIEW = "application_concat"
_ACTION_COMPONENTS = ("requested", "executed", "q_target")
_JOINT_WIDTH = 12
_ACTION_DIM = len(_ACTION_COMPONENTS) * _JOINT_WIDTH
_OBSERVATION_DIM = 46
_HISTORY_FRAMES = 5
_SPEED_ATOL_MPS = 1e-6


@dataclass(frozen=True)
class QSafeRuntimeResult:
    """Selected request plus all inputs and diagnostics needed for an audit."""

    selected_requested_action: np.ndarray
    selection: SelectionResult
    member_risk: np.ndarray
    member_state_risk: np.ndarray
    normalized_observation_history: np.ndarray
    nominal_action_features: np.ndarray
    candidate_action_features: np.ndarray
    candidate_mask: np.ndarray
    reward_q: np.ndarray
    artifact_command_speed_mps: float
    expected_command_speed_mps: float

    @property
    def selected_index(self) -> int:
        return self.selection.selected_index

    @property
    def intervened(self) -> bool:
        return self.selection.intervened


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _numeric_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _readonly(value: np.ndarray, *, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _freeze_selection(selection: SelectionResult) -> SelectionResult:
    updates: dict[str, np.ndarray] = {}
    for field in fields(selection):
        value = getattr(selection, field.name)
        if isinstance(value, np.ndarray):
            updates[field.name] = _readonly(value)
    return replace(selection, **updates)


def _validate_artifact(
    artifact: LoadedQSafeArtifact,
    expected_command_speed_mps: Any,
) -> tuple[float, float]:
    if not isinstance(artifact, LoadedQSafeArtifact):
        raise TypeError("artifact must be a LoadedQSafeArtifact")
    config = artifact.network_config
    normalization = artifact.normalization
    manifest = artifact.manifest
    if not isinstance(manifest, Mapping):
        raise ValueError("Q_safe artifact manifest must be a mapping")

    if manifest.get("feature_view") != _DEPLOYABLE_FEATURE_VIEW or (
        config.privileged_dim != 0
    ) or normalization.privileged_mean is not None or (
        normalization.privileged_std is not None
    ):
        raise ValueError("privileged Q_safe artifacts cannot run in deployment")

    if (
        config.observation_dim != _OBSERVATION_DIM
        or config.history_frames != _HISTORY_FRAMES
    ):
        raise ValueError("artifact observation contract does not match [5,46]")
    if config.action_mode != "selective_advantage":
        raise ValueError("deployment requires a selective_advantage Q_safe artifact")
    if (
        artifact.action_view != _ACTION_VIEW
        or tuple(artifact.action_components) != _ACTION_COMPONENTS
        or config.action_dim != _ACTION_DIM
    ):
        raise ValueError("artifact action contract is not requested/executed/q_target")

    expected_contract = {
        "view": _ACTION_VIEW,
        "components_in_order": list(_ACTION_COMPONENTS),
        "joint_width_per_component": _JOINT_WIDTH,
        "total_width": _ACTION_DIM,
    }
    manifest_contract = manifest.get("action_feature_contract")
    if manifest_contract != expected_contract:
        raise ValueError("artifact action feature contract is inconsistent")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("artifact provenance is required for command-speed checks")
    provenance_contract = provenance.get("action_feature_contract")
    if provenance_contract is not None:
        # Trainer provenance predates the joint-width field, so compare its
        # declared fields while still rejecting any causal mismatch.
        if not isinstance(provenance_contract, Mapping) or (
            provenance_contract.get("view") != _ACTION_VIEW
            or provenance_contract.get("components_in_order")
            != list(_ACTION_COMPONENTS)
            or provenance_contract.get("total_width") != _ACTION_DIM
        ):
            raise ValueError("artifact provenance action contract is inconsistent")

    if "command_vx" not in provenance:
        raise ValueError("artifact provenance has no command_vx")
    artifact_speed = _finite_float(
        provenance["command_vx"], "artifact provenance command_vx")
    expected_speed = _finite_float(
        expected_command_speed_mps, "expected_command_speed_mps")
    if not isclose(
        artifact_speed,
        expected_speed,
        rel_tol=0.0,
        abs_tol=_SPEED_ATOL_MPS,
    ):
        raise ValueError(
            "Q_safe command speed mismatch: artifact was trained at "
            f"{artifact_speed:.9g} m/s, runtime expects "
            f"{expected_speed:.9g} m/s"
        )

    if normalization.observation_mean.shape != (_OBSERVATION_DIM,) or (
        normalization.observation_std.shape != (_OBSERVATION_DIM,)
    ):
        raise ValueError("artifact normalization does not match 46D observation")
    return artifact_speed, expected_speed


def _ensemble_device_and_dtype(ensemble: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    tensors = list(ensemble.parameters()) + list(ensemble.buffers())
    if not tensors:
        return torch.device("cpu"), torch.float32
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError("Q_safe ensemble parameters span multiple devices")
    device = next(iter(devices))
    if device.type == "meta":
        raise ValueError("Q_safe ensemble cannot run from the meta device")
    floating_dtypes = {tensor.dtype for tensor in tensors if tensor.is_floating_point()}
    if len(floating_dtypes) > 1:
        raise ValueError("Q_safe ensemble parameters use mixed floating dtypes")
    dtype = next(iter(floating_dtypes), torch.float32)
    return device, dtype


def _validate_candidates(candidates: CandidateSet) -> tuple[np.ndarray, ...]:
    if not isinstance(candidates, CandidateSet):
        raise TypeError("candidates must be a CandidateSet")
    requested = _numeric_array(
        candidates.requested,
        name="candidate requested action",
        shape=(CANDIDATE_COUNT, _JOINT_WIDTH),
    )
    executed = _numeric_array(
        candidates.executed,
        name="candidate executed action",
        shape=(CANDIDATE_COUNT, _JOINT_WIDTH),
    )
    q_target = _numeric_array(
        candidates.q_target,
        name="candidate q_target action",
        shape=(CANDIDATE_COUNT, _JOINT_WIDTH),
    )
    if np.any(np.abs(requested) > 1.0 + 1e-6) or np.any(
        np.abs(executed) > 1.0 + 1e-6
    ):
        raise ValueError("requested and executed candidates must lie in [-1,1]")

    mask = np.asarray(candidates.mask)
    if mask.shape != (CANDIDATE_COUNT,) or mask.dtype != np.bool_:
        raise ValueError(
            f"candidate mask must be boolean shape ({CANDIDATE_COUNT},)"
        )
    mask = mask.astype(bool, copy=True)
    if not mask[0]:
        raise ValueError("nominal candidate zero must not be masked")
    minimum = candidates.manifest_protocol.get("minimum_unique_candidates")
    if isinstance(minimum, (bool, np.bool_)) or not isinstance(
        minimum, (int, np.integer)
    ):
        raise ValueError("candidate protocol has no valid minimum support")
    if not 1 <= int(minimum) <= CANDIDATE_COUNT or np.count_nonzero(mask) < int(
        minimum
    ):
        raise ValueError("candidate mask violates the minimum support contract")
    return requested, executed, q_target, mask


def run_qsafe_step(
    artifact: LoadedQSafeArtifact,
    observation_history: np.ndarray,
    candidates: CandidateSet,
    reward_q: np.ndarray,
    selector_config: SelectorConfig,
    *,
    expected_command_speed_mps: float,
) -> QSafeRuntimeResult:
    """Run one deterministic, deployable Q_safe selection step.

    ``expected_command_speed_mps`` is the current controller command and must
    match the artifact's single-speed training provenance.  Any malformed or
    non-finite input raises before an action can be returned.  Valid but
    unsupported/uncertain alternatives remain a normal fail-closed selector
    result whose selected action is candidate zero.
    """
    if not isinstance(selector_config, SelectorConfig):
        raise TypeError("selector_config must be a SelectorConfig")
    artifact_speed, expected_speed = _validate_artifact(
        artifact, expected_command_speed_mps)
    history = _numeric_array(
        observation_history,
        name="observation_history",
        shape=(_HISTORY_FRAMES, _OBSERVATION_DIM),
    )
    requested, executed, q_target, mask = _validate_candidates(candidates)
    reward = _numeric_array(
        reward_q,
        name="reward_q",
        shape=(CANDIDATE_COUNT,),
        dtype=np.dtype(np.float64),
    )

    normalized_history = (
        history - artifact.normalization.observation_mean[None, :]
    ) / artifact.normalization.observation_std[None, :]
    if not np.all(np.isfinite(normalized_history)):
        raise ValueError("normalized observation history contains non-finite values")

    raw_action_features = np.concatenate(
        (requested, executed, q_target), axis=1).astype(np.float32, copy=False)
    nominal_action_features = raw_action_features[0].copy()
    # Match TorchGroupedView exactly: invalid dense slots carry the finite
    # nominal tuple into the network; the original support mask still reaches
    # the selector and therefore cannot be bypassed by a low prediction.
    model_action_features = np.where(
        mask[:, None],
        raw_action_features,
        nominal_action_features[None, :],
    ).astype(np.float32, copy=False)

    device, dtype = _ensemble_device_and_dtype(artifact.ensemble)
    history_tensor = torch.as_tensor(
        normalized_history[None, ...], device=device, dtype=dtype)
    nominal_tensor = torch.as_tensor(
        nominal_action_features[None, ...], device=device, dtype=dtype)
    candidate_tensor = torch.as_tensor(
        model_action_features[None, ...], device=device, dtype=dtype)
    with torch.inference_mode():
        prediction = artifact.ensemble.predict(
            history_tensor,
            nominal_tensor,
            candidate_tensor,
        )
    if not isinstance(prediction.member_risk, torch.Tensor) or not isinstance(
        prediction.member_state_risk, torch.Tensor
    ):
        raise ValueError("Q_safe ensemble returned non-tensor diagnostics")
    member_risk = prediction.member_risk.detach().to("cpu", torch.float64).numpy()
    member_state_risk = (
        prediction.member_state_risk.detach().to("cpu", torch.float64).numpy()
    )
    if member_risk.ndim != 3 or member_risk.shape[1:] != (
        1,
        CANDIDATE_COUNT,
    ) or member_risk.shape[0] < 2:
        raise ValueError(
            "Q_safe ensemble member_risk must have shape [M,1,16] with M>=2"
        )
    if member_state_risk.shape != (member_risk.shape[0], 1):
        raise ValueError("Q_safe ensemble member_state_risk must have shape [M,1]")
    member_risk = member_risk[:, 0, :]
    member_state_risk = member_state_risk[:, 0]
    if not np.all(np.isfinite(member_risk)) or not np.all(
        np.isfinite(member_state_risk)
    ):
        raise ValueError("Q_safe ensemble returned non-finite risk")
    if np.any((member_risk < 0.0) | (member_risk > 1.0)) or np.any(
        (member_state_risk < 0.0) | (member_state_risk > 1.0)
    ):
        raise ValueError("Q_safe ensemble returned risk outside [0,1]")
    if not np.allclose(
        member_risk[:, 0], member_state_risk, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("candidate zero is not aligned with nominal state risk")

    selection = select_candidate(
        member_risk,
        CandidateBatch(
            requested=requested,
            executed=executed,
            q_target=q_target,
            reward_q=reward,
            mask=mask,
        ),
        selector_config,
        nominal_index=0,
    )
    selection = _freeze_selection(selection)
    selected_action = _readonly(
        requested[selection.selected_index], dtype=np.dtype(np.float32))
    return QSafeRuntimeResult(
        selected_requested_action=selected_action,
        selection=selection,
        member_risk=_readonly(member_risk, dtype=np.dtype(np.float64)),
        member_state_risk=_readonly(
            member_state_risk, dtype=np.dtype(np.float64)),
        normalized_observation_history=_readonly(
            normalized_history, dtype=np.dtype(np.float32)),
        nominal_action_features=_readonly(
            nominal_action_features, dtype=np.dtype(np.float32)),
        candidate_action_features=_readonly(
            model_action_features, dtype=np.dtype(np.float32)),
        candidate_mask=_readonly(mask, dtype=np.dtype(bool)),
        reward_q=_readonly(reward, dtype=np.dtype(np.float64)),
        artifact_command_speed_mps=artifact_speed,
        expected_command_speed_mps=expected_speed,
    )


__all__ = ["QSafeRuntimeResult", "run_qsafe_step"]
