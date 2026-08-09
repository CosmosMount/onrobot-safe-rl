"""Immutable option-aware action features for the preregistered K9 library.

The first action of a persistent recovery behavior is not its complete action
identity.  In particular, the three mature-actor candidates have identical
step-zero application tuples but different durations.  This module therefore
owns the one feature construction used by both grouped training data and the
runtime adapter:

``candidate program`` (46D)
    requested[12] || executed[12] || q_target[12] || behavior one-hot[9]
    || behavior_steps / 96

``model descriptor`` (82D)
    current nominal application tuple[36] || candidate program[46]

All validation is deliberately fail-closed.  The K9 order is semantic, not an
exchangeable set axis, so even a joint permutation must be rejected.  The
builder performs no I/O, consumes no RNG, and always returns detached,
read-only ``float32`` arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import re
from typing import Any, Mapping

import numpy as np

from safety_data.recovery_behaviors import RecoveryBehaviorConfig


RECOVERY_PROGRAM_FEATURE_SCHEMA_VERSION = (
    "qsafe.recovery_program_features.v1")
RECOVERY_PROGRAM_VIEW = "recovery_program_v1"
RECOVERY_PROGRAM_PROTOCOL_VERSION = (
    "qsafe.closed_loop_recovery_behaviors.v3")

RECOVERY_PROGRAM_NAMES = (
    "nominal",
    "mature_actor_L10",
    "mature_actor_L25",
    "mature_actor_L50",
    "joint_brake_L10",
    "halfway_neutral_L10",
    "halfway_neutral_L25",
    "ramp_neutral_L25",
    "ramp_crouch_L25",
)
RECOVERY_PROGRAM_BEHAVIOR_STEPS = (0, 10, 25, 50, 10, 10, 25, 25, 25)
RECOVERY_PROGRAM_HORIZON_STEPS = 96
RECOVERY_PROGRAM_NOMINAL_INDEX = 0
RECOVERY_PROGRAM_CANDIDATE_COUNT = 9
RECOVERY_PROGRAM_JOINT_WIDTH = 12
RECOVERY_PROGRAM_APPLICATION_DIM = 36
RECOVERY_PROGRAM_CANDIDATE_DIM = 46
RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM = 82
RECOVERY_PROGRAM_LIBRARY_FINGERPRINT_SHA256 = (
    "fcfb1fa541acf316f87dacf82b1fdeb9188d7a4b9df7f69544b567fb2c5d1045")

_APPLICATION_COMPONENTS = ("requested", "executed", "q_target")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

if len(RECOVERY_PROGRAM_NAMES) != RECOVERY_PROGRAM_CANDIDATE_COUNT or len(
        RECOVERY_PROGRAM_BEHAVIOR_STEPS) != RECOVERY_PROGRAM_CANDIDATE_COUNT:
    raise AssertionError("K9 recovery-program constants disagree")
if RECOVERY_PROGRAM_APPLICATION_DIM != (
        len(_APPLICATION_COMPONENTS) * RECOVERY_PROGRAM_JOINT_WIDTH):
    raise AssertionError("application feature width is inconsistent")
if RECOVERY_PROGRAM_CANDIDATE_DIM != (
        RECOVERY_PROGRAM_APPLICATION_DIM
        + RECOVERY_PROGRAM_CANDIDATE_COUNT
        + 1):
    raise AssertionError("candidate-program feature width is inconsistent")
if RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM != (
        RECOVERY_PROGRAM_APPLICATION_DIM + RECOVERY_PROGRAM_CANDIDATE_DIM):
    raise AssertionError("model descriptor width is inconsistent")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("feature manifest must be canonical JSON") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checked_fingerprint(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _manifest_payload(
    recovery_library_fingerprint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECOVERY_PROGRAM_FEATURE_SCHEMA_VERSION,
        "view": RECOVERY_PROGRAM_VIEW,
        "candidate_count": RECOVERY_PROGRAM_CANDIDATE_COUNT,
        "nominal_index": RECOVERY_PROGRAM_NOMINAL_INDEX,
        "ordered_names": list(RECOVERY_PROGRAM_NAMES),
        "behavior_steps_array": "candidate_behavior_steps",
        "behavior_steps": list(RECOVERY_PROGRAM_BEHAVIOR_STEPS),
        "horizon_steps": RECOVERY_PROGRAM_HORIZON_STEPS,
        "duration_denominator": RECOVERY_PROGRAM_HORIZON_STEPS,
        "application_components_in_order": list(_APPLICATION_COMPONENTS),
        "joint_width_per_component": RECOVERY_PROGRAM_JOINT_WIDTH,
        "candidate_program_components_in_order": [
            "application_tuple",
            "behavior_id_one_hot",
            "behavior_steps_over_horizon",
        ],
        "candidate_program_width": RECOVERY_PROGRAM_CANDIDATE_DIM,
        "model_descriptor_components_in_order": [
            "common_nominal_application_tuple",
            "candidate_recovery_program",
        ],
        "model_descriptor_width": RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
        "execution_binding": {
            "candidate_protocol_version": RECOVERY_PROGRAM_PROTOCOL_VERSION,
            "program_identity": "ordered_name_and_behavior_steps",
            "duration_unit": "policy_step",
            "reselection_during_option": "forbidden",
        },
        "candidate_mask_contract": "all_K9_preview_candidates_valid",
        "recovery_library_fingerprint_sha256": (
            recovery_library_fingerprint_sha256),
    }


def make_recovery_program_feature_manifest(
    recovery_library_fingerprint_sha256: str,
) -> dict[str, Any]:
    """Return the complete self-fingerprinted V4 feature contract.

    The recovery-library fingerprint is supplied by the collector/artifact
    boundary.  Callers must also pass that independently bound value to the
    feature builder, preventing a manifest from silently substituting another
    recovery implementation.
    """
    library_fingerprint = _checked_fingerprint(
        recovery_library_fingerprint_sha256,
        "recovery_library_fingerprint_sha256",
    )
    if library_fingerprint != RECOVERY_PROGRAM_LIBRARY_FINGERPRINT_SHA256:
        raise ValueError(
            "recovery_library_fingerprint_sha256 differs from the locked "
            "mature recovery library")
    payload = _manifest_payload(library_fingerprint)
    return payload | {"feature_contract_sha256": _canonical_sha256(payload)}


def bind_recovery_program_manifest(
    recovery_program_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a canonical full-manifest/fingerprint runtime binding."""
    if not isinstance(recovery_program_manifest, Mapping):
        raise TypeError("recovery_program_manifest must be a mapping")
    manifest = copy.deepcopy(dict(recovery_program_manifest))
    expected_keys = {
        "candidate_protocol",
        "candidate_protocol_sha256",
        "mature_policy_identity",
        "action_projection",
        "input_boundary",
        "privileged_inputs",
    }
    if set(manifest) != expected_keys:
        raise ValueError("recovery program manifest fields are incomplete")
    candidate_protocol = RecoveryBehaviorConfig().manifest_protocol()
    if manifest.get("candidate_protocol") != candidate_protocol or (
            manifest.get("candidate_protocol_sha256")
            != RecoveryBehaviorConfig().protocol_sha256()):
        raise ValueError("recovery program candidate protocol has drifted")
    if not isinstance(manifest.get("mature_policy_identity"), Mapping) or not (
            manifest["mature_policy_identity"]):
        raise ValueError("recovery program mature-policy identity is required")
    if not isinstance(manifest.get("action_projection"), Mapping) or not (
            manifest["action_projection"]):
        raise ValueError("recovery program action projection is required")
    if manifest.get("input_boundary") != "corrected_deployable_5x46_only" or (
            manifest.get("privileged_inputs") != "forbidden"):
        raise ValueError("recovery program deployable input boundary has drifted")
    fingerprint = _canonical_sha256(manifest)
    if fingerprint != RECOVERY_PROGRAM_LIBRARY_FINGERPRINT_SHA256:
        raise ValueError(
            "recovery program manifest does not match the preregistered "
            "mature-policy/action-projection library")
    return {
        "manifest": manifest,
        "fingerprint_sha256": fingerprint,
    }


def validate_recovery_program_binding(
    recovery_program_binding: Mapping[str, Any],
) -> str:
    """Validate a full recovery-program binding and return its fingerprint."""
    if not isinstance(recovery_program_binding, Mapping) or set(
            recovery_program_binding) != {"manifest", "fingerprint_sha256"}:
        raise ValueError(
            "recovery program binding must contain exact manifest/fingerprint")
    expected = bind_recovery_program_manifest(
        recovery_program_binding.get("manifest"))
    fingerprint = _checked_fingerprint(
        recovery_program_binding.get("fingerprint_sha256"),
        "recovery_program.fingerprint_sha256",
    )
    if dict(recovery_program_binding) != expected or fingerprint != expected[
            "fingerprint_sha256"]:
        raise ValueError("recovery program full-manifest fingerprint mismatch")
    return fingerprint


@dataclass(frozen=True)
class RecoveryProgramFeatures:
    """Validated single-state or batched K9 feature arrays."""

    candidate_program: np.ndarray
    nominal_descriptor: np.ndarray
    candidate_descriptor: np.ndarray
    candidate_mask: np.ndarray
    feature_contract_sha256: str
    recovery_library_fingerprint_sha256: str


def _require_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    return value


def _checked_action(value: Any, name: str) -> np.ndarray:
    array = _require_array(value, name)
    if array.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must have dtype float32")
    if array.ndim not in (2, 3) or array.shape[-2:] != (
            RECOVERY_PROGRAM_CANDIDATE_COUNT,
            RECOVERY_PROGRAM_JOINT_WIDTH,
    ):
        raise ValueError(f"{name} must have shape [9,12] or [B,9,12]")
    return array


def _validate_manifest(
    feature_manifest: Any,
    *,
    feature_manifest_fingerprint_sha256: Any,
    recovery_library_fingerprint_sha256: Any,
) -> tuple[str, str]:
    if not isinstance(feature_manifest, Mapping):
        raise TypeError("feature_manifest must be a mapping")
    library_fingerprint = _checked_fingerprint(
        recovery_library_fingerprint_sha256,
        "recovery_library_fingerprint_sha256",
    )
    expected_fingerprint = _checked_fingerprint(
        feature_manifest_fingerprint_sha256,
        "feature_manifest_fingerprint_sha256",
    )
    expected_manifest = make_recovery_program_feature_manifest(
        library_fingerprint)
    if set(feature_manifest) != set(expected_manifest):
        raise ValueError(
            "feature_manifest does not exactly match the locked V4 schema")
    payload = {
        key: value
        for key, value in feature_manifest.items()
        if key != "feature_contract_sha256"
    }
    expected_payload = {
        key: value
        for key, value in expected_manifest.items()
        if key != "feature_contract_sha256"
    }
    if payload != expected_payload:
        raise ValueError(
            "feature_manifest does not exactly match the locked V4 schema")
    embedded_fingerprint = _checked_fingerprint(
        feature_manifest.get("feature_contract_sha256"),
        "feature_manifest.feature_contract_sha256",
    )
    recomputed = _canonical_sha256(payload)
    if embedded_fingerprint != recomputed or expected_fingerprint != recomputed:
        raise ValueError("recovery-program feature manifest fingerprint mismatch")
    return recomputed, library_fingerprint


def _readonly(array: np.ndarray, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def build_recovery_program_features(
    *,
    candidate_requested: np.ndarray,
    candidate_executed: np.ndarray,
    candidate_q_target: np.ndarray,
    candidate_names: np.ndarray,
    candidate_behavior_steps: np.ndarray,
    candidate_mask: np.ndarray,
    nominal_index: int,
    feature_manifest: Mapping[str, Any],
    feature_manifest_fingerprint_sha256: str,
    recovery_library_fingerprint_sha256: str,
) -> RecoveryProgramFeatures:
    """Build the sole V4 option-aware action descriptor.

    Inputs are either one K9 state (``[9,*]``) or a batch
    (``[B,9,*]``).  Stage B is defined only for the complete previewable K9
    library, so every mask element must be true.  Names and durations remain
    mandatory for every slot and never weaken program-identity validation.
    """
    contract_fingerprint, library_fingerprint = _validate_manifest(
        feature_manifest,
        feature_manifest_fingerprint_sha256=(
            feature_manifest_fingerprint_sha256),
        recovery_library_fingerprint_sha256=(
            recovery_library_fingerprint_sha256),
    )
    if isinstance(nominal_index, (bool, np.bool_)) or not isinstance(
            nominal_index, (int, np.integer)) or int(nominal_index) != (
                RECOVERY_PROGRAM_NOMINAL_INDEX):
        raise ValueError("nominal_index must be the locked K9 index 0")

    requested = _checked_action(candidate_requested, "candidate_requested")
    executed = _checked_action(candidate_executed, "candidate_executed")
    q_target = _checked_action(candidate_q_target, "candidate_q_target")
    if requested.shape != executed.shape or requested.shape != q_target.shape:
        raise ValueError("candidate application arrays must have identical shapes")
    prefix = requested.shape[:-2]
    candidate_shape = prefix + (RECOVERY_PROGRAM_CANDIDATE_COUNT,)

    mask = _require_array(candidate_mask, "candidate_mask")
    if mask.dtype != np.dtype(np.bool_) or mask.shape != candidate_shape:
        raise ValueError(
            "candidate_mask must be boolean with shape [9] or [B,9]")
    if not np.all(mask):
        raise ValueError("candidate_mask must keep every locked K9 program valid")

    names = _require_array(candidate_names, "candidate_names")
    if names.dtype.kind != "U" or names.shape != candidate_shape:
        raise ValueError(
            "candidate_names must be a unicode array with shape [9] or [B,9]")
    expected_names = np.asarray(RECOVERY_PROGRAM_NAMES, dtype=names.dtype)
    if not np.array_equal(
            names,
            np.broadcast_to(expected_names, candidate_shape),
            equal_nan=False):
        raise ValueError("candidate_names differs from the locked K9 order")

    steps = _require_array(
        candidate_behavior_steps, "candidate_behavior_steps")
    if steps.dtype.kind not in "iu" or steps.shape != candidate_shape:
        raise ValueError(
            "candidate_behavior_steps must be an integer array with shape "
            "[9] or [B,9]")
    expected_steps = np.asarray(
        RECOVERY_PROGRAM_BEHAVIOR_STEPS, dtype=steps.dtype)
    if not np.array_equal(
            steps,
            np.broadcast_to(expected_steps, candidate_shape),
            equal_nan=False):
        raise ValueError(
            "candidate_behavior_steps differs from the locked K9 durations")

    for name, action in (
        ("candidate_requested", requested),
        ("candidate_executed", executed),
        ("candidate_q_target", q_target),
    ):
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{name} must contain only finite values")
    for name, action in (
        ("candidate_requested", requested),
        ("candidate_executed", executed),
    ):
        if np.any(action < -1.0 - 1e-6) or np.any(
                action > 1.0 + 1e-6):
            raise ValueError(f"{name} must lie in [-1,1]")

    application = np.concatenate(
        (requested, executed, q_target), axis=-1, dtype=np.float32)
    identity = np.eye(
        RECOVERY_PROGRAM_CANDIDATE_COUNT, dtype=np.float32)
    identity = np.broadcast_to(
        identity, prefix + identity.shape)
    duration = steps.astype(np.float32, copy=False)[..., None] / np.float32(
        RECOVERY_PROGRAM_HORIZON_STEPS)
    raw_program = np.concatenate(
        (application, identity, duration), axis=-1, dtype=np.float32)

    nominal_application = application[..., RECOVERY_PROGRAM_NOMINAL_INDEX, :]
    nominal_application_dense = np.broadcast_to(
        nominal_application[..., None, :],
        prefix
        + (
            RECOVERY_PROGRAM_CANDIDATE_COUNT,
            RECOVERY_PROGRAM_APPLICATION_DIM,
        ),
    )
    raw_descriptor = np.concatenate(
        (nominal_application_dense, raw_program), axis=-1, dtype=np.float32)
    nominal_descriptor = raw_descriptor[
        ..., RECOVERY_PROGRAM_NOMINAL_INDEX, :]

    candidate_descriptor = raw_descriptor.astype(np.float32, copy=False)
    candidate_program = raw_program.astype(np.float32, copy=False)

    if candidate_program.shape[-1] != RECOVERY_PROGRAM_CANDIDATE_DIM or (
            candidate_descriptor.shape[-1]
            != RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM):
        raise AssertionError("recovery-program builder produced a wrong width")
    if not np.all(np.isfinite(candidate_program)) or not np.all(
            np.isfinite(candidate_descriptor)):
        raise ValueError("recovery-program features contain non-finite values")
    if not np.array_equal(
            candidate_descriptor[..., RECOVERY_PROGRAM_NOMINAL_INDEX, :],
            nominal_descriptor,
            equal_nan=False):
        raise AssertionError("candidate zero is not centered on nominal")

    return RecoveryProgramFeatures(
        candidate_program=_readonly(candidate_program, np.dtype(np.float32)),
        nominal_descriptor=_readonly(
            nominal_descriptor, np.dtype(np.float32)),
        candidate_descriptor=_readonly(
            candidate_descriptor, np.dtype(np.float32)),
        candidate_mask=_readonly(mask, np.dtype(np.bool_)),
        feature_contract_sha256=contract_fingerprint,
        recovery_library_fingerprint_sha256=library_fingerprint,
    )


__all__ = [
    "RECOVERY_PROGRAM_APPLICATION_DIM",
    "RECOVERY_PROGRAM_BEHAVIOR_STEPS",
    "RECOVERY_PROGRAM_CANDIDATE_COUNT",
    "RECOVERY_PROGRAM_CANDIDATE_DIM",
    "RECOVERY_PROGRAM_FEATURE_SCHEMA_VERSION",
    "RECOVERY_PROGRAM_HORIZON_STEPS",
    "RECOVERY_PROGRAM_LIBRARY_FINGERPRINT_SHA256",
    "RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM",
    "RECOVERY_PROGRAM_NAMES",
    "RECOVERY_PROGRAM_NOMINAL_INDEX",
    "RECOVERY_PROGRAM_PROTOCOL_VERSION",
    "RECOVERY_PROGRAM_VIEW",
    "RecoveryProgramFeatures",
    "bind_recovery_program_manifest",
    "build_recovery_program_features",
    "make_recovery_program_feature_manifest",
    "validate_recovery_program_binding",
]
