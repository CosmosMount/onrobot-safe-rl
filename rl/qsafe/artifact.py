"""Self-describing, hash-verified Selective Advantage Q_safe artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

import numpy as np
import torch

from rl.qsafe.data import NormalizationStats
from rl.qsafe.loss import QSafeLossConfig
from rl.qsafe.network import QSafeEnsemble, QSafeNetworkConfig, SelectiveAdvantageQSafe
from rl.qsafe.recovery_program import (
    RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
    RECOVERY_PROGRAM_VIEW,
    make_recovery_program_feature_manifest,
    validate_recovery_program_binding,
)
from rl.qsafe.recovery_selector import RecoverySelectorBundle
from rl.qsafe.training import (
    RECOVERY_PROGRAM_V4_LOSS_CONFIG,
    RECOVERY_PROGRAM_V4_MEMBER_SEED_STRIDE,
    RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
    RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
    QSafeTrainingConfig,
    TrainedQSafeEnsemble,
)
from safety_data.paths import (
    assert_development_path,
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)


ARTIFACT_SCHEMA_VERSION = "qsafe.selective_advantage.v1"
_LOADED_ARTIFACT_TOKEN = object()
_RECOVERY_TEMPERATURE_MIN = math.exp(-4.0)
_RECOVERY_TEMPERATURE_MAX = math.exp(4.0)
_ACTION_FEATURE_COMPONENTS = {
    "requested": ("requested",),
    "application_concat": ("requested", "executed", "q_target"),
    RECOVERY_PROGRAM_VIEW: (
        "common_current_nominal_application_tuple",
        "candidate_recovery_program_v1",
    ),
}


def _action_feature_contract(
    action_view: str,
    provenance: Mapping[str, Any],
) -> tuple[tuple[str, ...], int, dict[str, Any]]:
    if action_view not in _ACTION_FEATURE_COMPONENTS:
        raise ValueError("trained ensemble has no supported action feature contract")
    components = _ACTION_FEATURE_COMPONENTS[action_view]
    if action_view != RECOVERY_PROGRAM_VIEW:
        width = 12 * len(components)
        return components, width, {
            "view": action_view,
            "components_in_order": list(components),
            "joint_width_per_component": 12,
            "total_width": width,
        }

    recovery = provenance.get("recovery_program")
    feature = provenance.get("recovery_program_feature_contract")
    if not isinstance(recovery, Mapping) or not isinstance(feature, Mapping):
        raise ValueError(
            "recovery_program_v1 artifact provenance requires recovery-program "
            "and feature-contract bindings")
    library_fingerprint = validate_recovery_program_binding(recovery)
    expected_feature = make_recovery_program_feature_manifest(
        library_fingerprint)
    if dict(feature) != expected_feature:
        raise ValueError(
            "recovery_program_v1 artifact feature contract is inconsistent")
    serialized_selector = provenance.get("recovery_selector_bundle")
    selector_sha256 = provenance.get("recovery_selector_bundle_sha256")
    if not isinstance(serialized_selector, Mapping):
        raise ValueError(
            "recovery_program_v1 artifact provenance requires a frozen "
            "selector bundle")
    try:
        selector_bundle = RecoverySelectorBundle.from_dict(serialized_selector)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "recovery_program_v1 artifact selector bundle is invalid") from exc
    if selector_sha256 != selector_bundle.bundle_sha256:
        raise ValueError(
            "recovery_program_v1 artifact selector bundle hash disagrees")
    contract = {
        "view": action_view,
        "components_in_order": list(components),
        "total_width": RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM,
        "feature_contract_sha256": expected_feature[
            "feature_contract_sha256"],
        "recovery_library_fingerprint_sha256": library_fingerprint,
        "recovery_selector_bundle_sha256": selector_bundle.bundle_sha256,
    }
    return components, RECOVERY_PROGRAM_MODEL_DESCRIPTOR_DIM, contract


def _validate_trained_ensemble_consistency(
    trained: TrainedQSafeEnsemble,
) -> None:
    """Prove that metadata members are the exact ensemble being serialized."""
    if not isinstance(trained.ensemble, QSafeEnsemble):
        raise TypeError("trained.ensemble must be a QSafeEnsemble")
    ensemble_members = list(trained.ensemble.members)
    if not trained.members or len(trained.members) != len(ensemble_members):
        raise ValueError("trained member metadata and ensemble disagree")
    for index, (metadata, ensemble_member) in enumerate(zip(
            trained.members, ensemble_members)):
        if metadata.model is not ensemble_member:
            raise ValueError(
                "trained member model object differs from ensemble member at "
                f"index {index}")
        metadata_state = metadata.model.state_dict()
        ensemble_state = ensemble_member.state_dict()
        if metadata_state.keys() != ensemble_state.keys() or any(
                not torch.equal(metadata_state[name], ensemble_state[name])
                for name in metadata_state):
            raise ValueError(
                f"trained member weights differ from ensemble at index {index}")
    temperatures = trained.ensemble.temperatures
    expected_temperatures = torch.as_tensor(
        [member.temperature for member in trained.members],
        dtype=temperatures.dtype,
        device=temperatures.device,
    )
    if temperatures.shape != expected_temperatures.shape or not torch.equal(
            temperatures, expected_temperatures):
        raise ValueError(
            "trained member temperatures differ from ensemble temperatures")


def _validate_trained_recovery_provenance(
    trained: TrainedQSafeEnsemble,
    provenance: Mapping[str, Any],
    recovery_selector_bundle: RecoverySelectorBundle | None,
) -> None:
    fields = (
        trained.recovery_program_binding,
        trained.recovery_program_feature_manifest,
        trained.recovery_program_feature_contract_sha256,
        trained.recovery_library_fingerprint_sha256,
    )
    if trained.action_view != RECOVERY_PROGRAM_VIEW:
        if any(value is not None for value in fields):
            raise ValueError(
                "legacy trained ensemble carries recovery-program provenance")
        if recovery_selector_bundle is not None:
            raise ValueError(
                "legacy trained ensemble must not carry a recovery selector")
        return
    binding, feature, feature_sha256, library_sha256 = fields
    if not isinstance(binding, Mapping) or not isinstance(feature, Mapping) or (
            not isinstance(feature_sha256, str)) or not isinstance(
                library_sha256, str):
        raise ValueError(
            "trained recovery_program_v1 ensemble is missing exact provenance")
    validated_library = validate_recovery_program_binding(binding)
    expected_feature = make_recovery_program_feature_manifest(
        validated_library)
    if dict(feature) != expected_feature or feature_sha256 != expected_feature[
            "feature_contract_sha256"] or library_sha256 != validated_library:
        raise ValueError(
            "trained recovery-program provenance is internally inconsistent")
    supplied_binding = provenance.get("recovery_program")
    supplied_feature = provenance.get("recovery_program_feature_contract")
    if not isinstance(supplied_binding, Mapping) or not isinstance(
            supplied_feature, Mapping) or dict(supplied_binding) != dict(
                binding) or dict(supplied_feature) != dict(feature):
        raise ValueError(
            "artifact recovery provenance differs from the training view")
    if not isinstance(recovery_selector_bundle, RecoverySelectorBundle):
        raise TypeError(
            "recovery_program_v1 artifact requires RecoverySelectorBundle")
    checked_selector = recovery_selector_bundle.validated()
    supplied_selector = provenance.get("recovery_selector_bundle")
    supplied_selector_sha256 = provenance.get(
        "recovery_selector_bundle_sha256")
    if not isinstance(supplied_selector, Mapping) or dict(
            supplied_selector) != checked_selector.to_dict() or (
            supplied_selector_sha256 != checked_selector.bundle_sha256):
        raise ValueError(
            "artifact selector provenance differs from the frozen bundle")
    command_vx = provenance.get("command_vx")
    if isinstance(command_vx, (bool, np.bool_)) or not isinstance(
            command_vx, (int, float, np.integer, np.floating)) or not (
            np.isfinite(float(command_vx))) or not np.isclose(
                float(command_vx), 0.30, rtol=0.0, atol=1e-6):
        raise ValueError(
            "recovery_program_v1 artifact provenance must bind command_vx=0.30")


def _validate_recovery_training_metadata(
    trained: TrainedQSafeEnsemble,
    network_config: QSafeNetworkConfig,
    training_config: QSafeTrainingConfig,
    loss_config: QSafeLossConfig,
) -> None:
    if trained.action_view != RECOVERY_PROGRAM_VIEW:
        return
    expected = (
        RECOVERY_PROGRAM_V4_NETWORK_CONFIG,
        RECOVERY_PROGRAM_V4_TRAINING_CONFIG,
        RECOVERY_PROGRAM_V4_LOSS_CONFIG,
    )
    actual = (network_config, training_config, loss_config)
    captured = (
        trained.network_config, trained.training_config, trained.loss_config)
    if actual != expected or captured != expected or captured != actual:
        raise ValueError(
            "recovery_program_v1 artifact requires exact captured V4 "
            "network/training/loss configurations")
    if len(trained.members) != training_config.ensemble_members:
        raise ValueError("recovery Q_safe artifact must contain five members")
    for index, member in enumerate(trained.members):
        if member.seed != (
                training_config.seed
                + RECOVERY_PROGRAM_V4_MEMBER_SEED_STRIDE * index):
            raise ValueError("recovery Q_safe member seed order has drifted")
        if len(member.epoch_loss) != training_config.epochs or not all(
                np.isfinite(float(value)) and float(value) >= 0.0
                for value in member.epoch_loss):
            raise ValueError("recovery Q_safe epoch metadata has drifted")
        if not np.isfinite(member.temperature) or not (
                _RECOVERY_TEMPERATURE_MIN <= float(member.temperature)
                <= _RECOVERY_TEMPERATURE_MAX):
            raise ValueError(
                "recovery Q_safe temperature must be within exp(-4)..exp(4)")
        if not member.bootstrap_trajectories:
            raise ValueError(
                "recovery Q_safe member lacks trajectory-bootstrap provenance")
    if trained.normalization is None or (
            trained.normalization.fit_content_sha256 is None) or (
            trained.normalization.fit_split is None) or (
            trained.train_split != trained.normalization.fit_split) or (
            trained.privileged_dim != 0) or (
            trained.command_vx is None) or not np.isclose(
                trained.command_vx, 0.30, rtol=0.0, atol=1e-6):
        raise ValueError(
            "recovery Q_safe preprocessing/command provenance is incomplete")


def _validate_loaded_recovery_metadata(
    *,
    manifest: Mapping[str, Any],
    network_config: QSafeNetworkConfig,
    normalization: NormalizationStats,
    entries: Any,
) -> None:
    """Recheck the complete V4 training contract at every artifact load."""
    action_contract = manifest.get("action_feature_contract")
    if not isinstance(action_contract, Mapping) or action_contract.get(
            "view") != RECOVERY_PROGRAM_VIEW:
        return
    try:
        training_config = QSafeTrainingConfig(**manifest["training_config"])
        loss_config = QSafeLossConfig(**manifest["loss_config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "recovery artifact training/loss metadata is invalid") from exc
    if (network_config != RECOVERY_PROGRAM_V4_NETWORK_CONFIG or
            training_config != RECOVERY_PROGRAM_V4_TRAINING_CONFIG or
            loss_config != RECOVERY_PROGRAM_V4_LOSS_CONFIG):
        raise ValueError(
            "loaded recovery_program_v1 artifact violates the exact V4 "
            "network/training/loss contract")
    if not isinstance(entries, list) or len(entries) != (
            RECOVERY_PROGRAM_V4_TRAINING_CONFIG.ensemble_members):
        raise ValueError("loaded recovery Q_safe artifact must contain five members")
    exact_fields = {
        "file", "seed", "temperature", "bootstrap_trajectories", "epoch_loss",
    }
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != exact_fields or (
                entry.get("file") != f"member_{index:02d}.pt") or (
                entry.get("seed") != (
                    RECOVERY_PROGRAM_V4_TRAINING_CONFIG.seed
                    + RECOVERY_PROGRAM_V4_MEMBER_SEED_STRIDE * index)):
            raise ValueError("loaded recovery Q_safe member identity drifted")
        temperature = entry.get("temperature")
        epoch_loss = entry.get("epoch_loss")
        bootstrap = entry.get("bootstrap_trajectories")
        if isinstance(temperature, bool) or not isinstance(
                temperature, (int, float)) or not np.isfinite(
                    float(temperature)) or not (
                        _RECOVERY_TEMPERATURE_MIN <= float(temperature)
                        <= _RECOVERY_TEMPERATURE_MAX):
            raise ValueError("loaded recovery Q_safe temperature is invalid")
        if not isinstance(epoch_loss, list) or len(epoch_loss) != (
                RECOVERY_PROGRAM_V4_TRAINING_CONFIG.epochs) or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(float(value)) or float(value) < 0.0
                    for value in epoch_loss):
            raise ValueError("loaded recovery Q_safe epoch metadata drifted")
        if not isinstance(bootstrap, list) or not bootstrap or any(
                not isinstance(value, str) or not value for value in bootstrap):
            raise ValueError(
                "loaded recovery Q_safe bootstrap provenance is invalid")
    if normalization.fit_content_sha256 is None or (
            normalization.fit_split is None):
        raise ValueError(
            "loaded recovery Q_safe normalization lacks fit-only provenance")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("loaded recovery Q_safe provenance is invalid")
    command_vx = provenance.get("command_vx")
    if isinstance(command_vx, bool) or not isinstance(command_vx, (int, float)) or (
            not np.isfinite(float(command_vx))) or not np.isclose(
                float(command_vx), 0.30, rtol=0.0, atol=1e-6):
        raise ValueError("loaded recovery Q_safe command provenance drifted")


@dataclass(frozen=True)
class LoadedQSafeArtifact:
    ensemble: QSafeEnsemble
    normalization: NormalizationStats
    network_config: QSafeNetworkConfig
    action_view: str
    action_components: tuple[str, ...]
    manifest: dict[str, Any]
    path: Path
    authorized_manifest_sha256: str | None = field(
        default=None, repr=False, compare=False)
    _load_token: object | None = field(default=None, repr=False, compare=False)
    _manifest_live_sha256: str | None = field(
        default=None, repr=False, compare=False)
    _normalization_live_sha256: str | None = field(
        default=None, repr=False, compare=False)
    _ensemble_live_attestation: tuple[Any, ...] | None = field(
        default=None, repr=False, compare=False)
    _ensemble_live_sha256: str | None = field(
        default=None, repr=False, compare=False)

    def require_live_integrity(self) -> None:
        """Reject hand-built or mutated objects at claim-bearing boundaries."""
        if self._load_token is not _LOADED_ARTIFACT_TOKEN:
            raise ValueError(
                "claim-bearing Q_safe artifact must come from load_qsafe_artifact")
        manifest_live_sha256 = _canonical_json_sha256(
            self.manifest, name="artifact manifest")
        if self._manifest_live_sha256 != manifest_live_sha256:
            raise ValueError("loaded Q_safe artifact manifest mutated after load")
        manifest_action_contract = self.manifest.get("action_feature_contract")
        manifest_action_view = (
            manifest_action_contract.get("view")
            if isinstance(manifest_action_contract, Mapping) else None)
        if manifest_action_view != self.action_view:
            raise ValueError("loaded Q_safe artifact action view mutated after load")
        if self.authorized_manifest_sha256 is not None and (
                not _is_lowercase_sha256(self.authorized_manifest_sha256) or
                self.authorized_manifest_sha256 != manifest_live_sha256):
            raise ValueError(
                "loaded Q_safe artifact authorization mutated after load")
        if manifest_action_view == RECOVERY_PROGRAM_VIEW and (
                self.authorized_manifest_sha256 is None):
            raise ValueError(
                "claim-bearing recovery Q_safe artifact lacks an authorized "
                "manifest hash")
        if self._normalization_live_sha256 != _normalization_live_sha256(
                self.normalization):
            raise ValueError(
                "loaded Q_safe normalization mutated after load")
        if self._ensemble_live_attestation != _ensemble_live_attestation(
                self.ensemble):
            raise ValueError(
                "loaded Q_safe ensemble structure or tensors mutated after load")
        if self._ensemble_live_sha256 != _ensemble_live_sha256(self.ensemble):
            raise ValueError(
                "loaded Q_safe ensemble values or callable surface mutated "
                "after load")

    @property
    def claim_identity_sha256(self) -> str:
        """Portable identity of the verified artifact manifest/components."""
        self.require_live_integrity()
        assert self._manifest_live_sha256 is not None
        return (
            self.authorized_manifest_sha256
            if self.authorized_manifest_sha256 is not None
            else self._manifest_live_sha256
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any, *, name: str) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be canonical JSON data") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_lowercase_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def _normalization_live_sha256(normalization: NormalizationStats) -> str:
    digest = hashlib.sha256(b"qsafe.normalization.live.v1\0")
    for name in (
        "observation_mean", "observation_std", "privileged_mean",
        "privileged_std",
    ):
        value = getattr(normalization, name)
        digest.update(name.encode("ascii") + b"\0")
        if value is None:
            digest.update(b"none\0")
            continue
        array = np.ascontiguousarray(value, dtype="<f4")
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(array.tobytes(order="C"))
    for name in ("fit_content_sha256", "fit_split"):
        value = getattr(normalization, name)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(b"none\0" if value is None else (
            str(value).encode("utf-8") + b"\0"))
    return digest.hexdigest()


def _tensor_live_attestation(
    name: str,
    value: torch.Tensor,
) -> tuple[Any, ...]:
    return (
        name,
        id(value),
        int(value.data_ptr()),
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
        int(value._version),
    )


def _ensemble_live_attestation(ensemble: QSafeEnsemble) -> tuple[Any, ...]:
    if not isinstance(ensemble, QSafeEnsemble):
        raise TypeError("loaded ensemble must be QSafeEnsemble")
    members = tuple(ensemble.members)
    if not members or not all(
            isinstance(member, SelectiveAdvantageQSafe) for member in members):
        raise TypeError(
            "loaded ensemble members must be SelectiveAdvantageQSafe")
    return (
        id(ensemble),
        tuple(id(member) for member in members),
        tuple(
            tuple(sorted(asdict(member.config).items()))
            for member in members
        ),
        tuple(_tensor_live_attestation(name, value) for name, value in (
            list(ensemble.named_parameters()) + list(ensemble.named_buffers())
        )),
    )


def _ensemble_live_sha256(ensemble: QSafeEnsemble) -> str:
    if not isinstance(ensemble, QSafeEnsemble):
        raise TypeError("loaded ensemble must be QSafeEnsemble")
    digest = hashlib.sha256(b"qsafe.ensemble.live.v1\0")
    members = tuple(ensemble.members)
    for index, member in enumerate(members):
        if not isinstance(member, SelectiveAdvantageQSafe):
            raise TypeError(
                "loaded ensemble members must be SelectiveAdvantageQSafe")
        digest.update(f"member_config[{index}]".encode("ascii") + b"\0")
        digest.update(_canonical_json_sha256(
            asdict(member.config), name="member network config").encode("ascii"))
    for module_name, module in ensemble.named_modules():
        for hook_name in (
            "_forward_hooks", "_forward_pre_hooks", "_backward_hooks",
            "_backward_pre_hooks",
        ):
            hooks = getattr(module, hook_name, None)
            if hooks:
                raise ValueError(
                    "claim-bearing Q_safe ensemble must not carry runtime hooks")
        if "forward" in vars(module) or "predict" in vars(module):
            raise ValueError(
                "claim-bearing Q_safe modules must not override callables on "
                f"an instance: {module_name or '<root>'}")
    for name, value in ensemble.state_dict().items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError("Q_safe state_dict must contain dense tensors")
        tensor = value.detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype="<u8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_normalization(path: Path, normalization: NormalizationStats) -> None:
    arrays: dict[str, np.ndarray] = {
        "observation_mean": normalization.observation_mean,
        "observation_std": normalization.observation_std,
    }
    if normalization.privileged_mean is not None:
        assert normalization.privileged_std is not None
        arrays["privileged_mean"] = normalization.privileged_mean
        arrays["privileged_std"] = normalization.privileged_std
    if normalization.fit_content_sha256 is not None:
        assert normalization.fit_split is not None
        arrays["fit_content_sha256"] = np.asarray(
            normalization.fit_content_sha256)
        arrays["fit_split"] = np.asarray(normalization.fit_split)
    np.savez(path, **arrays)


def _normalization_fit_provenance(
    normalization: NormalizationStats,
) -> dict[str, str] | None:
    if normalization.fit_content_sha256 is None:
        return None
    assert normalization.fit_split is not None
    return {
        "fit_content_sha256": normalization.fit_content_sha256,
        "fit_split": normalization.fit_split,
    }


def _npz_scalar_text(payload: Any, name: str) -> str:
    value = np.asarray(payload[name])
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(
            f"artifact normalization {name} must be one Unicode scalar")
    result = str(value.item())
    if not result:
        raise ValueError(f"artifact normalization {name} must be nonempty")
    return result


def save_qsafe_artifact(
    path: str | Path,
    trained: TrainedQSafeEnsemble,
    normalization: NormalizationStats,
    network_config: QSafeNetworkConfig,
    training_config: QSafeTrainingConfig,
    loss_config: QSafeLossConfig,
    *,
    provenance: Mapping[str, Any],
    recovery_selector_bundle: RecoverySelectorBundle | None = None,
    array_attachments: Mapping[str, np.ndarray] | None = None,
    pre_publish_check: Callable[[], None] | None = None,
) -> Path:
    """Save an artifact without overwriting any prior experimental result."""
    output = assert_development_path(assert_safe_evidence_output(path))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Q_safe artifact: {output}")
    _validate_trained_ensemble_consistency(trained)
    _validate_recovery_training_metadata(
        trained, network_config, training_config, loss_config)
    _validate_trained_recovery_provenance(
        trained, provenance, recovery_selector_bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        normalization_name = "normalization.npz"
        _write_normalization(temporary / normalization_name, normalization)
        component_hashes = {
            normalization_name: _sha256(temporary / normalization_name),
        }
        for filename, value in (array_attachments or {}).items():
            component = _safe_component(temporary, filename)
            if component.suffix != ".npy" or filename in component_hashes:
                raise ValueError(
                    "array attachment names must be unique local .npy files")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError("array attachments must not use object dtype")
            np.save(component, array, allow_pickle=False)
            component_hashes[filename] = _sha256(component)
        member_entries: list[dict[str, Any]] = []
        for index, member in enumerate(trained.members):
            filename = f"member_{index:02d}.pt"
            state = {
                name: value.detach().cpu()
                for name, value in member.model.state_dict().items()
            }
            torch.save(state, temporary / filename)
            component_hashes[filename] = _sha256(temporary / filename)
            member_entries.append({
                "file": filename,
                "seed": int(member.seed),
                "temperature": float(member.temperature),
                "bootstrap_trajectories": list(member.bootstrap_trajectories),
                "epoch_loss": [float(value) for value in member.epoch_loss],
            })
        effective_network = trained.members[0].model.config
        if any(member.model.config != effective_network for member in trained.members):
            raise ValueError("ensemble members do not share a network configuration")
        if network_config != effective_network:
            raise ValueError("declared network configuration differs from trained model")
        action_view = trained.action_view
        action_components, expected_action_dim, action_feature_contract = (
            _action_feature_contract(action_view, provenance))
        if trained.action_dim != expected_action_dim or (
                network_config.action_dim != expected_action_dim):
            raise ValueError("action feature contract and network width disagree")
        if trained.normalization is not None and not (
                trained.normalization.equivalent_to(normalization)):
            raise ValueError("artifact normalization differs from training provenance")
        manifest = _json_safe({
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "network_config": asdict(network_config),
            "training_config": asdict(training_config),
            "loss_config": asdict(loss_config),
            "feature_view": (
                "privileged_diagnostic_only"
                if network_config.privileged_dim else "deployable"),
            "action_feature_contract": action_feature_contract,
            "normalization_fit_provenance": _normalization_fit_provenance(
                normalization),
            "members": member_entries,
            "component_sha256": component_hashes,
            "provenance": dict(provenance),
        })
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if pre_publish_check is not None:
            pre_publish_check()
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def _safe_component(root: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError(f"invalid artifact component name: {filename!r}")
    component = assert_development_path(
        require_v3_audit_consumed_or_safe_input(root / filename))
    if component.parent != root:
        raise ValueError("artifact component escapes its directory")
    return component


def load_qsafe_artifact(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    expected_manifest_sha256: str | None = None,
) -> LoadedQSafeArtifact:
    """Load only an externally authorized recovery manifest and its components."""
    source = assert_development_path(
        require_v3_audit_consumed_or_safe_input(path))
    manifest_path = _safe_component(source, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported Q_safe artifact schema")
    manifest_sha256 = _canonical_json_sha256(
        manifest, name="artifact manifest")
    action_contract_at_authorization = manifest.get("action_feature_contract")
    recovery_artifact = isinstance(
        action_contract_at_authorization, Mapping) and (
            action_contract_at_authorization.get("view") ==
            RECOVERY_PROGRAM_VIEW)
    if expected_manifest_sha256 is not None and not _is_lowercase_sha256(
            expected_manifest_sha256):
        raise ValueError(
            "expected_manifest_sha256 must be an exact lowercase SHA-256")
    if recovery_artifact and expected_manifest_sha256 is None:
        raise ValueError(
            "recovery_program_v1 artifact requires expected_manifest_sha256")
    if expected_manifest_sha256 is not None and (
            expected_manifest_sha256 != manifest_sha256):
        raise ValueError("Q_safe artifact manifest authorization mismatch")
    hashes = manifest.get("component_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("artifact has no component hash manifest")
    if "normalization.npz" not in hashes:
        raise ValueError(
            "artifact normalization is absent from component hash manifest")
    for filename, expected in hashes.items():
        component = _safe_component(source, filename)
        if not component.is_file() or not isinstance(expected, str) or (
                _sha256(component) != expected):
            raise ValueError(f"Q_safe artifact hash mismatch: {filename}")

    normalization_path = _safe_component(source, "normalization.npz")
    require_v3_audit_consumed_or_safe_input(normalization_path)
    with np.load(normalization_path, allow_pickle=False) as payload:
        files = set(payload.files)
        if not {"observation_mean", "observation_std"}.issubset(files):
            raise ValueError("artifact normalization is incomplete")
        allowed_files = {
            "observation_mean", "observation_std", "privileged_mean",
            "privileged_std", "fit_content_sha256", "fit_split",
        }
        if not files.issubset(allowed_files):
            raise ValueError("artifact normalization has unexpected fields")
        if ("privileged_mean" in files) != ("privileged_std" in files):
            raise ValueError(
                "artifact privileged normalization is incomplete")
        if ("fit_content_sha256" in files) != ("fit_split" in files):
            raise ValueError(
                "artifact normalization fit provenance is incomplete")
        privileged_mean = (
            payload["privileged_mean"].copy()
            if "privileged_mean" in files else None)
        privileged_std = (
            payload["privileged_std"].copy()
            if "privileged_std" in files else None)
        fit_content_sha256 = (
            _npz_scalar_text(payload, "fit_content_sha256")
            if "fit_content_sha256" in files else None)
        fit_split = (
            _npz_scalar_text(payload, "fit_split")
            if "fit_split" in files else None)
        normalization = NormalizationStats(
            payload["observation_mean"].copy(),
            payload["observation_std"].copy(),
            privileged_mean,
            privileged_std,
            fit_content_sha256,
            fit_split,
        )

    network_config = QSafeNetworkConfig(**manifest["network_config"])
    action_contract = manifest.get("action_feature_contract")
    if not isinstance(action_contract, dict):
        raise ValueError("artifact has no action feature contract")
    action_view = action_contract.get("view")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("artifact provenance must be a mapping")
    action_components, expected_action_dim, expected_action_contract = (
        _action_feature_contract(action_view, provenance))
    if action_contract != expected_action_contract or (
            action_contract.get("total_width") != network_config.action_dim) or (
            network_config.action_dim != expected_action_dim):
        raise ValueError("artifact action feature contract is inconsistent")
    if manifest.get("normalization_fit_provenance") != (
            _normalization_fit_provenance(normalization)):
        raise ValueError(
            "artifact normalization fit provenance disagrees with manifest")
    if action_view == RECOVERY_PROGRAM_VIEW and (
            normalization.fit_content_sha256 is None):
        raise ValueError(
            "recovery_program_v1 artifact lacks fit-only normalization "
            "provenance")
    entries = manifest.get("members")
    _validate_loaded_recovery_metadata(
        manifest=manifest,
        network_config=network_config,
        normalization=normalization,
        entries=entries,
    )
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("deployable selector requires at least two ensemble members")
    models = []
    temperatures = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid artifact member entry")
        filename = entry.get("file")
        if filename not in hashes:
            raise ValueError("member is absent from component hash manifest")
        model = SelectiveAdvantageQSafe(network_config)
        state = torch.load(
            _safe_component(source, filename),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        models.append(model.to(device).eval())
        temperatures.append(float(entry["temperature"]))
    ensemble = QSafeEnsemble(models, temperatures=temperatures).to(device).eval()
    expected_privileged = network_config.privileged_dim > 0
    if expected_privileged != (normalization.privileged_mean is not None):
        raise ValueError("network and normalization feature views disagree")
    return LoadedQSafeArtifact(
        ensemble=ensemble,
        normalization=normalization,
        network_config=network_config,
        action_view=action_view,
        action_components=action_components,
        manifest=manifest,
        path=source,
        authorized_manifest_sha256=expected_manifest_sha256,
        _load_token=_LOADED_ARTIFACT_TOKEN,
        _manifest_live_sha256=manifest_sha256,
        _normalization_live_sha256=_normalization_live_sha256(normalization),
        _ensemble_live_attestation=_ensemble_live_attestation(ensemble),
        _ensemble_live_sha256=_ensemble_live_sha256(ensemble),
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "LoadedQSafeArtifact",
    "load_qsafe_artifact",
    "save_qsafe_artifact",
]
