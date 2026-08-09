"""Self-describing, hash-verified Selective Advantage Q_safe artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
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
from rl.qsafe.training import QSafeTrainingConfig, TrainedQSafeEnsemble
from safety_data.paths import (
    assert_development_path,
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)


ARTIFACT_SCHEMA_VERSION = "qsafe.selective_advantage.v1"
_ACTION_FEATURE_COMPONENTS = {
    "requested": ("requested",),
    "application_concat": ("requested", "executed", "q_target"),
}


@dataclass(frozen=True)
class LoadedQSafeArtifact:
    ensemble: QSafeEnsemble
    normalization: NormalizationStats
    network_config: QSafeNetworkConfig
    action_view: str
    action_components: tuple[str, ...]
    manifest: dict[str, Any]
    path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
    np.savez(path, **arrays)


def save_qsafe_artifact(
    path: str | Path,
    trained: TrainedQSafeEnsemble,
    normalization: NormalizationStats,
    network_config: QSafeNetworkConfig,
    training_config: QSafeTrainingConfig,
    loss_config: QSafeLossConfig,
    *,
    provenance: Mapping[str, Any],
    array_attachments: Mapping[str, np.ndarray] | None = None,
    pre_publish_check: Callable[[], None] | None = None,
) -> Path:
    """Save an artifact without overwriting any prior experimental result."""
    output = assert_development_path(assert_safe_evidence_output(path))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Q_safe artifact: {output}")
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
        if len(member_entries) != len(trained.ensemble.members):
            raise ValueError("trained member metadata and ensemble disagree")
        effective_network = trained.members[0].model.config
        if any(member.model.config != effective_network for member in trained.members):
            raise ValueError("ensemble members do not share a network configuration")
        if network_config != effective_network:
            raise ValueError("declared network configuration differs from trained model")
        action_view = trained.action_view
        if action_view not in _ACTION_FEATURE_COMPONENTS:
            raise ValueError("trained ensemble has no supported action feature contract")
        action_components = _ACTION_FEATURE_COMPONENTS[action_view]
        expected_action_dim = 12 * len(action_components)
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
            "action_feature_contract": {
                "view": action_view,
                "components_in_order": list(action_components),
                "joint_width_per_component": 12,
                "total_width": expected_action_dim,
            },
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
) -> LoadedQSafeArtifact:
    """Load only after all declared component hashes have been verified."""
    source = assert_development_path(
        require_v3_audit_consumed_or_safe_input(path))
    manifest_path = _safe_component(source, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported Q_safe artifact schema")
    hashes = manifest.get("component_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("artifact has no component hash manifest")
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
        privileged_mean = (
            payload["privileged_mean"].copy()
            if "privileged_mean" in files else None)
        privileged_std = (
            payload["privileged_std"].copy()
            if "privileged_std" in files else None)
        normalization = NormalizationStats(
            payload["observation_mean"].copy(),
            payload["observation_std"].copy(),
            privileged_mean,
            privileged_std,
        )

    network_config = QSafeNetworkConfig(**manifest["network_config"])
    action_contract = manifest.get("action_feature_contract")
    if not isinstance(action_contract, dict):
        raise ValueError("artifact has no action feature contract")
    action_view = action_contract.get("view")
    if action_view not in _ACTION_FEATURE_COMPONENTS:
        raise ValueError("artifact action feature view is unsupported")
    action_components = _ACTION_FEATURE_COMPONENTS[action_view]
    if action_contract.get("components_in_order") != list(action_components) or (
            action_contract.get("joint_width_per_component") != 12) or (
            action_contract.get("total_width") != network_config.action_dim) or (
            network_config.action_dim != 12 * len(action_components)):
        raise ValueError("artifact action feature contract is inconsistent")
    entries = manifest.get("members")
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
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "LoadedQSafeArtifact",
    "load_qsafe_artifact",
    "save_qsafe_artifact",
]
