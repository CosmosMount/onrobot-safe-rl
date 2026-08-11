"""Train a state-risk Q_safe trigger from direct natural-PPO supervision."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl.qsafe.network import QSafeNetworkConfig, SelectiveAdvantageQSafe


@dataclass(frozen=True)
class DirectTrainingConfig:
    ensemble_members: int = 5
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    state_risk_weight: float = 1.0
    calibration_steps: int = 200
    seed: int = 20260811
    frame_hidden_dim: int = 128
    state_hidden_dim: int = 128
    action_hidden_dim: int = 128

    def __post_init__(self) -> None:
        for name in (
                "ensemble_members", "epochs", "batch_size", "calibration_steps",
                "frame_hidden_dim", "state_hidden_dim", "action_hidden_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
                "learning_rate", "weight_decay", "state_risk_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class DirectDataset:
    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]
    file_sha256: str

    def role_mask(self, role: str) -> np.ndarray:
        return self.arrays["role"] == role.encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite direct Q_safe artifact: {destination}") from exc
    temporary.unlink()
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_direct_dataset(path: str | Path) -> DirectDataset:
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha256(path)
    if manifest.get("schema_version") != "qsafe.natural_ppo_direct_dataset.v1" or (
            manifest.get("dataset_file_sha256") != digest):
        raise ValueError("direct PPO dataset manifest does not bind the dataset")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    count = len(arrays["label"])
    required_shapes = {
        "observation_history": (count, 5, 46),
        "action_requested": (count, 12),
        "role": (count,),
        "pair_identity": (count,),
        "episode_identity": (count,),
    }
    for name, shape in required_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"direct PPO array {name} has an invalid shape")
    if not np.all(np.isfinite(arrays["observation_history"])) or not np.all(
            np.isfinite(arrays["action_requested"])):
        raise ValueError("direct PPO training features are non-finite")
    if set(np.unique(arrays["role"]).tolist()) != {
            b"fit", b"calibration", b"test"}:
        raise ValueError("direct PPO dataset must contain all three frozen roles")
    for role in (b"fit", b"calibration", b"test"):
        mask = arrays["role"] == role
        if not np.any(arrays["label"][mask]) or np.all(arrays["label"][mask]):
            raise ValueError(f"direct PPO role {role!r} is not binary")
    role_by_episode: dict[bytes, bytes] = {}
    for episode, role in zip(
            arrays["episode_identity"], arrays["role"], strict=True):
        previous = role_by_episode.setdefault(bytes(episode), bytes(role))
        if previous != bytes(role):
            raise ValueError("one PPO episode appears in multiple roles")
    pair_roles: dict[bytes, set[bytes]] = {}
    pair_labels: dict[bytes, list[bool]] = {}
    for pair, role, label in zip(
            arrays["pair_identity"], arrays["role"], arrays["label"], strict=True):
        key = bytes(pair)
        pair_roles.setdefault(key, set()).add(bytes(role))
        pair_labels.setdefault(key, []).append(bool(label))
    if any(len(value) != 1 for value in pair_roles.values()) or any(
            sorted(value) != [False, True] for value in pair_labels.values()):
        raise ValueError("direct PPO pairs must be balanced and role-local")
    if not np.allclose(arrays["command"], [0.3, 0.0, 0.0], atol=1e-6):
        raise ValueError("direct PPO dataset is not the +0.30 m/s task")
    return DirectDataset(arrays=arrays, manifest=manifest, file_sha256=digest)


def binary_auc(label: np.ndarray, score: np.ndarray) -> float:
    label = np.asarray(label, dtype=bool).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if label.shape != score.shape or not np.all(np.isfinite(score)):
        raise ValueError("binary AUC inputs are invalid")
    positives = int(label.sum())
    negatives = len(label) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("binary AUC requires both classes")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[label].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def expected_calibration_error(
    label: np.ndarray, probability: np.ndarray, bins: int = 10,
) -> float:
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if label.shape != probability.shape or not np.all(
            (probability >= 0.0) & (probability <= 1.0)):
        raise ValueError("ECE inputs are invalid")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(label)
    result = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability < edges[index + 1] if index + 1 < bins
            else probability <= edges[index + 1])
        if np.any(selected):
            result += float(np.sum(selected) / total) * abs(
                float(probability[selected].mean() - label[selected].mean()))
    return result


def paired_accuracy(
    pair_identity: np.ndarray, label: np.ndarray, score: np.ndarray,
) -> float:
    values: dict[bytes, dict[bool, float]] = {}
    for pair, target, prediction in zip(
            pair_identity, label, score, strict=True):
        values.setdefault(bytes(pair), {})[bool(target)] = float(prediction)
    if not values or any(set(value) != {False, True} for value in values.values()):
        raise ValueError("paired accuracy requires one positive and negative per pair")
    outcomes = [
        1.0 if value[True] > value[False]
        else 0.5 if value[True] == value[False] else 0.0
        for value in values.values()
    ]
    return float(np.mean(outcomes))


def _normalization(observation: np.ndarray) -> dict[str, np.ndarray]:
    observation_flat = observation.reshape(-1, observation.shape[-1])
    observation_mean = observation_flat.mean(axis=0, dtype=np.float64)
    observation_std = observation_flat.std(axis=0, dtype=np.float64)
    return {
        "observation_mean": observation_mean.astype(np.float32),
        "observation_std": np.maximum(observation_std, 1e-6).astype(np.float32),
    }


def _normalized_arrays(
    dataset: DirectDataset, normalization: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    arrays = dataset.arrays
    observation = (
        arrays["observation_history"] - normalization["observation_mean"]
    ) / normalization["observation_std"]
    return observation.astype(np.float32), arrays["label"]


def _fit_temperature(
    state_logits: torch.Tensor,
    label: torch.Tensor,
    steps: int,
) -> float:
    log_temperature = nn.Parameter(torch.zeros((), device=state_logits.device))
    optimizer = torch.optim.Adam([log_temperature], lr=0.03)
    detached_state = state_logits.detach()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(
            detached_state / temperature, label)
        loss.backward()
        optimizer.step()
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def _forward_state_logits(
    model: SelectiveAdvantageQSafe,
    observation: torch.Tensor,
) -> torch.Tensor:
    state = model.encode_state(observation)
    return model.state_risk_head(state).reshape(-1)


def _predict(
    models: list[SelectiveAdvantageQSafe],
    temperatures: list[float],
    observation: torch.Tensor,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    member_state = []
    with torch.inference_mode():
        for model, temperature in zip(models, temperatures, strict=True):
            state_chunks = []
            for start in range(0, len(observation), batch_size):
                state_logit = _forward_state_logits(
                    model, observation[start:start + batch_size])
                state_chunks.append(torch.sigmoid(state_logit / temperature).cpu())
            member_state.append(torch.cat(state_chunks).numpy())
    state = np.stack(member_state)
    return state.mean(axis=0), state.std(axis=0)


def _metrics(
    label: np.ndarray,
    pair_identity: np.ndarray,
    state_risk: np.ndarray,
) -> dict[str, float]:
    return {
        "state_auroc": binary_auc(label, state_risk),
        "state_pair_accuracy": paired_accuracy(pair_identity, label, state_risk),
        "state_ece": expected_calibration_error(label, state_risk),
        "state_accuracy_at_0.5": float(np.mean((state_risk >= 0.5) == label)),
    }


def _git_head() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True).stdout.strip()


def train_direct_qsafe(
    dataset_path: str | Path,
    output: str | Path,
    *,
    config: DirectTrainingConfig | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config = DirectTrainingConfig() if config is None else config
    dataset = load_direct_dataset(dataset_path)
    output = Path(output)
    report_path = output.with_suffix(".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("direct Q_safe training output path was already consumed")
    selected_device = torch.device(
        device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    fit_mask = dataset.role_mask("fit")
    calibration_mask = dataset.role_mask("calibration")
    test_mask = dataset.role_mask("test")
    normalization = _normalization(
        dataset.arrays["observation_history"][fit_mask])
    observation_np, label_np = _normalized_arrays(dataset, normalization)
    observation = torch.from_numpy(observation_np).to(selected_device)
    label = torch.from_numpy(label_np.astype(np.float32)).to(selected_device)
    calibration_indices = torch.from_numpy(
        np.flatnonzero(calibration_mask)).to(selected_device)
    test_indices_np = np.flatnonzero(test_mask)
    network_config = QSafeNetworkConfig(
        frame_hidden_dim=config.frame_hidden_dim,
        state_hidden_dim=config.state_hidden_dim,
        action_hidden_dim=config.action_hidden_dim,
        action_mode="pointwise",
    )
    models: list[SelectiveAdvantageQSafe] = []
    temperatures: list[float] = []
    fit_pairs = np.unique(dataset.arrays["pair_identity"][fit_mask])
    pair_to_indices = {
        bytes(pair): np.flatnonzero(
            fit_mask & (dataset.arrays["pair_identity"] == pair))
        for pair in fit_pairs
    }
    member_losses = []
    for member in range(config.ensemble_members):
        member_seed = config.seed + 1009 * member
        torch.manual_seed(member_seed)
        generator = np.random.default_rng(member_seed)
        sampled_pairs = generator.choice(fit_pairs, size=len(fit_pairs), replace=True)
        member_indices = np.concatenate([
            pair_to_indices[bytes(pair)] for pair in sampled_pairs
        ]).astype(np.int64)
        model = SelectiveAdvantageQSafe(network_config).to(selected_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay)
        final_loss = float("nan")
        for _ in range(config.epochs):
            generator.shuffle(member_indices)
            for start in range(0, len(member_indices), config.batch_size):
                indices = torch.from_numpy(
                    member_indices[start:start + config.batch_size]).to(selected_device)
                state_logit = _forward_state_logits(model, observation[indices])
                target = label[indices]
                state_loss = F.binary_cross_entropy_with_logits(state_logit, target)
                loss = config.state_risk_weight * state_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                final_loss = float(loss.detach().item())
        model.eval()
        with torch.no_grad():
            state_logit = _forward_state_logits(
                model, observation[calibration_indices])
        temperature = _fit_temperature(
            state_logit, label[calibration_indices], config.calibration_steps)
        models.append(model)
        temperatures.append(temperature)
        member_losses.append(final_loss)

    test_indices = torch.from_numpy(test_indices_np).to(selected_device)
    state_mean, state_std = _predict(
        models, temperatures, observation[test_indices], config.batch_size)
    test_metrics = _metrics(
        label_np[test_indices_np], dataset.arrays["pair_identity"][test_indices_np],
        state_mean)
    artifact = {
        "schema_version": "qsafe.natural_ppo_state_trigger_model.v2",
        "production_head": "state_risk_only",
        "executed_ppo_action_use": "provenance_and_diagnostics_only",
        "trainer_commit": _git_head(),
        "dataset_file_sha256": dataset.file_sha256,
        "dataset_manifest": dataset.manifest,
        "training_config": asdict(config),
        "network_config": asdict(network_config),
        "normalization": {
            name: torch.from_numpy(value) for name, value in normalization.items()
        },
        "member_state_dicts": [
            {name: value.detach().cpu() for name, value in model.state_dict().items()}
            for model in models
        ],
        "state_temperatures": temperatures,
        "heldout_ppo_test_metrics": test_metrics,
        "sac_model_test_consumed": False,
        "objective1_claim_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    torch.save(artifact, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary, output)
    output_sha256 = _sha256(output)
    report = {
        "schema_version": "qsafe.natural_ppo_state_trigger_training_report.v2",
        "model_file": output.name,
        "model_file_sha256": output_sha256,
        "dataset_file_sha256": dataset.file_sha256,
        "fit_samples": int(fit_mask.sum()),
        "calibration_samples": int(calibration_mask.sum()),
        "heldout_ppo_test_samples": int(test_mask.sum()),
        "member_final_losses": member_losses,
        "state_temperatures": temperatures,
        "heldout_ppo_test_metrics": test_metrics,
        "heldout_mean_state_uncertainty": float(state_std.mean()),
        "sac_model_test_consumed": False,
        "objective1_claim_eligible": False,
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary_report = report_path.with_name(
        f".{report_path.name}.tmp-{os.getpid()}")
    with temporary_report.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    _publish_no_clobber(temporary_report, report_path)
    return report
