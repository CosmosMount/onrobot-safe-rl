"""Lightweight, evidence-traceable loading of frozen DroQ actors.

This module deliberately builds only an optimizer-free inference policy.  It
never constructs a training agent, critic, optimizer, or replay buffer.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf
import torch

from rl.agents.droq.inference import DroQInferencePolicy
from rl.agents.inference import build_inference_policy
from safety_data.paths import assert_development_path
from train.config import load_app_config


POLICY_MANIFEST_VERSION = "qsafe.frozen_droq.v2"
_STEP_COMPONENT = re.compile(r"^step_(\d+)$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash tensor names, metadata, and values independent of serialization.

    The checkpoint-file digest remains useful provenance, but it also changes
    when an ignored optimizer payload or torch serialization detail changes.
    This digest identifies the actor weights that were actually loaded.
    """
    if not state_dict:
        raise ValueError("actor state_dict must be nonempty")
    if any(not isinstance(name, str) for name in state_dict):
        raise ValueError("actor state_dict keys must be strings")
    digest = hashlib.sha256(b"qsafe_actor_state_dict_v1\0")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"actor state_dict entry {name!r} must be a tensor")
        if tensor.layout != torch.strided:
            raise ValueError(
                f"actor state_dict entry {name!r} must be a dense tensor")
        detached = tensor.detach()
        if detached.is_meta:
            raise ValueError(
                f"actor state_dict entry {name!r} must contain materialized data")
        value = detached.cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all()):
            raise ValueError(
                f"actor state_dict entry {name!r} must contain finite values")
        metadata = json.dumps({
            "name": name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8")
        raw_value = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(raw_value).to_bytes(8, "little"))
        digest.update(raw_value)
    return digest.hexdigest()


def _actor_path(checkpoint: str | Path) -> Path:
    checked = assert_development_path(checkpoint)
    actor = checked / "actor.pt" if checked.is_dir() else checked
    actor = assert_development_path(actor)
    if actor.name != "actor.pt":
        raise ValueError(
            "frozen DroQ checkpoints must be an agent directory or actor.pt")
    if not actor.is_file():
        raise FileNotFoundError(actor)
    return actor


def _inferred_training_step(actor_path: Path) -> int | None:
    for parent in actor_path.parents:
        match = _STEP_COMPONENT.fullmatch(parent.name)
        if match is not None:
            return int(match.group(1))
    return None


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _jsonable(value: Any) -> Any:
    """Convert resolved config values to stable, JSON-compatible objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FrozenDroQPolicy:
    """A stateless wrapper around an optimizer-free frozen DroQ actor."""

    def __init__(
        self,
        policy: DroQInferencePolicy,
        manifest: Mapping[str, Any],
    ) -> None:
        self._policy = policy
        self._manifest = copy.deepcopy(dict(manifest))
        self._observation_dim = int(self._manifest["observation_dim"])
        self._action_dim = int(self._manifest["action_dim"])

    @property
    def training_step(self) -> int:
        return int(self._manifest["training_step"])

    @property
    def actor_sha256(self) -> str:
        return str(self._manifest["actor_sha256"])

    @property
    def actor_state_dict_sha256(self) -> str:
        """Return the serialization-independent digest of loaded weights."""
        return str(self._manifest["actor_state_dict_sha256"])

    def manifest(self) -> dict[str, Any]:
        """Return an isolated, JSON-serializable source-policy manifest."""
        return copy.deepcopy(self._manifest)

    def fingerprint(self) -> str:
        """Return the device-independent actor behavior fingerprint."""
        return str(self._manifest["policy_fingerprint_sha256"])

    def checkpoint_fingerprint(self) -> str:
        """Return provenance identity including files and recorded steps."""
        return str(self._manifest["checkpoint_fingerprint_sha256"])

    def _observation(self, value: np.ndarray) -> np.ndarray:
        observation = np.asarray(value, dtype=np.float32).reshape(-1)
        if observation.shape != (self._observation_dim,):
            raise ValueError(
                f"observation must have shape {(self._observation_dim,)}")
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation must be finite")
        return observation

    def _checked_action(self, value: np.ndarray) -> np.ndarray:
        action = np.asarray(value, dtype=np.float32).reshape(-1).copy()
        if action.shape != (self._action_dim,) or not np.all(np.isfinite(action)):
            raise RuntimeError("frozen DroQ actor returned an invalid action")
        return action

    def deterministic_action(self, observation: np.ndarray) -> np.ndarray:
        """Return the tanh of the actor mean without consuming any RNG."""
        decision = self._policy.decide(
            self._observation(observation), training=False)
        return self._checked_action(decision.action_requested)

    def sample_action(
        self,
        observation: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample once using only entropy drawn from ``rng``.

        ``torch.random.fork_rng`` restores the process-wide CPU generator (and
        the selected CUDA generator when relevant) even if inference raises.
        This keeps branch ordering from contaminating other stochastic code.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        torch_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        device = self._policy.device
        cuda_devices: list[int] = []
        if device.type == "cuda":
            cuda_devices = [
                torch.cuda.current_device()
                if device.index is None else int(device.index)]
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            # ``torch.manual_seed`` also seeds every CUDA device.  Seed the
            # CPU generator explicitly so fork_rng only needs to preserve the
            # selected inference device in addition to CPU.
            torch.random.default_generator.manual_seed(torch_seed)
            if cuda_devices:
                with torch.cuda.device(cuda_devices[0]):
                    torch.cuda.manual_seed(torch_seed)
            decision = self._policy.decide(
                self._observation(observation), training=True)
        return self._checked_action(decision.action_requested)

    def __call__(
        self,
        observation_history: np.ndarray,
        step: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample a native branch continuation from the newest history frame."""
        history = np.asarray(observation_history, dtype=np.float32)
        if history.ndim != 2 or history.shape[0] < 1:
            raise ValueError("observation_history must have shape [frames, dim]")
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
            raise ValueError("step must be a nonnegative integer")
        if int(step) < 0:
            raise ValueError("step must be a nonnegative integer")
        return self.sample_action(history[-1], rng)


def load_frozen_droq_policy(
    checkpoint: str | Path,
    config: str | Path,
    *,
    observation_dim: int,
    action_dim: int,
    training_step: int | None = None,
    device: str | torch.device = "cpu",
) -> FrozenDroQPolicy:
    """Load only ``actor.pt['network_state_dict']`` into inference policy.

    ``training_step`` is inferred from a ``step_<integer>`` ancestor when
    possible.  Otherwise it must be supplied explicitly; actor optimizer
    update counts are intentionally not conflated with environment steps.
    """
    actor_path = _actor_path(checkpoint)
    config_path = assert_development_path(config)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    observation_dim = _nonnegative_int(observation_dim, "observation_dim")
    action_dim = _nonnegative_int(action_dim, "action_dim")
    if observation_dim == 0 or action_dim == 0:
        raise ValueError("observation_dim and action_dim must be positive")

    inferred_step = _inferred_training_step(actor_path)
    if training_step is None:
        if inferred_step is None:
            raise ValueError(
                "cannot infer training step; pass training_step explicitly")
        source_training_step = inferred_step
    else:
        source_training_step = _nonnegative_int(training_step, "training_step")
        if inferred_step is not None and source_training_step != inferred_step:
            raise ValueError(
                f"training_step={source_training_step} disagrees with "
                f"checkpoint path step_{inferred_step:012d}")

    config_sha256_before = _sha256_file(config_path)
    _, train_cfg, loaded_cfg = load_app_config(path=config_path)
    config_sha256_after = _sha256_file(config_path)
    if config_sha256_after != config_sha256_before:
        raise RuntimeError("config changed while frozen actor was being loaded")
    loaded_agent_type = str(getattr(loaded_cfg, "agent_type", "")).lower()
    if loaded_agent_type != "droq":
        raise ValueError(
            f"frozen actor loader requires agent_type='droq', got "
            f"{loaded_agent_type!r}")
    source_resolved_cfg = OmegaConf.create(OmegaConf.to_container(
        loaded_cfg, resolve=True, throw_on_missing=True))
    resolved_cfg = OmegaConf.create(OmegaConf.to_container(
        source_resolved_cfg, resolve=True, throw_on_missing=True))
    runtime_device = torch.device(device)
    if runtime_device.type not in ("cpu", "cuda"):
        raise ValueError("frozen DroQ inference device must be CPU or CUDA")
    if runtime_device.type == "cuda" and runtime_device.index is None:
        runtime_device = torch.device("cuda:0")
    resolved_cfg.device_type = str(runtime_device)

    # Load and fingerprint the exact same immutable byte string.  Reading the
    # path once avoids a hash/load time-of-check/time-of-use mismatch.
    checkpoint_bytes = actor_path.read_bytes()
    actor_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint_data = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_data, Mapping):
        raise ValueError("actor.pt must contain a checkpoint mapping")
    state_dict = checkpoint_data.get("network_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("actor.pt is missing a nonempty network_state_dict")
    actor_state_dict_sha256 = _state_dict_sha256(state_dict)
    actor_update_step = _nonnegative_int(
        checkpoint_data.get("update_step", 0), "actor update_step")

    # Network constructors initialize temporary weights before the checkpoint
    # replaces them.  Do not let that irrelevant initialization advance the
    # application's global CPU RNG.
    initialization_cuda_devices = (
        [int(runtime_device.index)]
        if runtime_device.type == "cuda"
        else [])
    with torch.random.fork_rng(
            devices=initialization_cuda_devices, enabled=True):
        inference = build_inference_policy(
            observation_dim, action_dim, resolved_cfg)
    if not isinstance(inference, DroQInferencePolicy):
        raise TypeError("build_inference_policy did not return DroQ inference")
    if not 0 < int(inference.actor_observation_dim) <= observation_dim:
        raise ValueError(
            "actor_observation_dim must lie in [1, observation_dim], got "
            f"{inference.actor_observation_dim} for {observation_dim}")
    inference.load_snapshot({
        "snapshot_version": 0,
        "actor_state_dict": state_dict,
        "actor_steps": actor_update_step,
        "auxiliary_steps": 0,
    })

    resolved_agent_config = _jsonable(OmegaConf.to_container(
        source_resolved_cfg, resolve=True, throw_on_missing=True))
    resolved_train_config = _jsonable(asdict(train_cfg))
    behavior_config = {
        "agent_type": "droq",
        "observation_dim": observation_dim,
        "actor_observation_dim": int(inference.actor_observation_dim),
        "action_dim": action_dim,
        "hidden_dims": [int(value) for value in resolved_cfg.hidden_dims],
        "distribution": "NormalTanhPolicy",
        "log_std_min": -20.0,
        "log_std_max": 2.0,
    }
    fingerprint_basis = {
        "manifest_version": POLICY_MANIFEST_VERSION,
        "actor_state_dict_sha256": actor_state_dict_sha256,
        "behavior_config": behavior_config,
    }
    policy_fingerprint_sha256 = _canonical_sha256(fingerprint_basis)
    checkpoint_basis = {
        "manifest_version": POLICY_MANIFEST_VERSION,
        "policy_fingerprint_sha256": policy_fingerprint_sha256,
        "actor_sha256": actor_sha256,
        "config_sha256": config_sha256_after,
        "resolved_agent_config_sha256": _canonical_sha256(
            resolved_agent_config),
        "resolved_train_config_sha256": _canonical_sha256(
            resolved_train_config),
        "training_step": source_training_step,
        "actor_update_step": actor_update_step,
    }
    manifest = {
        **checkpoint_basis,
        **behavior_config,
        "actor_state_dict_sha256": actor_state_dict_sha256,
        "policy_fingerprint_sha256": policy_fingerprint_sha256,
        "checkpoint_fingerprint_sha256": _canonical_sha256(
            checkpoint_basis),
        "actor_path": str(actor_path),
        "config_path": str(config_path),
        "resolved_agent_config": resolved_agent_config,
        "train_config": resolved_train_config,
        "device": str(inference.device),
        "load_contract": "actor_pt_network_state_dict_only",
    }
    return FrozenDroQPolicy(inference, manifest)


__all__ = [
    "FrozenDroQPolicy",
    "POLICY_MANIFEST_VERSION",
    "load_frozen_droq_policy",
]
