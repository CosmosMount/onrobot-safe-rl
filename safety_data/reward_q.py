"""Optimizer-free, evidence-traceable DroQ reward-critic inference.

The online reward critic was trained against the normalized requested action
stored in the replay field named ``action``.  Accordingly, this module accepts
exactly one deployable observation and a matrix of requested actions.  Its
conservative value is the pointwise minimum across the online DroQ critic
ensemble, matching the reward-preservation gate used by SafeDroQ.

Only ``critic.pt['network_state_dict']`` is installed in an inference-only
``DroQEnsembleCritic``.  No agent, replay buffer, optimizer, target critic, or
training state is constructed.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf
import torch

from rl.agents.droq.network import DroQEnsembleCritic
from safety_data.paths import assert_development_path
from train.config import load_app_config


REWARD_Q_MANIFEST_VERSION = "qsafe.frozen_droq_reward_q.v1"
REWARD_Q_AGGREGATION = "pointwise_min_online_critic_ensemble"
REWARD_Q_ACTION_SEMANTICS = "normalized_requested_action_replay_field"
_STEP_COMPONENT = re.compile(r"^step_(\d+)$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_dict_sha256(state_dict: Mapping[str, Any]) -> str:
    """Hash the critic weights independent of torch serialization details."""
    if not state_dict:
        raise ValueError("critic state_dict must be nonempty")
    if any(not isinstance(name, str) for name in state_dict):
        raise ValueError("critic state_dict keys must be strings")
    digest = hashlib.sha256(b"qsafe_reward_critic_state_dict_v1\0")
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"critic state_dict entry {name!r} must be a tensor")
        if tensor.layout != torch.strided:
            raise ValueError(
                f"critic state_dict entry {name!r} must be a dense tensor")
        detached = tensor.detach()
        if detached.is_meta:
            raise ValueError(
                f"critic state_dict entry {name!r} must contain materialized data")
        value = detached.cpu().contiguous()
        if (value.is_floating_point() or value.is_complex()) and not bool(
                torch.isfinite(value).all()):
            raise ValueError(
                f"critic state_dict entry {name!r} must contain finite values")
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


def _critic_path(checkpoint: str | Path) -> Path:
    checked = assert_development_path(checkpoint)
    critic = checked / "critic.pt" if checked.is_dir() else checked
    critic = assert_development_path(critic)
    if critic.name != "critic.pt":
        raise ValueError(
            "frozen DroQ reward-Q checkpoints must be an agent directory "
            "or critic.pt")
    if not critic.is_file():
        raise FileNotFoundError(critic)
    return critic


def _inferred_training_step(critic_path: Path) -> int | None:
    for parent in critic_path.parents:
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


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _jsonable(value: Any) -> Any:
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


def _runtime_device(value: str | torch.device) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid frozen reward-Q device {value!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("frozen DroQ reward-Q device must be CPU or CUDA")
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU reward-Q device must not include an index")
        return torch.device("cpu")
    index = 0 if device.index is None else int(device.index)
    if not torch.cuda.is_available():
        raise ValueError("CUDA reward-Q device requested but CUDA is unavailable")
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA reward-Q device index {index} is unavailable")
    return torch.device(f"cuda:{index}")


@dataclass(frozen=True)
class RewardQEvaluation:
    """Reward-Q values for K requested actions at one observation.

    ``conservative`` always has shape ``[K]``.  ``per_critic`` is ``None``
    unless explicitly requested, in which case it has shape ``[num_qs, K]``.
    Arrays are independent CPU float32 copies.
    """

    conservative: np.ndarray
    per_critic: np.ndarray | None
    aggregation: str = REWARD_Q_AGGREGATION


class FrozenDroQRewardQ:
    """Inference-only online DroQ reward critic with immutable provenance."""

    def __init__(
        self,
        critic: DroQEnsembleCritic,
        manifest: Mapping[str, Any],
    ) -> None:
        self._critic = critic
        self._manifest = copy.deepcopy(dict(manifest))
        self._observation_dim = int(self._manifest["observation_dim"])
        self._action_dim = int(self._manifest["action_dim"])
        self._num_qs = int(self._manifest["num_qs"])
        self._device = next(critic.parameters()).device

    @property
    def training_step(self) -> int:
        return int(self._manifest["training_step"])

    @property
    def critic_sha256(self) -> str:
        return str(self._manifest["critic_sha256"])

    @property
    def critic_state_dict_sha256(self) -> str:
        return str(self._manifest["critic_state_dict_sha256"])

    @property
    def num_qs(self) -> int:
        return self._num_qs

    @property
    def device(self) -> torch.device:
        return self._device

    def manifest(self) -> dict[str, Any]:
        """Return an isolated, JSON-serializable critic manifest."""
        return copy.deepcopy(self._manifest)

    def fingerprint(self) -> str:
        """Return the device-independent loaded reward-Q fingerprint."""
        return str(self._manifest["reward_q_fingerprint_sha256"])

    def checkpoint_fingerprint(self) -> str:
        """Return provenance identity including files and recorded steps."""
        return str(self._manifest["checkpoint_fingerprint_sha256"])

    def _checked_observation(self, value: np.ndarray) -> np.ndarray:
        observation = np.asarray(value, dtype=np.float32).reshape(-1)
        if observation.shape != (self._observation_dim,):
            raise ValueError(
                f"observation must have shape {(self._observation_dim,)}")
        if not np.all(np.isfinite(observation)):
            raise ValueError("observation must be finite")
        return observation

    def _checked_actions(self, value: np.ndarray) -> np.ndarray:
        actions = np.asarray(value, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1:] != (self._action_dim,):
            raise ValueError(
                f"requested_actions must have shape [K, {self._action_dim}]")
        if actions.shape[0] == 0:
            raise ValueError("requested_actions must contain at least one action")
        if not np.all(np.isfinite(actions)):
            raise ValueError("requested_actions must be finite")
        if np.any(actions < -1.0) or np.any(actions > 1.0):
            raise ValueError("requested_actions must lie in [-1, 1]")
        return np.ascontiguousarray(actions)

    @torch.no_grad()
    def evaluate(
        self,
        observation: np.ndarray,
        requested_actions: np.ndarray,
        *,
        include_per_critic: bool = False,
    ) -> RewardQEvaluation:
        """Evaluate K requested actions without consuming any RNG.

        The same observation is repeated K times.  Conservative reward-Q is
        the minimum over online ensemble heads for each action; it is not a
        statistical lower confidence bound and it does not use target critics.
        """
        if not isinstance(include_per_critic, bool):
            raise TypeError("include_per_critic must be bool")
        observation_array = self._checked_observation(observation)
        action_array = self._checked_actions(requested_actions)
        observations = torch.as_tensor(
            observation_array, device=self._device).reshape(1, -1).expand(
                action_array.shape[0], -1)
        actions = torch.as_tensor(action_array, device=self._device)
        values, _ = self._critic(
            observations, actions, training=False)
        if values.shape != (self._num_qs, action_array.shape[0]):
            raise RuntimeError(
                "frozen DroQ critic returned an invalid ensemble shape")
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("frozen DroQ critic returned non-finite values")
        per_critic = values.detach().cpu().to(torch.float32).numpy().copy()
        conservative = np.min(per_critic, axis=0).astype(
            np.float32, copy=True)
        return RewardQEvaluation(
            conservative=conservative,
            per_critic=per_critic if include_per_critic else None,
        )

    def conservative_values(
        self,
        observation: np.ndarray,
        requested_actions: np.ndarray,
    ) -> np.ndarray:
        """Return only the pointwise ensemble-min reward-Q vector ``[K]``."""
        return self.evaluate(observation, requested_actions).conservative

    def __call__(
        self,
        observation: np.ndarray,
        requested_actions: np.ndarray,
    ) -> np.ndarray:
        return self.conservative_values(observation, requested_actions)


def load_frozen_droq_reward_q(
    checkpoint: str | Path,
    config: str | Path,
    *,
    observation_dim: int,
    action_dim: int,
    training_step: int | None = None,
    device: str | torch.device = "cpu",
) -> FrozenDroQRewardQ:
    """Load only ``critic.pt['network_state_dict']`` for reward-Q inference.

    ``training_step`` is inferred from a ``step_<integer>`` ancestor when
    possible.  The checkpoint's ``update_step`` is recorded separately because
    older asynchronous snapshots may leave that network-wrapper counter at
    zero even when the environment training step is known.
    """
    critic_path = _critic_path(checkpoint)
    config_path = assert_development_path(config)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    observation_dim = _positive_int(observation_dim, "observation_dim")
    action_dim = _positive_int(action_dim, "action_dim")
    runtime_device = _runtime_device(device)

    inferred_step = _inferred_training_step(critic_path)
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
        raise RuntimeError("config changed while frozen reward-Q was being loaded")
    loaded_agent_type = str(getattr(loaded_cfg, "agent_type", "")).lower()
    if loaded_agent_type != "droq":
        raise ValueError(
            f"frozen reward-Q loader requires agent_type='droq', got "
            f"{loaded_agent_type!r}")
    source_resolved_cfg = OmegaConf.create(OmegaConf.to_container(
        loaded_cfg, resolve=True, throw_on_missing=True))

    hidden_dims = [
        _positive_int(value, f"hidden_dims[{index}]")
        for index, value in enumerate(source_resolved_cfg.hidden_dims)
    ]
    if not hidden_dims:
        raise ValueError("hidden_dims must be nonempty")
    num_qs = _positive_int(source_resolved_cfg.num_qs, "num_qs")
    dropout_rate = float(source_resolved_cfg.critic_dropout_rate)
    if not np.isfinite(dropout_rate) or not 0.0 <= dropout_rate < 1.0:
        raise ValueError("critic_dropout_rate must lie in [0, 1)")
    layer_norm = bool(source_resolved_cfg.critic_layer_norm)

    # Hash and deserialize the same immutable bytes, preventing a hash/load
    # TOCTOU mismatch.  weights_only=True restricts checkpoint unpickling; any
    # bundled optimizer payload is ignored after parsing and never installed.
    checkpoint_bytes = critic_path.read_bytes()
    critic_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint_data = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_data, Mapping):
        raise ValueError("critic.pt must contain a checkpoint mapping")
    state_dict = checkpoint_data.get("network_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("critic.pt is missing a nonempty network_state_dict")
    critic_state_dict_sha256 = _state_dict_sha256(state_dict)
    critic_update_step = _nonnegative_int(
        checkpoint_data.get("update_step", 0), "critic update_step")

    initialization_cuda_devices = (
        [int(runtime_device.index)] if runtime_device.type == "cuda" else [])
    with torch.random.fork_rng(
            devices=initialization_cuda_devices, enabled=True):
        critic = DroQEnsembleCritic(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            num_qs=num_qs,
            dropout_rate=dropout_rate,
            use_layer_norm=layer_norm,
        ).to(runtime_device)
    try:
        critic.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "critic checkpoint architecture does not match resolved config "
            "and dimensions") from exc
    critic.eval()

    resolved_agent_config = _jsonable(OmegaConf.to_container(
        source_resolved_cfg, resolve=True, throw_on_missing=True))
    resolved_train_config = _jsonable(asdict(train_cfg))
    behavior_config = {
        "agent_type": "droq",
        "network_class": "DroQEnsembleCritic",
        "network_source": "online_reward_critic",
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "hidden_dims": hidden_dims,
        "num_qs": num_qs,
        "critic_dropout_rate": dropout_rate,
        "critic_layer_norm": layer_norm,
        "ensemble_aggregation": REWARD_Q_AGGREGATION,
        "action_semantics": REWARD_Q_ACTION_SEMANTICS,
    }
    fingerprint_basis = {
        "manifest_version": REWARD_Q_MANIFEST_VERSION,
        "critic_state_dict_sha256": critic_state_dict_sha256,
        "behavior_config": behavior_config,
    }
    reward_q_fingerprint_sha256 = _canonical_sha256(fingerprint_basis)
    checkpoint_basis = {
        "manifest_version": REWARD_Q_MANIFEST_VERSION,
        "reward_q_fingerprint_sha256": reward_q_fingerprint_sha256,
        "critic_sha256": critic_sha256,
        "config_sha256": config_sha256_after,
        "resolved_agent_config_sha256": _canonical_sha256(
            resolved_agent_config),
        "resolved_train_config_sha256": _canonical_sha256(
            resolved_train_config),
        "training_step": source_training_step,
        "critic_update_step": critic_update_step,
    }
    manifest = {
        **checkpoint_basis,
        **behavior_config,
        "critic_state_dict_sha256": critic_state_dict_sha256,
        "reward_q_fingerprint_sha256": reward_q_fingerprint_sha256,
        "checkpoint_fingerprint_sha256": _canonical_sha256(checkpoint_basis),
        "critic_path": str(critic_path),
        "config_path": str(config_path),
        "resolved_agent_config": resolved_agent_config,
        "train_config": resolved_train_config,
        "device": str(runtime_device),
        "load_contract": "critic_pt_network_state_dict_only",
    }
    return FrozenDroQRewardQ(critic, manifest)


__all__ = [
    "FrozenDroQRewardQ",
    "REWARD_Q_ACTION_SEMANTICS",
    "REWARD_Q_AGGREGATION",
    "REWARD_Q_MANIFEST_VERSION",
    "RewardQEvaluation",
    "load_frozen_droq_reward_q",
]
