"""Strict configuration loader for the isolated SQRL reproduction."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class EnvironmentConfig:
    observation_dim: int = 46
    observation_frames: int = 5
    action_dim: int = 12
    control_frequency: float = 50.0
    max_episode_steps: int = 500
    action_filter: bool = False
    max_joint_delta: float | None = None


@dataclass(frozen=True)
class ReplayConfig:
    task_capacity: int = 1_000_000
    safety_trajectories: int = 10
    batch_size: int = 256
    minimum_task_transitions: int = 1_000
    minimum_safety_transitions: int = 256


@dataclass(frozen=True)
class SQRLConfig:
    gamma_safe: float = 0.7
    epsilon_safe: float = 0.1
    mask_candidates: int = 100
    task_steps_per_cycle: int = 1_000
    safety_trajectories_per_cycle: int = 1
    safety_updates_per_cycle: int = 1
    safety_lagrange_initial: float = 0.0
    safety_lagrange_lr: float = 3e-4


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    device: str = "cpu"
    hidden_dims: tuple[int, ...] = (256, 256)
    checkpoint_interval: int = 25_000
    output_dir: str = "saved/reproductions/sqrl_go2"


@dataclass(frozen=True)
class DevelopmentProtocolConfig:
    pretrain_steps: int = 25_000
    target_steps: int = 10_000
    pretrain_seeds: tuple[int, ...] = (0, 1, 2)
    target_seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    phase: str
    move_speed: float
    environment: EnvironmentConfig
    replay: ReplayConfig
    sqrl: SQRLConfig
    training: TrainingConfig
    development_protocol: DevelopmentProtocolConfig

    @property
    def stacked_observation_dim(self) -> int:
        return self.environment.observation_dim * self.environment.observation_frames


T = TypeVar("T")


def _strict_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    known = {field.name for field in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    if "hidden_dims" in values:
        values["hidden_dims"] = tuple(int(x) for x in values["hidden_dims"])
    if "pretrain_seeds" in values:
        values["pretrain_seeds"] = tuple(int(x) for x in values["pretrain_seeds"])
    return cls(**values)


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with (path.parent / "base.yaml").open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}
    if path.name == "base.yaml":
        values = base
    else:
        with path.open("r", encoding="utf-8") as handle:
            values = _merge(base, yaml.safe_load(handle) or {})
    required = {"phase", "move_speed"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"missing experiment keys: {sorted(missing)}")
    known = {
        "phase", "move_speed", "environment", "replay", "sqrl", "training",
        "development_protocol",
        # The phase overlays are also consumed directly by the established Go2
        # runtime loader. These keys belong to that explicitly shared boundary.
        "reward_profile", "reward_command_vx", "train",
    }
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown experiment keys: {sorted(unknown)}")
    cfg = ExperimentConfig(
        phase=str(values["phase"]),
        move_speed=float(values["move_speed"]),
        environment=_strict_dataclass(EnvironmentConfig, values.get("environment", {})),
        replay=_strict_dataclass(ReplayConfig, values.get("replay", {})),
        sqrl=_strict_dataclass(SQRLConfig, values.get("sqrl", {})),
        training=_strict_dataclass(TrainingConfig, values.get("training", {})),
        development_protocol=_strict_dataclass(
            DevelopmentProtocolConfig, values.get("development_protocol", {})),
    )
    if cfg.phase not in {"pretrain", "target"}:
        raise ValueError("phase must be pretrain or target")
    if cfg.environment.action_filter or cfg.environment.max_joint_delta is not None:
        raise ValueError(
            "first SQRL reproduction requires identity requested/executed normalized action; "
            "keep action_filter=false and max_joint_delta=null")
    return cfg
