"""Contracts shared by optimizer-free asynchronous inference policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


def _action(value: np.ndarray, action_dim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    if result.shape != (action_dim,) or not np.all(np.isfinite(result)):
        raise ValueError("inference actions must be finite float32 vectors")
    return result


@dataclass(frozen=True)
class ActionDecision:
    """One immutable action decision made by a collector-owned policy."""

    action_nominal: np.ndarray
    action_requested: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        nominal = np.asarray(self.action_nominal, dtype=np.float32).reshape(-1).copy()
        requested = np.asarray(self.action_requested, dtype=np.float32).reshape(-1).copy()
        if nominal.shape != requested.shape:
            raise ValueError("nominal and requested actions must have equal shape")
        if not np.all(np.isfinite(nominal)) or not np.all(np.isfinite(requested)):
            raise ValueError("inference actions must be finite")
        nominal.setflags(write=False)
        requested.setflags(write=False)
        object.__setattr__(self, "action_nominal", nominal)
        object.__setattr__(self, "action_requested", requested)
        object.__setattr__(self, "metadata", dict(self.metadata))


class InferencePolicy(Protocol):
    snapshot_version: int
    actor_steps: int
    auxiliary_steps: int

    def load_snapshot(self, snapshot: dict[str, Any]) -> None: ...

    def decide(self, observation: np.ndarray, *, training: bool,
               action_nominal: np.ndarray | None = None) -> ActionDecision: ...

    def observe_transition(self, *, policy_step: bool, terminated: bool,
                           truncated: bool) -> dict[str, bool]: ...

    def transition_fields(self, decision: ActionDecision) -> dict[str, np.ndarray]: ...
