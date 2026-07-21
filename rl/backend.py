"""Common external interface for learner backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AgentBackend(Protocol):
    agent_type: str

    def sample_actions(
            self, observation: np.ndarray) -> tuple[np.ndarray, 'AgentBackend']:
        ...

    def eval_actions(self, observation: np.ndarray) -> np.ndarray:
        ...

    def update(self, batch: dict, utd_ratio: int) -> tuple['AgentBackend',
                                                          dict[str, float]]:
        ...

    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, state: dict) -> 'AgentBackend':
        ...
