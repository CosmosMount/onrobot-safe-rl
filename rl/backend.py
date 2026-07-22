"""Common external interface for learner backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AgentBackend(Protocol):
    agent_type: str
    owns_replay_buffer: bool

    def sample_actions(
            self, observation: np.ndarray) -> tuple[np.ndarray, 'AgentBackend']:
        ...

    def eval_actions(self, observation: np.ndarray) -> np.ndarray:
        ...

    def process_transition(self, transition: dict) -> None:
        ...

    def can_start_training(self) -> bool:
        ...

    def replay_size(self) -> int:
        ...

    def update(self, *args, **kwargs):
        ...

    def state_dict(self) -> dict:
        ...

    def load_state_dict(self, state: dict) -> 'AgentBackend':
        ...
