from abc import ABC, abstractmethod
from typing import Any, Generic, MutableMapping, TypeVar

import gymnasium as gym

from rl.utils.types import NDArray, Tensor
from rl.agents.base.update import PolicyUpdateRequest

Config = TypeVar("Config")


class BaseAgent(Generic[Config], ABC):
    def get_last_action_trace(self) -> dict[str, Any]:
        """Return metadata for the most recently sampled action."""
        return {}

    def get_update_step(self) -> int:
        """Return the current policy learner update step."""
        return 0

    def update_policy_steps(
        self,
        request: PolicyUpdateRequest,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement "
            "update_policy_steps()")

    def get_update_counters(self) -> dict[str, int]:
        return {}

    def get_policy_update_step(self) -> int:
        counters = self.get_update_counters()
        return int(counters.get("actor_steps", 0))

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: Config,
    ):
        """
        A generic agent class.
        """
        self._observation_space = observation_space
        self._action_space = action_space
        self._cfg = cfg

    @abstractmethod
    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        pass

    @abstractmethod
    def process_transition(
        self,
        transition: MutableMapping[str, Tensor],
    ) -> None:
        """Handle interaction samples (e.g., add to replay buffer)"""
        pass

    @abstractmethod
    def can_start_training(self) -> bool:
        """Whether the agent is ready to update (e.g., enough samples in buffer)"""
        pass

    @abstractmethod
    def update(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def save_replay_buffer(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def load_replay_buffer(self, path: str) -> None:
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        pass

    @property
    def observation_space(self) -> gym.spaces.Space[NDArray]:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space[NDArray]:
        return self._action_space

    @property
    def cfg(self) -> Config:
        return self._cfg
