from typing import Any

import gymnasium as gym

from rl.agents.base.agent import BaseAgent
from rl.utils.types import NDArray


def create_agent(
    observation_space: gym.spaces.Space[NDArray],
    action_space: gym.spaces.Space[NDArray],
    env_info: dict[str, Any],
    cfg: Any,
) -> BaseAgent[Any]:
    from omegaconf import OmegaConf

    cfg_dict = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("cfg must be a dictionary")
    cfg_dict = {str(k): v for k, v in cfg_dict.items()}
    agent_type = cfg_dict.pop("agent_type")

    agent: BaseAgent[Any]

    # sanity check
    if agent_type == "flashSAC":

        from rl.agents.flashsac.agent import (
            FlashSACAgent,
            FlashSACConfig,
        )

        config = FlashSACConfig(**cfg_dict)  # type: ignore
        agent = FlashSACAgent(observation_space, action_space, env_info, config)

    else:
        raise NotImplementedError

    return agent

