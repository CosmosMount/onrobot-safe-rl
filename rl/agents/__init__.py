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
    if agent_type == "flashsac":

        from rl.agents.flashsac.agent import (
            FlashSACAgent,
            FlashSACConfig,
        )

        config = FlashSACConfig(**cfg_dict)  # type: ignore
        agent = FlashSACAgent(observation_space, action_space, env_info, config)

    elif agent_type == "droq":

        from rl.agents.droq.agent import DroQAgent, DroQConfig

        config = DroQConfig(**cfg_dict)  # type: ignore
        agent = DroQAgent(observation_space, action_space, env_info, config)

    elif agent_type == "safe_droq":

        from rl.agents.safe_droq.agent import SafeDroQAgent, SafeDroQConfig

        config = SafeDroQConfig(**cfg_dict)  # type: ignore
        agent = SafeDroQAgent(
            observation_space, action_space, env_info, config)

    elif agent_type == "paper_sqrl":

        from rl.agents.paper_sqrl.agent import (
            PaperSQRLAgent,
            PaperSQRLConfig,
        )

        config = PaperSQRLConfig(**cfg_dict)  # type: ignore
        agent = PaperSQRLAgent(
            observation_space, action_space, env_info, config)

    else:
        raise NotImplementedError

    # Preserve the selected type on the concrete dataclass. Asynchronous
    # inference workers receive this optimizer-free config after OmegaConf has
    # been converted and otherwise cannot distinguish SAC from paper SQRL.
    setattr(agent.cfg, "agent_type", agent_type)
    return agent
