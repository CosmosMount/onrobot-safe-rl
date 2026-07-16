"""Versioned actor inference wrapper for collector loops."""

from __future__ import annotations

import numpy as np


class ActorRunner:
    def __init__(self, agent, policy_version: int = 0):
        self.agent = agent
        self.policy_version = policy_version

    def update_agent(self, agent) -> None:
        self.agent = agent
        self.policy_version += 1

    def sample_action(self, observation: np.ndarray):
        action, next_agent = self.agent.sample_actions(observation)
        self.agent = next_agent
        return action, self.policy_version
