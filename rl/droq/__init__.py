from __future__ import annotations

from flax import serialization

from rl.droq.agents.sac.sac_learner import SACLearner


class DroQLearner(SACLearner):
    @property
    def agent_type(self) -> str:
        return 'droq'

    def state_dict(self) -> dict:
        return serialization.to_state_dict(self)

    def load_state_dict(self, state: dict) -> 'DroQLearner':
        return serialization.from_state_dict(self, state)


__all__ = ['DroQLearner', 'SACLearner']
