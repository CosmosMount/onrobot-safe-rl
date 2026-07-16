from typing import Any, Dict

import numpy as np


def evaluate(agent, env: Any, num_episodes: int) -> Dict[str, float]:
    returns = []
    lengths = []
    for _ in range(num_episodes):
        observation = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            action = agent.eval_actions(observation)
            observation, reward, done, info = env.step(action)
            episode_return += reward
            if info.get('policy_step', True):
                episode_length += 1
        returns.append(episode_return)
        lengths.append(episode_length)

    return {
        'return': np.mean(returns),
        'length': np.mean(lengths),
    }
