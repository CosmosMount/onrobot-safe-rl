import gymnasium as gym
import numpy as np

from rl.agents import create_agent
from rl.agents.base.update import PolicyUpdateRequest
from train.config import load_app_config


def _agent():
    _, _, cfg = load_app_config(path="config/go2.yaml")
    cfg.device_type = "cpu"
    cfg.buffer_device_type = "cpu"
    cfg.hidden_dims = [16, 16]
    cfg.num_qs = 5
    cfg.num_min_qs = 2
    cfg.buffer_min_length = 4
    cfg.buffer_max_length = 32
    cfg.sample_batch_size = 4
    return create_agent(
        gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
        gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32), {}, cfg)


def _transition():
    return {
        "observation": np.random.randn(1, 4).astype(np.float32),
        "action": np.random.uniform(-1, 1, (1, 2)).astype(np.float32),
        "reward": np.asarray([0.1], dtype=np.float32),
        "terminated": np.asarray([0.0], dtype=np.float32),
        "truncated": np.asarray([0.0], dtype=np.float32),
        "next_observation": np.random.randn(1, 4).astype(np.float32),
    }


def test_categorical_droq_uses_droq_actor_and_head_only_critic_change():
    agent = _agent()
    assert type(agent._actor.network).__name__ == "DroQActor"
    assert len(agent._critic.network.qs) == 5
    assert agent._critic.network.qs[0].base.net[0].out_features == 16
    assert agent._critic.network.qs[0].logits.out_features == 101
    for _ in range(4):
        agent.process_transition(_transition())
    metrics = agent.update_policy_steps(PolicyUpdateRequest(1, 2))
    assert all(np.isfinite(value) for value in metrics.values())
    assert agent.export_inference_snapshot(snapshot_version=1)["agent_type"] == "categorical_droq"
