from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from rl.agents import create_agent
from rl.agents.inference import build_inference_policy
from train.config import load_app_config


def _config():
    _, train_cfg, cfg = load_app_config(path="config/go2_livesac.yaml")
    cfg.device_type = "cpu"
    cfg.buffer_device_type = "cpu"
    cfg.actor_hidden_dims = [16, 16]
    cfg.critic_hidden_dim = 16
    cfg.critic_num_blocks = 1
    cfg.critic_num_qs = 2
    cfg.critic_num_bins = 11
    cfg.buffer_min_length = 1
    cfg.buffer_max_length = 32
    cfg.sample_batch_size = 1
    return train_cfg, cfg


def test_livesac_profile_enables_async_collection():
    train_cfg, cfg = _config()
    assert train_cfg.async_collection
    assert train_cfg.agent == "livesac"
    assert train_cfg.utd_ratio == 5
    assert cfg.sampled_backup is True


def test_livesac_async_inference_snapshot_matches_actor():
    _, cfg = _config()
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    agent = create_agent(obs_space, action_space, {}, cfg)
    snapshot = agent.export_inference_snapshot(snapshot_version=3)
    policy = build_inference_policy(4, 2, cfg)
    policy.load_snapshot(snapshot)
    observation = np.zeros(4, dtype=np.float32)
    action = policy.decide(observation, training=False).action_requested
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))
    assert policy.snapshot_version == 3


def test_livesac_reward_normalization_is_independent_switch():
    _, cfg = _config()
    cfg.normalize_reward = False
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    agent = create_agent(obs_space, action_space, {}, cfg)
    assert agent._reward_normalizer is None
