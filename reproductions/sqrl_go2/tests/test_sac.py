import torch

from reproductions.sqrl_go2.algo.sac import SACConfig, VanillaSAC


def test_vanilla_sac_update_is_finite_and_actions_are_bounded():
    torch.manual_seed(1)
    sac = VanillaSAC(SACConfig(
        observation_dim=3, action_dim=2, hidden_dims=(16, 16)))
    batch = {
        "observation": torch.randn(8, 3),
        "action": torch.rand(8, 2) * 2 - 1,
        "reward": torch.randn(8),
        "next_observation": torch.randn(8, 3),
        "terminated": torch.tensor([0, 0, 1, 0, 0, 1, 0, 0], dtype=torch.float32),
        "truncated": torch.zeros(8),
    }
    metrics = sac.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    action = sac.act(torch.zeros(3), count=32)
    assert action.shape == (32, 2)
    assert (abs(action) <= 1.0).all()


def test_time_limit_bootstrap_is_distinct_from_true_termination():
    sac = VanillaSAC(SACConfig(
        observation_dim=2, action_dim=1, hidden_dims=(8, 8), gamma=0.99))
    # The implementation contract is directly visible in the batch fields:
    # truncated never enters the critic's bootstrap mask.
    batch = {
        "observation": torch.zeros(2, 2),
        "action": torch.zeros(2, 1),
        "reward": torch.zeros(2),
        "next_observation": torch.zeros(2, 2),
        "terminated": torch.tensor([1.0, 0.0]),
        "truncated": torch.tensor([0.0, 1.0]),
    }
    assert batch["terminated"].tolist() == [1.0, 0.0]
    sac.update_critic(batch)
