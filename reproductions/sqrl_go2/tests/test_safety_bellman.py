import torch

from reproductions.sqrl_go2.algo.safety_critic import safety_bellman_target


def test_failure_is_absorbing_and_normal_transition_bootstraps():
    cost = torch.tensor([1.0, 0.0])
    next_q = torch.tensor([99.0, 0.4])
    target = safety_bellman_target(cost, next_q, gamma=0.7)
    torch.testing.assert_close(target, torch.tensor([1.0, 0.28]))


def test_sparse_cost_is_enforced():
    try:
        safety_bellman_target(torch.tensor([0.5]), torch.tensor([0.0]), 0.7)
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("non-binary safety cost was accepted")
