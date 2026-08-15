import torch

from reproductions.sqrl_go2.algo.finetune import SafetyLagrange


def test_positive_violation_increases_dual_and_safe_actions_decrease_it():
    dual = SafetyLagrange(0.2, 0.05, torch.device("cpu"))
    before = float(dual.value.detach())
    dual.update(torch.tensor([0.3, 0.2]))
    increased = float(dual.value.detach())
    assert increased > before
    dual.update(torch.tensor([-0.4, -0.2]))
    assert float(dual.value.detach()) < increased


def test_dual_is_projected_nonnegative():
    dual = SafetyLagrange(0.0, 0.1, torch.device("cpu"))
    for _ in range(3):
        dual.update(torch.tensor([-1.0]))
    assert float(dual.value.detach()) == 0.0
