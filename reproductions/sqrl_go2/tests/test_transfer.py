import tempfile

import torch

from reproductions.sqrl_go2.algo.checkpoint import (
    load_pretrain_checkpoint, save_pretrain_checkpoint)
from reproductions.sqrl_go2.algo.sac import SACConfig, VanillaSAC
from reproductions.sqrl_go2.algo.safety_critic import (
    SafetyCriticConfig, SafetyCriticLearner)


def _models(seed):
    torch.manual_seed(seed)
    sac = VanillaSAC(SACConfig(
        observation_dim=3, action_dim=2, hidden_dims=(8, 8)))
    safety = SafetyCriticLearner(SafetyCriticConfig(
        observation_dim=3, action_dim=2, hidden_dims=(8, 8)))
    return sac, safety


def test_all_target_branches_receive_same_actor_but_fresh_task_critics():
    source_sac, source_safety = _models(1)
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/pretrain.pt"
        save_pretrain_checkpoint(path, source_sac, source_safety, {"seed": 1})
        actors = []
        for branch in ("sac_transfer", "sqrl_mask", "sqrl_full"):
            target_sac, target_safety = _models(10 + len(actors))
            load_pretrain_checkpoint(
                path, target_sac,
                None if branch == "sac_transfer" else target_safety, branch)
            actors.append([value.clone() for value in target_sac.actor.state_dict().values()])
            assert any(
                not torch.equal(a, b)
                for a, b in zip(source_sac.q1.parameters(), target_sac.q1.parameters()))
            if branch != "sac_transfer":
                assert not any(parameter.requires_grad for parameter in target_safety.critic.parameters())
        for left, right in zip(actors[0], actors[1]):
            torch.testing.assert_close(left, right)
        for left, right in zip(actors[0], actors[2]):
            torch.testing.assert_close(left, right)
