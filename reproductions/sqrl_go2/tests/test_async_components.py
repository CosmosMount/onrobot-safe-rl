import numpy as np
import torch

from reproductions.sqrl_go2.algo.sac import SACConfig, VanillaSAC
from reproductions.sqrl_go2.algo.safety_critic import (
    SafetyCriticConfig, SafetyCriticLearner)
from reproductions.sqrl_go2.env.adapter import ObservationStack
from reproductions.sqrl_go2.env.async_collector import inference_snapshot


def test_observation_stack_is_oldest_to_newest_and_resets_without_aliasing():
    stack = ObservationStack(3, 2)
    np.testing.assert_array_equal(
        stack.reset(np.asarray([1, 2], np.float32)), [1, 2, 1, 2, 1, 2])
    np.testing.assert_array_equal(
        stack.append(np.asarray([3, 4], np.float32)), [1, 2, 1, 2, 3, 4])


def test_async_snapshot_is_an_independent_cpu_copy():
    sac = VanillaSAC(SACConfig(
        observation_dim=3, action_dim=1, hidden_dims=(8, 8)))
    safety = SafetyCriticLearner(SafetyCriticConfig(
        observation_dim=3, action_dim=1, hidden_dims=(8, 8)))
    snapshot = inference_snapshot(sac.actor, safety.critic)
    source = next(iter(sac.actor.state_dict().values()))
    copied = next(iter(snapshot["actor"].values()))
    assert copied.device.type == "cpu"
    assert copied.data_ptr() != source.data_ptr()
    before = copied.clone()
    with torch.no_grad():
        source.add_(1.0)
    torch.testing.assert_close(copied, before)
