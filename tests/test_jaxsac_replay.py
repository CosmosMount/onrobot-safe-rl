import numpy as np

from train.train import Replay


def test_utd_sample_allows_replacement_from_one_minibatch():
    replay = Replay(32, (3,), (2,), seed=7)
    for i in range(4):
        replay.insert(
            np.full(3, i, np.float32),
            np.full(2, i, np.float32),
            float(i),
            1.0,
            np.full(3, i + 1, np.float32),
        )

    batch = replay.sample(4 * 20)
    assert batch.observations.shape == (80, 3)
    assert batch.actions.shape == (80, 2)
    assert batch.rewards.shape == (80,)
    # Sampling is explicitly with replacement; repeated source indices are
    # expected and are what makes UTD=20 possible with only 256 transitions.
    assert np.unique(batch.rewards).size <= 4


def test_delayed_failure_patches_recent_action_as_terminal():
    replay = Replay(8, (3,), (2,), seed=3)
    replay.insert(np.zeros(3), np.zeros(2), 5.0, 1.0, np.ones(3),
                  action_id=41)
    result = replay.patch_terminal(41, -1.0, np.full(3, 9.0))
    assert result is not None
    assert result["old_reward"] == 5.0
    assert replay.data["rewards"][result["index"]] == -1.0
    assert replay.data["discounts"][result["index"]] == 0.0
    np.testing.assert_allclose(
        replay.data["next_observations"][result["index"]], 9.0)
    assert replay.patch_terminal(999, 0.0, np.zeros(3)) is None
