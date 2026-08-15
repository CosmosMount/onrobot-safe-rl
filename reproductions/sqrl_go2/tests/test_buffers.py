import numpy as np

from reproductions.sqrl_go2.algo.buffers import SafetyReplayBuffer, Transition


def item(value, *, terminated=False, truncated=False):
    return Transition(
        np.asarray([value], np.float32), np.asarray([value], np.float32), 0.0,
        np.asarray([value + 1], np.float32), float(terminated),
        terminated, truncated)


def test_safety_buffer_retains_latest_complete_trajectories_only():
    replay = SafetyReplayBuffer(max_trajectories=2, seed=1)
    for episode in range(3):
        replay.add(item(episode * 2))
        replay.add(item(episode * 2 + 1, truncated=True))
    assert replay.trajectory_count == 2
    assert len(replay) == 4
    assert replay.fall_count == 0


def test_fall_cost_must_equal_termination():
    try:
        Transition(np.zeros(1), np.zeros(1), 0.0, np.zeros(1), 1.0, False, False)
    except ValueError as exc:
        assert "cost" in str(exc)
    else:
        raise AssertionError("inconsistent fall transition was accepted")
