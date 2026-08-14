import numpy as np

from train.train import _window_stats


def test_window_stats_converts_seconds_to_milliseconds():
    stats = _window_stats(np.asarray([0.010, 0.020, 0.030]), 100,
                          scale=1000.0)
    assert stats == {"p50": 20.0, "p95": 29.0, "max": 30.0}


def test_window_stats_keeps_action_period_milliseconds():
    stats = _window_stats(np.asarray([49.0, 50.0, 51.0]), 100)
    assert stats == {"p50": 50.0, "p95": 50.9, "max": 51.0}
