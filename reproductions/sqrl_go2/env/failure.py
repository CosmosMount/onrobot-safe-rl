"""The single sparse safety signal shared by both SQRL phases."""

from __future__ import annotations

from typing import Any, Mapping


def failure_cost(info: Mapping[str, Any], *, terminated: bool, truncated: bool) -> float:
    """Return c[t+1] for a transition entering a terminal Go2 fall.

    Time-limit truncation is explicitly non-failure.  The reproduction does
    not consume near-fall, pitch, roll, or recovery labels.
    """
    if terminated and truncated:
        raise ValueError("a transition cannot be both fall-terminated and time-limit truncated")
    if truncated:
        return 0.0
    reported_fall = bool(info.get("fallen", False) or info.get("inverted", False))
    # The canonical Go2 supervisor debounces inversion for several policy
    # ticks. During that debounce window ``inverted`` can be true while the
    # MDP has not yet entered its first-fall terminal state. The reproduction
    # cost is therefore bound to the supervisor's canonical termination, not
    # to instantaneous pose flags. A termination without either pose flag is
    # still inconsistent and fails closed.
    if terminated and not reported_fall:
        raise ValueError("Go2 first-fall termination lacks its fall predicate")
    return float(terminated)
