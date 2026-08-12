"""Protected statistics for the PPO same-state action-ranking gate."""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np


def stable_state_indices(identities: np.ndarray, count: int) -> np.ndarray:
    """Choose protected states by identity hash, independent of outcomes."""
    if count <= 0 or count > len(identities):
        raise ValueError("count must be positive and no larger than the pool")
    score = np.asarray([
        hashlib.sha256(b"qsafe.ppo.branch.state.v1\0" + bytes(value)).digest()
        for value in identities
    ], dtype="S32")
    return np.argsort(score, kind="stable")[:count]


def independent_oracle(fall: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discover with R1--R4 and evaluate the frozen choice on R5--R8."""
    value = np.asarray(fall, bool)
    if value.ndim != 3 or value.shape[2] != 8:
        raise ValueError("fall must have shape [state,candidate,8]")
    choice = np.argmin(value[:, :, :4].mean(axis=2), axis=1)
    evaluation = value[np.arange(len(value)), choice, 4:].mean(axis=1)
    return choice.astype(np.int16), evaluation


def selector_outcome(fall: np.ndarray, choice: np.ndarray) -> np.ndarray:
    value = np.asarray(fall, bool)
    selected = np.asarray(choice, np.int64)
    if value.ndim != 3 or value.shape[2] != 8 or selected.shape != (len(value),):
        raise ValueError("selector outcome inputs do not align")
    if np.any(selected < 0) or np.any(selected >= value.shape[1]):
        raise ValueError("selector choice is out of range")
    return value[np.arange(len(value)), selected, 4:].mean(axis=1)


def state_bootstrap_lcb(
    nominal: np.ndarray, selected: np.ndarray, *, seed: int = 20260812,
    draws: int = 20_000,
) -> float:
    """One-sided 95% LCB for nominal-minus-selected fall probability."""
    nominal = np.asarray(nominal, np.float64)
    selected = np.asarray(selected, np.float64)
    if nominal.shape != selected.shape or nominal.ndim != 1 or not len(nominal):
        raise ValueError("state outcomes must be matching nonempty vectors")
    rng = np.random.default_rng(seed)
    delta = nominal - selected
    values = np.empty(draws, np.float64)
    for start in range(0, draws, 1000):
        size = min(1000, draws - start)
        index = rng.integers(0, len(delta), size=(size, len(delta)))
        values[start:start + size] = delta[index].mean(axis=1)
    return float(np.quantile(values, 0.05))


def summarize_selector(
    fall: np.ndarray, choice: np.ndarray, *, bootstrap_seed: int,
) -> Mapping[str, float | int]:
    nominal = np.asarray(fall, bool)[:, 0, 4:].mean(axis=1)
    selected = selector_outcome(fall, choice)
    difference = nominal - selected
    intervened = np.asarray(choice) != 0
    return {
        "nominal_fall_rate": float(nominal.mean()),
        "selected_fall_rate": float(selected.mean()),
        "fall_reduction": float(difference.mean()),
        "fall_reduction_lcb95": state_bootstrap_lcb(
            nominal, selected, seed=bootstrap_seed),
        "rescue_states": int(np.sum(difference > 0)),
        "harm_states": int(np.sum(difference < 0)),
        "conditional_harm_rate": float(
            np.mean(difference[intervened] < 0) if np.any(intervened) else 0.0),
        "intervention_states": int(np.sum(intervened)),
    }
