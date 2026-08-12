"""Exact frozen-candidate reconstruction for R5--R16 diagnostics."""

from __future__ import annotations

import numpy as np

from safety_data.mjlab_natural_falls import MJLAB_TO_TARGET_JOINT


PERMUTATION = np.asarray(MJLAB_TO_TARGET_JOINT, np.int64)
INVERSE_PERMUTATION = np.argsort(PERMUTATION)


def reconstruct_frozen_candidates(
    action_requested_target_order: np.ndarray,
    critic_action_target_order: np.ndarray,
) -> dict[str, np.ndarray]:
    """Invert joint order only; never sample or modify a candidate."""
    requested = np.asarray(action_requested_target_order, np.float32)
    critic = np.asarray(critic_action_target_order, np.float32)
    if requested.shape != (16, 12) or critic.shape != (16, 12):
        raise ValueError("frozen state must contain exactly 16 12D candidates")
    if not np.all(np.isfinite(requested)) or not np.all(np.isfinite(critic)):
        raise ValueError("frozen candidates must be finite")
    return {
        "raw_internal": requested[:, INVERSE_PERMUTATION].copy(),
        "absolute_internal": critic[:, INVERSE_PERMUTATION].copy(),
        "critic_action": critic.copy(),
        "action_requested": requested.copy(),
    }


def verify_frozen_candidate_identity(
    expected_ids: np.ndarray,
    observed_ids: np.ndarray,
    expected_actions: np.ndarray,
    observed_actions: np.ndarray,
) -> None:
    if not np.array_equal(expected_ids, observed_ids):
        raise RuntimeError("candidate identity changed during replica extension")
    if not np.array_equal(expected_actions, observed_actions):
        raise RuntimeError("physical candidate action changed during replica extension")

