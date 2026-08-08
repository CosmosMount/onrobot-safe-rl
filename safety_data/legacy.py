"""Explicit, non-evidence adapter for legacy row-centric P17 files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from safety_data.paths import assert_development_path


def audit_legacy_p17(
    path: str | Path, *, acknowledge_legacy: bool = False,
) -> dict[str, Any]:
    """Inspect legacy shape/semantic defects without converting critical fields."""
    if not acknowledge_legacy:
        raise ValueError(
            "legacy P17 access is opt-in; pass acknowledge_legacy=True and never "
            "use the result as Phase-1 evidence")
    source = assert_development_path(path)
    with np.load(source, allow_pickle=False) as payload:
        fields = set(payload.files)
        history = (
            payload["observation_histories"].copy()
            if "observation_histories" in fields else None)
        has_application_triplet = {
            "actions", "actions_executed", "action_q_targets"
        }.issubset(fields)
    issues: list[str] = []
    if history is None:
        issues.append("missing_observation_histories")
        history_frames = None
    else:
        history_frames = int(history.shape[1]) if history.ndim == 3 else None
        if history.ndim != 3 or history.shape[1:] != (5, 46):
            issues.append("history_is_not_5x46")
    if not has_application_triplet:
        issues.append("missing_requested_executed_q_target_triplet")
    # q_send_history, trajectory identity, replicas and CRN cannot be inferred
    # safely from old row files, even if their observation tail happens to look
    # numerically plausible.
    issues.extend([
        "missing_explicit_q_send_history",
        "missing_trajectory_and_state_fingerprints",
        "missing_crn_replicas",
    ])
    return {
        "path": str(source),
        "legacy": True,
        "evidence_eligible": False,
        "history_frames": history_frames,
        "issues": issues,
    }
