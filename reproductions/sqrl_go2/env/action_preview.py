"""Side-effect-free mapping from policy candidates to executed critic actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from runtime.inference.actions import ActionApplier, ActionProjection


@dataclass(frozen=True)
class PreviewBatch:
    requested: np.ndarray
    critic_actions: np.ndarray
    q_targets: np.ndarray


class ActionPreview:
    """Mirror the runtime projection used to execute candidate actions.

    Candidate scoring always receives ``action_executed``.  Committing a
    chosen candidate advances the mirrored filter exactly once and returns the
    requested action that must be sent to the runtime.
    """

    def __init__(self, applier: ActionApplier):
        self.applier = applier

    def preview(self, candidates: np.ndarray, observation: np.ndarray) -> PreviewBatch:
        values = np.asarray(candidates, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("candidates must have shape [K, action_dim]")
        current_qpos = self._current_qpos(observation)
        projections = self.applier.preview_many(values, current_qpos)
        return self._batch(projections)

    def commit(self, candidate: np.ndarray, observation: np.ndarray) -> ActionProjection:
        return self.applier.project(
            np.asarray(candidate, dtype=np.float32), self._current_qpos(observation))

    def reset(self, qpos: np.ndarray) -> None:
        self.applier.reset_filter()
        self.applier.init_filter_history(np.asarray(qpos, dtype=np.float32))

    @staticmethod
    def _current_qpos(observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        # A raw observation is 46D.  Stacked observations are oldest-to-newest,
        # so the final frame contains the causal qpos for action projection.
        frame = values[-46:] if values.size >= 46 else values
        if frame.size < 12 or not np.all(np.isfinite(frame[:12])):
            raise ValueError("Go2 observation must contain finite joint_q in its first 12 entries")
        return frame[:12]

    @staticmethod
    def _batch(projections: tuple[ActionProjection, ...]) -> PreviewBatch:
        if not projections:
            raise ValueError("at least one candidate is required")
        return PreviewBatch(
            requested=np.stack([p.action_requested for p in projections]),
            critic_actions=np.stack([p.action_executed for p in projections]),
            q_targets=np.stack([p.action_q_target for p in projections]),
        )
