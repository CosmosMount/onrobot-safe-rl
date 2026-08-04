"""Scheduling helpers for integer and fractional update-to-data ratios."""

from __future__ import annotations

import math

from rl.agents.base.update import PolicyUpdateRequest


class UTDUpdateScheduler:
    """Turn a fractional UTD ratio into occasional integer update requests."""

    def __init__(self, utd_ratio: float) -> None:
        self.utd_ratio = float(utd_ratio)
        if not math.isfinite(self.utd_ratio) or self.utd_ratio <= 0:
            raise ValueError("utd_ratio must be finite and positive")
        self._budget = 0.0
        self._pending_policy_steps = 0

    def next_request(self) -> PolicyUpdateRequest | None:
        self._budget += self.utd_ratio
        self._pending_policy_steps += 1
        critic_updates = math.floor(self._budget + 1e-12)
        if critic_updates < 1:
            return None
        self._budget -= critic_updates
        policy_steps = self._pending_policy_steps
        self._pending_policy_steps = 0
        return PolicyUpdateRequest(
            policy_steps=policy_steps,
            critic_updates_per_policy_step=self.utd_ratio,
            critic_updates_override=critic_updates,
        )
