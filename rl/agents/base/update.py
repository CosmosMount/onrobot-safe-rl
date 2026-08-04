"""Shared update-budget requests and optimizer counters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyUpdateRequest:
    """Update budget produced by one or more real policy transitions."""

    policy_steps: int
    critic_updates_per_policy_step: float
    critic_updates_override: int | None = None

    def __post_init__(self) -> None:
        if self.policy_steps <= 0:
            raise ValueError("policy_steps must be positive")
        if self.critic_updates_per_policy_step <= 0:
            raise ValueError(
                "critic_updates_per_policy_step must be positive")
        if self.critic_updates_override is not None and self.critic_updates_override <= 0:
            raise ValueError("critic_updates_override must be positive")

    @property
    def critic_updates(self) -> int:
        if self.critic_updates_override is not None:
            return self.critic_updates_override
        updates = self.policy_steps * self.critic_updates_per_policy_step
        if not float(updates).is_integer():
            raise ValueError("fractional UTD requires critic_updates_override")
        return int(updates)


@dataclass
class UpdateCounters:
    policy_steps: int = 0
    critic_steps: int = 0
    actor_steps: int = 0
    temperature_steps: int = 0
    target_steps: int = 0
    auxiliary_steps: int = 0
    legacy_counters_inferred: bool = False

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "policy_steps": int(self.policy_steps),
            "critic_steps": int(self.critic_steps),
            "actor_steps": int(self.actor_steps),
            "temperature_steps": int(self.temperature_steps),
            "target_steps": int(self.target_steps),
            "auxiliary_steps": int(self.auxiliary_steps),
            "legacy_counters_inferred": bool(self.legacy_counters_inferred),
        }

    def load_state_dict(self, state: dict[str, int | bool]) -> None:
        for name in (
            "policy_steps", "critic_steps", "actor_steps",
            "temperature_steps", "target_steps", "auxiliary_steps",
        ):
            setattr(self, name, int(state.get(name, 0)))
        self.legacy_counters_inferred = bool(
            state.get("legacy_counters_inferred", False))
