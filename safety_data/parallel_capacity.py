"""Capacity-gate calculations for the natural-PPO GPU backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CapacityResult:
    environments: int
    policy_env_steps_per_second: float
    peak_vram_mib: int
    stable: bool
    nonfinite: bool = False
    external_force_nonzero: bool = False

    @property
    def base_eligible(self) -> bool:
        return bool(
            self.environments > 0
            and self.policy_env_steps_per_second > 0.0
            and self.peak_vram_mib <= 20480
            and self.stable
            and not self.nonfinite
            and not self.external_force_nonzero
        )


def select_capacity(
    results: Iterable[CapacityResult], *, minimum_gain: float = 0.15,
) -> CapacityResult | None:
    """Select the largest valid rung whose upgrade provides enough throughput."""
    ordered = sorted(results, key=lambda value: value.environments)
    selected: CapacityResult | None = None
    for result in ordered:
        if not result.base_eligible:
            continue
        if selected is None:
            selected = result
            continue
        gain = (
            result.policy_env_steps_per_second
            / selected.policy_env_steps_per_second - 1.0
        )
        if gain >= minimum_gain:
            selected = result
    return selected
