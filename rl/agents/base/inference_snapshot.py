"""Immutable CPU payloads exchanged between learner and collector."""

from __future__ import annotations

from typing import Any

import torch


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def inference_snapshot(*, agent_type: str, snapshot_version: int,
                       actor: torch.nn.Module, counters: dict[str, int],
                       auxiliary: torch.nn.Module | None = None,
                       critic: torch.nn.Module | None = None,
                       algorithm_state: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "agent_type": str(agent_type),
        "snapshot_version": int(snapshot_version),
        "actor_steps": int(counters.get("actor_steps", 0)),
        "critic_steps": int(counters.get("critic_steps", 0)),
        "temperature_steps": int(counters.get("temperature_steps", 0)),
        "auxiliary_steps": int(counters.get("auxiliary_steps", 0)),
        "actor_state_dict": cpu_state_dict(actor),
    }
    if auxiliary is not None:
        payload["safety_critic_state_dict"] = cpu_state_dict(auxiliary)
    if critic is not None:
        payload["critic_state_dict"] = cpu_state_dict(critic)
    if algorithm_state:
        payload["algorithm_state"] = dict(algorithm_state)
    return payload
