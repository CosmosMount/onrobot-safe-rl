"""Explicit pretrain outputs and paired target initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .finetune import initialize_target_branch
from .sac import VanillaSAC
from .safety_critic import SafetyCriticLearner


FORMAT_VERSION = 1


def save_pretrain_checkpoint(path: str | Path, sac: VanillaSAC,
                             safety: SafetyCriticLearner,
                             metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": FORMAT_VERSION,
        "phase": "pretrain",
        "sac": sac.checkpoint(),
        "safety": safety.checkpoint(),
        "metadata": dict(metadata),
    }, path)


def load_pretrain_checkpoint(path: str | Path, sac: VanillaSAC,
                             safety: SafetyCriticLearner | None,
                             branch: str) -> dict[str, Any]:
    payload = torch.load(path, map_location=sac.device, weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION or payload.get("phase") != "pretrain":
        raise ValueError("not an SQRL-Go2 pretrain checkpoint")
    # Paired target branches all transfer exactly the same actor and never the
    # pretrain task critics, task replay, or alpha.
    sac.load_checkpoint(payload["sac"], actor_only=True)
    if branch != "sac_transfer":
        if safety is None:
            raise ValueError(f"{branch} requires a safety learner")
        safety.load_checkpoint(payload["safety"], load_optimizer=False)
    initialize_target_branch(sac, safety, branch)
    return dict(payload.get("metadata", {}))
