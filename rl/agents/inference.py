"""Public factory for collector-owned inference policies."""
from __future__ import annotations
from typing import Any

from rl.agents.base.inference import InferencePolicy
from rl.agents.droq.inference import DroQInferencePolicy
from rl.agents.flashsac.inference import FlashSACInferencePolicy
from rl.agents.safe_droq.inference import SafeDroQInferencePolicy
from rl.agents.paper_sqrl.inference import PaperSQRLInferencePolicy
from rl.agents.livesac.inference import LiveSACInferencePolicy


def build_inference_policy(observation_dim: int, action_dim: int,
                           cfg: Any) -> InferencePolicy:
    agent_type = str(getattr(cfg, "agent_type", "")).lower()
    if bool(getattr(cfg, "asymmetric_observation", False)):
        # The config parser does not expose env_info in the collector. The
        # actor dimension is therefore optionally supplied by the caller.
        observation_dim = int(getattr(cfg, "actor_observation_dim", observation_dim))
    factories = {
        "droq": DroQInferencePolicy,
        "flashsac": FlashSACInferencePolicy,
        "safe_droq": SafeDroQInferencePolicy,
        "paper_sqrl": PaperSQRLInferencePolicy,
        "livesac": LiveSACInferencePolicy,
    }
    try:
        return factories[agent_type](observation_dim, action_dim, cfg)
    except KeyError as exc:
        raise ValueError(f"Async collector does not support agent_type={agent_type!r}") from exc
