"""Optional Weights & Biases logging."""

from __future__ import annotations

from typing import Any


class TrainLogger:
    def __init__(self,
                 enabled: bool = False,
                 project: str = 'go2_walk',
                 run_name: str | None = None,
                 config: dict[str, Any] | None = None):
        self.enabled = enabled
        self._wandb = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                'wandb is required when --wandb is set; pip install wandb'
            ) from exc
        self._wandb = wandb
        self._wandb.init(project=project, name=run_name, config=config or {})

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
