"""Optional Weights & Biases logging."""

from __future__ import annotations

import os
from typing import Any


class TrainLogger:
    def __init__(self,
                 enabled: bool = False,
                 project: str = 'go2_walk',
                 run_name: str | None = None,
                 config: dict[str, Any] | None = None):
        self.enabled = enabled
        self._wandb = None
        self.run_url: str | None = None
        self.run_path: str | None = None
        self.run_name: str | None = run_name
        self.mode: str | None = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                'wandb is required when --wandb is set; pip install wandb'
            ) from exc
        self._wandb = wandb
        # No API key → offline so training still records a local run.
        mode = os.environ.get('WANDB_MODE')
        if mode is None and not os.environ.get('WANDB_API_KEY'):
            try:
                logged_in = bool(wandb.api.api_key)
            except Exception:
                logged_in = False
            if not logged_in:
                mode = 'offline'
                os.environ['WANDB_MODE'] = 'offline'
        self.mode = mode
        init_kwargs: dict[str, Any] = {
            'project': project,
            'name': run_name,
            'config': config or {},
        }
        if mode:
            init_kwargs['mode'] = mode
        run = self._wandb.init(**init_kwargs)
        self.run_name = getattr(run, 'name', run_name)
        try:
            self.run_url = run.get_url()
        except Exception:
            self.run_url = None
        try:
            self.run_path = str(run.dir) if run is not None else None
        except Exception:
            self.run_path = None
        print(
            f'[wandb] init name={self.run_name} mode={self.mode or "online"} '
            f'url={self.run_url or "n/a"} dir={self.run_path or "n/a"}',
            flush=True)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            try:
                run = self._wandb.run
                if run is not None:
                    try:
                        self.run_url = run.get_url()
                    except Exception:
                        pass
                    print(
                        f'[wandb] finished name={self.run_name} '
                        f'url={self.run_url or "n/a"}',
                        flush=True)
            finally:
                self._wandb.finish()
