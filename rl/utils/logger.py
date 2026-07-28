from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf

class WandbTrainerLogger:
    def __init__(self, cfg: Any):
        import wandb

        self._wandb = wandb
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, throw_on_missing=True)
        wandb.init(
            project=cfg.project_name,
            entity=cfg.entity_name,
            group=cfg.group_name,
            config=dict_cfg,  # type: ignore
        )
        self.media_dict: dict[str, Any] = {}
        self.reset()

    def update_metric(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, (float, int)):
                self.average_meter_dict.update(k, v)
            elif isinstance(v, np.ndarray) and v.ndim == 5:
                self.media_dict[k] = self._wandb.Video(v, fps=30, format="gif")
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> None:
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        self._wandb.log(log_data, step=step)

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict.clear()


class AverageMeter:
    """
    Tracks and calculates the average and current values of a series of numbers.
    """

    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        # TODO: description for using n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format: str) -> str:
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class AverageMeterDict:
    """
    Manages a collection of AverageMeter instances,
    allowing for grouped tracking and averaging of multiple metrics.
    """

    def __init__(self, meters: Optional[dict[str, AverageMeter]] = None):
        self.meters = meters if meters else {}

    def __getitem__(self, key: str) -> AverageMeter:
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string: str = "{}") -> dict[str, float]:
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string: str = "{}") -> dict[str, float]:
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}
