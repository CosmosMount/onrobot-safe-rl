from __future__ import annotations

import torch


def make_support(num_bins: int = 101, min_v: float = -5.0, max_v: float = 5.0) -> torch.Tensor:
    return torch.linspace(min_v, max_v, num_bins, dtype=torch.float32)


def expected_q(log_probs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    return (log_probs.exp() * support.to(log_probs.device)).sum(dim=-1)


def select_min_distribution(qs: torch.Tensor, log_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = qs.argmin(dim=0)
    batch = torch.arange(qs.shape[1], device=qs.device)
    selected = log_probs[indices, batch]
    return selected, indices


def project_distribution(values: torch.Tensor, probabilities: torch.Tensor, *, num_bins: int = 101,
                         min_v: float = -5.0, max_v: float = 5.0) -> torch.Tensor:
    values = values.float().clamp(min_v, max_v)
    probabilities = probabilities.float()
    delta = (max_v - min_v) / (num_bins - 1)
    b = (values - min_v) / delta
    lower = b.floor().long().clamp(0, num_bins - 1)
    upper = b.ceil().long().clamp(0, num_bins - 1)
    target = torch.zeros((values.shape[0], num_bins), device=values.device, dtype=torch.float32)
    lo_weight = (upper.float() - b).where(lower != upper, torch.zeros_like(b))
    hi_weight = (b - lower.float()).where(lower != upper, torch.zeros_like(b))
    target.scatter_add_(1, lower, probabilities * lo_weight)
    target.scatter_add_(1, upper, probabilities * hi_weight)
    target.scatter_add_(1, lower, probabilities * (lower == upper).float())
    return target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def boundary_mass(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return probabilities[..., 0].mean(), probabilities[..., -1].mean()
