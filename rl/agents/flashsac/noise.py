from __future__ import annotations

import torch


def build_truncated_zeta_cdf(
    mu: float,
    max_n: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the normalized CDF of a truncated Zeta distribution."""
    mu_tensor = torch.tensor(mu, dtype=torch.float32)
    if not torch.isfinite(mu_tensor):
        raise ValueError("mu must be finite")
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if max_n <= 0:
        raise ValueError("max_n must be positive")

    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    pmf = ns.pow(-mu)
    pmf = pmf / pmf.sum()
    cdf = torch.cumsum(pmf, dim=0)
    cdf[-1] = 1.0
    return cdf


def sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """Sample an integer in [1, len(cdf)] from a valid 1D CDF."""
    if cdf.ndim != 1:
        raise ValueError("cdf must be one-dimensional")
    if cdf.numel() == 0:
        raise ValueError("cdf must not be empty")
    if not torch.all(torch.isfinite(cdf)):
        raise ValueError("cdf must contain only finite values")

    u = torch.rand((), device=cdf.device, dtype=cdf.dtype)
    index = torch.searchsorted(cdf, u, right=False)
    index = torch.clamp(index, max=cdf.numel() - 1)
    return (index + 1).to(dtype=torch.int32)
