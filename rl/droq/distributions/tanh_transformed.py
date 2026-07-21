from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


class MultivariateNormalDiag:
    def __init__(self, loc: jnp.ndarray, scale_diag: jnp.ndarray):
        self.loc = loc
        self.scale_diag = scale_diag

    def sample(self, seed: Any) -> jnp.ndarray:
        noise = jax.random.normal(seed, self.loc.shape, dtype=self.loc.dtype)
        return self.loc + noise * self.scale_diag

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        var = jnp.square(self.scale_diag)
        log_scale = jnp.log(self.scale_diag)
        log_prob = -0.5 * jnp.square(value - self.loc) / var
        log_prob -= log_scale + 0.5 * jnp.log(2.0 * jnp.pi)
        return jnp.sum(log_prob, axis=-1)

    def mode(self) -> jnp.ndarray:
        return self.loc


class TanhTransformedDistribution:
    def __init__(self, distribution: MultivariateNormalDiag, validate_args: bool = False):
        del validate_args
        self.distribution = distribution

    def sample(self, seed: Any) -> jnp.ndarray:
        return jnp.tanh(self.distribution.sample(seed))

    def sample_and_log_prob(self, seed: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        pre_tanh = self.distribution.sample(seed)
        action = jnp.tanh(pre_tanh)
        log_prob = self.distribution.log_prob(pre_tanh)
        log_prob -= _stable_tanh_log_det_jacobian(pre_tanh)
        return action, log_prob

    def log_prob(self, action: jnp.ndarray) -> jnp.ndarray:
        clipped = jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = jnp.arctanh(clipped)
        log_prob = self.distribution.log_prob(pre_tanh)
        log_prob -= jnp.sum(jnp.log1p(-jnp.square(clipped)), axis=-1)
        return log_prob

    def mode(self) -> jnp.ndarray:
        return jnp.tanh(self.distribution.mode())


def _stable_tanh_log_det_jacobian(pre_tanh: jnp.ndarray) -> jnp.ndarray:
    log_det = 2.0 * (jnp.log(2.0) - pre_tanh - jax.nn.softplus(-2.0 * pre_tanh))
    return jnp.sum(log_det, axis=-1)
