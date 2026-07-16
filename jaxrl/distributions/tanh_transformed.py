"""Tanh-squashed Gaussian without tensorflow_probability."""

from typing import Protocol

import jax
import jax.numpy as jnp


class _NormalLike(Protocol):

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        ...

    def mode(self) -> jnp.ndarray:
        ...

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        ...


class MultivariateNormalDiag:
    """Diagonal Gaussian used by the policy and critic heads."""

    def __init__(self, loc: jnp.ndarray, scale_diag: jnp.ndarray):
        self.loc = loc
        self.scale_diag = scale_diag

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        noise = jax.random.normal(seed, shape=self.loc.shape)
        return self.loc + self.scale_diag * noise

    def mode(self) -> jnp.ndarray:
        return self.loc

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        var = self.scale_diag**2
        log_2pi = jnp.log(2.0 * jnp.pi)
        return -0.5 * jnp.sum(
            ((value - self.loc)**2) / var + 2.0 * jnp.log(self.scale_diag) +
            log_2pi,
            axis=-1,
        )


class TanhTransformedDistribution:

    def __init__(self, distribution: _NormalLike):
        self.distribution = distribution

    def sample(self, seed: jax.Array) -> jnp.ndarray:
        return jnp.tanh(self.distribution.sample(seed))

    def mode(self) -> jnp.ndarray:
        return jnp.tanh(self.distribution.mode())

    def log_prob(self, value: jnp.ndarray) -> jnp.ndarray:
        value = jnp.clip(value, -0.999999, 0.999999)
        pre_tanh = jnp.arctanh(value)
        log_prob = self.distribution.log_prob(pre_tanh)
        log_prob -= jnp.sum(jnp.log(1.0 - value**2 + 1e-6), axis=-1)
        return log_prob
