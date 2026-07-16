import functools
from typing import Optional, Union

import flax.linen as nn
import jax.numpy as jnp

from jaxrl.distributions.tanh_transformed import (MultivariateNormalDiag,
                                                  TanhTransformedDistribution)
from jaxrl.networks.common import default_init

Distribution = Union[MultivariateNormalDiag, TanhTransformedDistribution]


class Normal(nn.Module):
    base_cls: nn.Module
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(x)
        log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(x)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = MultivariateNormalDiag(loc=means,
                                              scale_diag=jnp.exp(log_stds))

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)
