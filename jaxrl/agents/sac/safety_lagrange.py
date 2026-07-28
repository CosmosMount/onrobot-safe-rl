"""Non-negative Lagrange multiplier for SQRL fine-tuning."""

import flax.linen as nn
import jax.numpy as jnp


class SafetyLagrange(nn.Module):
    initial_nu: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_nu = self.param(
            'log_nu',
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_nu)))
        return jnp.exp(log_nu)
