import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jaxrl.distributions.tanh_transformed import (
    MultivariateNormalDiag,
    TanhTransformedDistribution,
)


def _distribution(loc=None, scale=None):
    if loc is None:
        loc = jnp.array([0.2, -0.3, 0.1], dtype=jnp.float32)
    if scale is None:
        scale = jnp.array([0.7, 1.1, 0.5], dtype=jnp.float32)
    return TanhTransformedDistribution(
        MultivariateNormalDiag(loc=loc, scale_diag=scale)
    )


class TanhDistributionTest(unittest.TestCase):

    def test_sample_and_log_prob_matches_inverse_formula(self):
        dist = _distribution()
        key = jax.random.PRNGKey(7)
        action, log_prob = dist.sample_and_log_prob(key)

        pre_tanh = jnp.arctanh(action)
        inverse_formula = dist.distribution.log_prob(pre_tanh)
        inverse_formula -= jnp.sum(jnp.log(1.0 - action**2), axis=-1)

        np.testing.assert_allclose(
            log_prob, inverse_formula, rtol=2e-5, atol=2e-5
        )

    def test_sample_and_log_prob_reuses_the_exact_sample(self):
        dist = _distribution()
        key = jax.random.PRNGKey(11)
        action, _ = dist.sample_and_log_prob(key)
        np.testing.assert_array_equal(action, dist.sample(key))

    def test_extreme_latents_have_finite_log_prob_and_gradient(self):
        scale = jnp.ones((3,), dtype=jnp.float32)

        def objective(loc):
            dist = _distribution(loc=loc, scale=scale)
            _, log_prob = dist.sample_and_log_prob(jax.random.PRNGKey(13))
            return log_prob

        extreme_loc = jnp.array([100.0, -100.0, 50.0], dtype=jnp.float32)
        value = objective(extreme_loc)
        gradient = jax.grad(objective)(extreme_loc)

        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))


if __name__ == '__main__':
    unittest.main()
