"""JAX helpers for RL data pipelines."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import frozen_dict


def tree_to_jnp(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree)


def freeze_jnp_batch(batch: dict) -> frozen_dict.FrozenDict:
    return frozen_dict.freeze(tree_to_jnp(batch))
