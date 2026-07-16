from typing import Any, Dict, Union

import flax
import jax
import numpy as np

DataType = Union[np.ndarray, Dict[str, 'DataType']]
PRNGKey = jax.Array
Params = flax.core.FrozenDict[str, Any]
