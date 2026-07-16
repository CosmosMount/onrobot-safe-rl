"""JIT warmup for DroQ agent before the online loop."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np


def warmup_agent(agent, env, batch_size: int, utd_ratio: int) -> tuple:
    """Compile sample_actions and update via dummy calls."""
    original_agent = agent
    obs = np.asarray(env.observation_spec.zeros(), dtype=np.float32)
    action, sampled_agent = agent.sample_actions(obs)
    _ = action

    total_batch = batch_size * utd_ratio
    dummy_batch = {
        'observations': jnp.tile(obs, (total_batch, 1)),
        'actions': jnp.tile(jnp.asarray(action), (total_batch, 1)),
        'rewards': jnp.zeros((total_batch, )),
        'masks': jnp.ones((total_batch, )),
        'dones': jnp.zeros((total_batch, ), dtype=jnp.bool_),
        'next_observations': jnp.tile(obs, (total_batch, 1)),
    }

    compile_t0 = time.perf_counter()
    warm_agent, update_info = sampled_agent.update(dummy_batch, utd_ratio)
    for v in update_info.values():
        if hasattr(v, 'block_until_ready'):
            v.block_until_ready()
    jax.block_until_ready(warm_agent)
    compile_ms = (time.perf_counter() - compile_t0) * 1000.0

    steady_t0 = time.perf_counter()
    warm_agent, update_info = sampled_agent.update(dummy_batch, utd_ratio)
    for v in update_info.values():
        if hasattr(v, 'block_until_ready'):
            v.block_until_ready()
    jax.block_until_ready(warm_agent)
    steady_ms = (time.perf_counter() - steady_t0) * 1000.0

    metrics = {
        k: float(v) if hasattr(v, 'item') else float(np.asarray(v))
        for k, v in update_info.items()
    }
    metrics['warmup_compile_ms'] = compile_ms
    metrics['warmup_steady_ms'] = steady_ms
    return original_agent, compile_ms, metrics
