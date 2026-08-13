"""Reference-compatible synchronous SAC/DroQ learner loop.

Matches ``walk_in_the_park/train_online.py``: action -> environment step ->
replay insert -> one learner call on ``batch_size * utd_ratio`` samples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxsac import create_agent, update
from jaxsac.actor import sample as sample_action
from jaxsac.batch import Batch
from jaxsac.config import AlgorithmConfig


def _block_agent(agent):
    """Wait for every device array, not only a scalar metric."""
    return jax.tree_util.tree_map(jax.block_until_ready, agent)


def _wandb_metrics(metrics):
    """Convert JAX/NumPy scalar metrics into values W&B can serialize."""
    result = {}
    for name, value in metrics.items():
        value = jax.device_get(value)
        if np.ndim(value) == 0:
            result[name] = float(value)
    return result


@dataclass
class Replay:
    capacity: int
    observation_shape: tuple
    action_shape: tuple
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed); self.size = 0; self.index = 0
        self.data = {"observations": np.empty((self.capacity, *self.observation_shape), np.float32),
                     "actions": np.empty((self.capacity, *self.action_shape), np.float32),
                     "next_observations": np.empty((self.capacity, *self.observation_shape), np.float32),
                     "rewards": np.empty(self.capacity, np.float32),
                     "discounts": np.empty(self.capacity, np.float32)}

    def insert(self, obs, action, reward, discount, next_obs):
        i = self.index
        self.data["observations"][i] = obs; self.data["actions"][i] = action
        self.data["rewards"][i] = reward; self.data["discounts"][i] = discount
        self.data["next_observations"][i] = next_obs
        self.index = (i + 1) % self.capacity; self.size = min(self.capacity, self.size + 1)

    def sample(self, n):
        if self.size < n: raise ValueError("not enough replay data")
        ids = self.rng.integers(self.size, size=n)
        return Batch(*(jnp.asarray(self.data[k][ids]) for k in Batch._fields))


def run(env, *, max_steps=1_000_000, start_training=10_000,
        batch_size=256, utd_ratio=1, replay_capacity=1_000_000,
        seed=42, config=None, enforce_20hz=True, benchmark_only=False,
        wandb_run=None, wandb_log_interval=100, progress_interval=100):
    if utd_ratio < 1:
        raise ValueError("walk_in_the_park requires UTD >= 1")
    config = config or AlgorithmConfig(
        actor_lr=3e-4, critic_lr=3e-4, temperature_lr=3e-4,
        hidden_dims=(256, 256), discount=0.99, num_qs=2,
        critic_dropout_rate=0.01, critic_layer_norm=True, tau=0.005,
        init_temperature=0.1, utd_ratio=utd_ratio)
    agent = create_agent(env.observation_space_shape, env.action_space_shape,
                         config, seed=seed)
    # Compile the exact UTD update before reset/action timing starts.  JAX's
    # first call can take seconds; allowing that compilation inside env.step
    # would violate the upstream 20 Hz control contract.  The synthetic batch
    # is discarded, so it has no replay or parameter effect.
    warmup_n = batch_size * utd_ratio
    warmup = Batch(
        observations=jnp.zeros((warmup_n, *env.observation_space_shape), jnp.float32),
        actions=jnp.zeros((warmup_n, *env.action_space_shape), jnp.float32),
        rewards=jnp.zeros((warmup_n,), jnp.float32),
        discounts=jnp.ones((warmup_n,), jnp.float32),
        next_observations=jnp.zeros((warmup_n, *env.observation_space_shape), jnp.float32),
    )
    warmup_agent, _ = update(agent, warmup, config)
    _block_agent(warmup_agent)
    benchmark_agent = agent
    benchmark_key = agent.rng
    # Compile actor inference before timing the 20 Hz steady-state budget.
    # The first XLA dispatch can take hundreds of milliseconds and is not a
    # per-step cost after startup; counting it would reject a valid runtime.
    benchmark_key, actor_warmup_key = jax.random.split(benchmark_key)
    benchmark_agent = benchmark_agent.replace(rng=benchmark_key)
    _ = jax.device_get(sample_action(
        benchmark_agent.actor.apply_fn, benchmark_agent.actor.params,
        jnp.zeros((1, *env.observation_space_shape), jnp.float32),
        actor_warmup_key)[0][0])
    _block_agent(benchmark_agent)
    benchmark_intervals = []
    for _ in range(3):
        benchmark_started = time.perf_counter()
        benchmark_key, sample_key = jax.random.split(benchmark_key)
        benchmark_agent = benchmark_agent.replace(rng=benchmark_key)
        _ = jax.device_get(sample_action(
            benchmark_agent.actor.apply_fn, benchmark_agent.actor.params,
            jnp.zeros((1, *env.observation_space_shape), jnp.float32),
            sample_key)[0][0])
        benchmark_agent, _ = update(benchmark_agent, warmup, config)
        _block_agent(benchmark_agent)
        benchmark_intervals.append(
            (time.perf_counter() - benchmark_started) * 1000.0)
    benchmark_ms = max(benchmark_intervals)
    if enforce_20hz and benchmark_ms >= 1000.0 / 20.0:
        raise RuntimeError(
            f"actor+UTD={utd_ratio} takes {benchmark_ms:.2f} ms; "
            "refusing to start because the 20 Hz action deadline cannot be guaranteed")
    if wandb_run is not None:
        wandb_run.log({
            "benchmark/actor_utd_max_ms": benchmark_ms,
            "benchmark/utd_ratio": utd_ratio,
            "benchmark/control_frequency_hz": 20.0,
        }, step=0)
    if benchmark_only:
        return agent, {
            "benchmark_only": True,
            "actor_utd_max_ms": benchmark_ms,
            "utd_ratio": utd_ratio,
            "control_frequency_hz": 20.0,
        }
    replay = Replay(replay_capacity, env.observation_space_shape,
                    env.action_space_shape, seed)
    print(
        f"[train] controller ready: action=20 Hz, UTD={utd_ratio}, "
        f"warmup={start_training} steps ({start_training / 20.0:.1f} s); "
        "starting synchronous rollout",
        flush=True)
    obs, _ = env.reset(); key = agent.rng
    print("[train] reset complete; waiting for policy states", flush=True)
    action_rng = np.random.default_rng(seed)
    action_dim = int(env.action_space_shape[-1])
    learner_intervals = []
    updates = 0
    for step in range(max_steps):
        if step < start_training:
            action = action_rng.uniform(-1.0, 1.0, action_dim).astype(np.float32)
        else:
            key, sample_key = jax.random.split(key)
            agent = agent.replace(rng=key)
            action = jax.device_get(
                jnp.asarray(sample_action(
                    agent.actor.apply_fn, agent.actor.params,
                    jnp.asarray(obs)[None], sample_key)[0][0]))
        if step == 0 or (step + 1) % max(1, progress_interval) == 0:
            print(
                f"[train] sending step={step + 1}/{max_steps} "
                f"phase={'warmup' if step < start_training else 'learning'}",
                flush=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # A recovery/stand-up motion is controller-generated, not an action
        # sampled from the policy.  Do not train on this transition; in
        # particular, repeated falls must not fill replay with recovery data.
        if not info.get("recovery_motion", False):
            replay.insert(obs, action, reward, 0.0 if terminated else 1.0, next_obs)
        obs = next_obs
        done = bool(terminated or truncated)
        if done:
            # Reset is an environment lifecycle phase, not a policy step.
            # Keep the upstream ordering (reset before learner update), but
            # never charge stand-up/recovery time to the 20 Hz action budget.
            obs, _ = env.reset()
        if step >= start_training:
            learner_started = time.perf_counter()
            agent, metrics = update(agent, replay.sample(batch_size * utd_ratio), config)
            _block_agent(agent)
            key = agent.rng
            learner_elapsed = time.perf_counter() - learner_started
            learner_intervals.append(learner_elapsed)
            if learner_elapsed > 1.0 / 20.0:
                raise RuntimeError(
                    f"learner deadline missed at step {step}: "
                    f"{learner_elapsed * 1000.0:.2f} ms")
            updates += 1
            if wandb_run is not None and (
                    step % max(1, wandb_log_interval) == 0):
                log_metrics = _wandb_metrics(metrics)
                log_metrics.update({
                    "train/update": updates,
                    "train/replay_size": replay.size,
                    "train/learner_ms": learner_elapsed * 1000.0,
                })
                wandb_run.log(log_metrics, step=step + 1)
        if (step == 0 or (step + 1) % max(1, progress_interval) == 0):
            print(
                f"[train] step={step + 1}/{max_steps} "
                f"phase={'warmup' if step < start_training else 'learning'} "
                f"updates={updates} reward={reward:.4f} "
                f"policy_seq={info.get('policy_sequence', '?')} "
                f"action_id={info.get('applied_action_id', '?')} "
                f"vx={info.get('reward/vx', float('nan')):.3f} "
                f"z={info.get('reward/position_z', float('nan')):.3f} "
                f"up={info.get('reward/body_up', float('nan')):.3f}"
                + (f" learner_ms={learner_intervals[-1] * 1000.0:.2f}"
                   if learner_intervals else ""),
                flush=True)
    action_intervals = np.asarray(getattr(env, "action_intervals_ms", []),
                                  dtype=np.float64)
    result = {"steps": max_steps, "updates": updates,
              # env.step waits for the next 20 Hz controller tick; charging
              # that wait plus the learner to one wall-clock budget creates a
              # false failure at the phase boundary. Measure action cadence
              # and learner latency independently instead.
              "interval_p50_ms": (float(np.percentile(action_intervals, 50))
                                   if action_intervals.size else None),
              "interval_p95_ms": (float(np.percentile(action_intervals, 95))
                                   if action_intervals.size else None),
              "interval_max_ms": (float(np.max(action_intervals))
                                  if action_intervals.size else None),
              "learner_p50_ms": (1000 * float(np.percentile(learner_intervals, 50))
                                  if learner_intervals else None),
              "learner_p95_ms": (1000 * float(np.percentile(learner_intervals, 95))
                                  if learner_intervals else None)}
    for name in ("action_intervals_ms", "state_intervals_ms"):
        values = np.asarray(getattr(env, name, []), dtype=np.float64)
        if values.size:
            result[f"{name}_p50"] = float(np.percentile(values, 50))
            result[f"{name}_p95"] = float(np.percentile(values, 95))
    if wandb_run is not None:
        wandb_run.summary.update({f"final/{name}": value
                                  for name, value in result.items()
                                  if value is not None})
    return agent, result
