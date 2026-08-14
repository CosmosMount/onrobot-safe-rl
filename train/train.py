"""Synchronous SAC/DroQ learner loop.

Each iteration performs: action -> environment step -> replay insert -> one
learner call on ``batch_size * utd_ratio`` samples.
"""

from __future__ import annotations

import time
import pickle
import shutil
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxsac import create_agent, update
from jaxsac.batch import Batch
from jaxsac.config import AlgorithmConfig


def _block_agent(agent):
    """Wait for every device array, not only a scalar metric."""
    return jax.tree_util.tree_map(jax.block_until_ready, agent)


def _block_tree(tree):
    """Synchronize every output of a JAX call, including auxiliary metrics."""
    return jax.tree_util.tree_map(jax.block_until_ready, tree)


@partial(jax.jit, static_argnames=('apply_fn',))
def _sample_action_jitted(apply_fn, params, observation, key):
    mean, log_std = apply_fn({'params': params}, observation)
    pre_tanh = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    return jnp.tanh(pre_tanh)


def _sample_action_blocking(agent, observation, key):
    """Sample the policy action and wait only for the action result.

    Rollout does not need the log probability.  Computing and synchronizing
    it here duplicated work on the 20 Hz critical path; the learner computes
    log probabilities when it performs the actor update.  The sampled action
    itself is unchanged: same actor, same RNG, same tanh-Gaussian transform.
    """
    actions = _sample_action_jitted(
        agent.actor.apply_fn, agent.actor.params, observation, key)
    return np.asarray(jax.device_get(actions)[0])


def _wandb_metrics(metrics):
    """Convert JAX/NumPy scalar metrics into values W&B can serialize."""
    result = {}
    for name, value in metrics.items():
        value = jax.device_get(value)
        if np.ndim(value) == 0:
            result[name] = float(value)
    return result


def _window_stats(values, limit, scale=1.0):
    """Return percentile stats after converting to the requested unit."""
    values = np.asarray(values[-max(1, limit):], dtype=np.float64) * scale
    return {
        "p50": float(np.percentile(values, 50)) if values.size else 0.0,
        "p95": float(np.percentile(values, 95)) if values.size else 0.0,
        "max": float(np.max(values)) if values.size else 0.0,
    }


def _clear_save_dir(save_dir):
    """Remove all learner checkpoints/replay files before a fresh run."""
    root = Path(save_dir).expanduser()
    resolved = root.resolve()
    if resolved in (Path("/"), Path.home().resolve(), Path.cwd().resolve()):
        raise ValueError(f"refusing to clear unsafe save_dir={resolved}")
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    print(f"[train] cleared saved state: {resolved}", flush=True)


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
        self.action_ids = np.zeros(self.capacity, np.uint64)
        self.action_to_index = {}

    def insert(self, obs, action, reward, discount, next_obs, action_id=None):
        i = self.index
        old_action_id = int(self.action_ids[i])
        if old_action_id and self.action_to_index.get(old_action_id) == i:
            del self.action_to_index[old_action_id]
        self.data["observations"][i] = obs; self.data["actions"][i] = action
        self.data["rewards"][i] = reward; self.data["discounts"][i] = discount
        self.data["next_observations"][i] = next_obs
        stored_action_id = 0 if action_id is None else int(action_id)
        self.action_ids[i] = stored_action_id
        if stored_action_id:
            self.action_to_index[stored_action_id] = i
        self.index = (i + 1) % self.capacity; self.size = min(self.capacity, self.size + 1)
        return i

    def patch_terminal(self, action_id, reward, next_obs):
        """Rewrite a recent transition when a delayed fall names its action."""
        action_id = int(action_id)
        i = self.action_to_index.get(action_id)
        if i is None or int(self.action_ids[i]) != action_id:
            return None
        old_reward = float(self.data["rewards"][i])
        self.data["rewards"][i] = reward
        self.data["discounts"][i] = 0.0
        self.data["next_observations"][i] = next_obs
        return {"index": i, "old_reward": old_reward}

    def sample(self, n):
        # UTD sampling is with replacement, as in walk_in_the_park.  The
        # buffer only needs enough distinct entries for one mini-batch.
        if self.size < 1: raise ValueError("not enough replay data")
        ids = self.rng.integers(self.size, size=n)
        return Batch(*(self.data[k][ids] for k in Batch._fields))


def run(env, *, max_steps=1_000_000, start_training=10_000,
        batch_size=256, utd_ratio=1, replay_capacity=1_000_000,
        seed=42, config=None, enforce_20hz=True, wandb_run=None,
        log_interval=100, progress_interval=100, save_dir=None,
        checkpoint_interval=1000, resume=False):
    if utd_ratio < 1:
        raise ValueError("training requires UTD >= 1")
    required_replay = batch_size
    if start_training < required_replay:
        raise ValueError(
            f"start_training={start_training} is smaller than the first "
            f"learner batch={required_replay}; increase warmup or lower UTD")
    config = config or AlgorithmConfig(
        actor_lr=3e-4, critic_lr=3e-4, temperature_lr=3e-4,
        hidden_dims=(256, 256), discount=0.99, num_qs=2,
        critic_dropout_rate=0.01, critic_layer_norm=True, tau=0.005,
        init_temperature=0.1, utd_ratio=utd_ratio, policy_delay=1,
        actor_q_aggregation="mean")
    if config.utd_ratio != utd_ratio:
        raise ValueError(
            f"UTD mismatch: train loop requested {utd_ratio}, "
            f"algorithm config contains {config.utd_ratio}")
    if config.policy_delay != 1:
        raise ValueError(
            "synchronous walk_in_the_park behavior requires one policy update "
            "per learner/action step (policy_delay=1)")
    agent = create_agent(env.observation_space_shape, env.action_space_shape,
                         config, seed=seed)
    replay = Replay(replay_capacity, env.observation_space_shape,
                    env.action_space_shape, seed)
    start_step = 0
    checkpoint_root = Path(save_dir) if save_dir else None
    if checkpoint_root is not None:
        _clear_save_dir(checkpoint_root)
        resume = False
    if resume and checkpoint_root is not None:
        checkpoint_dir = checkpoint_root / "checkpoints"
        candidates = sorted(checkpoint_dir.glob("agent_*.msgpack"))
        if candidates:
            latest = candidates[-1]
            from flax import serialization

            agent = serialization.from_bytes(agent, latest.read_bytes())
            start_step = int(latest.stem.split('_')[-1])
            replay_path = checkpoint_root / "replay" / f"replay_{start_step}.pkl"
            if not replay_path.exists():
                raise RuntimeError(
                    f"checkpoint {latest} has no matching replay buffer {replay_path}")
            with replay_path.open("rb") as stream:
                replay = pickle.load(stream)
            if start_step >= start_training and replay.size < required_replay:
                raise RuntimeError(
                    f"restored replay has {replay.size} transitions, "
                    f"need {required_replay}")

    # Compile and measure exactly the work that must fit between two 20 Hz
    # action ticks. On resume, use a real replay batch so the first live
    # update has the same host/device layout as this benchmark.
    benchmark_n = batch_size * config.utd_ratio
    if replay.size >= required_replay:
        benchmark = replay.sample(benchmark_n)
        benchmark_source = "replay"
        benchmark_observation = jnp.asarray(
            replay.data["observations"][replay.size - 1])[None]
    else:
        benchmark_key = jax.random.PRNGKey(seed + 1_000_003)
        benchmark_key, obs_key, action_key, reward_key, next_obs_key = (
            jax.random.split(benchmark_key, 5))
        # Keep benchmark inputs host-resident, matching Replay.sample() and
        # the first real update's NumPy-to-device transfer path.
        benchmark = Batch(
            observations=np.asarray(jax.device_get(jax.random.normal(
                obs_key, (benchmark_n, *env.observation_space_shape)))),
            actions=np.asarray(jax.device_get(jax.random.uniform(
                action_key, (benchmark_n, *env.action_space_shape),
                minval=-1.0, maxval=1.0))),
            rewards=np.asarray(jax.device_get(jax.random.normal(
                reward_key, (benchmark_n,)))),
            # Replay stores a terminal mask (1 for bootstrap, 0 for
            # terminal); jaxsac applies config.discount in the critic.
            discounts=np.ones((benchmark_n,), np.float32),
            next_observations=np.asarray(jax.device_get(jax.random.normal(
                next_obs_key, (benchmark_n, *env.observation_space_shape)))))
        benchmark_observation = np.asarray(jax.device_get(
            jax.random.normal(obs_key, (1, *env.observation_space_shape))))
        benchmark_source = "synthetic"

    # Warm the exact actor callable owned by the live agent before rollout.
    # The benchmark below advances a copied AgentState; warming only that copy
    # does not guarantee that the static ``apply_fn`` used by the first live
    # action has already compiled. If compilation is deferred until the first
    # action after warmup, its synchronization is charged to the 20 Hz cycle
    # (the observed failure was actor=40 ms while update_kernel=9 ms).
    live_actor_warmup_started = time.perf_counter()
    live_actor_warmup_key = jax.random.PRNGKey(seed + 2_000_003)
    _sample_action_blocking(
        agent, benchmark_observation, live_actor_warmup_key)
    live_actor_warmup_ms = (
        time.perf_counter() - live_actor_warmup_started) * 1000.0

    # Carry state and RNG through the benchmark. Reusing the original agent
    # made this differ from the first live actor -> update transition.
    benchmark_agent = agent
    benchmark_key, sample_key = jax.random.split(benchmark_agent.rng)
    benchmark_agent = benchmark_agent.replace(rng=benchmark_key)
    _sample_action_blocking(benchmark_agent, benchmark_observation, sample_key)
    warm_agent, warm_metrics = update(benchmark_agent, benchmark, config)
    _block_agent(warm_agent)
    _block_tree(warm_metrics)
    benchmark_agent = warm_agent
    benchmark_ms = []
    for _ in range(3):
        started = time.perf_counter()
        benchmark_key, sample_key = jax.random.split(benchmark_agent.rng)
        benchmark_agent = benchmark_agent.replace(rng=benchmark_key)
        _sample_action_blocking(
            benchmark_agent, benchmark_observation, sample_key)
        measured_agent, measured_metrics = update(
            benchmark_agent, benchmark, config)
        _block_agent(measured_agent)
        _block_tree(measured_metrics)
        benchmark_agent = measured_agent
        benchmark_ms.append((time.perf_counter() - started) * 1000.0)
    max_benchmark_ms = max(benchmark_ms)
    print(
        f"[benchmark] backend={jax.default_backend()} device={jax.devices()[0]} "
        f"actor+UTD_ms={benchmark_ms} max={max_benchmark_ms:.2f} "
        f"utd={config.utd_ratio} batch={batch_size} source={benchmark_source} "
        f"live_actor_warmup_ms={live_actor_warmup_ms:.2f}",
        flush=True)
    if enforce_20hz and max_benchmark_ms >= 50.0:
        raise RuntimeError(
            f"DroQ actor+UTD={config.utd_ratio} takes {max_benchmark_ms:.2f} ms; "
            "refusing to start because the 20 Hz action deadline cannot be guaranteed")
    if wandb_run is not None:
        startup_group_hz = 1000.0 / max_benchmark_ms
        wandb_run.log({"benchmark/startup_actor_utd_max_ms": max_benchmark_ms,
                        "benchmark/startup_live_actor_warmup_ms": (
                            live_actor_warmup_ms),
                        "benchmark/startup_learner_group_hz": startup_group_hz,
                        "benchmark/startup_critic_updates_hz": (
                            startup_group_hz * config.utd_ratio),
                        "benchmark/utd_ratio": config.utd_ratio,
                        "benchmark/policy_updates_per_action": 0.0,
                        "benchmark/control_frequency_hz": 20.0}, step=0)
    def save_training_state(step, current_agent, current_replay):
        if checkpoint_root is None or checkpoint_interval <= 0:
            return
        from flax import serialization

        checkpoint_dir = checkpoint_root / "checkpoints"
        replay_dir = checkpoint_root / "replay"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        replay_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / f"agent_{step:012d}.msgpack").write_bytes(
            serialization.to_bytes(current_agent))
        with (replay_dir / f"replay_{step}.pkl").open("wb") as stream:
            pickle.dump(current_replay, stream, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[train] controller ready: action=20 Hz, UTD={utd_ratio}, "
        f"start_step={start_step}, warmup={start_training} steps "
        f"({start_training / 20.0:.1f} s); "
        "starting synchronous rollout",
        flush=True)
    try:
        obs, _ = env.reset()
    except RuntimeError:
        safety_hold = getattr(env, "last_safety_hold_info", None)
        if wandb_run is not None and safety_hold is not None:
            wandb_run.log({
                "safety/standup_failures": 1,
                "safety/event": float(safety_hold["event"]),
                "safety/event_action_id": float(
                    safety_hold["event_action_id"]),
                "safety/event_confirm_ms": float(
                    safety_hold["event_confirm_ms"]),
                "safety/roll": safety_hold["roll"],
                "safety/pitch": safety_hold["pitch"],
                "safety/up_cos": safety_hold["up_cos"],
                "safety/acc_z": safety_hold["acc_z"],
            }, step=0)
        raise
    key = agent.rng
    print("[train] reset complete; waiting for policy states", flush=True)
    action_rng = np.random.default_rng(seed)
    action_dim = int(env.action_space_shape[-1])
    learner_intervals = []
    actor_intervals = []
    actor_utd_intervals = []
    recent_rewards = deque(maxlen=100)
    learning_rewards = deque(maxlen=100)
    recent_safety = deque(maxlen=20)  # one second at 20 Hz
    recent_action_saturation = deque(maxlen=100)
    updates = 0
    episode_return = 0.0
    episode_length = 0
    episode_vx = 0.0
    episode_cos_pitch = 0.0
    episode_dyaw = 0.0
    episode_action_l2 = 0.0
    episode_forward_reward = 0.0
    episode_yaw_penalty = 0.0
    episode_terms_by_action = {}
    attempted_steps = 0
    valid_steps = start_step
    fall_events = 0
    upside_down_events = 0
    standup_failures = 0
    causal_mismatches = 0
    terminal_patch_successes = 0
    terminal_patch_missing = 0
    event_action_lag_total = 0
    event_action_lag_count = 0
    seen_safety_events = set()
    previous_executed_action = None
    action_delta_rms = 0.0
    action_saturation_fraction = 0.0
    while valid_steps < max_steps:
        step = valid_steps
        actor_elapsed = 0.0
        # Recovery/stand-up ticks do not enter replay, so environment step
        # count can be ahead of the number of valid learner transitions.
        # Start policy execution and learning only when the actual replay
        # contains a complete UTD batch.
        replay_ready = replay.size >= start_training
        if not replay_ready:
            action = action_rng.uniform(-1.0, 1.0, action_dim).astype(np.float32)
        else:
            actor_started = time.perf_counter()
            key, sample_key = jax.random.split(key)
            agent = agent.replace(rng=key)
            action = _sample_action_blocking(
                agent, jnp.asarray(obs)[None], sample_key)
            actor_elapsed = time.perf_counter() - actor_started
            actor_intervals.append(actor_elapsed)
            if step == 0 or (step + 1) % max(1, progress_interval) == 0:
                print(
                    f"[train] sending step={step + 1}/{max_steps} "
                    f"phase={'warmup' if not replay_ready else 'learning'}",
                flush=True)
        attempted_steps += 1
        try:
            next_obs, reward, terminated, truncated, info = env.step(action)
        except RuntimeError:
            safety_hold = getattr(env, "last_safety_hold_info", None)
            if wandb_run is not None and safety_hold is not None:
                wandb_run.log({
                    "safety/standup_failures": standup_failures + 1,
                    "safety/event": float(safety_hold["event"]),
                    "safety/event_action_id": float(
                        safety_hold["event_action_id"]),
                    "safety/event_confirm_ms": float(
                        safety_hold["event_confirm_ms"]),
                    "safety/roll": safety_hold["roll"],
                    "safety/pitch": safety_hold["pitch"],
                    "safety/up_cos": safety_hold["up_cos"],
                    "safety/acc_z": safety_hold["acc_z"],
                }, step=max(1, valid_steps))
            raise
        # Recovery/stand-up targets are generated by the controller rather
        # than by this learner action.  They are lifecycle states, not MDP
        # transitions: inserting them would train the critic on an action
        # that was never executed and a next observation produced by a
        # different controller.  They are returned as truncated episodes by
        # the environment, so this also pauses learner updates until reset()
        # has returned to POLICY.  walk_in_the_park excludes these steps too.
        policy_transition = bool(info.get(
            "policy_transition", not info.get("recovery_motion", False)))
        patch_result = None
        patch_action_id = info.get("terminal_patch_action_id")
        if patch_action_id is not None:
            patch_result = replay.patch_terminal(
                patch_action_id, reward, next_obs)
            event_action_lag_total += int(info.get("event_action_lag", 0))
            event_action_lag_count += 1
            if patch_result is None:
                terminal_patch_missing += 1
                causal_mismatches += 1
            else:
                terminal_patch_successes += 1
        if policy_transition:
            replay.insert(obs, action, reward,
                          0.0 if terminated and not truncated else 1.0,
                          next_obs,
                          action_id=info.get("applied_action_id"))
            valid_steps += 1
            episode_return += float(reward)
            episode_length += 1
            recent_rewards.append(float(reward))
            if replay.size >= start_training:
                learning_rewards.append(float(reward))
            episode_vx += float(info.get("reward/vx", 0.0))
            episode_cos_pitch += float(info.get("reward/cos_pitch", 0.0))
            episode_dyaw += abs(float(info.get("reward/dyaw", 0.0)))
            episode_forward_reward += float(
                info.get("reward/forward_reward", 0.0))
            episode_yaw_penalty += float(
                info.get("reward/yaw_penalty", 0.0))
            episode_action_l2 += float(
                np.linalg.norm(action) / np.sqrt(action.size))
            episode_terms_by_action[int(info.get("applied_action_id", 0))] = {
                "vx": float(info.get("reward/vx", 0.0)),
                "cos_pitch": float(info.get("reward/cos_pitch", 0.0)),
                "abs_dyaw": abs(float(info.get("reward/dyaw", 0.0))),
                "forward_reward": float(
                    info.get("reward/forward_reward", 0.0)),
                "yaw_penalty": float(info.get("reward/yaw_penalty", 0.0)),
            }
            if previous_executed_action is not None:
                action_delta_rms = float(np.linalg.norm(
                    action - previous_executed_action) / np.sqrt(action.size))
            previous_executed_action = np.asarray(action).copy()
            action_saturation_fraction = float(
                np.mean(np.abs(action) >= 0.95))
            recent_action_saturation.append(
                (np.abs(action) >= 0.95).astype(np.float32))
            if not info.get("event", 0):
                recent_safety.append({
                    "roll": float(info.get("safety/roll", np.nan)),
                    "pitch": float(info.get("safety/pitch", np.nan)),
                    "vx": float(info.get("reward/vx", np.nan)),
                    "action_rms": float(
                        np.linalg.norm(action) / np.sqrt(action.size)),
                })
        elif patch_result is not None:
            # The prior transition was already counted in this episode; only
            # replace its reward contribution and terminal next state.
            episode_return += float(reward) - patch_result["old_reward"]
            old_terms = episode_terms_by_action.get(int(patch_action_id))
            if old_terms is not None:
                episode_vx += float(info.get("reward/vx", 0.0)) - old_terms["vx"]
                episode_cos_pitch += (
                    float(info.get("reward/cos_pitch", 0.0)) -
                    old_terms["cos_pitch"])
                episode_dyaw += (
                    abs(float(info.get("reward/dyaw", 0.0))) -
                    old_terms["abs_dyaw"])
                episode_forward_reward += (
                    float(info.get("reward/forward_reward", 0.0)) -
                    old_terms["forward_reward"])
                episode_yaw_penalty += (
                    float(info.get("reward/yaw_penalty", 0.0)) -
                    old_terms["yaw_penalty"])

        event_key = (info.get("event"), info.get("event_action_id"))
        if info.get("event", 0) and event_key not in seen_safety_events:
            seen_safety_events.add(event_key)
            if info["event"] == 1:
                fall_events += 1
            elif info["event"] == 2:
                upside_down_events += 1
            elif info["event"] == 3:
                standup_failures += 1
            print(
                f"[safety] event={info.get('event')} "
                f"event_action_id={info.get('event_action_id')} "
                f"applied_action_id={info.get('applied_action_id')} "
                f"confirm_ms={info.get('event_confirm_ms', 0)} "
                f"roll={info.get('safety/roll', float('nan')):.3f} "
                f"pitch={info.get('safety/pitch', float('nan')):.3f} "
                f"up_cos={info.get('safety/up_cos', float('nan')):.3f} "
                f"acc_z={info.get('safety/acc_z', float('nan')):.3f} "
                f"event_lag={info.get('event_action_lag', 0)} "
                f"patch={'ok' if patch_result is not None else 'missing' if patch_action_id is not None else 'n/a'} "
                f"causal_mismatch={info.get('causal_mismatch', False)}",
                flush=True)
        obs = next_obs
        done = bool(terminated or truncated)
        if done:
            if terminated and info.get("event") == 1:
                end_reason = "fallen_standup"
            elif terminated and info.get("event") == 2:
                end_reason = "upside_down_recovery"
            elif info.get("event") == 3:
                end_reason = "standup_failed"
            elif info.get("event") == 1 and patch_result is not None:
                end_reason = "fallen_standup_delayed"
            elif info.get("event") == 2 and patch_result is not None:
                end_reason = "upside_down_recovery_delayed"
            elif info.get("time_limit", False):
                end_reason = "time_limit"
            elif info.get("causal_mismatch", False):
                end_reason = "failure_causal_mismatch"
            else:
                end_reason = "lifecycle_truncation"
            episode_denominator = max(1, episode_length)
            if wandb_run is not None:
                wandb_run.log({"training/return": episode_return,
                                "training/length": episode_length,
                                "training/terminated": bool(terminated),
                                "training/truncated": bool(truncated),
                                "episode/mean_reward": (
                                    episode_return / episode_denominator),
                                "episode/valid_policy_length": episode_length,
                                "episode/full_horizon": float(
                                    end_reason == "time_limit" and
                                    episode_length >= env.max_episode_steps),
                                "episode/end_reason": end_reason,
                                "episode/mean_forward_reward": (
                                    episode_forward_reward / episode_denominator),
                                "episode/mean_yaw_penalty": (
                                    episode_yaw_penalty / episode_denominator)},
                               step=max(1, valid_steps))
                wandb_run.log({
                    "training/mean_vx": episode_vx / episode_denominator,
                    "training/mean_cos_pitch": episode_cos_pitch / episode_denominator,
                    "training/mean_abs_dyaw": episode_dyaw / episode_denominator,
                    "training/mean_action_rms": episode_action_l2 / episode_denominator,
                    "episode/mean_vx": episode_vx / episode_denominator,
                }, step=max(1, valid_steps))
            episode_return = 0.0
            episode_length = 0
            episode_vx = 0.0
            episode_cos_pitch = 0.0
            episode_dyaw = 0.0
            episode_action_l2 = 0.0
            episode_forward_reward = 0.0
            episode_yaw_penalty = 0.0
            episode_terms_by_action.clear()
            # Reset is an environment lifecycle phase, not a policy step.
            # Keep the reset-before-update ordering, but
            # never charge stand-up/recovery time to the 20 Hz action budget.
            try:
                obs, _ = env.reset()
            except RuntimeError:
                safety_hold = getattr(env, "last_safety_hold_info", None)
                if wandb_run is not None and safety_hold is not None:
                    wandb_run.log({
                        "safety/standup_failures": standup_failures + 1,
                        "safety/event": float(safety_hold["event"]),
                        "safety/event_action_id": float(
                            safety_hold["event_action_id"]),
                        "safety/event_confirm_ms": float(
                            safety_hold["event_confirm_ms"]),
                        "safety/roll": safety_hold["roll"],
                        "safety/pitch": safety_hold["pitch"],
                        "safety/up_cos": safety_hold["up_cos"],
                        "safety/acc_z": safety_hold["acc_z"],
                    }, step=max(1, valid_steps))
                raise
        if replay.size >= start_training and policy_transition:
            learner_started = time.perf_counter()
            sample_started = time.perf_counter()
            learner_batch = replay.sample(batch_size * utd_ratio)
            sample_ready = time.perf_counter()
            agent, metrics = update(agent, learner_batch, config)
            update_returned = time.perf_counter()
            _block_agent(agent)
            agent_blocked = time.perf_counter()
            _block_tree(metrics)
            metrics_blocked = time.perf_counter()
            if int(jax.device_get(metrics["actor_updates"])) != 1:
                raise RuntimeError(
                    "synchronous DroQ invariant violated: expected one policy "
                    "update for each action step")
            if int(jax.device_get(metrics["critic_updates"])) != config.utd_ratio:
                raise RuntimeError(
                    "synchronous DroQ invariant violated: critic UTD mismatch")
            key = agent.rng
            learner_elapsed = time.perf_counter() - learner_started
            update_call_elapsed = update_returned - learner_started
            replay_sample_elapsed = sample_ready - sample_started
            update_kernel_elapsed = update_returned - sample_ready
            agent_sync_elapsed = agent_blocked - update_returned
            metrics_sync_elapsed = metrics_blocked - agent_blocked
            learner_intervals.append(learner_elapsed)
            actor_utd_elapsed = actor_elapsed + learner_elapsed
            actor_utd_intervals.append(actor_utd_elapsed)
            if enforce_20hz and actor_utd_elapsed >= 0.050:
                raise RuntimeError(
                    f"DroQ actor+UTD takes {actor_utd_elapsed * 1000.0:.2f} ms "
                    f"(actor={actor_elapsed * 1000.0:.2f} ms, "
                    f"replay_sample={replay_sample_elapsed * 1000.0:.2f} ms, "
                    f"update_kernel={update_kernel_elapsed * 1000.0:.2f} ms, "
                    f"agent_sync={agent_sync_elapsed * 1000.0:.2f} ms, "
                    f"metrics_sync={metrics_sync_elapsed * 1000.0:.2f} ms); "
                    "the next 20 Hz action deadline would be missed")
            updates += 1
            if wandb_run is not None and (
                    valid_steps % max(1, log_interval) == 0):
                log_metrics = _wandb_metrics(metrics)
                if "q_mean" in log_metrics and "target_q_mean" in log_metrics:
                    log_metrics["policy/q_target_gap"] = (
                        log_metrics["q_mean"] - log_metrics["target_q_mean"])
                if "entropy" in log_metrics:
                    log_metrics["policy/entropy"] = log_metrics["entropy"]
                if "temperature" in log_metrics:
                    log_metrics["policy/temperature"] = log_metrics["temperature"]
                if ("q_mean" in log_metrics and learning_rewards and
                        np.mean(learning_rewards) > 0.0):
                    log_metrics["policy/q_scale_ratio"] = (
                        log_metrics["q_mean"] * (1.0 - config.discount) /
                        float(np.mean(learning_rewards)))
                log_metrics.update({
                    "train/update": updates,
                    "train/replay_size": replay.size,
                    "train/learner_ms": learner_elapsed * 1000.0,
                    # Unlike the startup synthetic benchmark, this is the
                    # measured actor/learner latency on real observations.
                    "benchmark/learner_ms": learner_elapsed * 1000.0,
                    "benchmark/actor_ms": actor_elapsed * 1000.0,
                    "benchmark/actor_utd_ms": actor_utd_elapsed * 1000.0,
                    "benchmark/update_call_ms": update_call_elapsed * 1000.0,
                    "benchmark/replay_sample_ms": replay_sample_elapsed * 1000.0,
                    "benchmark/update_kernel_ms": update_kernel_elapsed * 1000.0,
                    "benchmark/agent_sync_ms": agent_sync_elapsed * 1000.0,
                    "benchmark/metrics_sync_ms": metrics_sync_elapsed * 1000.0,
                })
                wandb_run.log(log_metrics, step=max(1, valid_steps))
        if wandb_run is not None and (
                (policy_transition and (
                    valid_steps == 1 or
                    valid_steps % max(1, log_interval) == 0)) or
                info.get("event", 0)):
            observation = np.asarray(obs, dtype=np.float32)
            # Actor/learner interval arrays are stored in seconds. Metrics
            # whose names end in _ms must be converted before logging.
            actor_stats = _window_stats(
                actor_intervals, log_interval, scale=1000.0)
            learner_stats = _window_stats(
                learner_intervals, log_interval, scale=1000.0)
            actor_utd_stats = _window_stats(
                actor_utd_intervals, log_interval, scale=1000.0)
            action_period_stats = _window_stats(
                getattr(env, "action_intervals_ms", []), log_interval)
            window_size = max(1, log_interval)
            learner_mean_ms = (
                float(np.mean(learner_intervals[-window_size:])) * 1000.0
                if learner_intervals else 0.0)
            actor_utd_mean_ms = (
                float(np.mean(actor_utd_intervals[-window_size:])) * 1000.0
                if actor_utd_intervals else 0.0)
            learner_group_hz = (
                1000.0 / learner_mean_ms if learner_mean_ms > 0.0 else 0.0)
            strict_cycle_hz = (
                1000.0 / actor_utd_mean_ms if actor_utd_mean_ms > 0.0 else 0.0)
            training_log = {
                # These are repeated every logging window so the dashboard
                # continuously shows the current real-time budget.
                "benchmark/utd_ratio": float(utd_ratio),
                "benchmark/policy_updates_per_action": (
                    float(updates / max(1, valid_steps - start_training + 1))
                    if replay.size >= start_training else 0.0),
                "benchmark/control_frequency_hz": (
                    1000.0 / action_period_stats["p50"]
                    if action_period_stats["p50"] > 0.0 else 0.0),
                "benchmark/actor_p50_ms": actor_stats["p50"],
                "benchmark/actor_p95_ms": actor_stats["p95"],
                "benchmark/actor_max_ms": actor_stats["max"],
                "benchmark/learner_p50_ms": learner_stats["p50"],
                "benchmark/learner_p95_ms": learner_stats["p95"],
                "benchmark/learner_max_ms": learner_stats["max"],
                "benchmark/actor_utd_p50_ms": actor_utd_stats["p50"],
                "benchmark/actor_utd_p95_ms": actor_utd_stats["p95"],
                "benchmark/actor_utd_max_ms": actor_utd_stats["max"],
                "benchmark/learner_group_hz": learner_group_hz,
                "benchmark/critic_updates_hz": (
                    learner_group_hz * float(utd_ratio)),
                "benchmark/strict_policy_cycle_hz": strict_cycle_hz,
                "benchmark/strict_deadline_margin_ms": (
                    50.0 - actor_utd_stats["p95"]),
                "benchmark/action_period_p50_ms": action_period_stats["p50"],
                "benchmark/action_period_p95_ms": action_period_stats["p95"],
                "benchmark/action_period_max_ms": action_period_stats["max"],
                "observation/finite": float(np.isfinite(observation).all()),
                "observation/mean": float(np.mean(observation)),
                "observation/std": float(np.std(observation)),
                "observation/min": float(np.min(observation)),
                "observation/max": float(np.max(observation)),
                "observation/rms": float(np.linalg.norm(observation) /
                                           np.sqrt(observation.size)),
                "training/valid_steps": valid_steps,
                "training/attempted_steps": attempted_steps,
                "training/replay_size": replay.size,
                "training/policy_transition": float(policy_transition),
                "training/step_reward": float(reward),
                "training/rolling_reward_100": (
                    float(np.mean(recent_rewards)) if recent_rewards else 0.0),
                "training/vx": float(info.get("reward/vx", np.nan)),
                "training/cos_pitch": float(info.get("reward/cos_pitch", np.nan)),
                "training/abs_dyaw": abs(float(info.get("reward/dyaw", np.nan))),
                "training/action_rms": float(np.linalg.norm(action) /
                                              np.sqrt(action.size)),
                "policy/action_delta_rms": action_delta_rms,
                "policy/action_saturation_fraction": action_saturation_fraction,
                "safety/fall_events": fall_events,
                "safety/upside_down_events": upside_down_events,
                "safety/standup_failures": standup_failures,
                "safety/causal_mismatches": causal_mismatches,
                "safety/terminal_patch_success": terminal_patch_successes,
                "safety/terminal_patch_missing": terminal_patch_missing,
                "safety/event_action_lag": (
                    float(event_action_lag_total / event_action_lag_count)
                    if event_action_lag_count else 0.0),
                "safety/falls_per_1000_valid_steps": (
                    1000.0 * fall_events / max(1, valid_steps)),
                "safety/recovery_per_1000_valid_steps": (
                    1000.0 * upside_down_events / max(1, valid_steps)),
                "safety/standup_failures_per_1000_valid_steps": (
                    1000.0 * standup_failures / max(1, valid_steps)),
                "safety/causal_mismatches_per_1000_valid_steps": (
                    1000.0 * causal_mismatches / max(1, valid_steps)),
                "safety/terminal_patch_missing_per_1000_valid_steps": (
                    1000.0 * terminal_patch_missing / max(1, valid_steps)),
                "safety/roll": float(info.get("safety/roll", np.nan)),
                "safety/pitch": float(info.get("safety/pitch", np.nan)),
                "safety/up_cos": float(info.get("safety/up_cos", np.nan)),
                "safety/acc_z": float(info.get("safety/acc_z", np.nan)),
                "safety/joint_tracking_error_rms": float(
                    info.get("safety/joint_tracking_error_rms", np.nan)),
                "safety/joint_tracking_error_max": float(
                    info.get("safety/joint_tracking_error_max", np.nan)),
                "safety/phase": float(info.get("phase", np.nan)),
                "safety/event": float(info.get("event", 0)),
                "safety/event_action_id": float(
                    info.get("event_action_id", 0)),
                "safety/event_confirm_ms": float(
                    info.get("event_confirm_ms", 0)),
                # Keep the legacy curve, but provide a learning-only curve so
                # the 10k random-action warmup cannot be mistaken for policy
                # performance.
            }
            if recent_action_saturation:
                saturation_window = np.asarray(
                    recent_action_saturation, dtype=np.float32)
                for joint_index, value in enumerate(
                        np.mean(saturation_window, axis=0)):
                    training_log[
                        f"policy/action_saturation_joint_{joint_index}"] = float(value)
            if info.get("event", 0) and recent_safety:
                pre_fall = list(recent_safety)
                training_log.update({
                    "safety/event_prev_1s_abs_roll": float(np.mean(
                        [abs(item["roll"]) for item in pre_fall])),
                    "safety/event_prev_1s_abs_pitch": float(np.mean(
                        [abs(item["pitch"]) for item in pre_fall])),
                    "safety/event_prev_1s_mean_vx": float(np.mean(
                        [item["vx"] for item in pre_fall])),
                    "safety/event_prev_1s_action_rms": float(np.mean(
                        [item["action_rms"] for item in pre_fall])),
                })
            if learning_rewards and policy_transition:
                training_log.update({
                    "training/learning_step_reward": float(reward),
                    "training/learning_rolling_reward_100": float(
                        np.mean(learning_rewards)),
                })
            wandb_run.log(training_log, step=max(1, valid_steps))
        if (policy_transition and
                valid_steps % max(1, checkpoint_interval) == 0 and
                checkpoint_root is not None):
            save_training_state(valid_steps, agent, replay)
        if (policy_transition and (
                valid_steps == 1 or
                valid_steps % max(1, progress_interval) == 0)):
            print(
                f"[train] step={valid_steps}/{max_steps} "
                f"attempted={attempted_steps} "
                f"phase={'warmup' if replay.size < start_training else 'learning'} "
                f"updates={updates} reward={reward:.4f} "
                f"policy_seq={info.get('policy_sequence', '?')} "
                f"action_id={info.get('applied_action_id', '?')} "
                f"vx={info.get('reward/vx', float('nan')):.3f} "
                f"cos_pitch={info.get('reward/cos_pitch', float('nan')):.3f} "
                f"dyaw={info.get('reward/dyaw', float('nan')):.3f}"
                + (f" learner_ms={learner_intervals[-1] * 1000.0:.2f}"
                   f" group_hz={1.0 / learner_intervals[-1]:.2f}"
                   f" critic_hz={utd_ratio / learner_intervals[-1]:.1f}"
                   f" cycle_hz={1.0 / actor_utd_intervals[-1]:.2f}"
                   if learner_intervals else ""),
                flush=True)
    action_intervals = np.asarray(getattr(env, "action_intervals_ms", []),
                                  dtype=np.float64)
    result = {"steps": valid_steps, "attempted_steps": attempted_steps,
              "updates": updates, "fall_events": fall_events,
              "upside_down_events": upside_down_events,
              "standup_failures": standup_failures,
              "causal_mismatches": causal_mismatches,
              "terminal_patch_successes": terminal_patch_successes,
              "terminal_patch_missing": terminal_patch_missing,
              "event_action_lag_mean": (
                  event_action_lag_total / event_action_lag_count
                  if event_action_lag_count else 0.0),
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
    result["benchmark_actor_utd_max_ms"] = float(max_benchmark_ms)
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
