"""End-to-end co-training and target-adaptation loops."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .buffers import TaskRollout, VectorRecentSafetyBuffer
from .checkpoint import (
    load_complete_checkpoint, module_sha256, save_checkpoint_no_clobber,
    verify_pretrain_lineage)
from .dual import ProjectedDual
from .io import append_jsonl, write_json_no_clobber
from .masking import select_masked_actions
from .mjlab import (
    advance_history, corrected_observation, forward_velocity,
    initialize_history, make_environment, project_environment_action,
    target_order_action, target_to_mjlab_action,
)
from .ppo import PpoConfig, PpoLearner
from .protocol import Protocol, reserve_output_directory
from .qsafe import SafetyCriticConfig, SafetyCriticLearner
from .resume import (
    capture_environment_state, capture_rng_state, restore_environment_state,
    restore_rng_state)


CHECKPOINT_EXPOSURES = (1_000_000, 2_000_000, 5_000_000,
                        10_000_000, 20_000_000, 30_000_000)


def _sample_candidates(actor, policy_observation: torch.Tensor,
                       candidates: int) -> torch.Tensor:
    distribution = actor.distribution(policy_observation)
    # Normal.sample(sample_shape) produces [K,B,A].
    return distribution.sample((int(candidates),)).permute(1, 0, 2)


def _masked(actor, safety_critic, qsafe_observation: torch.Tensor,
            policy_observation: torch.Tensor, cfg: Protocol):
    return select_masked_actions(
        qsafe_observation,
        sample_policy_actions=lambda _observation, count: _sample_candidates(
            actor, policy_observation, count),
        project_for_critic=target_order_action,
        critic=safety_critic,
        epsilon=cfg.epsilon_safe,
        candidates=cfg.mask_candidates,
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _paired_environment_step(wrapped: Any, action: torch.Tensor,
                             environment_rng: dict[str, object]):
    """Advance an environment-owned RNG stream without policy RNG coupling."""
    policy_rng = capture_rng_state()
    restore_rng_state(environment_rng)
    result = wrapped.step(action)
    next_environment_rng = capture_rng_state()
    restore_rng_state(policy_rng)
    return result, next_environment_rng


def _modules_finite(*modules: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(value).all().item())
               for module in modules for value in module.state_dict().values()
               if torch.is_tensor(value))


def _checkpoint_payload(*, phase: str, seed: int, iteration: int,
                        task_transitions: int, safety_transitions: int,
                        ppo: PpoLearner, safety: SafetyCriticLearner,
                        protocol_bundle: str, metrics: dict[str, Any],
                        run_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "ppo_sqrl_go2.checkpoint.v1",
        "ppo": ppo.state_dict(),
        "safety": safety.state_dict(),
        "rng": capture_rng_state(),
        "run_state": run_state,
        "metadata": {
            "phase": phase, "seed": seed, "iteration": iteration,
            "complete_iteration": True,
            "task_transitions": task_transitions,
            "safety_transitions": safety_transitions,
            "actor_sha256": module_sha256(ppo.actor),
            "safety_sha256": module_sha256(safety.critic),
            "protocol_bundle_sha256": protocol_bundle,
            "metrics": metrics,
        },
    }


def _episode_accumulators(count: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "return": torch.zeros(count, device=device),
        "length": torch.zeros(count, dtype=torch.long, device=device),
        "velocity": torch.zeros(count, device=device),
    }


def _update_episode_metrics(accumulator: dict[str, torch.Tensor], reward: torch.Tensor,
                            velocity: torch.Tensor, done: torch.Tensor) -> dict[str, float]:
    accumulator["return"] += reward
    accumulator["length"] += 1
    accumulator["velocity"] += velocity
    ended = done.to(torch.bool)
    result: dict[str, float] = {"episode/completed": float(ended.sum())}
    if bool(ended.any().item()):
        result.update({
            "episode/return_mean": float(accumulator["return"][ended].mean()),
            "episode/length_mean": float(accumulator["length"][ended].float().mean()),
            "episode/velocity_mean": float((
                accumulator["velocity"][ended]
                / accumulator["length"][ended].clamp_min(1)).mean()),
        })
        for value in accumulator.values():
            value[ended] = 0
    return result


def _merge_episode_metrics(total: dict[str, float], current: dict[str, float],
                           *, prefix: str = "") -> None:
    count = current.get("episode/completed", 0.0)
    total[f"{prefix}episode/completed"] = (
        total.get(f"{prefix}episode/completed", 0.0) + count)
    for field in ("return_mean", "length_mean", "velocity_mean"):
        key = f"episode/{field}"
        if key in current:
            sum_key = f"_{prefix}{key}_weighted_sum"
            total[sum_key] = total.get(sum_key, 0.0) + current[key] * count


def _finalize_episode_metrics(total: dict[str, float]) -> dict[str, float]:
    result = {key: value for key, value in total.items() if not key.startswith("_")}
    for prefix in ("", "safety_"):
        count = result.get(f"{prefix}episode/completed", 0.0)
        for field in ("return_mean", "length_mean", "velocity_mean"):
            sum_key = f"_{prefix}episode/{field}_weighted_sum"
            if count > 0 and sum_key in total:
                result[f"{prefix}episode/{field}"] = total[sum_key] / count
    return result


def run_cotrain(
    *, seed: int, output: str | Path, protocol_bundle: str,
    protocol: Protocol | None = None, device: str = "cuda:0",
    iterations: int | None = None, resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = protocol or Protocol()
    cfg.validate()
    total_iterations = cfg.pretrain_iterations if iterations is None else int(iterations)
    if total_iterations <= 0 or total_iterations > cfg.pretrain_iterations:
        raise ValueError("invalid cotrain iteration count")
    output = Path(output)
    resume_payload = None
    if resume_checkpoint is None:
        output = reserve_output_directory(output)
    else:
        if not output.is_dir() or (output / "manifest.json").exists():
            raise ValueError("resume requires an unfinished existing run directory")
        resume_payload = load_complete_checkpoint(
            resume_checkpoint, expected_seed=seed,
            expected_protocol_bundle=protocol_bundle)
    metrics_path = output / "metrics.jsonl"
    _seed_all(seed)
    environment = wrapped = None
    try:
        environment, wrapped, _, _ = make_environment(
            command_vx=cfg.pretrain_command,
            environments=cfg.task_envs + cfg.safety_envs,
            seed=seed, device=device)
        wrapped.episode_length_buf = torch.randint_like(
            wrapped.episode_length_buf, high=int(wrapped.max_episode_length))
        torch_device = torch.device(device)
        ppo = PpoLearner(PpoConfig(), device=torch_device, seed=seed)
        safety = SafetyCriticLearner(SafetyCriticConfig(
            gamma=cfg.gamma_safe, learning_rate=cfg.qsafe_lr,
            tau=cfg.qsafe_tau), device=torch_device)
        safety_buffer = VectorRecentSafetyBuffer(
            cfg.safety_envs, 500, cfg.recent_safety_trajectories,
            230, 47, 12, device=torch_device, seed=seed + 1)
        observations = wrapped.get_observations()
        observation = observations["actor"]
        critic_observation = observations["critic"]
        raw = corrected_observation(environment)
        history = initialize_history(raw)
        task_episode = _episode_accumulators(cfg.task_envs, torch_device)
        safety_episode = _episode_accumulators(cfg.safety_envs, torch_device)
        task_falls = 0
        safety_falls = 0
        safe_fraction_window: list[float] = []
        completed_safety_trajectories = 0
        start_iteration = 1
        if resume_payload is not None:
            resume_iteration = int(resume_payload["metadata"]["iteration"])
            if (resume_payload["metadata"].get("phase") != "pretrain"
                    or int(resume_payload["metadata"].get("task_transitions", -1))
                    != resume_iteration * cfg.task_envs * cfg.rollout_steps
                    or int(resume_payload["metadata"].get("safety_transitions", -1))
                    != resume_iteration * cfg.safety_envs * cfg.rollout_steps
                    or resume_iteration > total_iterations):
                raise ValueError("cotrain resume checkpoint step accounting differs")
            ppo.load_state_dict(resume_payload["ppo"])
            safety.load_state_dict(resume_payload["safety"])
            state = resume_payload["run_state"]
            restore_environment_state(environment, state["environment"])
            safety_buffer.load_state_dict(state["safety_buffer"])
            observation = state["observation"].to(torch_device)
            critic_observation = state["critic_observation"].to(torch_device)
            history = state["history"].to(torch_device)
            task_episode = {name: value.to(torch_device) for name, value in
                            state["task_episode"].items()}
            safety_episode = {name: value.to(torch_device) for name, value in
                              state["safety_episode"].items()}
            task_falls = int(state["task_falls"])
            safety_falls = int(state["safety_falls"])
            safe_fraction_window = list(state["safe_fraction_window"])
            completed_safety_trajectories = int(
                state["completed_safety_trajectories"])
            if safety_buffer.total_transitions != int(
                    resume_payload["metadata"]["safety_transitions"]):
                raise ValueError("safety buffer transition count differs from checkpoint")
            start_iteration = resume_iteration + 1
            restore_rng_state(resume_payload["rng"])
        for iteration in range(start_iteration, total_iterations + 1):
            rollout = TaskRollout(
                cfg.rollout_steps, cfg.task_envs, 47, 74, 230, 12, torch_device)
            iteration_safe_fraction = []
            iteration_acceptance = []
            iteration_no_safe = []
            iteration_attempts = []
            iteration_task_reward = []
            iteration_task_velocity = []
            episode_metrics: dict[str, float] = {}
            completed_this_iteration = 0
            for _ in range(cfg.rollout_steps):
                task_obs = observation[:cfg.task_envs]
                task_critic_obs = critic_observation[:cfg.task_envs]
                task_qsafe = history[:cfg.task_envs].reshape(cfg.task_envs, -1)
                safety_obs = observation[cfg.task_envs:]
                safety_qsafe = history[cfg.task_envs:].reshape(cfg.safety_envs, -1)
                with torch.no_grad():
                    task_action, task_logp, task_value, task_mean, task_std = ppo.act(
                        task_obs, task_critic_obs)
                    mask = _masked(
                        ppo.actor, safety.critic, safety_qsafe, safety_obs, cfg)
                    actions = torch.cat((
                        project_environment_action(task_action),
                        target_to_mjlab_action(mask.critic_action)), dim=0)
                next_observation_td, reward, done_long, extras = wrapped.step(actions)
                done = done_long.to(torch.bool)
                timeout = extras.get(
                    "time_outs", torch.zeros_like(done, dtype=torch.bool)).to(torch.bool)
                terminated = done & ~timeout
                next_observation = next_observation_td["actor"]
                next_critic_observation = next_observation_td["critic"]
                next_raw = corrected_observation(environment)
                next_history = advance_history(history, next_raw, done)
                velocity = forward_velocity(environment)
                task_reward = reward[:cfg.task_envs] + ppo.cfg.gamma * task_value * timeout[
                    :cfg.task_envs].float()
                rollout.add(
                    actor_observation=task_obs,
                    critic_observation=task_critic_obs,
                    qsafe_observation=task_qsafe,
                    action=task_action,
                    log_probability=task_logp,
                    value=task_value,
                    reward=task_reward,
                    done=done[:cfg.task_envs],
                    mean=task_mean,
                    std=task_std,
                    source="task")
                completed = safety_buffer.add_batch(
                    observation=safety_qsafe,
                    policy_observation=safety_obs,
                    action=mask.critic_action,
                    next_observation=next_history[cfg.task_envs:].reshape(
                        cfg.safety_envs, -1),
                    next_policy_observation=next_observation[cfg.task_envs:],
                    cost=terminated[cfg.task_envs:],
                    terminated=terminated[cfg.task_envs:],
                    truncated=timeout[cfg.task_envs:])
                completed_this_iteration += completed
                completed_safety_trajectories += completed
                task_falls += int(terminated[:cfg.task_envs].sum())
                safety_falls += int(terminated[cfg.task_envs:].sum())
                _merge_episode_metrics(episode_metrics, _update_episode_metrics(
                    task_episode, reward[:cfg.task_envs], velocity[:cfg.task_envs],
                    done[:cfg.task_envs]))
                _merge_episode_metrics(episode_metrics, _update_episode_metrics(
                    safety_episode, reward[cfg.task_envs:], velocity[cfg.task_envs:],
                    done[cfg.task_envs:]), prefix="safety_")
                iteration_safe_fraction.append(float(mask.candidate_safe_fraction.mean()))
                iteration_acceptance.append(float(mask.accepted.float().mean()))
                iteration_no_safe.append(float(mask.no_safe.float().mean()))
                iteration_attempts.append(float(mask.attempts.float().mean()))
                iteration_task_reward.append(float(reward[:cfg.task_envs].mean()))
                iteration_task_velocity.append(float(velocity[:cfg.task_envs].mean()))
                ppo.update_normalizers(
                    next_observation[:cfg.task_envs],
                    next_critic_observation[:cfg.task_envs])
                observation = next_observation
                critic_observation = next_critic_observation
                history = next_history
                if bool(torch.any(environment.sim.data.xfrc_applied != 0.0).item()):
                    raise RuntimeError("external force became non-zero")
            with torch.no_grad():
                last_value = ppo.value(critic_observation[:cfg.task_envs])
            rollout.finish(last_value, ppo.cfg.gamma, ppo.cfg.lam)
            safety_metrics: dict[str, float] = {}
            if len(safety_buffer) >= cfg.safety_batch_size:
                for _ in range(completed_this_iteration):
                    batch = safety_buffer.sample(cfg.safety_batch_size)
                    safety_metrics = safety.update(
                        batch,
                        lambda qsafe_obs, policy_obs: _masked(
                            ppo.actor, safety.critic, qsafe_obs, policy_obs, cfg
                        ).critic_action)
            ppo_metrics = ppo.update(rollout)
            task_transitions = iteration * cfg.task_envs * cfg.rollout_steps
            safety_transitions = iteration * cfg.safety_envs * cfg.rollout_steps
            mean_safe_fraction = float(np.mean(iteration_safe_fraction))
            safe_fraction_window.append(mean_safe_fraction)
            metric: dict[str, Any] = {
                "iteration": iteration,
                "task_transitions": task_transitions,
                "safety_transitions": safety_transitions,
                "task_falls": task_falls,
                "safety_falls": safety_falls,
                "safety_completed_trajectories": completed_safety_trajectories,
                "safety_buffer_size": len(safety_buffer),
                "safety_buffer_retained_falls": safety_buffer.retained_falls,
                "safety_buffer_fall_density": (
                    safety_buffer.retained_falls / max(1, len(safety_buffer))),
                "mask/candidate_safe_fraction": mean_safe_fraction,
                "mask/acceptance": float(np.mean(iteration_acceptance)),
                "mask/no_safe": float(np.mean(iteration_no_safe)),
                "mask/attempts": float(np.mean(iteration_attempts)),
                "reward/task_step_mean": float(np.mean(iteration_task_reward)),
                "forward_velocity/task_mean": float(np.mean(iteration_task_velocity)),
            } | _finalize_episode_metrics(episode_metrics) | safety_metrics | ppo_metrics
            if not all(np.isfinite(float(value)) for value in metric.values()):
                raise RuntimeError("non-finite co-training metric")
            append_jsonl(metrics_path, metric)
            if task_transitions in CHECKPOINT_EXPOSURES or iteration == total_iterations:
                payload = _checkpoint_payload(
                    phase="pretrain", seed=seed, iteration=iteration,
                    task_transitions=task_transitions,
                    safety_transitions=safety_transitions,
                    ppo=ppo, safety=safety,
                    protocol_bundle=protocol_bundle, metrics=metric,
                    run_state={
                        "environment": capture_environment_state(environment),
                        "safety_buffer": safety_buffer.state_dict(),
                        "observation": observation.clone(),
                        "critic_observation": critic_observation.clone(),
                        "history": history.clone(),
                        "task_episode": {
                            name: value.clone() for name, value in task_episode.items()},
                        "safety_episode": {
                            name: value.clone() for name, value in safety_episode.items()},
                        "task_falls": task_falls,
                        "safety_falls": safety_falls,
                        "safe_fraction_window": list(safe_fraction_window),
                        "completed_safety_trajectories": completed_safety_trajectories,
                    })
                save_checkpoint_no_clobber(
                    output / f"step_{task_transitions:012d}.pt", payload)
        final_window = max(1, int(np.ceil(0.2 * total_iterations)))
        final_safe_fraction = float(np.mean(safe_fraction_window[-final_window:]))
        manifest = {
            "schema_version": "ppo_sqrl_go2.cotrain_run.v1",
            "status": "finished", "seed": seed,
            "task_transitions": total_iterations * cfg.task_envs * cfg.rollout_steps,
            "safety_transitions": total_iterations * cfg.safety_envs * cfg.rollout_steps,
            "task_falls": task_falls, "safety_total_falls": safety_falls,
            "safety_updates": safety.updates,
            "safety_buffer_retained_falls": safety_buffer.retained_falls,
            "final_safe_fraction": final_safe_fraction,
            "all_numerics_finite": _modules_finite(
                ppo.actor, ppo.value, safety.critic, safety.target),
            "actor_sha256": module_sha256(ppo.actor),
            "safety_sha256": module_sha256(safety.critic),
            "protocol_bundle_sha256": protocol_bundle,
            "final_checkpoint": str(output / f"step_{total_iterations * cfg.task_envs * cfg.rollout_steps:012d}.pt"),
        }
        write_json_no_clobber(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        append_jsonl(output / "attempt_ledger.jsonl", {
            "status": "failed", "seed": seed, "error": repr(exc)})
        raise
    finally:
        if wrapped is not None:
            wrapped.close()


def _load_pretrain(path: str | Path, *, seed: int, device: torch.device,
                   protocol_bundle: str, protocol: Protocol,
                   require_complete: bool = True):
    payload = torch.load(path, map_location=device, weights_only=False)
    verify_pretrain_lineage(payload, seed=seed)
    if payload["metadata"].get("protocol_bundle_sha256") != protocol_bundle:
        raise ValueError("pretrain checkpoint protocol differs")
    if require_complete and (
            int(payload["metadata"].get("task_transitions", -1))
            != protocol.pretrain_task_transitions
            or int(payload["metadata"].get("safety_transitions", -1))
            != protocol.pretrain_safety_transitions):
        raise ValueError("target branch requires the complete fixed pretrain budget")
    ppo = PpoLearner(PpoConfig(), device=device, seed=seed)
    # Transfer actor only. Value and optimizer are branch-fresh by protocol.
    ppo.actor.load_state_dict(payload["ppo"]["actor"])
    safety = SafetyCriticLearner(SafetyCriticConfig(), device=device)
    safety.load_state_dict(payload["safety"], optimizer=False)
    safety.freeze()
    return payload, ppo, safety


def _rollout_mean_violation(ppo: PpoLearner, safety_critic,
                            rollout: TaskRollout, epsilon: float,
                            chunk: int = 4096) -> float:
    actor_obs = rollout.actor_observation.reshape(-1, 47)
    qsafe_obs = rollout.qsafe_observation.reshape(-1, 230)
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(actor_obs), chunk):
            action, _, _, _ = ppo.actor.sample(actor_obs[start:start + chunk])
            risk = safety_critic(
                qsafe_obs[start:start + chunk], target_order_action(action))
            total += float((risk - epsilon).sum())
            count += len(risk)
    return total / count


def run_target_branch(
    *, seed: int, branch: str, pretrain_checkpoint: str | Path,
    output: str | Path, pretrain_protocol_bundle: str,
    target_protocol_bundle: str, protocol: Protocol | None = None,
    device: str = "cuda:0", iterations: int | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = protocol or Protocol()
    cfg.validate()
    if branch not in {"ppo_transfer", "ppo_safe"}:
        raise ValueError("unknown target branch")
    total_iterations = cfg.target_iterations if iterations is None else int(iterations)
    if total_iterations <= 0 or total_iterations > cfg.target_iterations:
        raise ValueError("invalid target iteration count")
    output = Path(output)
    resume_payload = None
    if resume_checkpoint is None:
        output = reserve_output_directory(output)
    else:
        if not output.is_dir() or (output / "manifest.json").exists():
            raise ValueError("resume requires an unfinished existing target directory")
        resume_payload = load_complete_checkpoint(
            resume_checkpoint, expected_seed=seed,
            expected_protocol_bundle=target_protocol_bundle)
    _seed_all(seed)
    environment = wrapped = None
    try:
        torch_device = torch.device(device)
        source, ppo, safety = _load_pretrain(
            pretrain_checkpoint, seed=seed, device=torch_device,
            protocol_bundle=pretrain_protocol_bundle,
            protocol=cfg,
            require_complete=(iterations is None))
        source_actor_hash = source["metadata"]["actor_sha256"]
        source_safety_hash = source["metadata"]["safety_sha256"]
        environment, wrapped, _, _ = make_environment(
            command_vx=cfg.target_command, environments=cfg.target_envs,
            seed=seed, device=device)
        wrapped.episode_length_buf = torch.randint_like(
            wrapped.episode_length_buf, high=int(wrapped.max_episode_length))
        observations = wrapped.get_observations()
        observation = observations["actor"]
        critic_observation = observations["critic"]
        history = initialize_history(corrected_observation(environment))
        episode = _episode_accumulators(cfg.target_envs, torch_device)
        falls = 0
        dual = ProjectedDual(cfg.dual_lr, 0.0) if branch == "ppo_safe" else None
        metrics_path = output / "metrics.jsonl"
        environment_rng = capture_rng_state()
        # Branch-independent policy randomness; environment reset randomness is
        # kept in the separate stream above.
        _seed_all(70_000_000 + seed)
        start_iteration = 1
        if resume_payload is not None:
            if resume_payload["metadata"].get("branch") != branch:
                raise ValueError("target resume branch differs")
            resume_iteration = int(resume_payload["metadata"]["iteration"])
            if (int(resume_payload["metadata"].get("transitions", -1))
                    != resume_iteration * cfg.target_envs * cfg.rollout_steps
                    or resume_iteration > total_iterations):
                raise ValueError("target resume checkpoint step accounting differs")
            ppo.load_state_dict(resume_payload["ppo"])
            state = resume_payload["run_state"]
            restore_environment_state(environment, state["environment"])
            observation = state["observation"].to(torch_device)
            critic_observation = state["critic_observation"].to(torch_device)
            history = state["history"].to(torch_device)
            episode = {name: value.to(torch_device) for name, value in
                       state["episode"].items()}
            falls = int(state["falls"])
            environment_rng = state["environment_rng"]
            if dual is not None:
                dual.value = float(resume_payload["dual"])
            start_iteration = resume_iteration + 1
            restore_rng_state(resume_payload["rng"])
        for iteration in range(start_iteration, total_iterations + 1):
            rollout = TaskRollout(
                cfg.rollout_steps, cfg.target_envs, 47, 74, 230, 12, torch_device)
            episode_metrics: dict[str, float] = {}
            iteration_reward = []
            iteration_velocity = []
            iteration_tracking_error = []
            for _ in range(cfg.rollout_steps):
                qsafe_observation = history.reshape(cfg.target_envs, -1)
                with torch.no_grad():
                    action, logp, value, mean, std = ppo.act(
                        observation, critic_observation)
                (next_td, reward, done_long, extras), environment_rng = (
                    _paired_environment_step(
                        wrapped, project_environment_action(action),
                        environment_rng))
                done = done_long.to(torch.bool)
                timeout = extras.get(
                    "time_outs", torch.zeros_like(done, dtype=torch.bool)).to(torch.bool)
                terminated = done & ~timeout
                next_observation = next_td["actor"]
                next_critic_observation = next_td["critic"]
                next_history = advance_history(
                    history, corrected_observation(environment), done)
                velocity = forward_velocity(environment)
                training_reward = reward + ppo.cfg.gamma * value * timeout.float()
                rollout.add(
                    actor_observation=observation,
                    critic_observation=critic_observation,
                    qsafe_observation=qsafe_observation,
                    action=action, log_probability=logp, value=value,
                    reward=training_reward, done=done,
                    mean=mean, std=std, source="task")
                falls += int(terminated.sum())
                _merge_episode_metrics(
                    episode_metrics,
                    _update_episode_metrics(episode, reward, velocity, done))
                iteration_reward.append(float(reward.mean()))
                iteration_velocity.append(float(velocity.mean()))
                iteration_tracking_error.append(float(
                    (velocity - cfg.target_command).abs().mean()))
                ppo.update_normalizers(next_observation, next_critic_observation)
                observation = next_observation
                critic_observation = next_critic_observation
                history = next_history
            with torch.no_grad():
                last_value = ppo.value(critic_observation)
            rollout.finish(last_value, ppo.cfg.gamma, ppo.cfg.lam)
            violation = 0.0
            if dual is not None:
                violation = _rollout_mean_violation(
                    ppo, safety.critic, rollout, cfg.epsilon_safe)
                dual.update(violation)
            ppo_metrics = ppo.update(
                rollout, safety_critic=(safety.critic if dual is not None else None),
                dual=dual, epsilon_safe=cfg.epsilon_safe,
                to_critic_action=target_order_action)
            transitions = iteration * cfg.target_envs * cfg.rollout_steps
            metric: dict[str, Any] = {
                "iteration": iteration, "transitions": transitions,
                "falls": falls, "nu": 0.0 if dual is None else dual.value,
                "constraint_violation": violation,
                "reward/step_mean": float(np.mean(iteration_reward)),
                "forward_velocity/mean": float(np.mean(iteration_velocity)),
                "tracking_error/mean": float(np.mean(iteration_tracking_error)),
            } | _finalize_episode_metrics(episode_metrics) | ppo_metrics
            if not all(np.isfinite(float(value)) for value in metric.values()):
                raise RuntimeError("non-finite target metric")
            append_jsonl(metrics_path, metric)
            if transitions in cfg.evaluation_exposures[1:] or iteration == total_iterations:
                save_checkpoint_no_clobber(output / f"step_{transitions:012d}.pt", {
                    "schema_version": "ppo_sqrl_go2.target_checkpoint.v1",
                    "ppo": ppo.state_dict(),
                    "dual": None if dual is None else dual.value,
                    "rng": capture_rng_state(),
                    "run_state": {
                        "environment": capture_environment_state(environment),
                        "observation": observation.clone(),
                        "critic_observation": critic_observation.clone(),
                        "history": history.clone(),
                        "episode": {name: value.clone() for name, value in episode.items()},
                        "falls": falls,
                        "environment_rng": environment_rng,
                    },
                    "metadata": {
                        "seed": seed, "branch": branch,
                        "complete_iteration": True, "iteration": iteration,
                        "transitions": transitions,
                        "falls": falls,
                        "source_actor_sha256": source_actor_hash,
                        "source_safety_sha256": source_safety_hash,
                        "target_protocol_bundle_sha256": target_protocol_bundle,
                        "protocol_bundle_sha256": target_protocol_bundle,
                    },
                })
        manifest = {
            "schema_version": "ppo_sqrl_go2.target_run.v1",
            "status": "finished", "seed": seed, "branch": branch,
            "transitions": total_iterations * cfg.target_envs * cfg.rollout_steps,
            "falls": falls, "nu": 0.0 if dual is None else dual.value,
            "source_actor_sha256": source_actor_hash,
            "source_safety_sha256": source_safety_hash,
            "target_protocol_bundle_sha256": target_protocol_bundle,
            "pretrain_protocol_bundle_sha256": pretrain_protocol_bundle,
            "all_numerics_finite": _modules_finite(
                ppo.actor, ppo.value, safety.critic),
        }
        write_json_no_clobber(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        append_jsonl(output / "attempt_ledger.jsonl", {
            "status": "failed", "seed": seed, "branch": branch,
            "error": repr(exc)})
        raise
    finally:
        if wrapped is not None:
            wrapped.close()
