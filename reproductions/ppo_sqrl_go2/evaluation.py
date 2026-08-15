"""Frozen masked/unmasked evaluation that never feeds training storage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import module_sha256, verify_pretrain_lineage
from .masking import select_masked_actions
from .mjlab import (
    advance_history, corrected_observation, forward_velocity,
    initialize_history, make_environment, project_environment_action,
    target_order_action, target_to_mjlab_action,
)
from .ppo import PpoConfig, PpoLearner
from .protocol import Protocol, reserve_output_directory
from .qsafe import SafetyCriticConfig, SafetyCriticLearner
from .io import write_json_no_clobber


def _load_actor(checkpoint: dict[str, Any], device: torch.device) -> PpoLearner:
    learner = PpoLearner(PpoConfig(), device=device, seed=0)
    source = checkpoint["ppo"]
    actor_state = source["actor"] if "actor" in source else source
    learner.actor.load_state_dict(actor_state)
    learner.actor.eval()
    return learner


def _load_safety(pretrain: dict[str, Any], device: torch.device):
    learner = SafetyCriticLearner(SafetyCriticConfig(), device=device)
    learner.load_state_dict(pretrain["safety"], optimizer=False)
    return learner.freeze()


def evaluate_checkpoint(
    *, actor_checkpoint: str | Path, pretrain_checkpoint: str | Path,
    seed: int, branch: str, exposure: int, masked: bool, output: str | Path,
    pretrain_protocol_bundle: str, target_protocol_bundle: str,
    protocol: Protocol | None = None, device: str = "cuda:0",
    episodes: int | None = None,
) -> dict[str, Any]:
    cfg = protocol or Protocol()
    count = cfg.evaluation_episodes if episodes is None else int(episodes)
    if count <= 0:
        raise ValueError("evaluation episode count must be positive")
    output = reserve_output_directory(output)
    torch_device = torch.device(device)
    actor_payload = torch.load(actor_checkpoint, map_location=device, weights_only=False)
    pretrain_payload = torch.load(pretrain_checkpoint, map_location=device, weights_only=False)
    verify_pretrain_lineage(pretrain_payload, seed=seed)
    pretrain_metadata = pretrain_payload["metadata"]
    if pretrain_metadata.get("protocol_bundle_sha256") != pretrain_protocol_bundle:
        raise ValueError("evaluation pretrain protocol differs")
    if exposure == 0:
        if Path(actor_checkpoint).resolve() != Path(pretrain_checkpoint).resolve():
            raise ValueError("zero-exposure evaluation must use pretrain actor")
    else:
        metadata = actor_payload.get("metadata", {})
        if (int(metadata.get("seed", -1)) != seed
                or metadata.get("branch") != branch
                or int(metadata.get("transitions", -1)) != exposure
                or metadata.get("protocol_bundle_sha256") != target_protocol_bundle
                or metadata.get("source_actor_sha256")
                != pretrain_metadata.get("actor_sha256")
                or metadata.get("source_safety_sha256")
                != pretrain_metadata.get("safety_sha256")):
            raise ValueError("evaluation target checkpoint lineage differs")
    learner = _load_actor(actor_payload, torch_device)
    safety = _load_safety(pretrain_payload, torch_device)
    evaluation_seed = int(seed * 1_000_000 + exposure)
    torch.manual_seed(evaluation_seed)
    torch.cuda.manual_seed_all(evaluation_seed)
    environment = wrapped = None
    try:
        environment, wrapped, _, _ = make_environment(
            command_vx=cfg.target_command, environments=count,
            seed=evaluation_seed, device=device)
        observation = wrapped.get_observations()["actor"]
        history = initialize_history(corrected_observation(environment))
        finished = torch.zeros(count, dtype=torch.bool, device=torch_device)
        episode_return = torch.zeros(count, device=torch_device)
        episode_velocity = torch.zeros(count, device=torch_device)
        episode_length = torch.zeros(count, dtype=torch.long, device=torch_device)
        result_return = torch.zeros_like(episode_return)
        result_velocity = torch.zeros_like(episode_velocity)
        result_length = torch.zeros_like(episode_length)
        result_fall = torch.zeros(count, dtype=torch.bool, device=torch_device)
        mask_safe = []
        mask_acceptance = []
        mask_no_safe = []
        while not bool(finished.all().item()):
            active = ~finished
            with torch.no_grad():
                if masked:
                    qsafe = history.reshape(count, -1)
                    distribution = learner.actor.distribution(observation)
                    selected = select_masked_actions(
                        qsafe,
                        sample_policy_actions=lambda _obs, candidates: distribution.sample(
                            (candidates,)).permute(1, 0, 2),
                        project_for_critic=target_order_action,
                        critic=safety, epsilon=cfg.epsilon_safe,
                        candidates=cfg.mask_candidates)
                    action = target_to_mjlab_action(selected.critic_action)
                    mask_safe.append(float(selected.candidate_safe_fraction[active].mean()))
                    mask_acceptance.append(float(selected.accepted[active].float().mean()))
                    mask_no_safe.append(float(selected.no_safe[active].float().mean()))
                else:
                    action, _, _, _ = learner.actor.sample(observation)
                    action = project_environment_action(action)
            next_td, reward, done_long, extras = wrapped.step(action)
            done = done_long.to(torch.bool)
            timeout = extras.get(
                "time_outs", torch.zeros_like(done, dtype=torch.bool)).to(torch.bool)
            velocity = forward_velocity(environment)
            episode_return[active] += reward[active]
            episode_velocity[active] += velocity[active]
            episode_length[active] += 1
            first_done = active & done
            result_return[first_done] = episode_return[first_done]
            result_velocity[first_done] = (
                episode_velocity[first_done] / episode_length[first_done].clamp_min(1))
            result_length[first_done] = episode_length[first_done]
            result_fall[first_done] = ~timeout[first_done]
            finished |= first_done
            observation = next_td["actor"]
            history = advance_history(
                history, corrected_observation(environment), done)
        report = {
            "schema_version": "ppo_sqrl_go2.evaluation.v1",
            "seed": seed, "exposure": exposure, "masked": masked,
            "branch": branch,
            "episodes": count, "falls": int(result_fall.sum()),
            "mean_return": float(result_return.mean()),
            "mean_forward_velocity": float(result_velocity.mean()),
            "mean_tracking_error": float((result_velocity - cfg.target_command).abs().mean()),
            "mean_episode_length": float(result_length.float().mean()),
            "mask_candidate_safe_fraction": (
                None if not masked else float(np.mean(mask_safe))),
            "mask_acceptance": None if not masked else float(np.mean(mask_acceptance)),
            "mask_no_safe": None if not masked else float(np.mean(mask_no_safe)),
            "training_data_destination": None,
            "actor_sha256": module_sha256(learner.actor),
            "safety_sha256": module_sha256(safety),
            "pretrain_protocol_bundle_sha256": pretrain_protocol_bundle,
            "target_protocol_bundle_sha256": target_protocol_bundle,
        }
        write_json_no_clobber(output / "manifest.json", report)
        return report
    finally:
        if wrapped is not None:
            wrapped.close()
