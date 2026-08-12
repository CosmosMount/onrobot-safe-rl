#!/usr/bin/env python3
"""Train the matched critic on the existing SQRL/SAC trajectory replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.qsafe.ppo_sqrl_critic import (
    PpoSqrlCriticConfig, PpoSqrlSafetyCritic, sqrl_bellman_target)
from safety_data.ppo_reference_actor import (
    FrozenPpoReferenceActor, TARGET_OFFSET, TARGET_SCALE,
    sac_observation_to_ppo_actor_observation,
)
from safety_data.ppo_sqrl_protocol import ppo_sqrl_protocol_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _history(trajectory: list[dict], index: int, field: str) -> np.ndarray:
    values = np.stack([row[field] for row in trajectory[max(0, index - 4):index + 1]])
    if len(values) < 5:
        values = np.concatenate([np.repeat(values[:1], 5 - len(values), axis=0), values])
    return values.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    loaded = torch.load(args.replay, map_location="cpu", weights_only=False)
    trajectories = loaded.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("existing SQRL replay has no completed trajectories")
    rows = []
    for trajectory_id, trajectory in enumerate(trajectories):
        for index, row in enumerate(trajectory):
            observation = np.asarray(row["observation"], np.float32)
            action = np.clip(np.asarray(row["action"], np.float32), -1, 1)
            absolute = TARGET_OFFSET.numpy() + TARGET_SCALE.numpy() * action
            next_actor = sac_observation_to_ppo_actor_observation(
                np.asarray(row["next_observation"], np.float32)[None],
                episode_step=np.asarray([index + 1]))[0]
            rows.append({
                "history": _history(trajectory, index, "observation"),
                "action": absolute,
                "next_history": _history(trajectory, index, "next_observation"),
                "cost": bool(row["unsafe"]),
                "terminated": bool(row["unsafe"]),
                "truncated": bool(row["done"] and not row["unsafe"]),
                "next_actor": next_actor,
                "trajectory": trajectory_id,
            })
    arrays = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
    # Deterministic whole-trajectory roles; model training uses only fit.
    fit = (arrays["trajectory"] % 10) < 7
    observation_mean = arrays["history"][fit].mean((0, 1), dtype=np.float64).astype(np.float32)
    observation_std = np.maximum(
        arrays["history"][fit].std((0, 1), dtype=np.float64), 1e-6).astype(np.float32)
    action_mean = arrays["action"][fit].mean(0, dtype=np.float64).astype(np.float32)
    action_std = np.maximum(
        arrays["action"][fit].std(0, dtype=np.float64), 1e-6).astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    cfg = PpoSqrlCriticConfig(mode="action")
    model = PpoSqrlSafetyCritic(cfg).to(device)
    target = PpoSqrlSafetyCritic(cfg).to(device)
    target.load_state_dict(model.state_dict())
    actor = FrozenPpoReferenceActor(args.reference_checkpoint).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    fit_indices = np.flatnonzero(fit)
    updates = max(1, len(fit_indices) // 256)
    losses = []
    for _ in range(updates):
        index = rng.choice(fit_indices, 256, replace=True)
        history = torch.from_numpy(
            (arrays["history"][index] - observation_mean) / observation_std).to(device)
        next_history = torch.from_numpy(
            (arrays["next_history"][index] - observation_mean) / observation_std).to(device)
        action = torch.from_numpy(
            (arrays["action"][index] - action_mean) / action_std).to(device)
        with torch.no_grad():
            next_absolute = actor.critic_action(
                torch.from_numpy(arrays["next_actor"][index]).to(device),
                torch.zeros((len(index), 12), device=device), generator=generator)
            next_action = (next_absolute - torch.as_tensor(action_mean, device=device)) / torch.as_tensor(action_std, device=device)
            bellman = sqrl_bellman_target(
                torch.from_numpy(arrays["cost"][index]).to(device),
                torch.from_numpy(arrays["terminated"][index]).to(device),
                torch.from_numpy(arrays["truncated"][index]).to(device),
                target(next_history, next_action), gamma_safe=0.70)
        prediction = model(history, action)
        loss = F.mse_loss(prediction, bellman)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(target.parameters(), model.parameters(), strict=True):
                target_parameter.lerp_(parameter, 0.005)
        losses.append(float(loss.item()))
    artifact = {
        "schema_version": "qsafe.existing_sqrl_sac_data_matched_critic.v1",
        "protocol_sha256": ppo_sqrl_protocol_sha256(),
        "trainer_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
        "network_config": asdict(cfg),
        "model_state_dict": model.cpu().state_dict(),
        "normalization": {
            "observation_mean": torch.from_numpy(observation_mean),
            "observation_std": torch.from_numpy(observation_std),
            "action_mean": torch.from_numpy(action_mean),
            "action_std": torch.from_numpy(action_std),
        },
        "replay_sha256": _sha(args.replay),
        "transitions": len(rows),
        "fit_transitions": int(fit.sum()),
        "first_falls": int(arrays["cost"].sum()),
        "gradient_updates": updates,
        "losses": losses,
        "critic_action_semantic": "absolute_12d_joint_target_applied_to_pd_for_current_20ms_interval",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    torch.save(artifact, temporary)
    os.link(temporary, args.output)
    temporary.unlink()
    print(json.dumps({
        "transitions": len(rows), "first_falls": int(arrays["cost"].sum()),
        "gradient_updates": updates, "artifact_sha256": _sha(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
