#!/usr/bin/env python3
"""Fine-tune Q_safe with branch outcomes and same-state ranking loss."""

from __future__ import annotations

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from rl.agents import create_agent
from rl.agents.base.network import Network
from rl.agents.safe_droq.network import SafetyCritic
from safety_data.paths import (
    assert_development_path,
    assert_safe_evidence_output,
    require_v3_audit_consumed_or_safe_input,
)
from scripts.collect_native_grouped_qsafe import (
    _prepare_staged_outputs,
    _publish_staged_outputs,
)
from train.config import load_app_config


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    order = torch.argsort(probs)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(
        1, len(order) + 1, device=probs.device, dtype=torch.float32)
    positive = labels >= 0.5
    p = int(positive.sum())
    n = int((~positive).sum())
    auroc = float(
        ((ranks[positive].sum() - p * (p + 1) / 2) / (p * n)).item()
    ) if p and n else float("nan")
    return {
        "auroc": auroc,
        "brier": float(torch.mean(torch.square(probs - labels)).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config/go2_50hz_safe_adaptive_gated_v3.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--validation-dataset",
        help="Optional independently collected dataset; no states enter training.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-states", type=int, default=64)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-mode", choices=("binary", "discounted", "severity"),
        default="discounted")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--use-safety-context", action="store_true",
        help="Append normalized height, tilt, and contact count.")
    parser.add_argument(
        "--hidden-dims",
        help="Comma-separated safety MLP widths for fresh context critics.")
    parser.add_argument(
        "--history-frames", type=int, default=1,
        help="Use the latest N observation frames (1-4).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.config))
    checkpoint_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.checkpoint))
    dataset_path = assert_development_path(
        require_v3_audit_consumed_or_safe_input(args.dataset))
    validation_path = (
        None if args.validation_dataset is None
        else assert_development_path(
            require_v3_audit_consumed_or_safe_input(
                args.validation_dataset)))
    output = assert_development_path(assert_safe_evidence_output(args.output))
    report_path = assert_development_path(assert_safe_evidence_output(
        output.with_suffix(".metrics.json")))
    if output == report_path:
        raise ValueError("Q_safe checkpoint and metrics outputs must be distinct")
    existing = [
        path for path in (output, report_path)
        if os.path.lexists(os.fspath(path))]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    robot_cfg, _, agent_cfg = load_app_config(
        config_path, agent="safe_droq")
    agent_cfg.device_type = "cuda" if torch.cuda.is_available() else "cpu"
    agent_cfg.buffer_device_type = agent_cfg.device_type
    agent = create_agent(
        gym.spaces.Box(
            -100.0, 100.0, (robot_cfg.obs_dim,), dtype=np.float32),
        gym.spaces.Box(
            -1.0, 1.0, (robot_cfg.num_joints,), dtype=np.float32),
        {}, agent_cfg)
    agent.load(str(checkpoint_path))
    device = agent._device
    data = np.load(dataset_path)
    state_ids_np = data["state_ids"]
    unique_states = np.unique(state_ids_np)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_states)
    if args.validation_dataset:
        train_states = unique_states
        val_states = np.asarray([], dtype=unique_states.dtype)
    else:
        split = int(0.8 * len(unique_states))
        train_states = unique_states[:split]
        val_states = unique_states[split:]

    if not 1 <= args.history_frames <= 4:
        raise ValueError("--history-frames must be in [1, 4]")
    if args.history_frames > 1:
        observations_np = data["observation_histories"][
            :, -args.history_frames:, :].reshape(len(data["actions"]), -1)
    else:
        observations_np = data["observations"]
    fresh_critic = args.use_safety_context or args.history_frames > 1
    if args.use_safety_context:
        context = data["safety_contexts"].astype(np.float32).copy()
        context[:, 0] = (context[:, 0] - 0.30) / 0.10
        context[:, 1] = context[:, 1] / 0.70
        context[:, 2] = context[:, 2] / 16.0
        observations_np = np.concatenate(
            [observations_np, context], axis=-1)
    if fresh_critic:
        hidden_dims = (
            [int(value) for value in args.hidden_dims.split(",")]
            if args.hidden_dims else agent_cfg.safety_hidden_dims)
        critic_net = SafetyCritic(
            observations_np.shape[-1],
            robot_cfg.num_joints,
            hidden_dims).to(device)
        safety_critic = Network(
            network=critic_net,
            optimizer=optim.Adam(
                critic_net.parameters(), lr=agent_cfg.safety_lr))
    else:
        safety_critic = agent._safety_critic
    observations = torch.as_tensor(
        observations_np, device=device)
    actions = torch.as_tensor(data["actions"], device=device)
    binary_labels = torch.as_tensor(data["failures"], device=device)
    failure_steps = torch.as_tensor(
        data["failure_steps"], device=device, dtype=torch.float32)
    max_tilts = torch.as_tensor(
        data["max_tilts"], device=device, dtype=torch.float32)
    min_heights = torch.as_tensor(
        data["min_heights"], device=device, dtype=torch.float32)
    if args.target_mode == "discounted":
        labels = binary_labels * torch.pow(
            torch.full_like(failure_steps, args.gamma),
            torch.clamp(failure_steps - 1.0, min=0.0))
    elif args.target_mode == "severity":
        tilt_severity = max_tilts / float(
            robot_cfg.fallen_orientation_rad)
        height_severity = (0.30 - min_heights) / 0.12
        labels = torch.clamp(
            torch.maximum(tilt_severity, height_severity), 0.0, 1.0)
    else:
        labels = binary_labels
    state_ids = torch.as_tensor(state_ids_np, device=device)
    optimizer = safety_critic.optimizer
    assert optimizer is not None

    for epoch in range(args.epochs):
        rng.shuffle(train_states)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(train_states), args.batch_states):
            selected_states = torch.as_tensor(
                train_states[start:start + args.batch_states],
                device=device)
            mask = torch.isin(state_ids, selected_states)
            logits = safety_critic(
                observations=observations[mask],
                actions=actions[mask],
                training=True).reshape(-1)
            batch_labels = labels[mask]
            batch_binary = binary_labels[mask]
            positives = torch.clamp(batch_binary.sum(), min=1.0)
            negatives = torch.clamp(
                (1.0 - batch_binary).sum(), min=1.0)
            weights = torch.where(
                batch_binary > 0.5,
                0.5 / positives,
                0.5 / negatives) * len(batch_labels)
            classification = F.binary_cross_entropy_with_logits(
                logits, batch_labels, weight=weights)
            batch_state_ids = state_ids[mask]
            ranking_terms = []
            for state in selected_states:
                group = batch_state_ids == state
                group_logits = logits[group]
                group_targets = batch_labels[group]
                target_delta = (
                    group_targets[:, None] - group_targets[None, :])
                ordered = target_delta > 1e-6
                if bool(torch.any(ordered)):
                    logit_delta = (
                        group_logits[:, None] - group_logits[None, :])
                    ranking_terms.append(
                        F.softplus(-logit_delta[ordered]).mean())
            ranking = (
                torch.stack(ranking_terms).mean()
                if ranking_terms
                else torch.zeros((), device=device))
            loss = classification + args.ranking_weight * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        print(
            f"epoch={epoch + 1} loss={epoch_loss / max(batches, 1):.6f}")

    def evaluate(
        states: np.ndarray,
        *,
        eval_observations=observations,
        eval_actions=actions,
        eval_labels=labels,
        eval_binary_labels=binary_labels,
        eval_failure_steps=failure_steps,
        eval_state_ids=state_ids,
    ) -> dict[str, float]:
        mask = torch.isin(
            eval_state_ids, torch.as_tensor(states, device=device))
        with torch.no_grad():
            logits = safety_critic(
                observations=eval_observations[mask],
                actions=eval_actions[mask],
                training=False).reshape(-1)
        result = _metrics(logits, eval_binary_labels[mask])
        pair_correct = 0
        pair_total = 0
        masked_ids = eval_state_ids[mask]
        masked_labels = eval_binary_labels[mask]
        masked_failure_steps = eval_failure_steps[mask]
        ttf_correct = 0
        ttf_total = 0
        for state in torch.unique(masked_ids):
            group = masked_ids == state
            pos = logits[group & (masked_labels > 0.5)]
            neg = logits[group & (masked_labels <= 0.5)]
            if len(pos) and len(neg):
                pair_correct += int(
                    (pos[:, None] > neg[None, :]).sum().item())
                pair_total += len(pos) * len(neg)
            group_logits = logits[group]
            group_steps = masked_failure_steps[group]
            safer = group_steps[:, None] > group_steps[None, :]
            if bool(torch.any(safer)):
                ttf_correct += int(
                    (group_logits[:, None] < group_logits[None, :])[safer]
                    .sum().item())
                ttf_total += int(safer.sum().item())
        result["pair_accuracy"] = (
            pair_correct / pair_total if pair_total else float("nan"))
        result["states"] = int(len(states))
        result["ttf_pair_accuracy"] = (
            ttf_correct / ttf_total if ttf_total else float("nan"))
        return result

    if validation_path is not None:
        validation = np.load(validation_path)
        val_observations_np = validation["observations"]
        if args.history_frames > 1:
            val_observations_np = validation["observation_histories"][
                :, -args.history_frames:, :].reshape(
                    len(validation["actions"]), -1)
        else:
            val_observations_np = validation["observations"]
        if args.use_safety_context:
            val_context = validation[
                "safety_contexts"].astype(np.float32).copy()
            val_context[:, 0] = (val_context[:, 0] - 0.30) / 0.10
            val_context[:, 1] = val_context[:, 1] / 0.70
            val_context[:, 2] = val_context[:, 2] / 16.0
            val_observations_np = np.concatenate(
                [val_observations_np, val_context], axis=-1)
        val_observations = torch.as_tensor(
            val_observations_np, device=device)
        val_actions = torch.as_tensor(
            validation["actions"], device=device)
        val_labels = torch.as_tensor(
            validation["failures"], device=device)
        val_failure_steps = torch.as_tensor(
            validation["failure_steps"],
            device=device, dtype=torch.float32)
        val_state_ids = torch.as_tensor(
            validation["state_ids"], device=device)
        val_states = np.unique(validation["state_ids"])
        validation_metrics = evaluate(
            val_states,
            eval_observations=val_observations,
            eval_actions=val_actions,
            eval_labels=val_labels,
            eval_binary_labels=val_labels,
            eval_failure_steps=val_failure_steps,
            eval_state_ids=val_state_ids,
        )
    else:
        validation_metrics = evaluate(val_states)
    report = {
        "train": evaluate(train_states),
        "validation": validation_metrics,
        "dataset": str(dataset_path),
        "validation_dataset": (
            str(validation_path) if validation_path is not None else None),
        "use_safety_context": args.use_safety_context,
        "hidden_dims": args.hidden_dims,
        "history_frames": args.history_frames,
        "checkpoint": str(checkpoint_path),
    }
    staged = _prepare_staged_outputs((output, report_path))
    checkpoint_staging, report_staging = (item[0] for item in staged)
    try:
        safety_critic.save(str(checkpoint_staging))
        report_staging.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _publish_staged_outputs(staged)
    finally:
        for staging, _ in staged:
            staging.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
