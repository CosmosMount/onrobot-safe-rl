#!/usr/bin/env python3
"""Train matched SQRL Bellman critics on nested PPO transition cohorts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.qsafe.ppo_sqrl_critic import (
    PpoSqrlCriticConfig,
    PpoSqrlSafetyCritic,
    sqrl_bellman_target,
)
from safety_data.ppo_reference_actor import FrozenPpoReferenceActor
from safety_data.ppo_sqrl_master import validate_master_manifest
from safety_data.ppo_sqrl_protocol import (
    load_ppo_sqrl_protocol,
    ppo_sqrl_protocol_sha256,
)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.strip()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(path: Path, value: object, *, torch_value: bool = False) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if torch_value:
        torch.save(value, temporary)
    else:
        content = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _episode_keys(seed: np.ndarray, stage: np.ndarray, env: np.ndarray,
                  episode: np.ndarray) -> np.ndarray:
    return np.asarray([
        f"{int(a)}:{str(b)}:{int(c)}:{int(d)}"
        for a, b, c, d in zip(seed, stage, env, episode, strict=True)
    ], dtype="U64")


def load_nested_dataset(
    index_path: Path, selection_path: Path, budget: int,
) -> dict[str, np.ndarray]:
    index = json.loads(index_path.read_text())
    selection = json.loads(selection_path.read_text())
    selected_entry = next(
        item for item in selection["selections"] if item["nominal_budget"] == budget)
    selected = set(selected_entry["episode_keys"])
    role_by_key = {row["key"]: row["role"] for row in index["episodes"]}
    fields = [
        "observation_history_t", "critic_action", "next_observation_history",
        "c_t_plus_1", "terminated", "truncated", "next_policy_observation",
        "next_action_encoder_bias", "collector_stage", "ppo_seed", "env_id",
        "episode_id",
    ]
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    roles: list[np.ndarray] = []
    for source in index["sources"]:
        manifest_path = Path(source["manifest"])
        manifest = validate_master_manifest(manifest_path)
        for shard in manifest["shards"]:
            with np.load(manifest_path.parent / shard["path"], allow_pickle=False) as loaded:
                keys = _episode_keys(
                    loaded["ppo_seed"], loaded["collector_stage"],
                    loaded["env_id"], loaded["episode_id"])
                mask = np.fromiter((key in selected for key in keys), bool, len(keys))
                if not np.any(mask):
                    continue
                for name in fields:
                    chunks[name].append(loaded[name][mask].copy())
                roles.append(np.asarray([role_by_key[key] for key in keys[mask]], "U16"))
    arrays = {name: np.concatenate(values) for name, values in chunks.items()}
    arrays["role"] = np.concatenate(roles)
    if len(arrays["role"]) != selected_entry["realized_transitions"]:
        raise RuntimeError("loaded nested transition count differs from selection")
    return arrays


def auc(label: np.ndarray, score: np.ndarray) -> float:
    label = np.asarray(label, bool)
    order = np.argsort(score, kind="mergesort")
    rank = np.empty(len(score), np.float64)
    rank[order] = np.arange(1, len(score) + 1)
    positive = int(label.sum())
    negative = len(label) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    return float((rank[label].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def average_precision(label: np.ndarray, score: np.ndarray) -> float:
    label = np.asarray(label, bool)
    order = np.argsort(-score, kind="mergesort")
    target = label[order]
    positive = int(target.sum())
    if positive == 0:
        return float("nan")
    precision = np.cumsum(target) / np.arange(1, len(target) + 1)
    return float(precision[target].sum() / positive)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--budget", type=int, choices=(1_000_000, 3_000_000, 5_000_000), required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("action", "state_only", "shuffled"), default="action")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-transition", type=float, default=1 / 256)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    protocol = load_ppo_sqrl_protocol()
    reference = protocol["critic"]["reference_policy"]
    if args.reference_checkpoint.name != reference["checkpoint"] or not (
            args.reference_checkpoint.is_file()):
        raise ValueError("reference PPO checkpoint differs from protocol")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    arrays = load_nested_dataset(args.index, args.selection, args.budget)
    fit = arrays["role"] == "fit"
    calibration = arrays["role"] == "calibration"
    test = arrays["role"] == "test"
    if not np.any(fit) or not np.any(calibration) or not np.any(test):
        raise RuntimeError("nested dataset has an empty role")
    observation_mean = arrays["observation_history_t"][fit].mean(
        axis=(0, 1), dtype=np.float64).astype(np.float32)
    observation_std = arrays["observation_history_t"][fit].std(
        axis=(0, 1), dtype=np.float64).astype(np.float32)
    observation_std = np.maximum(observation_std, 1e-6)
    action_mean = arrays["critic_action"][fit].mean(axis=0, dtype=np.float64).astype(np.float32)
    action_std = arrays["critic_action"][fit].std(axis=0, dtype=np.float64).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = PpoSqrlCriticConfig(
        mode="state_only" if args.mode == "state_only" else "action")
    model = PpoSqrlSafetyCritic(config).to(device)
    target = PpoSqrlSafetyCritic(config).to(device)
    target.load_state_dict(model.state_dict())
    actor = FrozenPpoReferenceActor(args.reference_checkpoint).to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    fit_indices = np.flatnonzero(fit)
    rng = np.random.default_rng(args.seed)
    shuffled_action = arrays["critic_action"].copy()
    if args.mode == "shuffled":
        for stage in ("early", "boundary", "mature"):
            selected = np.flatnonzero(fit & (arrays["collector_stage"] == stage))
            shuffled_action[selected] = shuffled_action[rng.permutation(selected)]
    updates = max(1, round(len(fit_indices) * args.updates_per_transition))
    losses = []
    for update in range(updates):
        index = rng.choice(fit_indices, size=args.batch_size, replace=True)
        history = torch.from_numpy(
            (arrays["observation_history_t"][index] - observation_mean)
            / observation_std).to(device)
        next_history = torch.from_numpy(
            (arrays["next_observation_history"][index] - observation_mean)
            / observation_std).to(device)
        action_np = shuffled_action[index]
        action = torch.from_numpy((action_np - action_mean) / action_std).to(device)
        with torch.no_grad():
            next_policy_observation = torch.from_numpy(
                arrays["next_policy_observation"][index]).to(device)
            next_bias = torch.from_numpy(arrays["next_action_encoder_bias"][index]).to(device)
            next_action_absolute = actor.critic_action(
                next_policy_observation, next_bias, generator=generator)
            next_action = (next_action_absolute - torch.as_tensor(
                action_mean, device=device)) / torch.as_tensor(action_std, device=device)
            next_q = target(next_history, next_action)
            bellman = sqrl_bellman_target(
                torch.from_numpy(arrays["c_t_plus_1"][index]).to(device),
                torch.from_numpy(arrays["terminated"][index]).to(device),
                torch.from_numpy(arrays["truncated"][index]).to(device),
                next_q, gamma_safe=float(protocol["critic"]["gamma_safe"]))
        prediction = model(history, action)
        loss = F.mse_loss(prediction, bellman)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                    target.parameters(), model.parameters(), strict=True):
                target_parameter.lerp_(parameter, 0.005)
        if update % max(1, updates // 100) == 0 or update + 1 == updates:
            losses.append(float(loss.item()))

    def predict(mask: np.ndarray, *, permute: bool = False) -> np.ndarray:
        indices = np.flatnonzero(mask)
        result = []
        if permute:
            action_indices = rng.permutation(indices)
        else:
            action_indices = indices
        with torch.inference_mode():
            for start in range(0, len(indices), 4096):
                batch = indices[start:start + 4096]
                action_batch = action_indices[start:start + 4096]
                history = torch.from_numpy(
                    (arrays["observation_history_t"][batch] - observation_mean)
                    / observation_std).to(device)
                action = torch.from_numpy(
                    (arrays["critic_action"][action_batch] - action_mean)
                    / action_std).to(device)
                result.append(model(history, action).cpu().numpy())
        return np.concatenate(result)

    test_score = predict(test)
    permuted_score = predict(test, permute=True)
    test_label = arrays["c_t_plus_1"][test].astype(bool)
    metrics = {
        "immediate_cost_auroc": auc(test_label, test_score),
        "immediate_cost_auprc": average_precision(test_label, test_score),
        "immediate_cost_brier": float(np.mean((test_score - test_label) ** 2)),
        "test_time_action_permutation_mean_abs_change": float(np.mean(
            np.abs(test_score - permuted_score))),
        "fit_transitions": int(fit.sum()),
        "calibration_transitions": int(calibration.sum()),
        "test_transitions": int(test.sum()),
        "fit_first_falls": int(arrays["c_t_plus_1"][fit].sum()),
        "test_first_falls": int(test_label.sum()),
        "gradient_updates": updates,
    }
    artifact = {
        "schema_version": "qsafe.ppo_sqrl_bellman_critic.v1",
        "protocol_sha256": ppo_sqrl_protocol_sha256(),
        "trainer_commit": _git_head(),
        "mode": args.mode,
        "budget": args.budget,
        "gamma_safe": float(protocol["critic"]["gamma_safe"]),
        "cost_index": "c_t_plus_1",
        "critic_action_semantic": protocol["critic"]["critic_action"]["semantic"],
        "network_config": asdict(config),
        "model_state_dict": model.cpu().state_dict(),
        "target_state_dict": target.cpu().state_dict(),
        "normalization": {
            "observation_mean": torch.from_numpy(observation_mean),
            "observation_std": torch.from_numpy(observation_std),
            "action_mean": torch.from_numpy(action_mean),
            "action_std": torch.from_numpy(action_std),
        },
        "reference_checkpoint_sha256": _sha(args.reference_checkpoint),
        "training": {
            "batch_size": args.batch_size,
            "updates_per_transition": args.updates_per_transition,
            "gradient_updates": updates,
            "seed": args.seed,
            "sampled_loss_trace": losses,
        },
        "metrics": metrics,
    }
    _publish(args.output, artifact, torch_value=True)
    report = artifact.copy()
    for key in ("model_state_dict", "target_state_dict", "normalization"):
        report.pop(key)
    report["artifact_sha256"] = _sha(args.output)
    _publish(args.output.with_suffix(".report.json"), report)
    print(json.dumps(metrics, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
