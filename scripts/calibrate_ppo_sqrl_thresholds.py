#!/usr/bin/env python3
"""Freeze PPO SQRL thresholds on episode-disjoint calibration transitions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.qsafe.ppo_sqrl_critic import PpoSqrlCriticConfig, PpoSqrlSafetyCritic
from scripts.train_ppo_sqrl_critic import load_nested_dataset


def _threshold(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PpoSqrlCriticConfig(**artifact["network_config"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PpoSqrlSafetyCritic(cfg).to(device).eval()
    model.load_state_dict(artifact["model_state_dict"])
    mask = arrays["role"] == "calibration"
    positive = mask & arrays["c_t_plus_1"].astype(bool)
    indices = np.flatnonzero(positive)
    norm = {key: np.asarray(value) for key, value in artifact["normalization"].items()}
    scores = []
    with torch.inference_mode():
        for start in range(0, len(indices), 4096):
            selected = indices[start:start + 4096]
            history = ((arrays["observation_history_t"][selected]
                        - norm["observation_mean"]) / norm["observation_std"])
            action = ((arrays["critic_action"][selected] - norm["action_mean"])
                      / norm["action_std"])
            scores.append(model(
                torch.from_numpy(history).float().to(device),
                torch.from_numpy(action).float().to(device)).cpu().numpy())
    score = np.concatenate(scores)
    if len(score) < 5:
        raise RuntimeError("calibration set has fewer than five first-fall positives")
    # Largest threshold whose empirical first-fall recall is at least 90%.
    epsilon = float(np.quantile(score, 0.10, method="lower"))
    return {
        "epsilon_safe": epsilon,
        "calibration_transitions": int(mask.sum()),
        "calibration_first_falls": int(len(score)),
        "empirical_first_fall_recall": float(np.mean(score >= epsilon)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--critic", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    by_budget: dict[int, dict[str, np.ndarray]] = {}
    report = {
        "schema_version": "qsafe.ppo_sqrl_thresholds.v1",
        "rule": "largest_raw_q_threshold_with_empirical_first_fall_recall_at_least_0.90",
        "critics": {},
    }
    for path in args.critic:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        budget = int(artifact["budget"])
        if budget not in by_budget:
            by_budget[budget] = load_nested_dataset(
                args.index, args.selection, budget)
        report["critics"][path.stem] = _threshold(path, by_budget[budget])
        # Keep peak memory bounded when budgets are supplied in ascending order.
        previous = [value for value in by_budget if value < budget]
        for value in previous:
            del by_budget[value]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    os.link(temporary, args.output); temporary.unlink()
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
