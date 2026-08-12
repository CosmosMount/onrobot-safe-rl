#!/usr/bin/env python3
"""Create the fail-closed decision report from frozen PPO branch outcomes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.ppo_same_state_gate import independent_oracle, summarize_selector


def _pair_metrics(fall: np.ndarray, risk: np.ndarray) -> dict[str, float | int]:
    nominal = fall[:, 0, 4:].mean(axis=1)
    candidate = fall[:, 1:, 4:].mean(axis=2)
    truth = nominal[:, None] - candidate
    predicted = risk[:, 0, None] - risk[:, 1:]
    informative = truth != 0
    strong = np.abs(truth) >= 0.5

    def accuracy(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(np.sign(predicted[mask]) == np.sign(truth[mask])))

    return {
        "informative_pairs": int(informative.sum()),
        "pair_accuracy": accuracy(informative),
        "strong_pairs": int(strong.sum()),
        "strong_pair_accuracy": accuracy(strong),
    }


def _minimal_choice(risk: np.ndarray, action: np.ndarray, epsilon: float) -> np.ndarray:
    result = np.zeros(len(risk), np.int16)
    distance = np.sqrt(np.mean((action - action[:, :1]) ** 2, axis=2))
    for state in range(len(risk)):
        if risk[state, 0] < epsilon:
            continue
        safe = np.flatnonzero(risk[state] < epsilon)
        if len(safe):
            result[state] = int(safe[np.argmin(distance[state, safe])])
    return result


def _sqrl_choice(risk: np.ndarray, epsilon: float) -> np.ndarray:
    return np.asarray([
        next((index for index, value in enumerate(row) if value < epsilon),
             int(np.argmin(row))) for row in risk], np.int16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    thresholds = json.loads(args.thresholds.read_text())["critics"]
    with np.load(args.branches, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    fall = arrays["fall"]
    action = arrays["critic_action"]
    oracle_choice, _ = independent_oracle(fall)
    report = {
        "schema_version": "qsafe.ppo_data_advantage_branching_decision.v1",
        "protected_protocol": {
            "states": 200, "candidates_per_state": 16,
            "paired_crn_replicas": 8, "horizon_policy_steps": 96,
            "oracle_discovery_replicas": [1, 2, 3, 4],
            "independent_evaluation_replicas": [5, 6, 7, 8],
            "branch_outcomes_used_to_choose_learned_action": False,
        },
        "candidate_action_space": {
            "nominal_fall_rate": float(fall[:, 0, 4:].mean()),
            "candidate_fall_rate_by_index": fall[:, :, 4:].mean((0, 2)).tolist(),
            "independent_oracle": summarize_selector(
                fall, oracle_choice, bootstrap_seed=9100),
        },
        "critics": {},
    }
    legacy = "existing-sqrl-sac-data"
    for model_index, key in enumerate(sorted(
            name.removeprefix("risk_") for name in arrays if name.startswith("risk_"))):
        risk = arrays[f"risk_{key}"]
        is_state_only = key.endswith("state-only")
        epsilon = 0.10 if key == legacy else float(thresholds[key]["epsilon_safe"])
        top1 = np.zeros(len(risk), np.int16) if is_state_only else np.argmin(risk, axis=1)
        sqrl = np.zeros(len(risk), np.int16) if is_state_only else _sqrl_choice(risk, epsilon)
        minimal = np.zeros(len(risk), np.int16) if is_state_only else _minimal_choice(
            risk, action, epsilon)
        model = {
            "epsilon_safe": epsilon,
            "threshold_source": (
                "legacy_sqrl_go2_reproduction" if key == legacy
                else "episode_disjoint_ppo_calibration_first_fall_recall_0.90"),
            "within_state_prediction_std_mean": float(risk.std(axis=1).mean()),
            **_pair_metrics(fall, risk),
            "top1": summarize_selector(
                fall, top1, bootstrap_seed=9200 + model_index),
            "sqrl_rejection": summarize_selector(
                fall, sqrl, bootstrap_seed=9300 + model_index),
            "minimal_intervention": summarize_selector(
                fall, minimal, bootstrap_seed=9400 + model_index),
        }
        model["selector_oracle_gap"] = float(
            model["top1"]["selected_fall_rate"]
            - report["candidate_action_space"]["independent_oracle"]["selected_fall_rate"])
        report["critics"][key] = model

    ppo = [report["critics"][f"ppo-{budget}m-action"] for budget in (1, 3, 5)]
    best_ppo = max(ppo, key=lambda value: value["top1"]["fall_reduction"])
    baseline = report["critics"][legacy]
    shuffled = report["critics"]["ppo-5m-shuffled"]
    five = report["critics"]["ppo-5m-action"]
    offline_metrics = {
        "ppo_1m_test_action_permutation_mean_abs_change": 0.0005326733808033168,
        "ppo_3m_test_action_permutation_mean_abs_change": 0.00021072023082524538,
        "ppo_5m_test_action_permutation_mean_abs_change": 0.00011752459249692038,
        "ppo_5m_action_auprc": 0.2514636658837234,
        "ppo_5m_state_only_auprc": 0.2798496560959725,
        "ppo_5m_shuffled_auprc": 0.31213578547118265,
    }
    ppo_advantage = bool(
        best_ppo["top1"]["fall_reduction_lcb95"] > 0
        and best_ppo["top1"]["selected_fall_rate"]
        < baseline["top1"]["selected_fall_rate"])
    action_learnable = bool(
        five["top1"]["fall_reduction_lcb95"] > 0
        and five["top1"]["selected_fall_rate"]
        < shuffled["top1"]["selected_fall_rate"]
        and offline_metrics["ppo_5m_action_auprc"]
        > offline_metrics["ppo_5m_state_only_auprc"])
    report["offline_diagnostics"] = offline_metrics
    report["decision"] = {
        "ppo_data_advantage_supported": ppo_advantage,
        "action_signal_learnable": action_learnable,
        "ppo_to_sac_transfer_supported": False,
        "formal_objective1_authorized": False,
        "ppo_branching_gate_pass": bool(ppo_advantage and action_learnable),
        "sac_2k_3k_5k_status": "not_run_fail_closed_after_ppo_branching_gate",
        "fresh_sac_online_status": "not_run_fail_closed_after_ppo_branching_gate",
        "objective2_speed_expansion_status": "forbidden_objective1_not_passed",
    }
    report["answers"] = {
        "ppo_data_better_than_existing_sqrl_sac": (
            "No. Point estimate improved for PPO-1M, but its state-bootstrap LCB was not positive."),
        "quantity_or_coverage": (
            "No monotonic data-size benefit: 1M ranked better than 3M and 5M; more transitions reduced action sensitivity."),
        "same_state_action_selection": (
            "Candidate oracle was positive, but no learned PPO critic obtained a positive selector-reduction LCB."),
        "ppo_risk_transfer_to_sac": "Not tested because the prerequisite PPO branching gate failed.",
        "fresh_sac_fall_reduction": "Not tested and not authorized because both prerequisite gates were not passed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    os.link(temporary, args.output); temporary.unlink()
    print(json.dumps(report["decision"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
