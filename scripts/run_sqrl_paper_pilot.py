#!/usr/bin/env python3
"""Continue the seed-42 paper-aligned SQRL pilot through paired fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.inference.transport import SharedMemoryRingQueue

PRETRAIN_ROOT = ROOT / "saved/experiments/sqrl_paper/seed42/pretrain_strict_async_sac_v1"
SAC_PRETRAIN_ROOT = ROOT / "saved/experiments/sqrl_paper/seed42/pretrain_sac_async_v1"
SAC_ROOT = ROOT / "saved/experiments/sqrl_paper/seed42/finetune_sac_async_sac_v1"
SQRL_ROOT = ROOT / "saved/experiments/sqrl_paper/seed42/finetune_sqrl_async_sac_v1"
RESULT_PATH = ROOT / "saved/experiments/sqrl_paper/seed42/result.json"
LOG_ROOT = ROOT / "saved/experiments/sqrl_paper/seed42/orchestrator_logs"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def wait_for_run(root: Path, poll_seconds: float) -> Path:
    manifest_path = root / "manifest.json"
    while True:
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            status = manifest.get("status")
            if status == "finished":
                completed = int(manifest.get("completed_steps", 0))
                expected = int(manifest.get("max_steps", 500_000))
                if completed != expected:
                    raise RuntimeError(
                        f"pretrain finished at {completed}, expected {expected}")
                checkpoint = root / f"step_{completed:012d}"
                if not (checkpoint / "agent/actor.pt").exists():
                    raise RuntimeError(f"incomplete checkpoint: {checkpoint}")
                return checkpoint
            if status in {"stopped", "failed"}:
                raise RuntimeError(f"pretrain ended with status={status}")
        time.sleep(poll_seconds)


def run(config: str, checkpoint: Path | None = None, *,
        resume: bool = False) -> None:
    command = [
        sys.executable, "-m", "train",
        "--config", config,
        "--seed", "42",
        "--no-wandb",
    ]
    if checkpoint is not None:
        command.extend(["--initialize-from", str(checkpoint)])
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)


def restart_runtime(config: str, settle_seconds: float = 5.0) -> None:
    """Restart the fixed-rate runtime with the stage's speed/horizon config."""
    own_pid = os.getpid()
    victims: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == own_pid:
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"runtime.inference.runtime" in command:
            victims.append(int(proc.name))
    for pid in victims:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10.0
    while any(Path(f"/proc/{pid}").exists() for pid in victims):
        if time.monotonic() >= deadline:
            raise RuntimeError(f"runtime processes did not stop: {victims}")
        time.sleep(0.2)
    SharedMemoryRingQueue.unlink_existing("go2_runtime_state.ordered")
    subprocess.Popen(
        [sys.executable, "-m", "runtime.inference.runtime",
         "--config", config, "--ordered-state-queue"],
        cwd=ROOT,
        start_new_session=True,
    )
    time.sleep(settle_seconds)


def ensure_controller(settle_seconds: float = 5.0) -> None:
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"runtime/control/go2/build/go2_control" in command:
            return
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log = (LOG_ROOT / "controller.log").open("ab", buffering=0)
    subprocess.Popen(
        [str(ROOT / "runtime/control/go2/build/go2_control"),
         str(ROOT / "runtime/control/go2/go2.yaml")],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
    time.sleep(settle_seconds)


def require_finished(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    if manifest.get("status") != "finished":
        raise RuntimeError(f"run did not finish: {root}")
    if int(manifest["completed_steps"]) != int(manifest["max_steps"]):
        raise RuntimeError(f"run has incomplete policy steps: {root}")
    return manifest


def write_result(sqrl_pretrain_checkpoint: Path,
                 sac_pretrain_checkpoint: Path) -> None:
    sac = require_finished(SAC_ROOT)
    sqrl = require_finished(SQRL_ROOT)
    sac_pretrain = require_finished(SAC_PRETRAIN_ROOT)
    sqrl_pretrain = require_finished(PRETRAIN_ROOT)
    lineage = {
        "sac_actor": (
            sac["initial_actor_hash"] == sac_pretrain["final_actor_hash"]),
        "sac_reward_critic": (
            sac["initial_reward_critic_hash"]
            == sac_pretrain["final_reward_critic_hash"]),
        "sqrl_actor": (
            sqrl["initial_actor_hash"] == sqrl_pretrain["final_actor_hash"]),
        "sqrl_reward_critic": (
            sqrl["initial_reward_critic_hash"]
            == sqrl_pretrain["final_reward_critic_hash"]),
        "sqrl_safety_critic": (
            sqrl["initial_safety_critic_hash"]
            == sqrl_pretrain["final_safety_critic_hash"]),
    }
    if not all(lineage.values()):
        raise RuntimeError(f"target checkpoint lineage mismatch: {lineage}")
    result = {
        "protocol": "SQRL paper-aligned Go2 0.30->0.40 m/s pilot",
        "seed": 42,
        "sqrl_pretrain_checkpoint": str(sqrl_pretrain_checkpoint),
        "sac_pretrain_checkpoint": str(sac_pretrain_checkpoint),
        "note": ("Primary paper reproduction uses independently trained SAC "
                 "and SQRL pre-training policies, as in Section 7.1."),
        "checkpoint_lineage_verified": lineage,
        "sac": {
            key: sac.get(key) for key in (
                "policy_steps", "falls", "falls_per_1000_policy_steps",
                "episode_fall_rate", "episodes", "final_actor_hash")
        },
        "sqrl": {
            key: sqrl.get(key) for key in (
                "policy_steps", "falls", "falls_per_1000_policy_steps",
                "episode_fall_rate", "episodes", "policy_safety_active_steps",
                "policy_replacements", "policy_replacement_rate",
                "policy_no_safe", "policy_no_safe_rate", "final_actor_hash")
        },
        "fall_difference_sqrl_minus_sac": int(sqrl["falls"]) - int(sac["falls"]),
        "falls_per_1000_difference_sqrl_minus_sac": (
            float(sqrl["falls_per_1000_policy_steps"])
            - float(sac["falls_per_1000_policy_steps"])),
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    ensure_controller()
    sqrl_manifest = (
        read_json(PRETRAIN_ROOT / "manifest.json")
        if (PRETRAIN_ROOT / "manifest.json").exists() else None)
    if sqrl_manifest is None or sqrl_manifest.get("status") != "finished":
        restart_runtime("config/go2_50hz_sqrl_paper_pretrain.yaml")
        run(
            "config/go2_50hz_sqrl_paper_pretrain.yaml",
            resume=sqrl_manifest is not None)
    sqrl_checkpoint = wait_for_run(PRETRAIN_ROOT, args.poll_seconds)
    sqrl_target_manifest = (
        read_json(SQRL_ROOT / "manifest.json")
        if (SQRL_ROOT / "manifest.json").exists() else None)
    if (sqrl_target_manifest is None
            or sqrl_target_manifest.get("status") != "finished"):
        restart_runtime("config/go2_50hz_sqrl_paper_finetune.yaml")
        run(
            "config/go2_50hz_sqrl_paper_finetune.yaml",
            None if sqrl_target_manifest is not None else sqrl_checkpoint,
            resume=sqrl_target_manifest is not None)
    sac_manifest = (
        read_json(SAC_PRETRAIN_ROOT / "manifest.json")
        if (SAC_PRETRAIN_ROOT / "manifest.json").exists() else None)
    if sac_manifest is None or sac_manifest.get("status") != "finished":
        restart_runtime("config/go2_50hz_sqrl_paper_sac_pretrain.yaml")
        run(
            "config/go2_50hz_sqrl_paper_sac_pretrain.yaml",
            resume=sac_manifest is not None)
    sac_checkpoint = wait_for_run(SAC_PRETRAIN_ROOT, args.poll_seconds)
    sac_target_manifest = (
        read_json(SAC_ROOT / "manifest.json")
        if (SAC_ROOT / "manifest.json").exists() else None)
    if (sac_target_manifest is None
            or sac_target_manifest.get("status") != "finished"):
        restart_runtime("config/go2_50hz_sqrl_paper_sac_finetune.yaml")
        run(
            "config/go2_50hz_sqrl_paper_sac_finetune.yaml",
            None if sac_target_manifest is not None else sac_checkpoint,
            resume=sac_target_manifest is not None)
    write_result(sqrl_checkpoint, sac_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
