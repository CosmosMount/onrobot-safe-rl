#!/usr/bin/env python3
"""Train the official MjLab/RSL-RL Go2 PPO under the natural-fall protocol.

The production geometry uses 2,000 environments and 125 rollout steps, making
each completed PPO iteration exactly 250,000 policy-environment steps.  Thus
the registered 1M/2M/5M/10M/20M/30M checkpoints occur at exact iteration
boundaries.  This runner never enables the upstream push event.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_data.mjlab_natural_falls import MjlabNaturalFallCapture


CHECKPOINT_EXPOSURES = (0, 1_000_000, 2_000_000, 5_000_000,
                        10_000_000, 20_000_000, 30_000_000)


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=125)
    parser.add_argument("--exposure", type=int, default=30_000_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps_per_iteration = args.envs * args.rollout_steps
    if args.exposure <= 0 or args.exposure % steps_per_iteration:
        raise ValueError("exposure must be exactly divisible by envs*rollout-steps")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.os import dump_yaml

    cfg = load_env_cfg("Unitree-Go2-Flat")
    agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.envs
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.ranges.lin_vel_x = (0.40, 0.40)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    agent_cfg.seed = args.seed
    agent_cfg.num_steps_per_env = args.rollout_steps
    agent_cfg.max_iterations = args.exposure // steps_per_iteration
    # Every iteration is persisted.  The manifest below exposes only the
    # preregistered exposure boundaries and never selects by outcome.
    agent_cfg.save_interval = 1
    agent_cfg.upload_model = False
    agent_cfg.logger = "tensorboard"

    args.output.mkdir(parents=True, exist_ok=False)
    dump_yaml(args.output / "env.yaml", asdict(cfg))
    dump_yaml(args.output / "agent.yaml", asdict(agent_cfg))
    capture = MjlabNaturalFallCapture(
        args.envs, args.output / "natural-falls", seed=args.seed)

    class CapturingEnvironment(ManagerBasedRlEnv):
        def step(self, action: torch.Tensor):
            capture.before_step(self, action)
            return super().step(action)

        def _reset_idx(self, env_ids: torch.Tensor) -> None:
            capture.before_reset(self, env_ids)
            super()._reset_idx(env_ids)
            capture.after_reset(env_ids)

    environment = CapturingEnvironment(cfg=cfg, device="cuda:0")
    if "push_robot" in environment.cfg.events:
        raise RuntimeError("natural PPO runner unexpectedly contains push_robot")
    wrapped = RslRlVecEnvWrapper(environment, clip_actions=agent_cfg.clip_actions)
    capture.arm(environment)
    runner = MjlabOnPolicyRunner(
        wrapped, asdict(agent_cfg), str(args.output), device="cuda:0")
    initial = args.output / "model_initial.pt"
    runner.save(str(initial), infos={"policy_env_steps": 0})
    started = time.perf_counter()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)
    elapsed = time.perf_counter() - started
    fall_manifest = capture.close({
        "seed": args.seed,
        "environments": args.envs,
        "fixed_exposure": args.exposure,
        "command_vx_mps": 0.40,
        "push_event": False,
    })

    entries = []
    for exposure in CHECKPOINT_EXPOSURES:
        if exposure > args.exposure:
            continue
        if exposure == 0:
            path = initial
            iteration = -1
        else:
            completed_iterations = exposure // steps_per_iteration
            iteration = completed_iterations - 1
            path = args.output / f"model_{iteration}.pt"
        if not path.is_file():
            raise RuntimeError(f"missing exact-exposure checkpoint {path}")
        entries.append({
            "policy_env_steps": exposure,
            "completed_iterations": 0 if exposure == 0 else iteration + 1,
            "path": path.name,
            "sha256": _sha256(path),
        })

    repository = REPOSITORY_ROOT
    upstream = Path.cwd()
    manifest = {
        "schema_version": "qsafe.natural_ppo_training.v1",
        "run_scope": (
            "fixed_30m_production" if args.exposure == 30_000_000
            else "development_pilot_not_claim_eligible"),
        "algorithm": "rsl_rl_clipped_ppo",
        "training_from_zero": True,
        "seed": args.seed,
        "environments": args.envs,
        "rollout_steps": args.rollout_steps,
        "steps_per_iteration": steps_per_iteration,
        "fixed_exposure": args.exposure,
        "elapsed_seconds": elapsed,
        "external_push_event": False,
        "natural_fall_archive": {
            "manifest": str(fall_manifest.relative_to(args.output)),
            "manifest_sha256": _sha256(fall_manifest),
            "recorded_falls": capture.fall_count,
        },
        "command_distribution": {
            "type": "constant", "vx": 0.40,
            "vy": 0.0, "yaw_rate": 0.0,
        },
        "checkpoint_selection_uses_outcomes": False,
        "checkpoints": entries,
        "generator_commit": _git_head(repository),
        "unitree_rl_mjlab_commit": _git_head(upstream),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True, indent=2))
    environment.close()


if __name__ == "__main__":
    main()
