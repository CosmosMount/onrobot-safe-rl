#!/usr/bin/env python3
"""Roll out a frozen PPO boundary policy and capture fresh 97-step fall windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.mjlab_natural_falls import MjlabNaturalFallCapture
from safety_data.mjlab_target_alignment import (
    configure_target_aligned_go2, validate_target_aligned_go2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ppo-seed", type=int, choices=(137, 138), required=True)
    parser.add_argument("--rollout-seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=2000)
    parser.add_argument("--aggregate-transitions", type=int, required=True)
    parser.add_argument("--normal-events", type=int, default=400)
    args = parser.parse_args()
    rollout_seed = args.ppo_seed if args.rollout_seed is None else args.rollout_seed
    if args.aggregate_transitions % args.envs:
        raise ValueError("aggregate transitions must be divisible by environments")
    if args.output.exists() or not args.checkpoint.is_file():
        raise FileExistsError("output exists or checkpoint is missing")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    cfg = configure_target_aligned_go2(load_env_cfg("Unitree-Go2-Flat"))
    agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
    cfg.seed = rollout_seed
    cfg.scene.num_envs = args.envs
    validate_target_aligned_go2(cfg)
    agent_cfg.seed = rollout_seed
    agent_cfg.upload_model = False
    args.output.mkdir(parents=True, exist_ok=False)
    capture = MjlabNaturalFallCapture(
        args.envs, args.output / "natural-falls", seed=rollout_seed,
        export_ring_steps=97, max_normal_events=args.normal_events,
        preview_policy_steps=1,
    )

    class CapturingEnvironment(ManagerBasedRlEnv):
        def step(self, action: torch.Tensor):
            capture.before_step(self, action)
            return super().step(action)

        def _reset_idx(self, env_ids: torch.Tensor) -> None:
            capture.before_reset(self, env_ids)
            super()._reset_idx(env_ids)
            capture.after_reset(env_ids)

    environment = CapturingEnvironment(cfg=cfg, device="cuda:0")
    if "push_robot" in environment.cfg.events or bool(torch.any(
            environment.sim.data.xfrc_applied != 0.0).item()):
        raise RuntimeError("counterfactual state collector contains external force")
    mujoco.mj_saveModel(environment.sim.mj_model, str(args.output / "model.mjb"))
    wrapped = RslRlVecEnvWrapper(environment, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), device="cuda:0")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    observation = wrapped.get_observations()
    capture.arm(environment)
    with torch.inference_mode():
        for _ in range(args.aggregate_transitions // args.envs):
            action = policy(observation, stochastic_output=True)
            observation, _, _, _ = wrapped.step(action)
    manifest = capture.close({
        "seed": args.ppo_seed,
        "collector_seed": args.ppo_seed,
        "rollout_seed": rollout_seed,
        "environments": args.envs,
        "fixed_frozen_policy_exposure": args.aggregate_transitions,
        "checkpoint": str(args.checkpoint.resolve()),
        "command_vx_mps": 0.30,
        "push_event": False,
        "export_ring_steps": 97,
    })
    report = {
        "manifest": str(manifest.resolve()),
        "falls": capture.fall_count,
        "normals": capture.normal_count,
        "aggregate_transitions": args.aggregate_transitions,
    }
    (args.output / "collection-result.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    environment.close()


if __name__ == "__main__":
    main()
