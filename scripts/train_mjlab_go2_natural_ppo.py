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
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_data.mjlab_natural_falls import MjlabNaturalFallCapture
from safety_data.mjlab_target_alignment import (
    configure_target_aligned_go2,
    target_alignment_manifest,
    validate_target_aligned_go2,
)


CHECKPOINT_EXPOSURES = (0, 1_000_000, 2_000_000, 5_000_000,
                        10_000_000, 20_000_000, 30_000_000)


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _git_status(path: Path) -> bytes:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=path, check=True, capture_output=True,
    ).stdout


def require_clean_production_worktree(exposure: int, status: bytes) -> None:
    if exposure == 30_000_000 and status:
        raise RuntimeError("30M production PPO requires a clean generator worktree")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"authorization must be a JSON object: {path}")
    return value


def validate_preflight_authorizations(
    *,
    capacity_path: Path,
    model_contract_path: Path,
    parity_path: Path,
    production_envs: int,
) -> dict:
    capacity = _load_json(capacity_path)
    model = _load_json(model_contract_path)
    parity = _load_json(parity_path)
    if capacity.get("schema_version") != "qsafe.mjlab_capacity_authorization.v1" or (
            capacity.get("authorized") is not True):
        raise ValueError("PPO capacity is not authorized")
    if capacity.get("production_envs") != production_envs or int(
            capacity.get("selected_capacity_envs", 0)) < production_envs:
        raise ValueError("PPO environment count differs from capacity authorization")
    if model.get("schema_version") != "qsafe.mjlab_target_model_contract.v1" or (
            model.get("pass") is not True):
        raise ValueError("target model contract did not pass")
    if parity.get("schema_version") != "qsafe.mjlab_native_parity.v1" or (
            parity.get("pass") is not True):
        raise ValueError("native/Warp parity did not pass")
    if int(parity.get("states", 0)) < 100 or int(
            parity.get("policy_steps_per_state", 0)) < 100 or float(
            parity.get("fall_predicate_agreement", 0.0)) != 1.0:
        raise ValueError("native/Warp parity corpus is too small or disagrees on falls")
    if model.get("external_force_nonzero") is not False or (
            parity.get("external_force_nonzero") is not False):
        raise ValueError("preflight evidence contains an external force")
    expected = target_alignment_manifest()["contract_sha256"]
    contracts = {
        str(capacity.get("target_alignment_contract_sha256")),
        str(model.get("target_alignment", {}).get("contract_sha256")),
        str(parity.get("target_alignment", {}).get("contract_sha256")),
    }
    if contracts != {expected}:
        raise ValueError("preflight target-alignment contracts differ")
    versions = {
        json.dumps(capacity.get("backend_versions"), sort_keys=True),
        json.dumps(model.get("versions"), sort_keys=True),
        json.dumps(parity.get("versions"), sort_keys=True),
    }
    if len(versions) != 1:
        raise ValueError("preflight backend versions differ")
    return {
        "capacity": {"path": str(capacity_path), "sha256": _sha256(capacity_path)},
        "target_model_contract": {
            "path": str(model_contract_path), "sha256": _sha256(model_contract_path)},
        "native_warp_parity": {"path": str(parity_path), "sha256": _sha256(parity_path)},
        "target_alignment_contract_sha256": expected,
        "backend_versions": capacity["backend_versions"],
    }


def _write_json_no_clobber(path: Path, value: dict) -> None:
    content = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite training manifest: {path}") from exc
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=125)
    parser.add_argument("--exposure", type=int, default=30_000_000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--capacity-authorization", type=Path,
        default=Path("saved/qsafe_development/natural_ppo/capacity-030-target-aligned/capacity-authorization-v1.json"))
    parser.add_argument(
        "--model-contract", type=Path,
        default=Path("saved/qsafe_development/natural_ppo/parity/mjlab-target-model-contract-v1.json"))
    parser.add_argument(
        "--parity-report", type=Path,
        default=Path("saved/qsafe_development/natural_ppo/parity/mjlab-native-target-aligned-validation-seed137-v1.json"))
    args = parser.parse_args()
    steps_per_iteration = args.envs * args.rollout_steps
    if args.exposure <= 0 or args.exposure % steps_per_iteration:
        raise ValueError("exposure must be exactly divisible by envs*rollout-steps")
    require_clean_production_worktree(
        args.exposure, _git_status(REPOSITORY_ROOT))
    launch_commit = _git_head(REPOSITORY_ROOT)
    launch_worktree_clean = not bool(_git_status(REPOSITORY_ROOT))
    preflight = validate_preflight_authorizations(
        capacity_path=args.capacity_authorization,
        model_contract_path=args.model_contract,
        parity_path=args.parity_report,
        production_envs=args.envs,
    )

    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    import src  # type: ignore
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.utils.os import dump_yaml

    cfg = configure_target_aligned_go2(load_env_cfg("Unitree-Go2-Flat"))
    agent_cfg = load_rl_cfg("Unitree-Go2-Flat")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.envs
    validate_target_aligned_go2(cfg)
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
        args.envs, args.output / "natural-falls", seed=args.seed,
        rollout_steps=args.rollout_steps)

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
    compiled_model = args.output / "target-aligned-model.mjb"
    mujoco.mj_saveModel(environment.sim.mj_model, str(compiled_model))
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
        "command_vx_mps": 0.30,
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
    upstream = Path(src.__file__).resolve().parents[1]
    if args.exposure == 30_000_000 and (
            _git_head(repository) != launch_commit or _git_status(repository)):
        raise RuntimeError("production generator changed during PPO collection")
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
            "type": "constant", "vx": 0.30,
            "vy": 0.0, "yaw_rate": 0.0,
        },
        "target_alignment": target_alignment_manifest(),
        "preflight_authorizations": preflight,
        "compiled_model": {
            "path": compiled_model.name,
            "sha256": _sha256(compiled_model),
        },
        "resolved_configs": {
            "environment": {"path": "env.yaml", "sha256": _sha256(args.output / "env.yaml")},
            "agent": {"path": "agent.yaml", "sha256": _sha256(args.output / "agent.yaml")},
        },
        "checkpoint_selection_uses_outcomes": False,
        "checkpoints": entries,
        "generator_commit": launch_commit,
        "generator_worktree_clean_at_launch": launch_worktree_clean,
        "unitree_rl_mjlab_commit": _git_head(upstream),
    }
    _write_json_no_clobber(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True, indent=2))
    environment.close()


if __name__ == "__main__":
    main()
