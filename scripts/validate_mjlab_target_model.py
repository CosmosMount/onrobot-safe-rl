#!/usr/bin/env python3
"""Validate compiled target-aligned MjLab mechanics against native SAC XML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_data.mjlab_target_alignment import (
    configure_target_aligned_go2,
    target_alignment_manifest,
    validate_target_aligned_go2,
)
from safety_data.mjlab_target_model_contract import validate_compiled_target_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if not args.target_model.is_file():
        raise FileNotFoundError(args.target_model)

    import mujoco
    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = configure_target_aligned_go2(load_env_cfg("Unitree-Go2-Flat"))
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    validate_target_aligned_go2(cfg)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        target = mujoco.MjModel.from_xml_path(str(args.target_model))
        action_term = env.action_manager.get_term("joint_pos")
        report = validate_compiled_target_model(
            env.sim.mj_model,
            target,
            action_joint_names=action_term.target_names,
        )
        report.update({
            "target_model_path": str(args.target_model.resolve()),
            "target_model_sha256": _sha256(args.target_model),
            "target_alignment": target_alignment_manifest(),
            "external_force_nonzero": bool(
                env.sim.data.xfrc_applied.ne(0).any().item()),
            "versions": {
                "mujoco": mujoco.__version__,
                "warp": __import__("warp").__version__,
                "mujoco_warp": __import__("mujoco_warp").__version__,
            },
        })
        report["pass"] = bool(report["pass"] and not report[
            "external_force_nonzero"])
    finally:
        env.close()

    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    print(rendered, end="")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(rendered.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, args.output)
    except FileExistsError as exc:
        raise FileExistsError("model-contract output path was already consumed") from exc
    temporary.unlink()
    directory = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if not report["pass"]:
        raise RuntimeError("target-aligned MjLab/native model contract failed")


if __name__ == "__main__":
    main()
