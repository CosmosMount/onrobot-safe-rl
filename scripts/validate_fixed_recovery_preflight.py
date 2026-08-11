#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from safety_data.fixed_recovery_motion import FixedRecoveryConfig
from safety_data.fixed_recovery_preflight import evaluate_standing_fixed_recovery
from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_no_clobber(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite preflight report: {path}") from exc
    temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path,
        default=Path("/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml"))
    parser.add_argument(
        "--robot-config", type=Path,
        default=Path("config/go2_50hz_safe_adaptive_gated_v3.yaml"))
    parser.add_argument(
        "--controller-config", type=Path,
        default=Path("runtime/control/go2/go2.yaml"))
    parser.add_argument(
        "--controller-source", type=Path,
        default=Path("runtime/control/go2/motions/src/recovery.cpp"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()

    # MuJoCo selects its GL backend at import time.  The native environment
    # imports MuJoCo during construction, so choose EGL before that happens.
    if args.video is not None and "DISPLAY" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "egl")

    robot, train, _ = load_app_config(args.robot_config, agent="safe_droq")
    env = MujocoSnapshotEnv(
        args.model, robot, policy_frequency=train.control_frequency,
        max_joint_delta=train.max_joint_delta)
    env.reset_standing(settle_seconds=1.0)
    recovery = FixedRecoveryConfig.from_controller_yaml(args.controller_config)
    frames: list[np.ndarray] = []
    renderer = None
    camera = None
    if args.video is not None:
        import mujoco

        renderer = mujoco.Renderer(env.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = env.data.xpos[env.base_body_id]
        camera.distance = 1.7
        camera.azimuth = 135.0
        camera.elevation = -20.0

    def capture(tick: int, current_env: MujocoSnapshotEnv, execution: object) -> None:
        del execution
        if renderer is not None and camera is not None and tick % 10 == 0:
            camera.lookat[:] = current_env.data.xpos[current_env.base_body_id]
            renderer.update_scene(current_env.data, camera=camera)
            frames.append(renderer.render().copy())

    result = evaluate_standing_fixed_recovery(
        env, recovery, frame_callback=capture if renderer is not None else None)
    if renderer is not None:
        renderer.close()
    if args.video is not None:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        if args.video.exists():
            raise FileExistsError(f"refusing to overwrite video: {args.video}")
        imageio.mimsave(args.video, frames, fps=50, macro_block_size=1)

    report = {
        "schema_version": "qsafe.original_fixed_recovery_preflight.v1",
        "claim": "prevention_negative_control_only",
        "objective1_authorized": False,
        "fall_predicate_active_during_recovery": True,
        "fall_exemption_used": False,
        "model_path": str(args.model.resolve()),
        "model_root_xml_sha256": _sha256(args.model),
        "simulator": env.simulator_fingerprint(),
        "recovery": recovery.manifest(
            control_hz=500.0,
            controller_yaml=args.controller_config,
            controller_source=args.controller_source,
        ),
        "result": result.manifest(),
        "video": None if args.video is None else str(args.video.resolve()),
    }
    rendered = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _write_no_clobber(args.output, rendered)
    print(rendered.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
