#!/usr/bin/env python3
"""Render audit videos from an immutable natural-PPO archive.

The renderer only replays qpos exported by the collector.  It never advances
physics, runs recovery, or invents transitions between archived states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value: np.generic) -> str:
    return bytes(value).decode("ascii")


def _find_full_fall(archive: Path) -> tuple[Path, int, int]:
    """Choose the first complete 64-step pre-fall trajectory deterministically."""
    for path in sorted(archive.glob("falls-*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            rows = np.flatnonzero(arrays["trajectory_length"] == 65)
            if rows.size:
                return path, int(rows[0]), 65
    raise ValueError("archive contains no complete 64-step pre-fall trajectory")


def _writer(path: Path, fps: int):
    import imageio.v2 as imageio

    if path.exists():
        raise FileExistsError(f"refusing to overwrite video: {path}")
    return imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p",
        macro_block_size=1, quality=8,
    )


def _annotate(frame: np.ndarray, lines: Iterable[str], color: str = "white") -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = "\n".join(lines)
    box = draw.multiline_textbbox((8, 7), text, spacing=3)
    draw.rectangle((4, 3, box[2] + 4, box[3] + 5), fill=(0, 0, 0))
    draw.multiline_text((8, 7), text, fill=color, spacing=3)
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    run = args.run.resolve()
    archive = run / "natural-falls"
    model_path = run / "target-aligned-model.mjb"
    parallel_path = archive / "parallel-preview.npz"
    normal_path = archive / "normal-preview.npz"
    manifest_path = archive / "manifest.json"
    run_manifest_path = run / "manifest.json"
    for path in (model_path, parallel_path, normal_path, manifest_path,
                 run_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    seed = int(manifest["provenance"]["seed"])

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "parallel": output_dir / f"parallel-ppo-training-seed{seed}-v1.mp4",
        "normal": output_dir / f"normal-sequence-seed{seed}-v1.mp4",
        "fall": output_dir / f"fall-sequence-seed{seed}-v1.mp4",
        "report": output_dir / f"video-audit-seed{seed}-v1.json",
        "contact_sheet": output_dir / f"video-audit-contact-sheet-seed{seed}-v1.png",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("one or more audit outputs already exist")

    # MuJoCo chooses the GL backend when imported.
    if "DISPLAY" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_binary_path(str(model_path))
    if (model.nq, model.nv) != (19, 18):
        raise ValueError("compiled model is not the exported single-Go2 model")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/base_link")
    renderer = mujoco.Renderer(model, height=240, width=320)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.7
    camera.azimuth = 135.0
    camera.elevation = -20.0

    def render(qpos: np.ndarray, qvel: np.ndarray | None = None) -> np.ndarray:
        data.qpos[:] = qpos
        if qvel is not None:
            data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
        camera.lookat[:] = data.xpos[base_id]
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    contact_frames: list[np.ndarray] = []
    with np.load(parallel_path, allow_pickle=False) as arrays:
        steps, environments = arrays["qpos"].shape[:2]
        if environments != 4:
            raise ValueError("parallel preview must bind exactly four environments")
        with _writer(outputs["parallel"], args.fps) as video:
            for step in range(steps):
                cells = []
                for env in range(environments):
                    fell = bool(arrays["fall_after_action"][step, env])
                    qpos = (arrays["terminal_qpos"][step, env] if fell
                            else arrays["qpos"][step, env])
                    qvel = None if fell else arrays["qvel"][step, env]
                    frame = render(qpos, qvel)
                    frame = _annotate(frame, (
                        f"env {env}  episode {int(arrays['episode_id'][step, env])}",
                        f"PPO step {step}  episode step {int(arrays['episode_step'][step, env])}",
                        "FALL -> immediate reset" if fell else "training transition",
                    ), color="red" if fell else "white")
                    cells.append(frame)
                tiled = np.concatenate((np.concatenate(cells[:2], axis=1),
                                        np.concatenate(cells[2:], axis=1)), axis=0)
                video.append_data(tiled)
                if step in (0, steps // 2, steps - 1):
                    contact_frames.append(tiled)

    with np.load(normal_path, allow_pickle=False) as arrays:
        if bool(arrays["fall_within_96_steps"]):
            raise ValueError("normal preview is labeled as a future fall")
        normal_env = int(arrays["environment_id"])
        normal_episode = int(arrays["episode_id"])
        normal_frames = int(arrays["qpos"].shape[0])
        with _writer(outputs["normal"], args.fps) as video:
            for step in range(normal_frames):
                frame = render(arrays["qpos"][step], arrays["qvel"][step])
                frame = _annotate(frame, (
                    f"NORMAL  env {normal_env}  episode {normal_episode}",
                    f"archived frame {step + 1}/{normal_frames}",
                    "survived next 96 policy steps",
                ))
                video.append_data(frame)
                if step == normal_frames // 2:
                    contact_frames.append(frame)

    fall_path, fall_row, fall_length = _find_full_fall(archive)
    with np.load(fall_path, allow_pickle=False) as arrays:
        fall_identity = _decode(arrays["identity"][fall_row])
        fall_env = int(arrays["environment_id"][fall_row])
        fall_episode = int(arrays["episode_id"][fall_row])
        qpos = np.concatenate((arrays["trajectory_qpos"][fall_row, :fall_length],
                               arrays["terminal_qpos"][fall_row][None]), axis=0)
        qvel = np.concatenate((arrays["trajectory_qvel"][fall_row, :fall_length],
                               arrays["terminal_qvel"][fall_row][None]), axis=0)
        action_count = int(arrays["trajectory_action_requested"][fall_row, :fall_length].shape[0])
        with _writer(outputs["fall"], args.fps) as video:
            for step in range(len(qpos)):
                terminal = step == len(qpos) - 1
                frame = render(qpos[step], qvel[step])
                frame = _annotate(frame, (
                    f"FALL  env {fall_env}  episode {fall_episode}",
                    f"archived frame {step + 1}/{len(qpos)}",
                    "terminal -> reset; no recovery" if terminal else "pre-fall PPO transition",
                ), color="red" if terminal else "white")
                video.append_data(frame)
                if step in (0, len(qpos) // 2, len(qpos) - 1):
                    contact_frames.append(frame)

    renderer.close()
    from PIL import Image

    thumbnails = [
        np.asarray(Image.fromarray(frame).resize((320, 240)))
        for frame in contact_frames
    ]
    thumbnails.extend([np.zeros((240, 320, 3), dtype=np.uint8)] * (9 - len(thumbnails)))
    sheet = np.concatenate([
        np.concatenate(thumbnails[row:row + 3], axis=1)
        for row in range(0, 9, 3)
    ], axis=0)
    Image.fromarray(sheet).save(outputs["contact_sheet"])

    report = {
        "schema_version": "qsafe.natural_ppo_video_audit.v1",
        "claim": "kinematic_replay_of_archived_actual_ppo_states",
        "physics_advanced_during_render": False,
        "recovery_executed": False,
        "fall_handling": "terminal_recorded_once_then_reset_in_same_vector_step",
        "archive_manifest_sha256": _sha256(manifest_path),
        "generator_commit": run_manifest["generator_commit"],
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "parallel": {
            "source": str(parallel_path), "source_sha256": _sha256(parallel_path),
            "environment_count_shown": 4, "policy_frames": steps,
            "video": str(outputs["parallel"]),
        },
        "normal": {
            "source": str(normal_path), "source_sha256": _sha256(normal_path),
            "environment_id": normal_env, "episode_id": normal_episode,
            "policy_frames": normal_frames, "future_nonterminal_steps": 96,
            "video": str(outputs["normal"]),
        },
        "fall": {
            "source": str(fall_path), "source_sha256": _sha256(fall_path),
            "row": fall_row, "identity": fall_identity,
            "environment_id": fall_env, "episode_id": fall_episode,
            "preterminal_policy_frames": fall_length,
            "executed_action_records": action_count,
            "terminal_frames": 1, "video": str(outputs["fall"]),
        },
        "fps": args.fps,
    }
    for name in ("parallel", "normal", "fall"):
        report[name]["output_sha256"] = _sha256(outputs[name])
    report["contact_sheet"] = {
        "path": str(outputs["contact_sheet"]),
        "output_sha256": _sha256(outputs["contact_sheet"]),
    }
    with outputs["report"].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    for name in ("parallel", "normal", "fall", "report", "contact_sheet"):
        print(outputs[name])


if __name__ == "__main__":
    main()
