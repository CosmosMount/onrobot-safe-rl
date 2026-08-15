"""Evaluate converged target policies and render their MuJoCo trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train.config import load_app_config
from train.mujoco_snapshot_env import MujocoSnapshotEnv

from ..algo.safety_critic import SafetyCriticConfig, SafetyCriticLearner
from ..algo.safety_policy import SafetyPolicy
from ..config import load_config
from .common import build_core


BRANCHES = ("sac_transfer", "sqrl_mask", "sqrl_full")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _annotate(frame: np.ndarray, lines: tuple[str, ...], *, fall: bool) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    # The minimal Pillow build ships a bitmap font whose multiline bbox API
    # rejects custom spacing. Its fixed metrics are enough for audit overlays.
    line_height = 13
    width = max(len(line) for line in lines) * 7 + 14
    height = len(lines) * line_height + 12
    draw.rectangle((7, 5, width, height), fill=(0, 0, 0))
    color = (255, 80, 80) if fall else "white"
    for index, line in enumerate(lines):
        draw.text((12, 9 + index * line_height), line, fill=color)
    return np.asarray(image)


def _load_branch(root: Path, branch: str, cfg: Any, device: str):
    sac, safety, _, _, _ = build_core(cfg, seed=0, device=device)
    payload = torch.load(
        root / f"target_040_{branch}" / "final_sac.pt",
        map_location=sac.device, weights_only=False)
    sac.load_checkpoint(payload)
    sac.actor.eval()
    if branch == "sac_transfer":
        return sac, None
    pretrain = torch.load(
        root / "pretrain_030" / "final.pt",
        map_location=sac.device, weights_only=False)
    safety.load_checkpoint(pretrain["safety"], load_optimizer=False)
    safety.freeze()
    safety.critic.eval()
    return sac, SafetyPolicy(
        sac.actor, safety.critic, cfg.sqrl.epsilon_safe,
        cfg.sqrl.mask_candidates, sac.device)


def _select_action(sac, policy, observation: np.ndarray, env: MujocoSnapshotEnv):
    if policy is None:
        return sac.act(observation, deterministic=False, count=1)[0], None

    class Preview:
        def __init__(self, actions: np.ndarray):
            projections = env.action_applier.preview_many(
                actions, env.data.qpos[env.qpos_addresses])
            self.requested = np.stack([item.action_requested for item in projections])
            self.critic_actions = np.stack([item.action_executed for item in projections])
            self.q_targets = np.stack([item.action_q_target for item in projections])

    result = policy.select(observation, Preview)
    return result.requested_action, result


def _render_video(model, qpos: np.ndarray, qvel: np.ndarray,
                  velocities: np.ndarray, falls: np.ndarray,
                  episodes: np.ndarray, branch: str, output: Path, fps: int) -> None:
    if "DISPLAY" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco

    data = mujoco.MjData(model)
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.7
    camera.azimuth = 135.0
    camera.elevation = -20.0
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
            output, fps=fps, codec="libx264", pixelformat="yuv420p",
            macro_block_size=1, quality=8) as writer:
        for index in range(len(qpos)):
            data.qpos[:] = qpos[index]
            data.qvel[:] = qvel[index]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.xpos[base_id]
            renderer.update_scene(data, camera=camera)
            frame = renderer.render().copy()
            writer.append_data(_annotate(frame, (
                f"SQRL-Go2  {branch}",
                f"evaluation frame {index + 1}/{len(qpos)}  episode {episodes[index]}",
                f"forward velocity {velocities[index]:+.3f} m/s  target +0.400 m/s",
                "FALL -> reset" if falls[index] else "converged checkpoint",
            ), fall=bool(falls[index])))
    renderer.close()


def evaluate_branch(*, root: Path, branch: str, cfg: Any, robot: Any,
                    model_path: Path, episodes: int, episode_steps: int,
                    video_frames: int, eval_seed: int, device: str,
                    output_dir: Path, fps: int) -> dict[str, Any]:
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)
    sac, policy = _load_branch(root, branch, cfg, device)
    env = MujocoSnapshotEnv(
        model_path, robot, policy_frequency=cfg.environment.control_frequency,
        max_joint_delta=None, use_action_filter=False)
    rng = np.random.default_rng(eval_seed)
    velocities: list[float] = []
    episode_means: list[float] = []
    falls = 0
    mask_steps = 0
    mask_accepts = 0
    no_safe = 0
    video_qpos: list[np.ndarray] = []
    video_qvel: list[np.ndarray] = []
    video_velocity: list[float] = []
    video_fall: list[bool] = []
    video_episode: list[int] = []
    for episode in range(episodes):
        env.reset_standing(settle_seconds=1.0, rng=rng)
        observation = env.record_observation().reshape(-1)
        current: list[float] = []
        for _ in range(episode_steps):
            action, mask = _select_action(sac, policy, observation, env)
            result = env.step(action)
            velocity = float(env.robot_state().body_velocity[0])
            fell = bool(result.failure)
            velocities.append(velocity)
            current.append(velocity)
            if mask is not None:
                mask_steps += 1
                mask_accepts += int(mask.accepted)
                no_safe += int(mask.no_safe_candidate)
            if len(video_qpos) < video_frames:
                video_qpos.append(env.data.qpos.copy())
                video_qvel.append(env.data.qvel.copy())
                video_velocity.append(velocity)
                video_fall.append(fell)
                video_episode.append(episode)
            if fell:
                falls += 1
                break
            observation = env.record_observation().reshape(-1)
        episode_means.append(float(np.mean(current)))
    video_path = output_dir / f"sqrl_go2_target_040_{branch}.mp4"
    _render_video(
        env.model, np.asarray(video_qpos), np.asarray(video_qvel),
        np.asarray(video_velocity), np.asarray(video_fall),
        np.asarray(video_episode), branch, video_path, fps)
    return {
        "branch": branch,
        "evaluation_seed": eval_seed,
        "episodes": episodes,
        "maximum_steps_per_episode": episode_steps,
        "evaluated_policy_steps": len(velocities),
        "falls": falls,
        "mean_forward_velocity_mps": float(np.mean(velocities)),
        "std_forward_velocity_mps": float(np.std(velocities)),
        "mean_episode_forward_velocity_mps": float(np.mean(episode_means)),
        "std_episode_forward_velocity_mps": float(np.std(episode_means)),
        "mask_acceptance_rate": (
            mask_accepts / mask_steps if mask_steps else None),
        "no_safe_candidate_rate": no_safe / mask_steps if mask_steps else None,
        "video": str(video_path.resolve()),
        "video_sha256": _sha256(video_path),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="reproductions/sqrl_go2/config/target_040.yaml")
    parser.add_argument(
        "--checkpoint-root", type=Path,
        default=Path("saved/reproductions/sqrl_go2/seed_0"))
    parser.add_argument(
        "--model", type=Path,
        default=Path("/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument("--video-frames", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args(argv)
    if min(args.episodes, args.episode_steps, args.video_frames, args.fps) <= 0:
        raise ValueError("evaluation and video sizes must be positive")
    # MujocoSnapshotEnv imports MuJoCo during environment construction, before
    # the renderer exists. Select the headless backend before that first import.
    if "DISPLAY" not in os.environ:
        os.environ.setdefault("MUJOCO_GL", "egl")
    cfg = load_config(args.config)
    robot, _, _ = load_app_config(args.config)
    results = [evaluate_branch(
        root=args.checkpoint_root, branch=branch, cfg=cfg, robot=robot,
        model_path=args.model, episodes=args.episodes,
        episode_steps=args.episode_steps, video_frames=args.video_frames,
        eval_seed=args.eval_seed, device=args.device,
        output_dir=args.output_dir, fps=args.fps)
        for branch in BRANCHES]
    report = {
        "schema_version": "sqrl_go2.converged_video_evaluation.v1",
        "task_speed_mps": cfg.move_speed,
        "policy_semantics": "fixed-seed stochastic actor; SQRL uses 100-candidate rejection sampling",
        "velocity_definition": "signed post-action body-frame forward velocity over policy steps",
        "backend": "in-process MuJoCo scene_empty at 50 Hz with project Go2 PD/action semantics",
        "model": str(args.model.resolve()),
        "model_sha256": _sha256(args.model),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "sqrl_go2_target_040_video_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
