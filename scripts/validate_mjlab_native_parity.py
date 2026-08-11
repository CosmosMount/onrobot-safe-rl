#!/usr/bin/env python3
"""Compare MuJoCo-Warp worlds with native MuJoCo on the identical MjModel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fall(qpos: np.ndarray) -> np.ndarray:
    quat = qpos[:, 3:7]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quat.T
    roll = np.arctan2(2.0 * (w * x + y * z),
                      1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return (qpos[:, 2] < 0.18) | (np.maximum(np.abs(roll), np.abs(pitch)) >= 1.047198)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=100)
    parser.add_argument("--policy-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-qpos-error", type=float, default=5e-3)
    parser.add_argument("--p99-qpos-error", type=float, default=1e-3)
    parser.add_argument("--max-qvel-error", type=float, default=1e-1)
    parser.add_argument("--p99-qvel-error", type=float, default=1e-2)
    parser.add_argument("--min-contact-presence-agreement", type=float, default=0.95)
    parser.add_argument("--min-contact-pair-jaccard", type=float, default=0.90)
    args = parser.parse_args()
    if args.states <= 0 or args.policy_steps <= 0:
        raise ValueError("states and policy steps must be positive")

    import mujoco
    import mjlab.tasks  # noqa: F401
    import src.tasks  # type: ignore  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Unitree-Go2-Flat")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.states
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.ranges.lin_vel_x = (0.30, 0.30)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    if bool(torch.any(env.sim.data.xfrc_applied != 0.0).item()):
        raise RuntimeError("parity corpus contains external force")

    with tempfile.TemporaryDirectory() as temporary:
        model_path = Path(temporary) / "model.mjb"
        mujoco.mj_saveModel(env.sim.mj_model, str(model_path))
        model_binary_sha256 = _sha256(model_path)
        native_models = []
        native_data = []
        expanded = sorted(env.sim.expanded_fields)
        expanded_values = {
            name: getattr(env.sim.model, name).detach().cpu().numpy()
            for name in expanded
        }
        initial_qpos = env.sim.data.qpos.detach().cpu().numpy()
        initial_qvel = env.sim.data.qvel.detach().cpu().numpy()
        initial_ctrl = env.sim.data.ctrl.detach().cpu().numpy()
        initial_time = env.sim.data.time.detach().cpu().numpy()
        for world in range(args.states):
            model = mujoco.MjModel.from_binary_path(str(model_path))
            for name, values in expanded_values.items():
                target = getattr(model, name)
                value = values[world]
                if target.shape != value.shape:
                    raise RuntimeError(
                        f"expanded model field {name} shape differs: "
                        f"{target.shape} versus {value.shape}")
                np.copyto(target, value)
            data = mujoco.MjData(model)
            data.qpos[:] = initial_qpos[world]
            data.qvel[:] = initial_qvel[world]
            data.ctrl[:] = initial_ctrl[world]
            data.time = float(initial_time[world])
            mujoco.mj_forward(model, data)
            native_models.append(model)
            native_data.append(data)

        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(args.seed + 991)
        qpos_errors = []
        qvel_errors = []
        contact_equal = 0
        contact_presence_equal = 0
        contact_pair_jaccard = []
        fall_equal = 0
        comparisons = 0
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.policy_steps):
                action = 0.4 * (2.0 * torch.rand(
                    (args.states, 12), generator=generator, device="cuda:0") - 1.0)
                env.action_manager.process_action(action)
                for _ in range(cfg.decimation):
                    env.action_manager.apply_action()
                    env.scene.write_data_to_sim()
                    controls = env.sim.data.ctrl.detach().cpu().numpy()
                    for world, (model, data) in enumerate(zip(
                            native_models, native_data, strict=True)):
                        data.ctrl[:] = controls[world]
                        mujoco.mj_step(model, data)
                    env.sim.step()
                    env.scene.update(dt=env.physics_dt)
                env.sim.forward()
                gpu_qpos = env.sim.data.qpos.detach().cpu().numpy()
                gpu_qvel = env.sim.data.qvel.detach().cpu().numpy()
                native_qpos = np.stack([data.qpos for data in native_data])
                native_qvel = np.stack([data.qvel for data in native_data])
                qpos_errors.append(np.max(np.abs(gpu_qpos - native_qpos), axis=1))
                qvel_errors.append(np.max(np.abs(gpu_qvel - native_qvel), axis=1))
                active_contacts = int(env.sim.wp_data.nacon.numpy()[0])
                contact_world = env.sim.wp_data.contact.worldid.numpy()[
                    :active_contacts]
                gpu_ncon = np.bincount(
                    contact_world, minlength=args.states)[:args.states]
                gpu_geom = env.sim.wp_data.contact.geom.numpy()[:active_contacts]
                native_ncon = np.asarray([data.ncon for data in native_data])
                contact_equal += int(np.sum(gpu_ncon == native_ncon))
                contact_presence_equal += int(np.sum(
                    (gpu_ncon > 0) == (native_ncon > 0)))
                for world, data in enumerate(native_data):
                    gpu_pairs = {
                        tuple(sorted(map(int, pair)))
                        for pair in gpu_geom[contact_world == world]
                    }
                    native_pairs = {
                        tuple(sorted(map(int, pair)))
                        for pair in data.contact.geom[:data.ncon]
                    }
                    union = gpu_pairs | native_pairs
                    contact_pair_jaccard.append(
                        1.0 if not union else len(gpu_pairs & native_pairs) / len(union))
                fall_equal += int(np.sum(_fall(gpu_qpos) == _fall(native_qpos)))
                comparisons += args.states
        elapsed = time.perf_counter() - started
        qpos_error = np.concatenate(qpos_errors)
        qvel_error = np.concatenate(qvel_errors)
        report = {
            "schema_version": "qsafe.mjlab_native_parity.v1",
            "states": args.states,
            "policy_steps_per_state": args.policy_steps,
            "physics_substeps_per_policy_step": cfg.decimation,
            "comparisons": comparisons,
            "elapsed_seconds": elapsed,
            "expanded_model_fields": expanded,
            "model_binary_sha256": model_binary_sha256,
            "versions": {
                "mujoco": mujoco.__version__,
                "warp": __import__("warp").__version__,
                "mujoco_warp": __import__("mujoco_warp").__version__,
            },
            "max_abs_qpos_error": float(qpos_error.max()),
            "p99_abs_qpos_error": float(np.quantile(qpos_error, 0.99)),
            "max_abs_qvel_error": float(qvel_error.max()),
            "p99_abs_qvel_error": float(np.quantile(qvel_error, 0.99)),
            "contact_count_exact_agreement": contact_equal / comparisons,
            "contact_presence_agreement": contact_presence_equal / comparisons,
            "mean_contact_pair_jaccard": float(np.mean(contact_pair_jaccard)),
            "minimum_contact_pair_jaccard": float(np.min(contact_pair_jaccard)),
            "fall_predicate_agreement": fall_equal / comparisons,
            "nonfinite": not bool(np.all(np.isfinite(qpos_error))
                                  and np.all(np.isfinite(qvel_error))),
            "external_force_nonzero": False,
            "command_vx_mps": 0.3,
            "thresholds": {
                "max_qpos_error": args.max_qpos_error,
                "p99_qpos_error": args.p99_qpos_error,
                "max_qvel_error": args.max_qvel_error,
                "p99_qvel_error": args.p99_qvel_error,
                "min_contact_presence_agreement": (
                    args.min_contact_presence_agreement),
                "min_mean_contact_pair_jaccard": args.min_contact_pair_jaccard,
                "required_fall_predicate_agreement": 1.0,
            },
        }
        report["pass"] = bool(
            not report["nonfinite"]
            and report["max_abs_qpos_error"] <= args.max_qpos_error
            and report["p99_abs_qpos_error"] <= args.p99_qpos_error
            and report["max_abs_qvel_error"] <= args.max_qvel_error
            and report["p99_abs_qvel_error"] <= args.p99_qvel_error
            and report["contact_presence_agreement"]
            >= args.min_contact_presence_agreement
            and report["mean_contact_pair_jaccard"]
            >= args.min_contact_pair_jaccard
            and report["fall_predicate_agreement"] == 1.0
        )
    env.close()
    rendered = json.dumps(report, sort_keys=True, indent=2)
    print(rendered)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(
        f".{args.output.name}.tmp-{os.getpid()}")
    with temporary_output.open("xb") as stream:
        stream.write((rendered + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary_output, args.output)
    except FileExistsError as exc:
        raise FileExistsError("parity output path was already consumed") from exc
    temporary_output.unlink()
    directory = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if not report["pass"]:
        raise RuntimeError("MuJoCo-Warp/native parity gate failed")


if __name__ == "__main__":
    main()
