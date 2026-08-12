#!/usr/bin/env python3
"""Collect the frozen 600-state L1/L4/L8 PPO residual-option oracle."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.counterfactual_firewall import assert_development_artifact
from safety_data.mjlab_natural_falls import MJLAB_TO_TARGET_JOINT
from safety_data.ppo_reference_actor import FrozenPpoReferenceActor
from safety_data.short_option_candidates import (
    ACTION_SCALE, BETA, apply_closed_loop_residual, option_candidate_layout,
    project_physical_target, select_farthest_residuals,
)


TARGET_OFFSET = np.asarray([0.05, 0.7, -1.4] * 4, np.float32)
PERMUTATION = np.asarray(MJLAB_TO_TARGET_JOINT, np.int64)
INVERSE_PERMUTATION = np.argsort(PERMUTATION)
HORIZON = 96
REPLICAS = 8
_WORKER: dict[str, object] = {}


def _u64(namespace: bytes, identity: bytes, *parts: int) -> int:
    digest = hashlib.sha256(namespace + b"\0" + identity)
    for part in parts:
        digest.update(int(part).to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load_roster(path: Path) -> list[dict[str, object]]:
    path = assert_development_artifact(path)
    with np.load(path, allow_pickle=False) as data:
        rows = [{name: data[name][row].item() for name in data.files}
                for row in range(len(data["state_id"]))]
    if len(rows) != 600:
        raise RuntimeError("short-option roster must contain exactly 600 states")
    identities = [str(row["state_id"]) for row in rows]
    episodes = [str(row["episode_key"]) for row in rows]
    if len(set(identities)) != 600 or len(set(episodes)) != 600:
        raise RuntimeError("short-option roster identities are not episode-disjoint")
    if any(str(row["risk_stratum"]) != "boundary" for row in rows):
        raise RuntimeError("short-option roster contains a non-Boundary state")
    counts = {seed: sum(int(row["collector_seed"]) == seed for row in rows)
              for seed in (137, 138)}
    if counts != {137: 300, 138: 300}:
        raise RuntimeError("short-option roster is not 1:1 across collectors")
    return rows


def _load_state(row: dict[str, object]) -> dict[str, object]:
    path = assert_development_artifact(str(row["archive_path"]))
    with np.load(path, allow_pickle=False) as data:
        source_row = int(row["archive_row"])
        index = int(row["trajectory_index"])
        if index < 0:
            raise RuntimeError("short-option state must be a pre-fall trajectory state")
        at = lambda name: data["trajectory_" + name][source_row, index].copy()
        return {
            **row,
            "identity_bytes": str(row["state_id"]).encode("ascii"),
            "qpos": at("qpos"), "qvel": at("qvel"), "ctrl": at("ctrl"),
            "time": float(at("time")), "act": at("act"),
            "qacc": at("qacc_warmstart"),
            "history": at("observation_history"),
            "episode_step": int(at("episode_step")),
            "geom_friction": data["randomized_geom_friction"][source_row].copy(),
            "body_ipos": data["randomized_body_ipos"][source_row].copy(),
            "encoder_bias": data["randomized_encoder_bias"][source_row].copy(),
            "randomization_identity": bytes(data["rng_identity"][source_row]),
        }


def _actor_observation_from_history(state: dict[str, object]) -> np.ndarray:
    latest = np.asarray(state["history"], np.float32)[-1]
    bias_internal = np.asarray(state["encoder_bias"], np.float32)
    q_target = latest[:12] + bias_internal[PERMUTATION]
    dq_target = latest[12:24]
    angular = latest[24:27]
    quat = latest[30:34].copy()
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    w, x, y, z = quat
    gravity = np.asarray([2 * (x * z - w * y), 2 * (y * z + w * x),
                          1 - 2 * (x * x + y * y)], np.float32)
    phase = (int(state["episode_step"]) * 0.02 % 0.6) / 0.6
    previous_target = latest[34:46]
    previous_internal = (
        previous_target[INVERSE_PERMUTATION] + bias_internal - TARGET_OFFSET
    ) / ACTION_SCALE
    return np.concatenate((
        angular, gravity, [0.3, 0.0, 0.0],
        [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        q_target[INVERSE_PERMUTATION] - TARGET_OFFSET,
        dq_target[INVERSE_PERMUTATION], previous_internal,
    )).astype(np.float32)


def _physical_from_raw(
    raw_internal: np.ndarray, bias_internal: np.ndarray,
    lower_target: np.ndarray, upper_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unprojected_internal = TARGET_OFFSET + ACTION_SCALE * raw_internal - bias_internal
    return project_physical_target(
        unprojected_internal[PERMUTATION], lower_target, upper_target)


def _raw_from_physical(target: np.ndarray, bias_internal: np.ndarray) -> np.ndarray:
    return ((np.asarray(target, np.float32)[INVERSE_PERMUTATION]
             + bias_internal - TARGET_OFFSET) / ACTION_SCALE).astype(np.float32)


def _make_candidates(
    state: dict[str, object], actor: FrozenPpoReferenceActor,
    lower_target: np.ndarray, upper_target: np.ndarray,
) -> dict[str, np.ndarray]:
    observation = _actor_observation_from_history(state)
    generator = torch.Generator(device="cpu").manual_seed(
        _u64(b"qsafe.short-option.candidates.v1", state["identity_bytes"]))
    with torch.inference_mode():
        raw = actor.requested_action(
            torch.from_numpy(observation).repeat(65, 1), generator=generator).numpy()
    bias = np.asarray(state["encoder_bias"], np.float32)
    physical, saturation = [], []
    for value in raw:
        projected, clipped = _physical_from_raw(value, bias, lower_target, upper_target)
        physical.append(projected); saturation.append(clipped)
    physical_array = np.asarray(physical, np.float32)
    selection = select_farthest_residuals(physical_array[0], physical_array[1:])
    duration, direction = option_candidate_layout(selection.residuals)
    initial_targets = np.concatenate((
        selection.nominal[None],
        np.concatenate([selection.selected_targets] * 3, axis=0),
    ), axis=0)
    requested_internal = np.stack([
        _raw_from_physical(value, bias) for value in initial_targets])
    requested_target = requested_internal[:, PERMUTATION]
    selected_proposal_saturation = np.asarray(saturation, bool)[
        np.concatenate(([0], selection.proposal_indices.astype(np.int64) + 1))]
    initial_saturation = np.concatenate((
        selected_proposal_saturation[:1],
        np.concatenate([selected_proposal_saturation[1:]] * 3, axis=0),
    ), axis=0)
    return {
        "raw_internal": requested_internal.astype(np.float32),
        "action_requested": requested_target.astype(np.float32),
        "critic_action": initial_targets.astype(np.float32),
        "residual": np.concatenate((
            np.zeros((1, 12), np.float32),
            np.concatenate([selection.residuals] * 3, axis=0),
        )),
        "duration": duration,
        "direction": direction,
        "proposal_index": np.concatenate((
            [-1], np.concatenate([selection.proposal_indices] * 3),
        )).astype(np.int16),
        "initial_projection_saturation": initial_saturation,
        "direction_distance": np.concatenate((
            [0.0], np.concatenate([selection.nominal_distance] * 3),
        )).astype(np.float32),
    }


def _worker_init(model_path: str, checkpoint137: str, checkpoint138: str) -> None:
    import mujoco
    torch.set_num_threads(1)
    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    actors = {137: FrozenPpoReferenceActor(checkpoint137).eval(),
              138: FrozenPpoReferenceActor(checkpoint138).eval()}
    joint_names = [f"robot/{leg}_{joint}_joint"
                   for leg in ("FL", "FR", "RL", "RR")
                   for joint in ("hip", "thigh", "calf")]
    joint_ids = np.asarray([mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names], np.int32)
    if np.any(joint_ids < 0) or not np.all(model.jnt_limited[joint_ids]):
        raise RuntimeError("target model lacks the twelve hard joint limits")
    actuator = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]
                           for joint in joint_ids], np.int32)
    gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")
    _WORKER.update({
        "mujoco": mujoco, "model": model, "data": data, "actors": actors,
        "qpos_adr": model.jnt_qposadr[joint_ids],
        "qvel_adr": model.jnt_dofadr[joint_ids],
        "actuator": actuator, "gyro": gyro,
        "joint_lower": model.jnt_range[joint_ids, 0].astype(np.float32),
        "joint_upper": model.jnt_range[joint_ids, 1].astype(np.float32),
    })


def _pose() -> tuple[float, float, float, float, float]:
    model, data = _WORKER["model"], _WORKER["data"]
    quat = np.asarray(data.qpos[3:7], np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    w, x, y, z = quat
    roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    sensor = int(_WORKER["gyro"])
    adr, dim = model.sensor_adr[sensor], model.sensor_dim[sensor]
    angular = float(np.linalg.norm(data.sensordata[adr:adr + dim]))
    height = float(data.qpos[2])
    return roll, pitch, angular, height, max(abs(roll), abs(pitch))


def _apply_target(target_target_order: np.ndarray) -> tuple[bool, tuple[float, ...]]:
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    for _ in range(10):
        q, dq = data.qpos[_WORKER["qpos_adr"]], data.qvel[_WORKER["qvel_adr"]]
        torque = 60.0 * (target_target_order - q) - 5.0 * dq
        actuator = _WORKER["actuator"]
        data.ctrl[actuator] = np.clip(torque, model.actuator_ctrlrange[actuator, 0],
                                      model.actuator_ctrlrange[actuator, 1])
        mujoco.mj_step(model, data)
    pose = _pose()
    return bool(pose[3] < 0.18 or pose[4] >= 1.047198), pose


def _actor_observation(
    previous_raw_internal: np.ndarray, episode_step: int,
    encoder_bias_internal: np.ndarray, sensor_rng: np.random.Generator,
) -> np.ndarray:
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    mujoco.mj_forward(model, data)
    sensor = int(_WORKER["gyro"])
    adr, dim = model.sensor_adr[sensor], model.sensor_dim[sensor]
    angular = np.asarray(data.sensordata[adr:adr + dim], np.float32)
    angular = angular + sensor_rng.uniform(-0.2, 0.2, 3).astype(np.float32)
    quat = np.asarray(data.qpos[3:7], np.float32)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    w, x, y, z = quat
    gravity = np.asarray([2 * (x * z - w * y), 2 * (y * z + w * x),
                          1 - 2 * (x * x + y * y)], np.float32)
    gravity += sensor_rng.uniform(-0.05, 0.05, 3).astype(np.float32)
    q_target = np.asarray(data.qpos[_WORKER["qpos_adr"]], np.float32)
    q_target += encoder_bias_internal[PERMUTATION]
    dq_target = np.asarray(data.qvel[_WORKER["qvel_adr"]], np.float32)
    q_internal = q_target[INVERSE_PERMUTATION]
    dq_internal = dq_target[INVERSE_PERMUTATION]
    q_internal += sensor_rng.uniform(-0.01, 0.01, 12).astype(np.float32)
    dq_internal += sensor_rng.uniform(-1.5, 1.5, 12).astype(np.float32)
    phase = (episode_step * 0.02 % 0.6) / 0.6
    return np.concatenate((
        angular, gravity, [0.3, 0.0, 0.0],
        [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        q_internal - TARGET_OFFSET, dq_internal, previous_raw_internal,
    )).astype(np.float32)


def _run_state(task: tuple[int, dict[str, object], dict[str, np.ndarray]]):
    index, state, candidates = task
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    model.geom_friction[:] = state["geom_friction"]
    model.body_ipos[:] = state["body_ipos"]
    mujoco.mj_setConst(model, data)
    actor = _WORKER["actors"][int(state["collector_seed"])]
    bias = np.asarray(state["encoder_bias"], np.float32)
    lower, upper = _WORKER["joint_lower"], _WORKER["joint_upper"]
    fall = np.zeros((16, REPLICAS), bool)
    first = np.full((16, REPLICAS), HORIZON + 1, np.int16)
    replacement_sum = np.zeros((16, REPLICAS), np.float32)
    replacement_max = np.zeros((16, REPLICAS), np.float32)
    projection_saturation = np.zeros((16, REPLICAS), np.int16)
    joint_limit_saturation = np.zeros((16, REPLICAS), np.int16)
    active_steps_executed = np.zeros((16, REPLICAS), np.int8)
    max_roll = np.zeros((16, REPLICAS), np.float32)
    max_pitch = np.zeros((16, REPLICAS), np.float32)
    max_angular = np.zeros((16, REPLICAS), np.float32)
    min_height = np.full((16, REPLICAS), np.inf, np.float32)
    for candidate in range(16):
        duration = int(candidates["duration"][candidate])
        residual = candidates["residual"][candidate]
        for replica in range(REPLICAS):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = state["qpos"]; data.qvel[:] = state["qvel"]
            data.ctrl[:] = state["ctrl"]; data.time = state["time"]
            if len(data.act):
                data.act[:] = state["act"]
            data.qacc_warmstart[:] = state["qacc"]
            mujoco.mj_forward(model, data)
            crn_seed = _u64(
                b"qsafe.short-option.crn.v1", state["identity_bytes"], replica)
            policy_generator = torch.Generator(device="cpu").manual_seed(crn_seed)
            sensor_rng = np.random.default_rng(_u64(
                b"qsafe.short-option.sensor.v1", state["identity_bytes"], replica))
            previous_raw = candidates["raw_internal"][candidate].copy()
            for step in range(HORIZON):
                if step == 0:
                    nominal_target = candidates["critic_action"][0]
                    target = candidates["critic_action"][candidate]
                    clipped = candidates["initial_projection_saturation"][candidate]
                else:
                    observation = _actor_observation(
                        previous_raw, int(state["episode_step"]) + step, bias, sensor_rng)
                    with torch.inference_mode():
                        nominal_raw = actor.requested_action(
                            torch.from_numpy(observation[None]),
                            generator=policy_generator).numpy()[0]
                    nominal_target, nominal_clipped = _physical_from_raw(
                        nominal_raw, bias, lower, upper)
                    if duration > 1 and step < duration:
                        target, clipped = apply_closed_loop_residual(
                            nominal_target, residual, duration=duration,
                            option_step=step, joint_lower=lower, joint_upper=upper)
                    else:
                        target, clipped = nominal_target, nominal_clipped
                previous_raw = _raw_from_physical(target, bias)
                active = step == 0 or (duration > 1 and step < duration)
                if active:
                    active_steps_executed[candidate, replica] += 1
                    magnitude = float(np.sqrt(np.mean(
                        np.square((target - nominal_target) / ACTION_SCALE))))
                    replacement_sum[candidate, replica] += magnitude
                    replacement_max[candidate, replica] = max(
                        replacement_max[candidate, replica], magnitude)
                    projection_saturation[candidate, replica] += int(np.sum(clipped))
                    joint_limit_saturation[candidate, replica] += int(np.sum(
                        np.isclose(target, lower, atol=1e-6)
                        | np.isclose(target, upper, atol=1e-6)))
                failed, pose = _apply_target(target)
                if active:
                    roll, pitch, angular, height, _ = pose
                    max_roll[candidate, replica] = max(
                        max_roll[candidate, replica], abs(roll))
                    max_pitch[candidate, replica] = max(
                        max_pitch[candidate, replica], abs(pitch))
                    max_angular[candidate, replica] = max(
                        max_angular[candidate, replica], angular)
                    min_height[candidate, replica] = min(
                        min_height[candidate, replica], height)
                if failed:
                    fall[candidate, replica] = True
                    first[candidate, replica] = step + 1
                    break
    return (index, fall, first, replacement_sum, replacement_max,
            projection_saturation, joint_limit_saturation,
            active_steps_executed, max_roll, max_pitch, max_angular, min_height)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--checkpoint137", type=Path, required=True)
    parser.add_argument("--checkpoint138", type=Path, required=True)
    parser.add_argument("--model-binary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = _load_roster(args.roster)
    states = [_load_state(row) for row in rows]

    import mujoco
    model = mujoco.MjModel.from_binary_path(str(args.model_binary))
    joint_ids = np.asarray([mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{leg}_{joint}_joint")
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")], np.int32)
    lower, upper = model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]
    actors = {137: FrozenPpoReferenceActor(args.checkpoint137).eval(),
              138: FrozenPpoReferenceActor(args.checkpoint138).eval()}
    candidates = [_make_candidates(
        state, actors[int(state["collector_seed"])], lower, upper) for state in states]

    shape = (600, 16, REPLICAS)
    arrays3 = {
        "h96_fall": np.zeros(shape, bool),
        "first_fall_step": np.full(shape, HORIZON + 1, np.int16),
        "replacement_magnitude_sum": np.zeros(shape, np.float32),
        "replacement_magnitude_max": np.zeros(shape, np.float32),
        "projection_saturation_count": np.zeros(shape, np.int16),
        "joint_limit_saturation_count": np.zeros(shape, np.int16),
        "option_active_steps_executed": np.zeros(shape, np.int8),
        "option_max_abs_roll": np.zeros(shape, np.float32),
        "option_max_abs_pitch": np.zeros(shape, np.float32),
        "option_max_angular_velocity": np.zeros(shape, np.float32),
        "option_min_base_height": np.full(shape, np.inf, np.float32),
    }
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_worker_init,
        initargs=(str(args.model_binary), str(args.checkpoint137),
                  str(args.checkpoint138)),
    ) as executor:
        futures = [executor.submit(_run_state, (i, states[i], candidates[i]))
                   for i in range(600)]
        for future in as_completed(futures):
            values = future.result()
            index = values[0]
            for name, value in zip(arrays3, values[1:], strict=True):
                if value.shape != (16, REPLICAS) or not np.all(np.isfinite(value)):
                    raise RuntimeError(f"incomplete or invalid state group: {name}")
                arrays3[name][index] = value

    duration, direction = option_candidate_layout(np.zeros((5, 12), np.float32))
    output = {
        "state_id": np.asarray([row["state_id"] for row in rows], "S64"),
        "episode_id": np.asarray([row["episode_key"] for row in rows], "S64"),
        "risk_stratum": np.asarray(["boundary"] * 600, "U8"),
        "collector_seed": np.asarray([row["collector_seed"] for row in rows], np.int16),
        "rollout_seed": np.asarray([row["rollout_seed"] for row in rows], np.int32),
        "observation_history": np.stack([state["history"] for state in states]),
        "candidate_index": np.broadcast_to(np.arange(16), (600, 16)),
        "candidate_duration": np.broadcast_to(duration, (600, 16)),
        "candidate_direction": np.broadcast_to(direction, (600, 16)),
        "proposal_index": np.stack([value["proposal_index"] for value in candidates]),
        "candidate_direction_distance": np.stack(
            [value["direction_distance"] for value in candidates]),
        "action_requested": np.stack([value["action_requested"] for value in candidates]),
        "critic_action": np.stack([value["critic_action"] for value in candidates]),
        "absolute_q_target": np.stack([value["critic_action"] for value in candidates]),
        "residual": np.stack([value["residual"] for value in candidates]),
        "replica_id": np.broadcast_to(np.arange(1, 9), shape),
        "crn_id": np.asarray([[[hashlib.sha256(
            b"qsafe.short-option.crn.id.v1\0" + state["identity_bytes"]
            + replica.to_bytes(2, "little")).hexdigest()
            for replica in range(8)] for _ in range(16)]
            for state in states], "S64"),
        "randomization_identity": np.asarray(
            [state["randomization_identity"] for state in states], "S64"),
        **arrays3,
    }
    if not np.all(output["first_fall_step"] == np.where(
            output["h96_fall"], output["first_fall_step"], 97)):
        raise RuntimeError("H96 first-fall sentinel mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **output)
    os.link(temporary, args.output); temporary.unlink()
    report = {
        "schema_version": "qsafe.short_option_oracle_dataset.v1",
        "generator_commit": _git_commit(),
        "states": 600, "candidates": 16, "replicas": 8,
        "branches": int(np.prod(shape)), "falls": int(output["h96_fall"].sum()),
        "protected_outcomes_read_or_generated": False,
        "safety_critic_trained": False, "sac_transfer_run": False,
        "output": str(args.output),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
