#!/usr/bin/env python3
"""Collect grouped H96 same-state PPO action branches with paired CRN."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from safety_data.counterfactual_candidates import select_diverse_candidates
from safety_data.counterfactual_firewall import (
    assert_development_artifact, load_identity_denylist, reject_denied_identities,
)
from safety_data.mjlab_natural_falls import MJLAB_TO_TARGET_JOINT
from safety_data.ppo_reference_actor import (
    FrozenPpoReferenceActor, sac_observation_to_ppo_actor_observation,
)


TARGET_OFFSET = np.asarray([0.05, 0.7, -1.4] * 4, np.float32)
TARGET_SCALE = np.asarray([0.2, 0.4, 0.4] * 4, np.float32)
PERMUTATION = np.asarray(MJLAB_TO_TARGET_JOINT, np.int64)
_WORKER: dict[str, object] = {}


def _u64(namespace: bytes, identity: bytes, *parts: int) -> int:
    digest = hashlib.sha256(namespace + b"\0" + identity)
    for part in parts:
        digest.update(int(part).to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def _load_roster(path: Path, denylist: frozenset[str]) -> list[dict[str, object]]:
    path = assert_development_artifact(path)
    with np.load(path, allow_pickle=False) as data:
        rows = [{name: data[name][row].item() for name in data.files}
                for row in range(len(data["state_id"]))]
    identities = [str(row["state_id"]) for row in rows]
    reject_denied_identities(identities, denylist)
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate state identity")
    if any(row["split"] == "protected" for row in rows):
        raise RuntimeError("development branch collector cannot consume protected roster")
    return rows


def _load_state(row: dict[str, object]) -> dict[str, object]:
    path = assert_development_artifact(str(row["archive_path"]))
    with np.load(path, allow_pickle=False) as data:
        source_row = int(row["archive_row"])
        index = int(row["trajectory_index"])
        prefix = "trajectory_" if index >= 0 else ""
        at = (lambda name: data[prefix + name][source_row, index].copy()) if index >= 0 else (
            lambda name: data[name][source_row].copy())
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


def _candidates(state: dict[str, object], actor: FrozenPpoReferenceActor) -> dict[str, np.ndarray]:
    actor_observation = sac_observation_to_ppo_actor_observation(
        np.asarray(state["history"], np.float32)[-1:],
        episode_step=np.asarray([state["episode_step"]]))
    generator = torch.Generator(device="cpu").manual_seed(
        _u64(b"qsafe.counterfactual.candidates.v2", state["identity_bytes"]))
    with torch.inference_mode():
        raw = actor.requested_action(
            torch.from_numpy(actor_observation).repeat(65, 1), generator=generator,
        ).numpy()
    bias = np.asarray(state["encoder_bias"], np.float32)
    absolute_internal = TARGET_OFFSET + TARGET_SCALE * raw - bias
    absolute_target = absolute_internal[:, PERMUTATION]
    selection = select_diverse_candidates(absolute_target[0], absolute_target[1:])
    proposal = selection.proposal_indices.astype(np.int64) + 1
    selected_rows = np.concatenate(([0], proposal))
    distance = np.concatenate(([0.0], selection.distance)).astype(np.float32)
    distance_bin = np.concatenate((np.asarray(["nominal"]), selection.distance_bin))
    return {
        "raw_internal": raw[selected_rows].astype(np.float32),
        "absolute_internal": absolute_internal[selected_rows].astype(np.float32),
        "critic_action": absolute_target[selected_rows].astype(np.float32),
        "action_requested": raw[selected_rows][:, PERMUTATION].astype(np.float32),
        "action_pre_projection": raw[selected_rows][:, PERMUTATION].astype(np.float32),
        "distance": distance, "distance_bin": distance_bin,
        "proposal_index": np.concatenate(([-1], selection.proposal_indices)).astype(np.int16),
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
    actuator = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == joint)[0]
                           for joint in joint_ids], np.int32)
    gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")
    _WORKER.update({"mujoco": mujoco, "model": model, "data": data, "actors": actors,
                    "qpos_adr": model.jnt_qposadr[joint_ids],
                    "qvel_adr": model.jnt_dofadr[joint_ids],
                    "actuator": actuator, "gyro": gyro})


def _actor_observation(previous: np.ndarray, episode_step: int,
                       encoder_bias: np.ndarray) -> np.ndarray:
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    mujoco.mj_forward(model, data)
    sensor = int(_WORKER["gyro"])
    adr, dim = model.sensor_adr[sensor], model.sensor_dim[sensor]
    angular = np.asarray(data.sensordata[adr:adr + dim], np.float32)
    quat = np.asarray(data.qpos[3:7], np.float32)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    w, x, y, z = quat
    gravity = np.asarray([2 * (x * z - w * y), 2 * (y * z + w * x),
                          1 - 2 * (x * x + y * y)], np.float32)
    q = np.asarray(data.qpos[_WORKER["qpos_adr"]], np.float32) + encoder_bias
    dq = np.asarray(data.qvel[_WORKER["qvel_adr"]], np.float32)
    phase = (episode_step * 0.02 % 0.6) / 0.6
    return np.concatenate((angular, gravity, [0.3, 0.0, 0.0],
        [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        q - TARGET_OFFSET, dq, previous)).astype(np.float32)


def _apply_target(target: np.ndarray) -> bool:
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    for _ in range(10):
        q, dq = data.qpos[_WORKER["qpos_adr"]], data.qvel[_WORKER["qvel_adr"]]
        torque = 60.0 * (target - q) - 5.0 * dq
        actuator = _WORKER["actuator"]
        data.ctrl[actuator] = np.clip(torque, model.actuator_ctrlrange[actuator, 0],
                                      model.actuator_ctrlrange[actuator, 1])
        mujoco.mj_step(model, data)
    quat = np.asarray(data.qpos[3:7], np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return bool(data.qpos[2] < 0.18 or max(abs(roll), abs(pitch)) >= 1.047198)


def _run_state(task: tuple[int, dict[str, object], dict[str, np.ndarray], int]):
    index, state, candidates, replicas = task
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    model.geom_friction[:] = state["geom_friction"]
    model.body_ipos[:] = state["body_ipos"]
    mujoco.mj_setConst(model, data)
    fall = np.zeros((16, replicas), bool)
    first = np.full((16, replicas), 97, np.int16)
    seed = int(state["collector_seed"])
    actor = _WORKER["actors"][seed]
    bias = np.asarray(state["encoder_bias"], np.float32)
    for candidate in range(16):
        for replica in range(replicas):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = state["qpos"]; data.qvel[:] = state["qvel"]
            data.ctrl[:] = state["ctrl"]; data.time = state["time"]
            if len(data.act): data.act[:] = state["act"]
            data.qacc_warmstart[:] = state["qacc"]
            mujoco.mj_forward(model, data)
            if _apply_target(candidates["absolute_internal"][candidate]):
                fall[candidate, replica] = True; first[candidate, replica] = 1
                continue
            generator = torch.Generator(device="cpu").manual_seed(_u64(
                b"qsafe.counterfactual.crn.v2", state["identity_bytes"], replica))
            previous = candidates["raw_internal"][candidate]
            for step in range(2, 97):
                observation = _actor_observation(
                    previous, int(state["episode_step"]) + step - 1, bias)
                with torch.inference_mode():
                    previous = actor.requested_action(
                        torch.from_numpy(observation[None]), generator=generator).numpy()[0]
                target = TARGET_OFFSET + TARGET_SCALE * previous - bias
                if _apply_target(target):
                    fall[candidate, replica] = True; first[candidate, replica] = step
                    break
    return index, fall, first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--checkpoint137", type=Path, required=True)
    parser.add_argument("--checkpoint138", type=Path, required=True)
    parser.add_argument("--model-binary", type=Path, required=True)
    parser.add_argument("--round-one-denylist", type=Path, required=True)
    parser.add_argument("--replicas", type=int, choices=(4, 8), required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = _load_roster(args.roster, load_identity_denylist(args.round_one_denylist))
    states = [_load_state(row) for row in rows]
    actors = {137: FrozenPpoReferenceActor(args.checkpoint137).eval(),
              138: FrozenPpoReferenceActor(args.checkpoint138).eval()}
    candidates = [_candidates(state, actors[int(state["collector_seed"])]) for state in states]
    falls = np.zeros((len(states), 16, args.replicas), bool)
    first = np.full((len(states), 16, args.replicas), 97, np.int16)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
            initargs=(str(args.model_binary), str(args.checkpoint137),
                      str(args.checkpoint138))) as executor:
        futures = [executor.submit(_run_state, (i, state, candidates[i], args.replicas))
                   for i, state in enumerate(states)]
        for future in as_completed(futures):
            index, state_fall, state_first = future.result()
            if state_fall.shape != (16, args.replicas):
                raise RuntimeError("incomplete state group")
            falls[index], first[index] = state_fall, state_first
    arrays = {
        "state_id": np.asarray([row["state_id"] for row in rows], "S64"),
        "snapshot_identity": np.asarray([row["state_id"] for row in rows], "S64"),
        "episode_id": np.asarray([row["episode_key"] for row in rows], "S64"),
        "split": np.asarray([row["split"] for row in rows], "U12"),
        "risk_stratum": np.asarray([row["risk_stratum"] for row in rows], "U8"),
        "collector_seed": np.asarray([row["collector_seed"] for row in rows], np.int16),
        "collector_checkpoint": np.asarray([f"seed{row['collector_seed']}/model_19.pt"
                                             for row in rows], "U32"),
        "observation_history": np.stack([state["history"] for state in states]),
        "candidate_index": np.broadcast_to(np.arange(16), (len(states), 16)),
        "candidate_id": np.asarray([[hashlib.sha256(
            b"qsafe.counterfactual.candidate.v2\0" + state["identity_bytes"]
            + candidate.to_bytes(2, "little")).hexdigest()
            for candidate in range(16)] for state in states], "S64"),
        "candidate_distance": np.stack([value["distance"] for value in candidates]),
        "candidate_distance_bin": np.stack([value["distance_bin"] for value in candidates]),
        "candidate_kind": np.broadcast_to(
            np.asarray(["nominal"] + ["ppo_stochastic"] * 15, "U16"), (len(states), 16)),
        "action_requested": np.stack([value["action_requested"] for value in candidates]),
        "action_pre_projection": np.stack([value["action_pre_projection"] for value in candidates]),
        "critic_action": np.stack([value["critic_action"] for value in candidates]),
        "absolute_q_target": np.stack([value["critic_action"] for value in candidates]),
        "replica_id": np.broadcast_to(np.arange(1, args.replicas + 1),
                                       (len(states), 16, args.replicas)),
        "crn_id": np.asarray([[[hashlib.sha256(
            b"qsafe.counterfactual.crn.id.v2\0" + state["identity_bytes"]
            + replica.to_bytes(2, "little")).hexdigest()
            for replica in range(1, args.replicas + 1)] for _ in range(16)]
            for state in states], "S64"),
        "h96_fall": falls, "first_fall_step": first,
        "terminal_reason": np.where(falls, "fall", "h96_safe"),
        "randomization_identity": np.asarray(
            [state["randomization_identity"] for state in states], "S64"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.link(temporary, args.output); temporary.unlink()
    print(json.dumps({"states": len(states), "branches": int(falls.size),
                      "falls": int(falls.sum()), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
