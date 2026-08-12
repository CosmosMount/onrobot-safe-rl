#!/usr/bin/env python3
"""Run the protected 200-state PPO H96 action-ranking gate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import glob
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

from rl.qsafe.ppo_sqrl_critic import PpoSqrlCriticConfig, PpoSqrlSafetyCritic
from safety_data.mjlab_natural_falls import MJLAB_TO_TARGET_JOINT
from safety_data.ppo_reference_actor import (
    FrozenPpoReferenceActor, sac_observation_to_ppo_actor_observation,
)
from safety_data.ppo_same_state_gate import (
    independent_oracle, stable_state_indices, summarize_selector,
)


OFFSETS = (16, 32, 64)
TARGET_OFFSET_INTERNAL = np.asarray([0.05, 0.7, -1.4] * 4, np.float32)
TARGET_SCALE_INTERNAL = np.asarray([0.2, 0.4, 0.4] * 4, np.float32)
TARGET_PERMUTATION = np.asarray(MJLAB_TO_TARGET_JOINT, np.int64)
_WORKER: dict[str, object] = {}


def _u64(domain: bytes, identity: bytes, *parts: int) -> int:
    digest = hashlib.sha256(domain + b"\0" + identity)
    for part in parts:
        digest.update(int(part).to_bytes(8, "little", signed=False))
    return int.from_bytes(digest.digest()[:8], "little") & ((1 << 63) - 1)


def _load_pool(roots: list[Path]) -> list[dict[str, np.ndarray | int | bytes]]:
    pool = []
    for source_seed, root in enumerate(roots):
        for path in sorted(root.glob("falls-*.npz")):
            with np.load(path, allow_pickle=False) as z:
                for row in range(len(z["identity"])):
                    iteration = int(z["ppo_iteration"][row])
                    if not 5 <= iteration <= 20:
                        continue
                    identity = bytes(z["identity"][row])
                    available = [offset for offset in OFFSETS if int(
                        z["trajectory_length"][row]) >= offset]
                    if not available:
                        continue
                    offset = available[_u64(b"qsafe.ppo.branch.offset.v1", identity) % len(available)]
                    offset_slot = (1, 2, 4, 8, 16, 32, 64).index(offset)
                    index = int(z["prefall_trajectory_index"][row, offset_slot])
                    pool.append({
                        "identity": identity,
                        "source_seed": source_seed,
                        "offset": offset,
                        "qpos": z["trajectory_qpos"][row, index].copy(),
                        "qvel": z["trajectory_qvel"][row, index].copy(),
                        "ctrl": z["trajectory_ctrl"][row, index].copy(),
                        "time": float(z["trajectory_time"][row, index]),
                        "qacc": z["trajectory_qacc_warmstart"][row, index].copy(),
                        "history": z["trajectory_observation_history"][row, index].copy(),
                        "episode_step": int(z["trajectory_episode_step"][row, index]),
                        "geom_friction": z["randomized_geom_friction"][row].copy(),
                        "body_ipos": z["randomized_body_ipos"][row].copy(),
                        "encoder_bias": z["randomized_encoder_bias"][row].copy(),
                    })
    if len(pool) < 200:
        raise RuntimeError(f"boundary snapshot pool has only {len(pool)} states")
    indices = stable_state_indices(
        np.asarray([row["identity"] for row in pool], dtype="S64"), 200)
    return [pool[int(index)] for index in indices]


def _candidate_actions(
    states: list[dict[str, object]], checkpoint: Path,
) -> tuple[np.ndarray, np.ndarray]:
    actor = FrozenPpoReferenceActor(checkpoint).eval()
    raw_internal = np.empty((len(states), 16, 12), np.float32)
    absolute_internal = np.empty_like(raw_internal)
    with torch.inference_mode():
        for state_index, state in enumerate(states):
            observation = sac_observation_to_ppo_actor_observation(
                np.asarray(state["history"], np.float32)[-1: ],
                episode_step=np.asarray([state["episode_step"]]))
            generator = torch.Generator(device="cpu").manual_seed(
                _u64(b"qsafe.ppo.branch.candidate.v1", state["identity"]))
            policy_observation = torch.from_numpy(observation).repeat(16, 1)
            raw = actor.requested_action(
                policy_observation, generator=generator).cpu().numpy()
            bias_internal = np.asarray(state["encoder_bias"], np.float32)
            raw_internal[state_index] = raw
            absolute_internal[state_index] = (
                TARGET_OFFSET_INTERNAL + TARGET_SCALE_INTERNAL * raw - bias_internal)
    return raw_internal, absolute_internal


def _worker_init(model_path: str, checkpoint: str) -> None:
    import mujoco
    torch.set_num_threads(1)
    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    actor = FrozenPpoReferenceActor(checkpoint).eval()
    joint_names = [
        f"robot/{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    ]
    joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in joint_names], np.int32)
    qpos_adr = model.jnt_qposadr[joint_ids]
    qvel_adr = model.jnt_dofadr[joint_ids]
    actuator_for_joint = np.empty(12, np.int32)
    for joint_index, joint_id in enumerate(joint_ids):
        found = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
        if len(found) != 1:
            raise RuntimeError("MJB actuator/joint mapping is ambiguous")
        actuator_for_joint[joint_index] = found[0]
    gyro_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")
    _WORKER.update({
        "mujoco": mujoco, "model": model, "data": data, "actor": actor,
        "qpos_adr": qpos_adr, "qvel_adr": qvel_adr,
        "actuator": actuator_for_joint, "gyro_id": gyro_id,
    })


def _actor_observation(previous_raw: np.ndarray, episode_step: int,
                       encoder_bias: np.ndarray) -> np.ndarray:
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    mujoco.mj_forward(model, data)
    sensor_id = int(_WORKER["gyro_id"])
    adr, dim = model.sensor_adr[sensor_id], model.sensor_dim[sensor_id]
    angular = np.asarray(data.sensordata[adr:adr + dim], np.float32)
    quat = np.asarray(data.qpos[3:7], np.float32)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    w, x, y, z = quat
    gravity = np.asarray([
        2 * (x * z - w * y), 2 * (y * z + w * x),
        1 - 2 * (x * x + y * y)], np.float32)
    q = np.asarray(data.qpos[_WORKER["qpos_adr"]], np.float32) + encoder_bias
    dq = np.asarray(data.qvel[_WORKER["qvel_adr"]], np.float32)
    phase = (episode_step * 0.02 % 0.6) / 0.6
    return np.concatenate([
        angular, gravity, [0.3, 0.0, 0.0],
        [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        q - TARGET_OFFSET_INTERNAL, dq, previous_raw,
    ]).astype(np.float32)


def _apply_target(target: np.ndarray) -> bool:
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    for _ in range(10):
        q = data.qpos[_WORKER["qpos_adr"]]
        dq = data.qvel[_WORKER["qvel_adr"]]
        torque = 60.0 * (target - q) - 5.0 * dq
        actuator = _WORKER["actuator"]
        data.ctrl[actuator] = np.clip(
            torque, model.actuator_ctrlrange[actuator, 0],
            model.actuator_ctrlrange[actuator, 1])
        mujoco.mj_step(model, data)
    quat = np.asarray(data.qpos[3:7], np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return bool(data.qpos[2] < 0.18 or max(abs(roll), abs(pitch)) >= 1.047198)


def _run_state(task: tuple[int, dict[str, object], np.ndarray, np.ndarray]) -> tuple[int, np.ndarray]:
    state_index, state, raw_candidates, target_candidates = task
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    model.geom_friction[:] = state["geom_friction"]
    model.body_ipos[:] = state["body_ipos"]
    mujoco.mj_setConst(model, data)
    outcome = np.zeros((16, 8), bool)
    bias_internal = np.asarray(state["encoder_bias"], np.float32)
    for candidate in range(16):
        for replica in range(8):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
            data.ctrl[:] = state["ctrl"]
            data.time = state["time"]
            data.qacc_warmstart[:] = state["qacc"]
            mujoco.mj_forward(model, data)
            if _apply_target(target_candidates[candidate]):
                outcome[candidate, replica] = True
                continue
            generator = torch.Generator(device="cpu").manual_seed(_u64(
                b"qsafe.ppo.branch.continuation.v1", state["identity"], replica))
            previous = raw_candidates[candidate]
            for continuation_step in range(1, 96):
                actor_observation = _actor_observation(
                    previous, int(state["episode_step"]) + continuation_step,
                    bias_internal)
                with torch.inference_mode():
                    previous = _WORKER["actor"].requested_action(
                        torch.from_numpy(actor_observation[None]),
                        generator=generator).numpy()[0]
                target = (TARGET_OFFSET_INTERNAL + TARGET_SCALE_INTERNAL * previous
                          - bias_internal)
                if _apply_target(target):
                    outcome[candidate, replica] = True
                    break
    return state_index, outcome


class _Predictor:
    def __init__(self, path: Path):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PpoSqrlCriticConfig(**artifact["network_config"])
        self.model = PpoSqrlSafetyCritic(cfg).eval()
        self.model.load_state_dict(artifact["model_state_dict"])
        self.norm = {key: np.asarray(value) for key, value in artifact["normalization"].items()}

    def __call__(self, histories: np.ndarray, actions: np.ndarray) -> np.ndarray:
        count, candidates = actions.shape[:2]
        history = np.repeat(histories[:, None], candidates, axis=1).reshape(-1, 5, 46)
        action = actions.reshape(-1, 12)
        history = (history - self.norm["observation_mean"]) / self.norm["observation_std"]
        action = (action - self.norm["action_mean"]) / self.norm["action_std"]
        result = []
        with torch.inference_mode():
            for start in range(0, len(action), 4096):
                result.append(self.model(
                    torch.from_numpy(history[start:start + 4096]).float(),
                    torch.from_numpy(action[start:start + 4096]).float()).numpy())
        return np.concatenate(result).reshape(count, candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fall-root", type=Path, action="append", required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--model-binary", type=Path, required=True)
    parser.add_argument("--critic", type=Path, action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".report.json").exists():
        raise FileExistsError("branching result already exists")
    states = _load_pool(args.fall_root)
    raw, absolute_internal = _candidate_actions(states, args.reference_checkpoint)
    absolute_target = absolute_internal[:, :, TARGET_PERMUTATION]
    predictions = {
        path.stem: _Predictor(path)(
            np.stack([state["history"] for state in states]), absolute_target)
        for path in args.critic
    }
    outcomes = np.zeros((200, 16, 8), bool)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_worker_init,
        initargs=(str(args.model_binary), str(args.reference_checkpoint)),
    ) as executor:
        futures = [executor.submit(
            _run_state, (index, state, raw[index], absolute_internal[index]))
            for index, state in enumerate(states)]
        for completed, future in enumerate(as_completed(futures), 1):
            index, outcome = future.result()
            outcomes[index] = outcome
            if completed % 20 == 0:
                print(json.dumps({"completed_states": completed}), flush=True)
    oracle_choice, oracle_outcome = independent_oracle(outcomes)
    report = {
        "schema_version": "qsafe.ppo_independent_same_state_gate.v1",
        "states": 200, "candidates": 16, "replicas": 8, "horizon": 96,
        "oracle": summarize_selector(outcomes, oracle_choice, bootstrap_seed=8100),
        "critics": {},
    }
    for index, (name, risk) in enumerate(predictions.items()):
        lowest = np.argmin(risk, axis=1)
        # Frozen engineering threshold from the original SQRL Go2 reproduction.
        epsilon = 0.10
        sqrl = np.asarray([
            next((candidate for candidate in range(16) if row[candidate] < epsilon),
                 int(np.argmin(row))) for row in risk], np.int16)
        minimal = np.where(risk[:, 0] < epsilon, 0, np.argmin(risk, axis=1))
        report["critics"][name] = {
            "epsilon_safe": epsilon,
            "top1": summarize_selector(outcomes, lowest, bootstrap_seed=8200 + index),
            "sqrl_rejection": summarize_selector(outcomes, sqrl, bootstrap_seed=8300 + index),
            "minimal_intervention": summarize_selector(
                outcomes, minimal, bootstrap_seed=8400 + index),
            "mean_within_state_risk_std": float(risk.std(axis=1).mean()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(
        temporary, identity=np.asarray([state["identity"] for state in states], "S64"),
        offset=np.asarray([state["offset"] for state in states], np.int16),
        source_seed=np.asarray([state["source_seed"] for state in states], np.int8),
        critic_action=absolute_target, fall=outcomes,
        oracle_choice=oracle_choice, **{f"risk_{key}": value for key, value in predictions.items()})
    os.link(temporary, args.output); temporary.unlink()
    report_path = args.output.with_suffix(".report.json")
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    tmp_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    tmp_report.write_bytes(content); os.link(tmp_report, report_path); tmp_report.unlink()
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
