#!/usr/bin/env python3
"""Run only R5--R16 for 400 states using exact previously frozen candidates."""

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

from safety_data.counterfactual_firewall import (
    assert_development_artifact, load_identity_denylist, reject_denied_identities,
)
from safety_data.counterfactual_replica_extension import (
    reconstruct_frozen_candidates, verify_frozen_candidate_identity,
)
from scripts.collect_ppo_counterfactual_branches import (
    TARGET_OFFSET, TARGET_SCALE, _WORKER, _actor_observation, _apply_target,
    _load_state, _u64, _worker_init,
)


ADDITIONAL_REPLICAS_ZERO_BASED = tuple(range(4, 16))


def _load_original_states(
    development_roster: Path, diagnostic_ids: np.ndarray,
) -> list[dict[str, object]]:
    with np.load(assert_development_artifact(development_roster),
                 allow_pickle=False) as data:
        by_identity = {
            str(data["state_id"][row].item()):
            {name: data[name][row].item() for name in data.files}
            for row in range(len(data["state_id"]))
        }
    rows = []
    for value in diagnostic_ids:
        identity = bytes(value).decode("ascii")
        if identity not in by_identity:
            raise RuntimeError("diagnostic state is absent from development roster")
        rows.append(by_identity[identity])
    return [_load_state(row) for row in rows]


def _run_extension_state(task):
    index, state, candidates = task
    model, data, mujoco = _WORKER["model"], _WORKER["data"], _WORKER["mujoco"]
    model.geom_friction[:] = state["geom_friction"]
    model.body_ipos[:] = state["body_ipos"]
    mujoco.mj_setConst(model, data)
    count = len(ADDITIONAL_REPLICAS_ZERO_BASED)
    fall = np.zeros((16, count), bool)
    first = np.full((16, count), 97, np.int16)
    actor = _WORKER["actors"][int(state["collector_seed"])]
    bias = np.asarray(state["encoder_bias"], np.float32)
    for candidate in range(16):
        for output_replica, replica in enumerate(ADDITIONAL_REPLICAS_ZERO_BASED):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
            data.ctrl[:] = state["ctrl"]
            data.time = state["time"]
            if len(data.act):
                data.act[:] = state["act"]
            data.qacc_warmstart[:] = state["qacc"]
            mujoco.mj_forward(model, data)
            if _apply_target(candidates["absolute_internal"][candidate]):
                fall[candidate, output_replica] = True
                first[candidate, output_replica] = 1
                continue
            generator = torch.Generator(device="cpu").manual_seed(_u64(
                b"qsafe.counterfactual.crn.v2", state["identity_bytes"], replica))
            previous = candidates["raw_internal"][candidate]
            for step in range(2, 97):
                observation = _actor_observation(
                    previous, int(state["episode_step"]) + step - 1, bias)
                with torch.inference_mode():
                    previous = actor.requested_action(
                        torch.from_numpy(observation[None]),
                        generator=generator).numpy()[0]
                target = TARGET_OFFSET + TARGET_SCALE * previous - bias
                if _apply_target(target):
                    fall[candidate, output_replica] = True
                    first[candidate, output_replica] = step
                    break
    return index, fall, first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-roster", type=Path, required=True)
    parser.add_argument("--development-dataset", type=Path, required=True)
    parser.add_argument("--development-roster", type=Path, required=True)
    parser.add_argument("--checkpoint137", type=Path, required=True)
    parser.add_argument("--checkpoint138", type=Path, required=True)
    parser.add_argument("--model-binary", type=Path, required=True)
    parser.add_argument("--round-one-denylist", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(assert_development_artifact(args.diagnostic_roster),
                 allow_pickle=False) as roster:
        arrays = {name: roster[name] for name in roster.files}
    identities = [bytes(value).decode("ascii") for value in arrays["state_id"]]
    reject_denied_identities(
        identities, load_identity_denylist(args.round_one_denylist))
    if len(identities) != 400 or len(set(identities)) != 400:
        raise RuntimeError("diagnostic roster identity contract failed")
    with np.load(assert_development_artifact(args.development_dataset),
                 allow_pickle=False) as development:
        source_rows = arrays["source_row"].astype(np.int64)
        verify_frozen_candidate_identity(
            development["candidate_id"][source_rows], arrays["candidate_id"],
            development["critic_action"][source_rows], arrays["critic_action"])
    states = _load_original_states(args.development_roster, arrays["state_id"])
    candidates = [reconstruct_frozen_candidates(
        arrays["action_requested"][state], arrays["critic_action"][state])
        for state in range(400)]
    fall = np.zeros((400, 16, 12), bool)
    first = np.full((400, 16, 12), 97, np.int16)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_worker_init,
        initargs=(str(args.model_binary), str(args.checkpoint137),
                  str(args.checkpoint138)),
    ) as executor:
        futures = [executor.submit(_run_extension_state, (index, state,
                    candidates[index])) for index, state in enumerate(states)]
        for future in as_completed(futures):
            index, state_fall, state_first = future.result()
            if state_fall.shape != (16, 12) or state_first.shape != (16, 12):
                raise RuntimeError("incomplete replica-extension state group")
            fall[index] = state_fall
            first[index] = state_first
    replica_id = np.broadcast_to(np.arange(5, 17), (400, 16, 12))
    crn_id = np.asarray([[[hashlib.sha256(
        b"qsafe.counterfactual.crn.id.v2\0" + bytes(identity)
        + replica.to_bytes(2, "little")).hexdigest()
        for replica in range(5, 17)] for _ in range(16)]
        for identity in arrays["state_id"]], "S64")
    output = {
        "state_id": arrays["state_id"],
        "candidate_id": arrays["candidate_id"],
        "critic_action": arrays["critic_action"],
        "replica_id": replica_id,
        "crn_id": crn_id,
        "h96_fall": fall,
        "first_fall_step": first,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **output)
    os.link(temporary, args.output)
    temporary.unlink()
    print(json.dumps({"states": 400, "additional_branches": int(fall.size),
                      "falls": int(fall.sum()), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
