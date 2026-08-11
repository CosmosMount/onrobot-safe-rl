"""Result-blind natural-fall recording for parallel PPO state proposals.

The recorder is backend independent.  A simulator adapter supplies one
``NaturalPpoFrame`` per policy step and calls :meth:`finish_episode` exactly
once at a terminal boundary.  PPO fall outcomes are deliberately not Q_safe
labels; the stored snapshots are proposal states for later target-SAC
same-state branching.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
from typing import Any, Iterable, Mapping

import numpy as np


PREFALL_OFFSETS = (1, 2, 4, 8, 16, 32, 64)
HISTORY_SHAPE = (5, 46)
RING_POLICY_STEPS = 65
NORMAL_TERMINAL_DISTANCE = 96


def _array(value: Any, *, dtype: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


def _vector(value: Any, *, dtype: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result.copy()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


@dataclass(frozen=True)
class NaturalPpoFrame:
    environment_id: int
    episode_id: int
    episode_step: int
    global_policy_step: int
    ppo_training_step: int
    integration_state: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    observation_history: np.ndarray
    previous_action_requested: np.ndarray
    previous_action_executed: np.ndarray
    previous_action_q_target: np.ndarray
    randomization: Mapping[str, Any]
    rng_identity: int
    external_force: np.ndarray

    def __post_init__(self) -> None:
        for name in ("environment_id", "episode_id", "episode_step",
                     "global_policy_step", "ppo_training_step", "rng_identity"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        object.__setattr__(self, "integration_state", _vector(
            self.integration_state, dtype=np.float64, name="integration_state"))
        object.__setattr__(self, "qpos", _vector(
            self.qpos, dtype=np.float64, name="qpos"))
        object.__setattr__(self, "qvel", _vector(
            self.qvel, dtype=np.float64, name="qvel"))
        object.__setattr__(self, "act", np.asarray(
            self.act, dtype=np.float64).reshape(-1).copy())
        object.__setattr__(self, "ctrl", _vector(
            self.ctrl, dtype=np.float64, name="ctrl"))
        object.__setattr__(self, "observation_history", _array(
            self.observation_history, dtype=np.float32, shape=HISTORY_SHAPE,
            name="observation_history"))
        for name in ("previous_action_requested", "previous_action_executed",
                     "previous_action_q_target"):
            object.__setattr__(self, name, _array(
                getattr(self, name), dtype=np.float32, shape=(12,), name=name))
        force = np.asarray(self.external_force, dtype=np.float64)
        if force.size == 0 or not np.all(np.isfinite(force)):
            raise ValueError("external_force must be a finite non-empty array")
        if np.any(force != 0.0):
            raise ValueError("natural PPO collection forbids non-zero external force")
        object.__setattr__(self, "external_force", force.copy())
        _canonical_json(dict(self.randomization))

    @property
    def identity(self) -> str:
        raw = np.asarray([
            self.environment_id, self.episode_id, self.episode_step,
            self.global_policy_step, self.ppo_training_step, self.rng_identity,
        ], dtype=np.uint64).tobytes()
        return hashlib.sha256(b"qsafe.natural_ppo_frame.v1\0" + raw).hexdigest()

    @property
    def stratum(self) -> str:
        return hashlib.sha256(
            b"qsafe.natural_ppo_stratum.v1\0"
            + str(self.ppo_training_step).encode("ascii") + b"\0"
            + _canonical_json(dict(self.randomization)).encode("ascii")
        ).hexdigest()


@dataclass(frozen=True)
class NaturalFallEvent:
    terminal: NaturalPpoFrame
    trajectory: tuple[NaturalPpoFrame, ...]
    prefall: tuple[NaturalPpoFrame | None, ...]

    @property
    def availability(self) -> np.ndarray:
        return np.asarray([frame is not None for frame in self.prefall], dtype=bool)


class NaturalFallRecorder:
    """Track independent vector environments without repeated fallen frames."""

    def __init__(self, environment_ids: Iterable[int]) -> None:
        ids = tuple(int(value) for value in environment_ids)
        if len(ids) == 0 or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError("environment ids must be unique non-negative integers")
        self._buffers = {value: deque(maxlen=RING_POLICY_STEPS) for value in ids}
        self._terminal = {value: False for value in ids}
        self._seen_identities: set[str] = set()
        self.independent_fall_episodes = 0
        self.recorded_falls = 0

    def append(self, frame: NaturalPpoFrame) -> None:
        environment_id = int(frame.environment_id)
        if environment_id not in self._buffers:
            raise KeyError(f"unknown environment id {environment_id}")
        if self._terminal[environment_id]:
            raise RuntimeError("terminal environment must reset before another policy step")
        if frame.identity in self._seen_identities:
            raise RuntimeError("duplicate natural-PPO snapshot identity")
        buffer = self._buffers[environment_id]
        if buffer:
            previous = buffer[-1]
            if frame.episode_id != previous.episode_id:
                raise RuntimeError("episode changed without finish_episode/reset")
            if frame.episode_step != previous.episode_step + 1:
                raise RuntimeError("episode steps must be contiguous")
        self._seen_identities.add(frame.identity)
        buffer.append(frame)

    def finish_episode(self, environment_id: int, *, fell: bool) -> NaturalFallEvent | None:
        environment_id = int(environment_id)
        if environment_id not in self._buffers:
            raise KeyError(f"unknown environment id {environment_id}")
        if self._terminal[environment_id]:
            raise RuntimeError("episode terminal was already recorded")
        buffer = self._buffers[environment_id]
        if not buffer:
            raise RuntimeError("cannot terminate an empty episode")
        self._terminal[environment_id] = True
        if not fell:
            return None
        self.independent_fall_episodes += 1
        terminal = buffer[-1]
        prefall: list[NaturalPpoFrame | None] = []
        frames = tuple(buffer)
        for offset in PREFALL_OFFSETS:
            index = len(frames) - 1 - offset
            prefall.append(frames[index] if index >= 0 else None)
        event = NaturalFallEvent(terminal, frames, tuple(prefall))
        self.recorded_falls += 1
        return event

    def reset(self, environment_id: int) -> None:
        environment_id = int(environment_id)
        if environment_id not in self._buffers:
            raise KeyError(f"unknown environment id {environment_id}")
        if not self._terminal[environment_id]:
            raise RuntimeError("reset is allowed only after finish_episode")
        self._buffers[environment_id].clear()
        self._terminal[environment_id] = False

    def assert_complete(self) -> None:
        if self.recorded_falls != self.independent_fall_episodes:
            raise RuntimeError("recorded fall count drifted from terminal fall episodes")


class DeterministicNormalReservoir:
    """Keep outcome-blind, hash-ranked normal candidates by source stratum."""

    def __init__(self) -> None:
        self._values: dict[str, list[tuple[str, NaturalPpoFrame]]] = defaultdict(list)
        self._used: set[str] = set()

    def add_episode(self, frames: Iterable[NaturalPpoFrame], *, fell: bool) -> None:
        values = tuple(frames)
        terminal_step = values[-1].episode_step if values else -1
        for frame in values:
            distance = terminal_step - frame.episode_step
            if fell and distance <= NORMAL_TERMINAL_DISTANCE:
                continue
            rank = hashlib.sha256(
                b"qsafe.natural_normal_rank.v1\0" + frame.identity.encode("ascii")
            ).hexdigest()
            self._values[frame.stratum].append((rank, frame))

    def match(self, frame: NaturalPpoFrame) -> NaturalPpoFrame | None:
        values = self._values.get(frame.stratum, [])
        for _, candidate in sorted(values, key=lambda item: item[0]):
            if candidate.identity not in self._used:
                self._used.add(candidate.identity)
                return candidate
        return None


class NaturalFallShardWriter:
    """Atomically publish fixed-size compressed fall-event shards."""

    def __init__(self, root: str | Path, *, events_per_shard: int = 1024) -> None:
        self.root = Path(root)
        if events_per_shard <= 0:
            raise ValueError("events_per_shard must be positive")
        self.events_per_shard = int(events_per_shard)
        self._pending: list[NaturalFallEvent] = []
        self._shards: list[dict[str, Any]] = []
        self._closed = False

    def add(self, event: NaturalFallEvent) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        self._pending.append(event)
        if len(self._pending) == self.events_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        index = len(self._shards)
        path = self.root / f"falls-{index:06d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
        events = tuple(self._pending)
        reference = events[0].terminal
        dimensions = {
            "integration_state": reference.integration_state.shape,
            "qpos": reference.qpos.shape,
            "qvel": reference.qvel.shape,
            "act": reference.act.shape,
            "ctrl": reference.ctrl.shape,
        }
        for event in events:
            for frame in event.trajectory:
                for name, shape in dimensions.items():
                    if getattr(frame, name).shape != shape:
                        raise ValueError(f"{name} shape drifted within a shard")

        def trajectory_array(name: str, dtype: Any) -> np.ndarray:
            shape = dimensions.get(name, getattr(reference, name).shape)
            result = np.zeros((len(events), RING_POLICY_STEPS, *shape), dtype=dtype)
            for event_index, event in enumerate(events):
                for frame_index, frame in enumerate(event.trajectory):
                    result[event_index, frame_index] = getattr(frame, name)
            return result

        def prefall_array(name: str, dtype: Any) -> np.ndarray:
            shape = dimensions.get(name, getattr(reference, name).shape)
            result = np.zeros((len(events), len(PREFALL_OFFSETS), *shape), dtype=dtype)
            for event_index, event in enumerate(events):
                for offset_index, frame in enumerate(event.prefall):
                    if frame is not None:
                        result[event_index, offset_index] = getattr(frame, name)
            return result

        np.savez_compressed(
            temporary,
            terminal_identity=np.asarray(
                [event.terminal.identity for event in events], dtype="S64"),
            environment_id=np.asarray(
                [event.terminal.environment_id for event in events], dtype=np.int32),
            episode_id=np.asarray(
                [event.terminal.episode_id for event in events], dtype=np.int64),
            terminal_episode_step=np.asarray(
                [event.terminal.episode_step for event in events], dtype=np.int32),
            ppo_training_step=np.asarray(
                [event.terminal.ppo_training_step for event in events], dtype=np.int64),
            availability=np.stack([event.availability for event in events]),
            prefall_identity=np.asarray([
                [b"" if frame is None else frame.identity.encode("ascii")
                 for frame in event.prefall] for event in events], dtype="S64"),
            trajectory_length=np.asarray(
                [len(event.trajectory) for event in events], dtype=np.int16),
            trajectory_integration_state=trajectory_array(
                "integration_state", np.float64),
            trajectory_qpos=trajectory_array("qpos", np.float64),
            trajectory_qvel=trajectory_array("qvel", np.float64),
            trajectory_act=trajectory_array("act", np.float64),
            trajectory_ctrl=trajectory_array("ctrl", np.float64),
            trajectory_observation_history=trajectory_array(
                "observation_history", np.float32),
            trajectory_previous_action_requested=trajectory_array(
                "previous_action_requested", np.float32),
            trajectory_previous_action_executed=trajectory_array(
                "previous_action_executed", np.float32),
            trajectory_previous_action_q_target=trajectory_array(
                "previous_action_q_target", np.float32),
            trajectory_rng_identity=np.asarray([
                [frame.rng_identity for frame in event.trajectory]
                + [0] * (RING_POLICY_STEPS - len(event.trajectory))
                for event in events], dtype=np.uint64),
            trajectory_randomization_json=np.asarray([
                [_canonical_json(dict(frame.randomization)).encode("ascii")
                 for frame in event.trajectory]
                + [b""] * (RING_POLICY_STEPS - len(event.trajectory))
                for event in events], dtype="S4096"),
            prefall_integration_state=prefall_array(
                "integration_state", np.float64),
            prefall_qpos=prefall_array("qpos", np.float64),
            prefall_qvel=prefall_array("qvel", np.float64),
            prefall_act=prefall_array("act", np.float64),
            prefall_ctrl=prefall_array("ctrl", np.float64),
            prefall_observation_history=prefall_array(
                "observation_history", np.float32),
            prefall_previous_action_requested=prefall_array(
                "previous_action_requested", np.float32),
            prefall_previous_action_executed=prefall_array(
                "previous_action_executed", np.float32),
            prefall_previous_action_q_target=prefall_array(
                "previous_action_q_target", np.float32),
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
            content = stream.read()
        digest = hashlib.sha256(content).hexdigest()
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        self._shards.append({
            "path": path.name,
            "sha256": digest,
            "event_count": len(events),
        })
        self._pending.clear()

    def close(self, *, provenance: Mapping[str, Any]) -> Path:
        if self._closed:
            raise RuntimeError("writer is already closed")
        self._flush()
        manifest = {
            "schema_version": "qsafe.natural_ppo_fall_archive.v1",
            "ppo_outcomes_are_qsafe_labels": False,
            "external_force": "forbidden",
            "prefall_offsets": list(PREFALL_OFFSETS),
            "event_count": sum(item["event_count"] for item in self._shards),
            "shards": self._shards,
            "provenance": dict(provenance),
        }
        path = self.root / "manifest.json"
        _atomic_bytes(path, (_canonical_json(manifest) + "\n").encode("utf-8"))
        self._closed = True
        return path
