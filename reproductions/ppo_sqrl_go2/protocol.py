"""Frozen protocol values and immutable-lock helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from typing import Iterable, Mapping


AUDIT_SEEDS = (12, 16, 18, 10, 13)
POSITIVE_AUDIT_SEEDS = (12, 16, 18)
COTRAIN_SEEDS = (0, 1, 2)
TARGET_SEEDS = (10, 11, 12, 13, 14, 15)
TARGET_BRANCHES = ("ppo_transfer", "ppo_safe")


@dataclass(frozen=True)
class Protocol:
    task_envs: int = 1600
    safety_envs: int = 400
    target_envs: int = 2000
    rollout_steps: int = 125
    pretrain_task_transitions: int = 30_000_000
    pretrain_safety_transitions: int = 7_500_000
    target_transitions: int = 10_000_000
    pretrain_command: float = 0.30
    target_command: float = 0.40
    gamma_safe: float = 0.70
    epsilon_safe: float = 0.10
    qsafe_tau: float = 0.005
    qsafe_lr: float = 3e-4
    dual_lr: float = 3e-4
    mask_candidates: int = 100
    recent_safety_trajectories: int = 10
    safety_batch_size: int = 256
    audit_states_per_seed: int = 200
    audit_candidates: int = 16
    audit_replicas: int = 8
    audit_horizon: int = 96
    evaluation_episodes: int = 200
    evaluation_exposures: tuple[int, ...] = (
        0, 2_000_000, 4_000_000, 6_000_000, 8_000_000, 10_000_000)
    bootstrap_seed: int = 20260814
    bootstrap_replicates: int = 100_000
    final_safe_fraction_low: float = 0.005
    final_safe_fraction_high: float = 0.995

    def validate(self) -> None:
        if self.task_envs + self.safety_envs != 2000:
            raise ValueError("cotrain allocation must contain exactly 2000 environments")
        if self.pretrain_task_transitions % (self.task_envs * self.rollout_steps):
            raise ValueError("pretrain task budget must end on an iteration boundary")
        if self.pretrain_safety_transitions % (self.safety_envs * self.rollout_steps):
            raise ValueError("pretrain safety budget must end on an iteration boundary")
        task_iterations = self.pretrain_task_transitions // (
            self.task_envs * self.rollout_steps)
        safety_iterations = self.pretrain_safety_transitions // (
            self.safety_envs * self.rollout_steps)
        if task_iterations != safety_iterations:
            raise ValueError("task and safety budgets must describe the same iterations")
        if self.target_transitions % (self.target_envs * self.rollout_steps):
            raise ValueError("target budget must end on an iteration boundary")
        if not 0 < self.final_safe_fraction_low < self.final_safe_fraction_high < 1:
            raise ValueError("invalid non-degeneracy interval")

    @property
    def pretrain_iterations(self) -> int:
        self.validate()
        return self.pretrain_task_transitions // (self.task_envs * self.rollout_steps)

    @property
    def target_iterations(self) -> int:
        self.validate()
        return self.target_transitions // (self.target_envs * self.rollout_steps)


def target_branch_order(seed: int) -> tuple[str, str]:
    if seed not in TARGET_SEEDS:
        raise ValueError(f"seed {seed} is not a target seed")
    return TARGET_BRANCHES if seed % 2 == 0 else tuple(reversed(TARGET_BRANCHES))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_no_clobber(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
               + "\n").encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite protocol artifact {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def create_protocol_lock(
    output: str | Path,
    *,
    protocol_id: str,
    files: Iterable[str | Path],
    external_hashes: Mapping[str, str] | None = None,
    protocol: Protocol | None = None,
) -> dict[str, object]:
    cfg = protocol or Protocol()
    cfg.validate()
    entries = []
    for raw in sorted({str(Path(item).resolve()) for item in files}):
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append({"path": raw, "sha256": sha256_file(path)})
    payload: dict[str, object] = {
        "schema_version": "ppo_sqrl_go2.protocol_lock.v1",
        "protocol_id": protocol_id,
        "protocol": asdict(cfg),
        "files": entries,
        "external_hashes": dict(sorted((external_hashes or {}).items())),
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    payload["bundle_sha256"] = hashlib.sha256(encoded).hexdigest()
    _atomic_json_no_clobber(Path(output), payload)
    return payload


def verify_protocol_lock(path: str | Path) -> dict[str, object]:
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if lock.get("schema_version") != "ppo_sqrl_go2.protocol_lock.v1":
        raise ValueError("unknown protocol lock schema")
    digest = lock.pop("bundle_sha256", None)
    encoded = json.dumps(lock, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    if digest != hashlib.sha256(encoded).hexdigest():
        raise ValueError("protocol bundle hash changed")
    lock["bundle_sha256"] = digest
    for item in lock["files"]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"locked file changed: {item['path']}")
    for raw_path, expected in lock.get("external_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"locked external file changed: {path}")
    return lock


def reserve_output_directory(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=False)
    return result
