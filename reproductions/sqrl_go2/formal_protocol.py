"""Immutable contract and executable hashing for the SQRL-Go2 formal study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_ID = "sqrl_go2_formal_v2"
FORMAL_SEEDS = tuple(range(10, 20))
PRETRAIN_STEPS = 25_000
TARGET_STEPS = 10_000
BRANCHES = ("sac_transfer", "sqrl_mask", "sqrl_full")
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 20_260_813
NU_ACTIVE_THRESHOLD = 1e-8
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "saved/reproductions/sqrl_go2/formal_v2"

SIMULATOR = Path(
    "/home/xyz/code/unitree_mujoco/simulate/build/unitree_mujoco")
SIMULATOR_MODEL = Path(
    "/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml")
CONTROLLER = REPO_ROOT / "runtime/control/go2/build/go2_control"
CONTROLLER_CONFIG = REPO_ROOT / "runtime/control/go2/go2.yaml"


def branch_order(seed: int) -> tuple[str, ...]:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"seed {seed} is not in the frozen formal roster")
    orders = (
        ("sac_transfer", "sqrl_mask", "sqrl_full"),
        ("sqrl_mask", "sqrl_full", "sac_transfer"),
        ("sqrl_full", "sac_transfer", "sqrl_mask"),
    )
    return orders[(seed - FORMAL_SEEDS[0]) % len(orders)]


def protocol_payload() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "amends_protocol": "sqrl_go2_formal_v1",
        "amendment_reason": (
            "formal-v1 invalidated: runtime startup incorrectly required the "
            "action shared-memory mailbox to survive an earlier process session"),
        "formal_seeds": list(FORMAL_SEEDS),
        "pretrain_steps": PRETRAIN_STEPS,
        "target_steps": TARGET_STEPS,
        "pretrain_command_mps": 0.30,
        "target_command_mps": 0.40,
        "branches": list(BRANCHES),
        "branch_orders": {str(seed): list(branch_order(seed)) for seed in FORMAL_SEEDS},
        "primary_comparison": "sac_transfer_minus_sqrl_full",
        "speed_reward_tracking_are_gating": False,
        "bootstrap": {
            "unit": "paired_seed",
            "rng": "numpy.PCG64",
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "quantile_method": "linear",
            "two_sided_ci_quantiles": [0.025, 0.975],
            "one_sided_lcb_quantile": 0.05,
        },
        "gates": {
            "formal_go2_sqrl_reproduced": {
                "mean_paired_reduction_gt": 0.0,
                "one_sided_lcb_gt": 0.0,
                "positive_seeds_at_least": 8,
                "pooled_relative_reduction_at_least": 0.30,
            },
            "sqrl_masking_effect_supported": {
                "one_sided_lcb_gt": 0.0,
                "pooled_relative_reduction_at_least": 0.30,
            },
            "sqrl_lagrangian_effect_supported": {
                "full_vs_mask_one_sided_lcb_gt": 0.0,
                "nu_active_threshold": NU_ACTIVE_THRESHOLD,
                "nu_active_seeds_at_least": 2,
            },
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executable_paths() -> tuple[Path, ...]:
    paths = set((REPO_ROOT / "reproductions/sqrl_go2").rglob("*.py"))
    paths.update((REPO_ROOT / "reproductions/sqrl_go2/config").glob("*.yaml"))
    paths.update((REPO_ROOT / "runtime/inference").rglob("*.py"))
    paths.update({
        REPO_ROOT / "train/config.py",
        REPO_ROOT / "train/ordered_runtime.py",
        CONTROLLER,
        CONTROLLER_CONFIG,
        SIMULATOR,
        SIMULATOR_MODEL,
    })
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"formal executable inputs are missing: {missing}")
    return tuple(sorted(paths, key=lambda path: str(path)))


def _lock_name(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_lock() -> dict[str, Any]:
    files = {_lock_name(path): _sha256(path) for path in executable_paths()}
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return {
        "schema_version": "sqrl_go2.formal_protocol_lock.v1",
        "protocol": protocol_payload(),
        "executable_files": files,
        "executable_bundle_sha256": digest.hexdigest(),
    }


def write_lock(path: Path) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal lock: {path}")
    lock = build_lock()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def verify_lock(path: Path) -> dict[str, Any]:
    stored = json.loads(path.read_text(encoding="utf-8"))
    current = build_lock()
    if stored != current:
        stored_files = stored.get("executable_files", {})
        current_files = current["executable_files"]
        changed = sorted(
            name for name in set(stored_files) | set(current_files)
            if stored_files.get(name) != current_files.get(name))
        raise RuntimeError(f"formal executable lock drifted: {changed}")
    return stored
