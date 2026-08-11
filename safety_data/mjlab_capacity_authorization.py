"""Compile immutable MuJoCo-Warp capacity measurements into one authorization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


CAPACITY_TIERS = (256, 512, 1024, 2048)
MINIMUM_UPGRADE_GAIN = 0.15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"capacity report must be a JSON object: {path}")
    return value


def _validate_report(
    value: dict[str, Any], *, envs: int, minimum_seconds: float,
) -> None:
    if value.get("schema_version") != "qsafe.mjlab_go2_capacity.v1":
        raise ValueError("capacity report schema mismatch")
    if value.get("envs") != envs or value.get("pass") is not True:
        raise ValueError(f"capacity tier {envs} did not pass")
    if float(value.get("elapsed_seconds", -1.0)) < minimum_seconds:
        raise ValueError(f"capacity tier {envs} is shorter than required")
    if value.get("generator_worktree_clean_at_launch") is not True:
        raise ValueError("formal capacity generator was dirty at launch")
    if value.get("external_force_nonzero") is not False or (
            value.get("push_event_present") is not False):
        raise ValueError("capacity report contains an external perturbation")
    if value.get("nonfinite") is not False or value.get("gpu_sampling_error") is not None:
        raise ValueError("capacity report contains runtime integrity failure")
    if int(value.get("peak_total_gpu_memory_mib", 1 << 30)) > 20480:
        raise ValueError("capacity report exceeds the 20-GiB VRAM gate")
    if float(value.get("memory_growth_mib", 1e9)) > 128.0:
        raise ValueError("capacity report exceeds the resource-growth gate")
    if float(value.get("policy_env_steps_per_second", 0.0)) <= 0.0:
        raise ValueError("capacity report has no positive throughput")


def compile_capacity_authorization(
    five_minute_reports: Iterable[str | Path],
    stability_report: str | Path,
    *,
    production_envs: int = 2000,
) -> dict[str, Any]:
    paths = tuple(Path(path) for path in five_minute_reports)
    if len(paths) != len(CAPACITY_TIERS):
        raise ValueError("exactly four capacity-tier reports are required")
    reports = tuple(_load(path) for path in paths)
    for tier, report in zip(CAPACITY_TIERS, reports, strict=True):
        _validate_report(report, envs=tier, minimum_seconds=300.0)

    commits = {str(report.get("generator_commit")) for report in reports}
    contracts = {
        str(report.get("target_alignment", {}).get("contract_sha256"))
        for report in reports
    }
    versions = {
        json.dumps(report.get("versions"), sort_keys=True)
        for report in reports
    }
    if len(commits) != 1 or len(contracts) != 1 or len(versions) != 1 or (
            None in commits or "None" in commits or "None" in contracts):
        raise ValueError("capacity ladder provenance is not identical")

    throughput = [float(report["policy_env_steps_per_second"]) for report in reports]
    gains = [
        throughput[index] / throughput[index - 1] - 1.0
        for index in range(1, len(throughput))
    ]
    selected = CAPACITY_TIERS[0]
    for tier, gain in zip(CAPACITY_TIERS[1:], gains, strict=True):
        if gain >= MINIMUM_UPGRADE_GAIN:
            selected = tier
        else:
            break
    if production_envs <= 0 or production_envs > selected:
        raise ValueError("production_envs must be positive and no larger than selected capacity")

    stability_path = Path(stability_report)
    stability = _load(stability_path)
    _validate_report(stability, envs=selected, minimum_seconds=1800.0)
    if str(stability.get("generator_commit")) not in commits or str(
            stability.get("target_alignment", {}).get("contract_sha256")) not in contracts:
        raise ValueError("stability report provenance differs from the capacity ladder")
    if json.dumps(stability.get("versions"), sort_keys=True) not in versions:
        raise ValueError("stability report backend versions differ from the capacity ladder")

    return {
        "schema_version": "qsafe.mjlab_capacity_authorization.v1",
        "authorized": True,
        "generator_commit": next(iter(commits)),
        "target_alignment_contract_sha256": next(iter(contracts)),
        "backend_versions": json.loads(next(iter(versions))),
        "tiers": [
            {
                "envs": tier,
                "throughput_policy_env_steps_per_second": speed,
                "peak_total_gpu_memory_mib": int(report["peak_total_gpu_memory_mib"]),
                "elapsed_seconds": float(report["elapsed_seconds"]),
                "report": str(path),
                "report_sha256": _sha256(path),
            }
            for tier, speed, report, path in zip(
                CAPACITY_TIERS, throughput, reports, paths, strict=True)
        ],
        "upgrade_gains": [
            {"from_envs": before, "to_envs": after, "gain_fraction": gain}
            for before, after, gain in zip(
                CAPACITY_TIERS[:-1], CAPACITY_TIERS[1:], gains, strict=True)
        ],
        "minimum_upgrade_gain_fraction": MINIMUM_UPGRADE_GAIN,
        "selected_capacity_envs": selected,
        "production_envs": int(production_envs),
        "production_envs_reason": "exact_250k_exposure_iterations_with_125_rollout_steps",
        "stability": {
            "report": str(stability_path),
            "report_sha256": _sha256(stability_path),
            "elapsed_seconds": float(stability["elapsed_seconds"]),
            "throughput_policy_env_steps_per_second": float(
                stability["policy_env_steps_per_second"]),
            "peak_total_gpu_memory_mib": int(stability["peak_total_gpu_memory_mib"]),
            "memory_growth_mib": float(stability["memory_growth_mib"]),
            "mean_gpu_utilization_percent": float(
                stability["mean_gpu_utilization_percent"]),
        },
    }


def publish_capacity_authorization(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite capacity authorization: {destination}") from exc
    temporary.unlink()
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CAPACITY_TIERS", "MINIMUM_UPGRADE_GAIN", "compile_capacity_authorization",
    "publish_capacity_authorization",
]
