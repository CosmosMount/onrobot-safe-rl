"""Create the three immutable protocol locks before claim-bearing runs."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys

from ..protocol import create_protocol_lock, sha256_file


ROOT = Path(__file__).resolve().parents[3]


def _source_files() -> list[Path]:
    package = ROOT / "reproductions/ppo_sqrl_go2"
    return sorted(path for path in package.rglob("*")
                  if path.is_file() and path.suffix in {".py", ".yaml", ".md"})


def _external_hashes(phase: str) -> dict[str, str]:
    paths = [
        ROOT / "safety_data/mjlab_target_alignment.py",
        ROOT / "safety_data/mjlab_natural_falls.py",
        ROOT / "safety_data/native.py",
        ROOT / "train/config.py",
        ROOT / "train/mujoco_snapshot_env.py",
        Path("/home/xyz/code/unitree_mujoco/unitree_robots/go2/scene_empty.xml"),
        Path("/home/xyz/Desktop/xluo/unitree_rl_mjlab/src/tasks/velocity/config/go2/env_cfgs.py"),
        Path("/home/xyz/Desktop/xluo/unitree_rl_mjlab/src/tasks/velocity/config/go2/rl_cfg.py"),
        ROOT / "saved/qsafe_development/natural_ppo/capacity-030-target-aligned/capacity-authorization-v1.json",
        ROOT / "saved/qsafe_development/natural_ppo/parity/mjlab-target-model-contract-v1.json",
        ROOT / "saved/qsafe_development/natural_ppo/parity/mjlab-native-target-aligned-validation-seed137-v1.json",
        Path(sys.executable),
    ]
    for package in ("torch", "numpy", "mujoco", "mujoco-mjx", "mjlab", "rsl-rl-lib"):
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        for filename in ("METADATA", "RECORD"):
            candidate = Path(distribution._path) / filename
            if candidate.is_file():
                paths.append(candidate)
    if phase == "mechanism":
        paths.extend([
            ROOT / "saved/reproductions/sqrl_go2/formal_v2/formal_protocol_lock.json",
            ROOT / "reproductions/sqrl_go2/config/target_040.yaml",
        ])
    return {str(path.resolve()): sha256_file(path) for path in paths}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("mechanism", "cotrain", "target"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    files = _source_files()
    create_protocol_lock(
        args.output, protocol_id=f"ppo_sqrl_go2_{args.phase}_v1",
        files=files, external_hashes=_external_hashes(args.phase))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
