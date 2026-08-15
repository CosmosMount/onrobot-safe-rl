"""Serial, locked execution of the ten-seed SQRL-Go2 formal protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from ..diagnostics.formal_results import publish
from ..formal_protocol import (
    CONTROLLER,
    CONTROLLER_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    FORMAL_SEEDS,
    PRETRAIN_STEPS,
    PROTOCOL_ID,
    REPO_ROOT,
    SIMULATOR,
    TARGET_STEPS,
    branch_order,
    verify_lock,
)


RETRY_REASONS = (
    "process_crash", "machine_interruption", "corrupt_output",
)


def _append_ledger(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _run_key(seed: int, name: str) -> str:
    return f"seed_{seed}/{name}"


def _run_directory(root: Path, seed: int, name: str) -> Path:
    return root / f"seed_{seed}" / name


def _is_complete(directory: Path, *, steps: int, phase: str,
                 seed: int, branch: str | None) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        line_count = sum(1 for line in (directory / "metrics.jsonl").open(
            encoding="utf-8") if line.strip())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    expected = (
        manifest.get("status") == "finished"
        and manifest.get("phase") == phase
        and manifest.get("seed") == seed
        and manifest.get("completed_steps") == steps
        and manifest.get("protocol_id") == PROTOCOL_ID
        and line_count == steps)
    if branch is not None:
        expected = expected and manifest.get("branch") == branch
    return bool(expected)


def _archive_failed(root: Path, directory: Path, key: str, reason: str) -> Path:
    attempts = root / "attempts" / key
    attempts.mkdir(parents=True, exist_ok=True)
    index = len(list(attempts.glob("attempt_*"))) + 1
    destination = attempts / f"attempt_{index:03d}_{reason}"
    shutil.move(str(directory), str(destination))
    return destination


def _matching_processes() -> list[str]:
    matches: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            values = [item.decode("utf-8", errors="replace") for item in argv if item]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not values:
            continue
        executable = Path(values[0]).resolve()
        is_runtime = any(
            values[index:index + 2] == ["-m", "runtime.inference"]
            for index in range(len(values) - 1))
        if executable in {SIMULATOR.resolve(), CONTROLLER.resolve()} or is_runtime:
            matches.append(f"{entry.name} {' '.join(values)}")
    return matches


def _terminate(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5.0)


def _launch_stack(log_dir: Path) -> tuple[subprocess.Popen[Any], subprocess.Popen[Any]]:
    existing = _matching_processes()
    if existing:
        raise RuntimeError(f"formal stack requires exclusive ownership; found: {existing}")
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.setdefault("DISPLAY", ":1")
    environment.setdefault("XAUTHORITY", str(Path.home() / ".Xauthority"))
    simulator_log = (log_dir / "simulator.log").open("wb")
    controller_log = (log_dir / "controller.log").open("wb")
    simulator = subprocess.Popen([
        str(SIMULATOR), "-r", "go2", "-s", "scene_empty.xml",
        "-i", "1", "-n", "lo",
    ], cwd=SIMULATOR.parent, env=environment, stdout=simulator_log,
       stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3.0)
    if simulator.poll() is not None:
        raise RuntimeError(f"simulator exited during startup: {simulator.returncode}")
    controller = subprocess.Popen([
        str(CONTROLLER), str(CONTROLLER_CONFIG),
    ], cwd=REPO_ROOT, env=environment, stdout=controller_log,
       stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3.0)
    if controller.poll() is not None:
        _terminate(simulator)
        raise RuntimeError(f"controller exited during startup: {controller.returncode}")
    return simulator, controller


def _execute(command: list[str], log_path: Path) -> int:
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            return process.wait()
        except BaseException:
            _terminate(process)
            raise


def _command(root: Path, device: str, seed: int,
             name: str, branch: str | None) -> list[str]:
    common = [
        "--seed", str(seed), "--device", device,
        "--output-root", str(root), "--protocol-id", PROTOCOL_ID,
        "--launch-runtime",
    ]
    if branch is None:
        return [
            sys.executable, "-m", "reproductions.sqrl_go2.runners.run_pretrain",
            "--config", "reproductions/sqrl_go2/config/pretrain_030.yaml",
            "--steps", str(PRETRAIN_STEPS), *common,
        ]
    checkpoint = root / f"seed_{seed}/pretrain_030/final.pt"
    return [
        sys.executable, "-m", "reproductions.sqrl_go2.runners.run_target",
        "--config", "reproductions/sqrl_go2/config/target_040.yaml",
        "--pretrain-checkpoint", str(checkpoint),
        "--branch", branch, "--steps", str(TARGET_STEPS), *common,
    ]


def _jobs() -> list[tuple[int, str, str | None, int, str]]:
    jobs: list[tuple[int, str, str | None, int, str]] = []
    for seed in FORMAL_SEEDS:
        jobs.append((seed, "pretrain_030", None, PRETRAIN_STEPS, "pretrain"))
        jobs.extend((
            seed, f"target_040_{branch}", branch, TARGET_STEPS, "target")
            for branch in branch_order(seed))
    return jobs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-run", default=None,
                        help="Exact failed key, e.g. seed_10/pretrain_030")
    parser.add_argument("--retry-reason", choices=RETRY_REASONS)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    lock_path = root / "formal_protocol_lock.json"
    verify_lock(lock_path)
    if (args.retry_run is None) != (args.retry_reason is None):
        raise SystemExit("--retry-run and --retry-reason must be supplied together")
    ledger = root / "campaign_ledger.jsonl"
    jobs = _jobs()
    for position, (seed, name, branch, steps, phase) in enumerate(jobs, start=1):
        verify_lock(lock_path)
        key = _run_key(seed, name)
        directory = _run_directory(root, seed, name)
        if directory.exists():
            if _is_complete(
                    directory, steps=steps, phase=phase,
                    seed=seed, branch=branch):
                if not args.resume:
                    raise FileExistsError(
                        f"completed run exists; use --resume to verify and skip: {key}")
                print(f"[formal {position}/{len(jobs)}] verified complete: {key}", flush=True)
                continue
            if args.retry_run != key:
                raise RuntimeError(
                    f"incomplete/failed run requires explicit --retry-run {key} "
                    "and --retry-reason")
            archived = _archive_failed(root, directory, key, args.retry_reason)
            _append_ledger(ledger, {
                "event": "retry_authorized", "run": key,
                "reason": args.retry_reason, "archived_to": str(archived),
                "time_unix": time.time(),
            })
            args.retry_run = None
            args.retry_reason = None
        print(f"[formal {position}/{len(jobs)}] starting: {key}", flush=True)
        attempt_log = root / "orchestrator_logs" / key
        if attempt_log.exists():
            suffix = len(list(attempt_log.parent.glob(f"{attempt_log.name}_attempt*"))) + 1
            attempt_log = attempt_log.with_name(f"{attempt_log.name}_attempt{suffix:03d}")
        _append_ledger(ledger, {
            "event": "started", "run": key, "position": position,
            "time_unix": time.time(),
        })
        simulator = controller = None
        try:
            simulator, controller = _launch_stack(attempt_log)
            returncode = _execute(
                _command(root, args.device, seed, name, branch),
                attempt_log / "runner.log")
        finally:
            _terminate(controller)
            _terminate(simulator)
            time.sleep(2.0)
        if returncode != 0 or not _is_complete(
                directory, steps=steps, phase=phase,
                seed=seed, branch=branch):
            _append_ledger(ledger, {
                "event": "execution_failed", "run": key,
                "returncode": returncode, "time_unix": time.time(),
            })
            raise RuntimeError(
                f"formal execution failed for {key}; inspect {attempt_log}")
        _append_ledger(ledger, {
            "event": "finished", "run": key,
            "time_unix": time.time(),
        })
        print(f"[formal {position}/{len(jobs)}] finished: {key}", flush=True)
    verify_lock(lock_path)
    report = root / "report"
    if report.exists():
        raise FileExistsError(f"formal report already exists: {report}")
    result = publish(root, report)
    print(json.dumps(result["flags"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
