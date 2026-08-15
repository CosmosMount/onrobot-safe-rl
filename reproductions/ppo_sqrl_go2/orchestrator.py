"""Serial, fail-closed orchestrator for the three preregistered stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

from .io import append_jsonl
from .protocol import (
    AUDIT_SEEDS, COTRAIN_SEEDS, TARGET_SEEDS, Protocol,
    target_branch_order, verify_protocol_lock,
)
from .reporting import (
    summarize_cotrain, summarize_mechanism, summarize_target,
    write_campaign_charts, write_final_report,
)


REPOSITORY = Path(__file__).resolve().parents[2]
SYSTEM_PYTHON = Path("/usr/bin/python3")
MJLAB_PYTHON = Path("/home/xyz/micromamba/envs/safesac/bin/python")
FORMAL_ROOT = REPOSITORY / "saved/reproductions/sqrl_go2/formal_v2"


class Campaign:
    def __init__(self, root: Path, *, resume: bool, retry_failed: bool,
                 device: str):
        self.root = root
        self.resume = resume
        self.retry_failed = retry_failed
        self.device = device
        self.logs = root / "process_logs"
        self.ledger = root / "campaign_ledger.jsonl"
        self.locks = {
            phase: root / "locks" / f"{phase}_v1.lock.json"
            for phase in ("mechanism", "cotrain", "target")
        }

    def initialize(self) -> None:
        if self.resume:
            if not self.root.is_dir():
                raise FileNotFoundError("resume campaign root does not exist")
        else:
            self.root.mkdir(parents=True, exist_ok=False)
        self.logs.mkdir(exist_ok=True)
        (self.root / "locks").mkdir(exist_ok=True)
        runtimes = {
            "mechanism": SYSTEM_PYTHON,
            "cotrain": MJLAB_PYTHON,
            "target": MJLAB_PYTHON,
        }
        for phase, lock in self.locks.items():
            if lock.exists():
                verify_protocol_lock(lock)
                continue
            self._execute(
                runtimes[phase], f"create_lock_{phase}",
                lock.parent / f".{phase}.lock-run",
                ["-m", "reproductions.ppo_sqrl_go2.runners.create_locks",
                 "--phase", phase, "--output", str(lock)],
                output_is_run=False,
            )

    def _archive_failed(self, output: Path) -> None:
        index = 1
        while True:
            archived = output.with_name(f"{output.name}.failed_attempt_{index:03d}")
            if not archived.exists():
                output.rename(archived)
                append_jsonl(self.ledger, {
                    "event": "failed_attempt_archived", "source": str(output),
                    "archive": str(archived)})
                return
            index += 1

    def _execute(self, python: Path, label: str, output: Path,
                 arguments: Sequence[str], *, output_is_run: bool = True) -> None:
        manifest = output / "manifest.json" if output_is_run else None
        if manifest is not None and manifest.is_file():
            if not self.resume:
                raise FileExistsError(f"completed output is immutable: {output}")
            json.loads(manifest.read_text(encoding="utf-8"))
            return
        resume_args: list[str] = []
        if output_is_run and output.exists():
            failed = (output / "attempt_ledger.jsonl").exists()
            if failed:
                if not self.retry_failed:
                    raise RuntimeError(
                        f"failed run {output} requires --retry-failed after cause registration")
                self._archive_failed(output)
            elif self.resume:
                checkpoints = sorted(output.glob("step_*.pt"))
                if checkpoints:
                    resume_args = ["--resume-checkpoint", str(checkpoints[-1])]
                else:
                    raise RuntimeError(
                        f"incomplete run has no complete iteration checkpoint: {output}")
            else:
                raise FileExistsError(f"refusing to overwrite incomplete output {output}")
        log = self.logs / f"{label}.log"
        if log.exists():
            if not self.resume and not self.retry_failed:
                raise FileExistsError(log)
            suffix = 1
            while log.with_name(f"{log.stem}.attempt_{suffix:03d}.log").exists():
                suffix += 1
            log = log.with_name(f"{log.stem}.attempt_{suffix:03d}.log")
        command = [str(python), *arguments, *resume_args]
        append_jsonl(self.ledger, {
            "event": "process_started", "label": label,
            "output": str(output), "command": command})
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPOSITORY)
        with log.open("x", encoding="utf-8") as stream:
            result = subprocess.run(
                command, cwd=REPOSITORY, env=environment,
                stdout=stream, stderr=subprocess.STDOUT, check=False)
        append_jsonl(self.ledger, {
            "event": "process_finished", "label": label,
            "returncode": result.returncode, "log": str(log)})
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {log}")

    def mechanism(self) -> dict[str, object]:
        root = self.root / "mechanism_v1"
        root.mkdir(exist_ok=True)
        for seed in AUDIT_SEEDS:
            self._execute(
                SYSTEM_PYTHON, f"mechanism_seed_{seed}", root / f"seed_{seed}",
                ["-m", "reproductions.ppo_sqrl_go2.runners.run_audit",
                 "--seed", str(seed), "--formal-root", str(FORMAL_ROOT),
                 "--output", str(root / f"seed_{seed}"),
                 "--lock", str(self.locks["mechanism"])])
        summary = root / "summary.json"
        if summary.exists():
            return json.loads(summary.read_text(encoding="utf-8"))
        return summarize_mechanism(root, summary)

    def _pretrain(self, seed: int, root: Path, label: str) -> Path:
        output = root / f"seed_{seed}"
        self._execute(
            MJLAB_PYTHON, label, output,
            ["-m", "reproductions.ppo_sqrl_go2.runners.run_cotrain",
             "--seed", str(seed), "--output", str(output),
             "--lock", str(self.locks["cotrain"]), "--device", self.device])
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        return Path(manifest["final_checkpoint"])

    def cotrain(self) -> dict[str, object]:
        root = self.root / "cotrain_v1"
        root.mkdir(exist_ok=True)
        for seed in COTRAIN_SEEDS:
            self._pretrain(seed, root, f"cotrain_seed_{seed}")
        summary = root / "summary.json"
        if summary.exists():
            return json.loads(summary.read_text(encoding="utf-8"))
        return summarize_cotrain(root, summary)

    def _evaluate(self, *, seed: int, branch: str, exposure: int,
                  actor: Path, pretrain: Path, root: Path) -> None:
        for masked in (False, True):
            arm = "masked" if masked else "unmasked"
            output = root / f"step_{exposure:08d}" / arm
            arguments = [
                "-m", "reproductions.ppo_sqrl_go2.runners.evaluate",
                "--actor-checkpoint", str(actor),
                "--pretrain-checkpoint", str(pretrain),
                "--pretrain-lock", str(self.locks["cotrain"]),
                "--target-lock", str(self.locks["target"]),
                "--seed", str(seed), "--branch", branch,
                "--exposure", str(exposure), "--output", str(output),
                "--device", self.device,
            ]
            if masked:
                arguments.append("--masked")
            self._execute(
                MJLAB_PYTHON, f"eval_seed_{seed}_{branch}_{exposure}_{arm}",
                output, arguments)

    def target(self) -> dict[str, object]:
        pretrain_root = self.root / "target_pretrain_v1"
        target_root = self.root / "target_v1"
        evaluation_root = self.root / "target_evaluation_v1"
        pretrain_root.mkdir(exist_ok=True)
        target_root.mkdir(exist_ok=True)
        evaluation_root.mkdir(exist_ok=True)
        cfg = Protocol()
        for seed in TARGET_SEEDS:
            pretrain = self._pretrain(
                seed, pretrain_root, f"target_pretrain_seed_{seed}")
            for branch in target_branch_order(seed):
                branch_root = target_root / f"seed_{seed}" / branch
                self._evaluate(
                    seed=seed, branch=branch, exposure=0,
                    actor=pretrain, pretrain=pretrain,
                    root=evaluation_root / f"seed_{seed}" / branch)
                self._execute(
                    MJLAB_PYTHON, f"target_seed_{seed}_{branch}", branch_root,
                    ["-m", "reproductions.ppo_sqrl_go2.runners.run_target",
                     "--seed", str(seed), "--branch", branch,
                     "--pretrain-checkpoint", str(pretrain),
                     "--pretrain-lock", str(self.locks["cotrain"]),
                     "--target-lock", str(self.locks["target"]),
                     "--output", str(branch_root), "--device", self.device])
                for exposure in cfg.evaluation_exposures[1:]:
                    checkpoint = branch_root / f"step_{exposure:012d}.pt"
                    self._evaluate(
                        seed=seed, branch=branch, exposure=exposure,
                        actor=checkpoint, pretrain=pretrain,
                        root=evaluation_root / f"seed_{seed}" / branch)
        summary = target_root / "summary.json"
        if summary.exists():
            return json.loads(summary.read_text(encoding="utf-8"))
        return summarize_target(target_root, summary)

    def run(self, *, prepare_only: bool) -> dict[str, object] | None:
        self.initialize()
        if prepare_only:
            return None
        mechanism = self.mechanism()
        cotrain = self.cotrain()
        if bool(cotrain["ppo_sqrl_cotrain_stable"]):
            target = self.target()
        else:
            target = None
            append_jsonl(self.ledger, {
                "event": "target_stopped", "reason": "cotrain_stability_gate_failed"})
        final = self.root / "final_report"
        if final.exists():
            return json.loads((final / "final_results.json").read_text())
        charts = self.root / "charts"
        if not charts.exists():
            write_campaign_charts(self.root, charts)
        return write_final_report(
            mechanism=mechanism, cotrain=cotrain, target=target, output=final)
