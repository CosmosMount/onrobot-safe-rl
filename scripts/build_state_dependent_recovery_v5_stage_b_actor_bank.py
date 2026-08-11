#!/usr/bin/env python3
"""Prepare or compile the frozen V5 Stage-B SAC actor bank."""

from __future__ import annotations

import argparse
import json

from train.state_dependent_recovery_v5_stage_b_actor_bank import (
    PRODUCTION_ACTION_DIM,
    PRODUCTION_OBSERVATION_DIM,
    compile_actor_bank_manifest,
    compile_reduced7_actor_bank_manifest,
    prepare_actor_run_contracts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 14 fixed SAC-from-zero run contracts or compile their "
            "42 exact policy-only checkpoints without outcome filtering."))
    subparsers = parser.add_subparsers(dest="operation", required=True)

    def common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--supplement", required=True)
        target.add_argument(
            "--protocol",
            default="config/qsafe_state_dependent_recovery_v5.yaml")
        target.add_argument("--stage-a-report", required=True)
        target.add_argument(
            "--training-config",
            default="config/go2_50hz_sqrl_paper_sac_pretrain.yaml")
        target.add_argument("--actor-root", required=True)
        target.add_argument("--contracts-root", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="publish all 14 no-clobber run contracts")
    common(prepare)
    prepare.add_argument(
        "--generator-commit", required=True,
        help="full clean implementation commit bound to every actor run")

    compile_parser = subparsers.add_parser(
        "compile", help="validate all 42 identities and publish the bank")
    common(compile_parser)
    compile_parser.add_argument("--output", required=True)
    reduced = subparsers.add_parser(
        "compile-reduced7",
        help="compile the explicit pre-outcome seven-seed roster amendment",
    )
    reduced.add_argument("--amendment", required=True)
    reduced.add_argument("--training-config", required=True)
    reduced.add_argument("--actor-root", required=True)
    reduced.add_argument("--contracts-root", required=True)
    reduced.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "compile-reduced7":
        manifest, file_sha256 = compile_reduced7_actor_bank_manifest(
            amendment_path=args.amendment,
            training_config_path=args.training_config,
            actor_root=args.actor_root,
            contracts_root=args.contracts_root,
            output_path=args.output,
            observation_dim=PRODUCTION_OBSERVATION_DIM,
            action_dim=PRODUCTION_ACTION_DIM,
        )
        print(json.dumps({
            "operation": "compile-reduced7",
            "actor_bank_manifest": args.output,
            "actor_bank_manifest_file_sha256": file_sha256,
            "actor_bank_contract_sha256": manifest[
                "actor_bank_contract_sha256"],
            "identity_count": manifest["identity_count"],
        }, sort_keys=True, indent=2))
        return 0
    common = {
        "supplement_path": args.supplement,
        "protocol_path": args.protocol,
        "stage_a_report_path": args.stage_a_report,
        "training_config_path": args.training_config,
        "actor_root": args.actor_root,
        "contracts_root": args.contracts_root,
    }
    if args.operation == "prepare":
        outputs = prepare_actor_run_contracts(
            **common, generator_commit=args.generator_commit)
        print(json.dumps({
            "operation": "prepare",
            "run_contract_count": len(outputs),
            "run_contracts": [str(path) for path in outputs],
        }, sort_keys=True, indent=2))
        return 0
    manifest, file_sha256 = compile_actor_bank_manifest(
        **common,
        output_path=args.output,
        observation_dim=PRODUCTION_OBSERVATION_DIM,
        action_dim=PRODUCTION_ACTION_DIM,
    )
    print(json.dumps({
        "operation": "compile",
        "actor_bank_manifest": args.output,
        "actor_bank_manifest_file_sha256": file_sha256,
        "actor_bank_contract_sha256": manifest[
            "actor_bank_contract_sha256"],
        "identity_count": manifest["identity_count"],
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
