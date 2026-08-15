"""Atomic complete-iteration checkpoints and seed-specific lineage."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def save_checkpoint_no_clobber(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite checkpoint {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_complete_checkpoint(path: str | Path, *, expected_seed: int,
                             expected_protocol_bundle: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("complete_iteration") is not True:
        raise ValueError("resume checkpoint is not a complete iteration")
    if int(metadata.get("seed", -1)) != expected_seed:
        raise ValueError("resume checkpoint seed differs")
    if metadata.get("protocol_bundle_sha256") != expected_protocol_bundle:
        raise ValueError("resume checkpoint protocol differs")
    return payload


def verify_pretrain_lineage(payload: dict[str, Any], *, seed: int,
                            actor_hash: str | None = None,
                            safety_hash: str | None = None) -> None:
    metadata = payload.get("metadata", {})
    if metadata.get("phase") != "pretrain" or int(metadata.get("seed", -1)) != seed:
        raise ValueError("target branch did not receive its seed-specific pretrain")
    if actor_hash is not None and metadata.get("actor_sha256") != actor_hash:
        raise ValueError("pretrain actor lineage hash differs")
    if safety_hash is not None and metadata.get("safety_sha256") != safety_hash:
        raise ValueError("pretrain Q_safe lineage hash differs")
