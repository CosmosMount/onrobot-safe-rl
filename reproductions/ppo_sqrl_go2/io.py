"""Append-only metrics, manifests, and failure ledgers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping


def append_jsonl(path: str | Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_no_clobber(path: str | Path, value: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
               + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)
