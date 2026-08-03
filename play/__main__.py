"""Run deterministic policy rollouts from a saved checkpoint.

Example::

    python -m play --agent droq \
        --checkpoint saved/checkpoints_46d/step_000000005000
"""

from __future__ import annotations

import sys

from train.main import main


def main_play(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(argument == "--mode" or argument.startswith("--mode=") for argument in args):
        raise SystemExit("python -m play already selects --mode play; remove --mode")
    return main(["--mode", "play", *args])


if __name__ == "__main__":
    raise SystemExit(main_play())
