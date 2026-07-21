"""Plot the exact random-action distribution used before SAC training starts."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare


JOINT_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)


def sample_actions(seed: int, count: int, scale: float) -> np.ndarray:
    """Mirror BaseEnv.sample_action and train.loop's exploration scaling."""
    rng = jax.random.PRNGKey(seed)
    actions = np.empty((count, len(JOINT_NAMES)), dtype=np.float32)
    for step in range(count):
        rng, key = jax.random.split(rng)
        actions[step] = np.asarray(
            jax.random.uniform(
                key, shape=(len(JOINT_NAMES),), minval=-1.0, maxval=1.0
            ),
            dtype=np.float32,
        ) * scale
    return np.clip(actions, -1.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("train/artifacts/random_action_distribution.png"),
    )
    args = parser.parse_args()

    actions = sample_actions(args.seed, args.samples, args.scale)
    limit = min(abs(args.scale), 1.0)
    edges = np.linspace(-limit, limit, args.bins + 1)

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), sharex=True, sharey=True)
    rows = []
    for joint, (name, ax) in enumerate(zip(JOINT_NAMES, axes.flat)):
        counts, _, _ = ax.hist(
            actions[:, joint], bins=edges, color="#3977b8", edgecolor="white"
        )
        expected = np.full(args.bins, args.samples / args.bins)
        chi2, p_value = chisquare(counts, expected)
        rows.append((name, actions[:, joint].mean(), actions[:, joint].std(),
                     int(counts.min()), int(counts.max()), p_value))
        ax.axhline(args.samples / args.bins, color="#d1495b", linestyle="--",
                   linewidth=1)
        ax.set_title(f"{name}  p={p_value:.3f}")
        ax.grid(alpha=0.2)

    fig.supxlabel("Normalized action")
    fig.supylabel("Count per bin")
    fig.suptitle(
        f"SAC random exploration: {args.samples} samples/joint, "
        f"{args.bins} bins, seed={args.seed}, scale={args.scale}\n"
        "Red dashed line = expected count under a uniform distribution"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")

    csv_path = args.output.with_suffix(".csv")
    np.savetxt(
        csv_path,
        actions,
        delimiter=",",
        header=",".join(JOINT_NAMES),
        comments="",
    )
    print("joint,mean,std,min_bin_count,max_bin_count,chi_square_p")
    for row in rows:
        print(f"{row[0]},{row[1]:.6f},{row[2]:.6f},"
              f"{row[3]},{row[4]},{row[5]:.6f}")
    print(f"plot={args.output}")
    print(f"samples={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
