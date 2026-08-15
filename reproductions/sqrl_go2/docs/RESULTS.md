# Development results

Status: the frozen development protocol passed. These runs are a semantic and
engineering gate, not a formal multi-seed reproduction claim.

## Frozen protocol

- Pre-training: 25,000 environment steps at 0.30 m/s, seeds 0, 1, and 2.
- Target audit: 10,000 environment steps at 0.40 m/s, seed 0.
- All target branches start from the exact seed-0 pre-trained actor.
- Both SQRL branches start from the exact seed-0 Q_safe; target Q_safe is frozen.
- Task critics, entropy temperature, and task replay start fresh in each target branch.

The machine-readable audit is produced by:

```bash
python3 -m reproductions.sqrl_go2.diagnostics.development_gate \
  --output saved/reproductions/sqrl_go2/development_gate.json
```

All eight gate checks passed: run completeness, task learning, safety signal,
non-degenerate masks, weight lineage, finite metrics, and execution of the full
branch's dual update path.

## Pre-training screening

| Seed | Total falls | Falls in final 5k | Final-5k mean reward | Final-5k tracking error | Mask acceptance | No-safe-candidate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50 | 0 | 4.790 | 0.099 | 0.910 | 0.090 |
| 1 | 35 | 0 | 5.061 | 0.090 | 0.378 | 0.622 |
| 2 | 12 | 0 | 4.376 | 0.106 | 0.105 | 0.895 |

Every seed observed terminal falls in `D_safe`, performed Q_safe updates, and
ended with no falls in its final 5,000 steps. The large cross-seed variation in
mask acceptance is a reason to require paired multi-seed formal experiments
before making an efficacy claim.

## Paired target semantic audit (seed 0)

| Branch | Falls | Falls / 1k | Final-5k mean reward | Final-5k tracking error | Mask acceptance | No-safe-candidate |
|---|---:|---:|---:|---:|---:|---:|
| SAC-transfer | 48 | 4.8 | 4.726 | 0.092 | — | — |
| SQRL-mask | 10 | 1.0 | 5.331 | 0.109 | 0.971 | 0.029 |
| SQRL-full | 16 | 1.6 | 5.668 | 0.101 | 0.935 | 0.065 |

Relative to SAC-transfer, the observed fall reductions are 79.2% for SQRL-mask
and 66.7% for SQRL-full. These percentages describe this one development seed
only; they are not confidence-bounded estimates.

The full branch executed 9,001 dual updates. Its actor constraint violation was
always negative (range -1.063 to -0.609), so projected `nu` correctly remained
zero. Thus this audit validates the Lagrangian chain mechanically, but does not
show an active Lagrangian penalty. The target safety benefit in this seed is
therefore attributable primarily to behavior-time masking.

## Interpretation boundary

The development gate establishes that the isolated implementation runs end to
end and that none of the three target branches is degenerate. It does not
establish statistical significance, reproduce the paper's original robot/task,
or justify a formal SQRL performance claim. Reward, tracking error, and actual
speed are diagnostics rather than gates. The locked formal-v1 study uses the
now-frozen 25k/10k budgets and paired seeds 10 through 19 without tuning on
formal outcomes.
