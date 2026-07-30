# P16: 0.30 m/s SAC vs transferred Q_safe

Both primary runs used seed 42, a 50 Hz runtime, and exactly 15,000
policy transitions. Their initial actor and reward-critic hashes match. The
transferred Q_safe remained frozen for the full masking run.

| Group | Policy steps | Falls | Falls/1k | Episode fall rate | Avg return | Avg length | Avg velocity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q_safe logging source (diagnostic) | 3,689 | 11 | 2.982 | 0.550 | n/a | n/a | 0.124 |
| Pure SAC | 15,000 | 56 | 3.733 | 0.683 | 1,306.704 | 179.756 | 0.215 |
| SAC + frozen Q_safe masking | 15,000 | 85 | 5.667 | 0.552 | 638.497 | 95.760 | 0.252 |

The masking run produced 29 more falls than pure SAC, a 51.8% increase.
It replaced 2,351 actions (15.67%) and abstained because no candidate was
below the threshold on 6,034 steps (40.23%). Replacement failure rates over
8/16/32 subsequent policy steps were 0.98%/2.98%/10.90%.

This experiment does not demonstrate fall reduction. The lower episode fall
rate for masking is caused by many more, shorter episodes and must not be
interpreted as improved safety; falls per fixed policy-step budget is the
primary metric. The source AUROC was measured on balanced training batches,
not an episode-disjoint action-ranking validation set, so it did not establish
that Q_safe could safely rank same-state candidate actions.

W&B runs:

- Source logging: `7zcxznwx`
- Pure SAC: `j7t6hn83`
- Frozen Q_safe masking: `t3ad2kq6`
