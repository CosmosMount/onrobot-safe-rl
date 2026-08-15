# Formal paired-seed protocol

`sqrl_go2_formal_v2` freezes the development implementation before observing
any formal result. The formal roster is seeds 10 through 19. Each seed produces
one 25,000-step 0.30 m/s pretrain checkpoint and three 10,000-step 0.40 m/s
target branches cloned from that checkpoint.

The primary comparison is SQRL-full versus SAC-transfer. SQRL-mask is a
mechanism ablation. Reward, tracking error, and actual forward velocity are
recorded diagnostics and never gate admission or success.

Formal-v1 was invalidated in full after an interrupted process session exposed
an infrastructure bug: runtime startup assumed the action shared-memory mailbox
survived a prior Python process. Formal-v2 creates and clears that mailbox before
runtime startup. No formal-v1 outcome is mixed into formal-v2.

## Execution

After tests pass, create the executable lock exactly once:

```bash
python3 -m reproductions.sqrl_go2.diagnostics.create_formal_lock
```

Run the serial campaign:

```bash
python3 -m reproductions.sqrl_go2.runners.run_formal --device cuda
```

An interrupted campaign may verify and skip immutable completed runs with
`--resume`. An incomplete run is never retried implicitly. Supply its exact key
and one allowed execution-failure reason, for example:

```bash
python3 -m reproductions.sqrl_go2.runners.run_formal --device cuda --resume \
  --retry-run seed_10/pretrain_030 --retry-reason machine_interruption
```

Performance, pessimistic masks, zero dual values, or poor reward are not retry
reasons. A software fix invalidates all of formal-v1 and requires an amended
protocol with every formal seed rerun.

## Locked statistics

The statistical unit is one complete paired seed. PCG64 seed 20260813 draws
100,000 ten-seed bootstrap resamples. Two-sided intervals use percentile
2.5/97.5 and the one-sided 95% lower confidence bound uses percentile 5, all
with NumPy's linear quantile method.

Formal SQRL reproduction requires SQRL-full versus SAC-transfer to have a
positive mean and lower confidence bound, at least 8/10 strictly improved
seeds, and at least 30% pooled relative fall reduction. Masking support requires
a positive lower bound and 30% pooled reduction. Lagrangian support requires a
positive Full-versus-Mask lower bound and `nu > 1e-8` in at least two seeds.
