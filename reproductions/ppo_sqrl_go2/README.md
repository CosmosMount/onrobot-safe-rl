# PPO-SQRL Go2 validation

This package is an isolated three-stage validation of policy/safety-critic
co-adaptation. It imports the target-aligned MjLab Go2 environment, exact fall
predicate, controller projection semantics, and RSL-RL MLP, Gaussian, and
normalization components. It owns the safety-aware PPO update, rollout storage,
recent safety buffer, Q_safe, protocol locks, runners, statistics, and reports.

The package never changes `reproductions/sqrl_go2`, existing agent families, or
the controller. Protocol outputs live under
`saved/reproductions/ppo_sqrl_go2/` and are immutable once published.

Entrypoints are exposed under `reproductions.ppo_sqrl_go2.runners`.

Prepare and freeze all three locks without launching a claim-bearing run:

```bash
python3 -m reproductions.ppo_sqrl_go2.runners.run_campaign --prepare-only
```

Run the serial campaign:

```bash
python3 -m reproductions.ppo_sqrl_go2.runners.run_campaign
```

An interrupted campaign can be continued only from a complete iteration:

```bash
python3 -m reproductions.ppo_sqrl_go2.runners.run_campaign --resume
```

A recorded failed attempt is never silently resumed. After the failure cause is
explicitly registered, `--resume --retry-failed` archives that whole run and
starts the run again from iteration zero. Any locked source change requires a
new versioned campaign root and amendment; v1 and amended results must not be
combined.
