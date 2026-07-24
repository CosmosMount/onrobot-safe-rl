# Go2 Safe RL Research Pipeline

## Invariants

- The converged SAC/DroQ reward learner remains the baseline.
- Safety learning is auxiliary until the evaluation gate passes.
- Scripted standup/recovery is a reset and emergency supervisor, not policy
  competence. Recovery transitions never enter reward replay.
- `python -m train --mode in_process --config config/go2.yaml` remains valid.

## Completed foundation

1. Safety costs, unsafe/near-failure labels, termination reasons and
   intervention masks are recorded in every policy transition.
2. Safety replay is split into recent, boundary, failure and recovery buffers.
3. An independent sigmoid Q_safe critic is trained without modifying the SAC
   actor, reward critic or executed action.
4. `safety_eval` exports rollout CSV, JSON metrics and an SVG report, and blocks
   shielding unless AUROC and pre-failure warning-rise gates pass.
5. `safety_collect` freezes the reward learner and collects reproducible
   perturbed-policy data.

## Q_safe propagation improvement (current)

The first held-out disturbed evaluation produced AUROC 0.739 but only a 0.0277
increase in mean Q_safe before failure, below the required 0.05. The following
changes target slow risk propagation without changing the baseline learner.

### 1. H-step future-failure supervision

For transitions within `H=32` steps of a hard failure, replay supplies
`future_failure_labels=1`. Q_safe adds an auxiliary binary cross-entropy loss:

```text
L_safe = L_n_step_td + beta * BCE(Q_safe(s,a), future_failure_within_H)
```

Initial `beta=0.5`. This directly teaches early warning while retaining the
Bellman objective.

### 2. Eight-step safety target

Replay incrementally records the next observation, unsafe event and bootstrap
mask over up to eight same-episode policy transitions:

```text
y_8 = I[t:t+7] + (1-I[t:t+7]) gamma_safe^n
      mask_n Q_safe_target(s_t+n, a_t+n)
```

Partial sequences at the replay frontier use their available `n <= 8` steps.
No target crosses an episode or recovery boundary.

### 3. Behavior-matched backup actions

Safety collection stores its Gaussian action-noise standard deviation. Target
actions use the same behavior distribution:

```text
a' = clip(sample(pi(.|s')) + Normal(0, behavior_noise_std), -1, 1)
```

Ordinary training and clean evaluation use zero noise. This prevents disturbed
current actions from being backed up under an unrealistically clean policy.

## Validation gate

After collecting at least 30 diverse failures across several seeds and noise
levels, evaluate on held-out seeds. Action masking remains blocked unless:

- both future-failure positive and negative samples exist;
- Q_safe AUROC is at least 0.70;
- pre-failure mean Q_safe exceeds normal mean Q_safe by at least 0.05;
- return and failure metrics are based on independent evaluation rollouts.

Use `--rollout-seed` to keep collection and evaluation noise streams disjoint.
Safety collection checkpoints after every completed episode so an interrupted
multi-episode batch retains all completed data.

## Stage 2: SQRL action masking

Action masking is an opt-in wrapper (`safety_eval --safety-mask`) and does not
change actor parameters. For each state it builds `K=32` candidate actions,
evaluates Q_safe, and retains candidates at or below `epsilon_safe=0.30`.
Candidates include the policy mean, previous action, a contracted previous
action, local samples around the policy mean and ordinary policy samples.
Safe candidates use normalized reward Q minus risk and action-delta penalties.
If none pass, an acceptable previous action is reused; otherwise the
minimum-risk candidate is executed. Evaluation records mask rate,
no-safe-candidate rate, selected Q_safe, action delta and fallback type. The
ordinary training entrypoint and unmasked evaluation remain unchanged.

### Masking validation status

An early K=16 experiment appeared to reduce falls by 25%, but each candidate
was incorrectly given an independently sampled target disturbance before
selection. That let the selector choose a favorable noise realization and is
an oracle comparison, so the result is withdrawn.

The corrected evaluation selects the commanded action first and applies one
held-out disturbance afterward, identically to the unmasked baseline. On seed
9010 with action noise 0.50 and epsilon 0.10:

| metric | unmasked | structured K=32 |
|---|---:|---:|
| falls | 2/2 | 2/2 |
| mean episode length | 200.0 | 179.5 |
| mean return | 1425.2 | 1304.9 |
| no-safe-candidate rate | 0% | 17.8% |

Structured candidates and temporal scoring are implemented, but this fair
experiment does not improve safety. The likely next requirement is training
Q_safe on the masked-policy distribution and reducing its false
no-safe-candidate rate before claiming a shielding benefit.

`safety_collect --safety-mask` closes this distribution loop while keeping the
reward learner frozen. It selects the command with the structured shield,
applies one target disturbance afterward, stores resulting transitions in
safety replay, and updates Q_safe online. Completed successful trajectories
provide hard negatives; masked failures retain H-step positive backfill.

The first loop-closure trial added two masked failures and one 400-step masked
success trajectory. On held-out seed 9010, however, Q_safe AUROC fell to 0.537
and masking retained 2/2 falls. This small update set is insufficient and
temporarily worsens calibration; checkpoint 15,327 remains the reference
critic. Further loop closure should be performed as a larger, balanced dataset
(not a few sequential online episodes) before another shielding claim.
