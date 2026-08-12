# Action-candidate oracle development result v1

Evidence role: consumed development diagnosis only. This report cannot pass
the candidate oracle gate, authorize Q_safe(s,a) training, or support
Objective 1.

## Design

- Three existing SAC actors: seeds 47, 48, and 49.
- One source per actor; 20 pre-outcome state-risk proposals per source.
- 60 source-state groups total.
- 24 executable actions per state: nominal, local SAC actions, symmetric local
  perturbations, and eight deployable state-dependent support/capture actions.
- Eight paired H96 SAC continuations per candidate, split before analysis into
  four discovery and four audit replicas.
- No external force, impulse, settle transition, get-up, or persistent recovery.
- No selector or action critic was trained.

## Result

- Nominal audit fall rate: 23.33%.
- Discovery-selected oracle audit fall rate: 24.17%.
- Reduction: -0.83 percentage points.
- One-sided 95% LCB: -5.42 percentage points.
- Actor effects: seed 47 = -5.00 pp, seed 48 = +3.75 pp, seed 49 = -1.25 pp.
- Candidate oracle gate: failed.
- `model_training_authorized=false`.
- `objective1_pass=false`; `phase2_authorized=false`.

The same-replica empirical minimum showed headroom (actor reductions 19.38,
27.50, and 1.25 pp), but that quantity reuses outcomes for selection and is
optimistically biased. It is not accepted as the gate result.

## Diagnosis and required correction

The state-only proposal window selected false-positive states for some actor
distributions. Seed 49 had only 1.25% nominal risk over the full eight
replicas, leaving almost no preventable failures. The next candidate-space
development cohort must use natural trajectory identity to select early
pre-fall states with a nontrivial time-to-fall margin. Natural fall timing may
admit a source state, but it still cannot label any unexecuted candidate; all
Q_safe(s,a) labels remain same-state H96 branches.

Authoritative machine report:
`saved/qsafe_development/natural_ppo/production-30m-seed137-v1/action-oracle-development/development-oracle-g60-r8-v1.json`.
