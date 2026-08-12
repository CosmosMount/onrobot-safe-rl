# Action-conditioned Q_safe candidate-space development result

Status: development headroom found; protected oracle gate not yet passed; model
training remains forbidden.

This experiment uses 60 early pre-fall natural-SAC source states from three SAC
actors (47/48/49), K=24 executable one-step actions, eight paired H96 replicas
per action, no force/impulse/settle/recovery, and SAC continuation after the
first candidate action. The three source shards were regenerated under protocol
revision 4; no state-only or fixed-recovery outcome is used.

## Literal post-hoc action oracle

For each source state, the oracle inspected all eight branch outcomes of every
candidate and selected the concrete action with the lowest empirical H96 fall
risk. It is an upper bound on candidate-space headroom, not a deployable
selector.

| Metric | Result |
|---|---:|
| nominal H96 fall rate | 51.25% |
| post-hoc oracle-best H96 fall rate | 30.00% |
| absolute reduction | 21.25 pp |
| actor/source/state cluster-bootstrap one-sided 95% LCB | 17.50 pp |
| actor 47 reduction | 21.25 pp |
| actor 48 reduction | 19.38 pp |
| actor 49 reduction | 23.13 pp |

The action space therefore contains substantial state-dependent counterfactual
headroom on this development cohort. The selected action is not one global
recovery: nominal was best for 13/60 states and the other 47 states spread over
17 local, support, tilt and capture-step action kinds.

## Stability diagnostics and decision

Selecting on four discovery replicas and evaluating on four held-out audit
replicas reduced fall by only 0.42 pp (50.00% to 49.58%). The same-CRN
per-realization oracle reduced 35.00 pp, but it is explicitly unattainable
because it knows future continuation randomness and is not a gate statistic.

The protected gate requires at least two unseen actors, four unseen sources,
120 state groups and 32 replicas per action. This development run has only 60
groups and eight replicas, so:

```text
candidate_oracle_gate_pass = false
model_training_authorized = false
objective1_pass = false
phase2_authorized = false
```

No Q_safe(s,a) selector was trained from these outcomes. The next admissible
operation is protected candidate-space branching with more replicas; threshold
tuning and fixed recovery remain forbidden.

Machine-readable report:
`saved/qsafe_development/natural_ppo/production-30m-seed137-v1/action-oracle-early-prefall-development-v4/development-posthoc-oracle-g60-r8-v2.json`.
