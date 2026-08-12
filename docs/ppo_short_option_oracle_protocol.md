# PPO short closed-loop residual-option oracle protocol

Date frozen: 2026-08-13 (Asia/Shanghai)

This development-only experiment tests one question: did the previous
same-state experiment fail because a single 20 ms action has too little causal
control authority? It does not train a critic, open or generate protected
outcomes, or run any SAC experiment.

The machine-readable source of truth is
`config/qsafe_short_option_oracle_v1.yaml`. It was frozen before collecting the
new source episodes or any option outcome.

Two frozen PPO boundary actors use independent fresh rollout seeds and fixed
32M-transition exposure each. Exactly 300 Boundary episodes per actor are
selected by identity hash, with one state per episode and a deterministic
32--64 step pre-fall offset. Previous development, diagnostic, and protected
episodes are excluded. No state is protected in this experiment.

At each state, the actor produces one nominal action and 64 stochastic
proposals. After conversion to physical absolute joint targets and clipping to
the MuJoCo hard joint limits, five residuals are selected with deterministic
greedy farthest-point sampling. The selection API accepts actions and identity
only; it cannot receive branch outcomes.

For residual `delta`, an option of length `L` executes

```text
Proj[PPO_nominal(s[t+k]) + beta[L,k] * delta]
```

at each active policy cycle. The schedules are fixed to linear decay:

```text
L1: 1
L4: 1, 3/4, 1/2, 1/4
L8: 1, 7/8, 6/8, 5/8, 4/8, 3/8, 2/8, 1/8
```

After the option, the matching frozen stochastic PPO actor continues to H96.
There is no recovery or get-up behavior. Each state has one nominal plus five
directions at each of L1/L4/L8, for 16 candidates. Every candidate gets eight
paired-CRN replicas. R1--R4 discover an empirical oracle separately within
each duration family; only R5--R8 evaluate it.

A long family passes only if its independent oracle reduction is at least
3 percentage points, its one-sided state-bootstrap 95% LCB is positive, its
split-half ordering-agreement two-sided 95% CI is entirely above 0.5, and
rescue states exceed harm states. A passing L4/L8 family is "clearly better"
than L1 only when the paired long-minus-L1 oracle-reduction one-sided 95% LCB
is positive.

The fixed outputs are:

```text
short_option_candidate_space_supported
one_step_action_timescale_insufficient
```

No duration, residual scale, direction rule, beta schedule, threshold, or
candidate family may be changed after seeing this cohort's outcomes.
