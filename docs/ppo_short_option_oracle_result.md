# PPO short closed-loop residual-option oracle result

Date: 2026-08-13 (Asia/Shanghai)

## Decision

The short residual-option candidate space failed its preregistered primary
gate:

```text
short_option_candidate_space_supported=false
one_step_action_timescale_insufficient=false
persistent_residual_option_route_stopped=true
critic_training_authorized=false
```

The experiment does not prove that 20 ms is intrinsically sufficient. It
shows that extending five frozen PPO residual directions to 80 or 160 ms with
the preregistered linear beta schedule does not create the required causal
headroom. No safety critic was trained, no protected outcome was opened or
generated, and no SAC transfer was run.

## Fresh cohort and branch integrity

The two frozen PPO boundary actors were rolled out under independent new seeds
4137 and 4138. Each received exactly 32M aggregate environment transitions in
2,000 parallel environments at +0.30 m/s, with zero external force and normal
first-fall terminal/reset behavior. They produced 544 and 630 independent
natural falls respectively.

Exactly 600 new Boundary episodes were selected outcome-blind: 300 per PPO
collector, one state per episode, with offsets in the frozen 32--64 step
interval. None came from the previous diagnostic or protected cohorts.

The dataset contains exactly 600 states, 16 candidates, and 8 replicas:
76,800 complete paired-CRN H96 branches. The audit verified 4,800 unique
state/replica CRN streams, exact sharing across candidates, exact reuse of the
same five residual directions across L1/L4/L8, and correct first-fall indexing
with 97 as the no-fall sentinel. There were 25,827 branch falls.

## Independent oracle result

R1--R4 selected the best action separately inside each duration family;
R5--R8 alone evaluated it. Confidence intervals use state-group bootstrap.

| Family | Nominal fall | Oracle fall | Reduction | One-sided 95% LCB | Ordering agreement (95% CI) | Reproducible states | Rescue / harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1, 20 ms | 29.75% | 27.67% | 2.08 pp | 0.83 pp | 0.531 `[0.522, 0.541]` | 42.17% | 96 / 61 |
| L4, 80 ms | 29.75% | 28.83% | 0.92 pp | -0.33 pp | 0.552 `[0.542, 0.562]` | 50.50% | 85 / 76 |
| L8, 160 ms | 29.75% | 28.00% | 1.75 pp | 0.54 pp | 0.590 `[0.579, 0.601]` | 65.33% | 86 / 59 |

L8 produces the most reproducible ordering and has positive independent
headroom, but its 1.75 pp reduction misses the fixed 3 pp minimum. L4 also
misses 3 pp and its LCB crosses zero. Relative to L1, L4 is worse by 1.17 pp
and L8 is worse by 0.33 pp; neither paired comparison has positive LCB.
Therefore, neither long option is clearly better than the one-step family.

The collector split shows why the pooled conclusion must remain conservative:

| Family | seed137 reduction | seed138 reduction |
|---|---:|---:|
| L1 | 1.58 pp | 2.58 pp |
| L4 | 0.25 pp | 1.58 pp |
| L8 | 0.50 pp | 3.00 pp |

Only seed138/L8 reaches 3 pp; seed137/L8 is 0.5 pp with an LCB below zero.
The predefined pooled 600-state gate cannot be passed by one collector alone.

## Physical diagnostics

The five selected residuals were intentionally broad: normalized physical
distance from nominal had mean 1.311, median 1.290, and 5--95% interval
`[1.104, 1.583]`. The result is therefore not explained by selecting only
near-nominal proposals.

| Family | Mean replacement per active step | Mean branch maximum | Projection saturation | Joint-limit saturation |
|---|---:|---:|---:|---:|
| L1 | 0.637 | 0.637 | 1.96% | 1.96% |
| L4 | 0.386 | 0.626 | 2.72% | 2.72% |
| L8 | 0.335 | 0.605 | 2.88% | 2.88% |

Longer persistence increases motion transients rather than supplying a clear
safety advantage. Across all candidates, the 95th-percentile maximum absolute
roll rises from 0.333 rad at L1 to 0.380 at L4 and 0.492 at L8; maximum angular
velocity rises from 3.65 to 4.67 and 5.21 rad/s. These diagnostics were
recorded after candidate freezing and were never used for candidate selection.

## Consequence

The persistent frozen-residual route stops here. The result does not authorize
training `Q_safe(s,o)`. A next cycle may study a separately preregistered,
state-conditioned support/capture-step option family, using a new development
cohort. This cohort cannot be used to tune duration, residual magnitude,
direction selection, beta schedule, or a threshold.

## Reproduction

```bash
/home/xyz/micromamba/envs/safesac/bin/python scripts/collect_frozen_ppo_counterfactual_states.py --checkpoint saved/qsafe_development/natural_ppo/production-30m-seed137-v1/model_19.pt --ppo-seed 137 --rollout-seed 4137 --output saved/qsafe_short_option_v1/fresh-boundary-seed137-32m --aggregate-transitions 32000000 --normal-events 0

/home/xyz/micromamba/envs/safesac/bin/python scripts/collect_frozen_ppo_counterfactual_states.py --checkpoint saved/qsafe_ppo_sqrl_v1/ppo-seed138-30m/model_19.pt --ppo-seed 138 --rollout-seed 4138 --output saved/qsafe_short_option_v1/fresh-boundary-seed138-32m --aggregate-transitions 32000000 --normal-events 0

python3 scripts/plan_short_option_boundary_roster.py --seed137-root saved/qsafe_short_option_v1/fresh-boundary-seed137-32m --seed138-root saved/qsafe_short_option_v1/fresh-boundary-seed138-32m --output saved/qsafe_short_option_v1/boundary-600-roster.npz

/home/xyz/micromamba/envs/safesac/bin/python scripts/collect_short_option_oracle.py --roster saved/qsafe_short_option_v1/boundary-600-roster.npz --checkpoint137 saved/qsafe_development/natural_ppo/production-30m-seed137-v1/model_19.pt --checkpoint138 saved/qsafe_ppo_sqrl_v1/ppo-seed138-30m/model_19.pt --model-binary saved/qsafe_short_option_v1/fresh-boundary-seed137-32m/model.mjb --workers 16 --output saved/qsafe_short_option_v1/short-option-r8-h96-v2.npz

python3 scripts/analyze_short_option_oracle.py --dataset saved/qsafe_short_option_v1/short-option-r8-h96-v2.npz --output saved/qsafe_short_option_v1/short-option-oracle-report-v3.json
```
