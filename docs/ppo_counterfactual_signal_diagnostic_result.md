# PPO same-state counterfactual signal diagnosis

Date: 2026-08-12 (Asia/Shanghai)

## Decision

The failed R4 informativeness result was diagnosed without training a safety
critic, opening or generating protected outcomes, or starting SAC transfer.
All three preregistered causal flags are false:

```text
r4_label_noise_likely=false
h96_credit_dilution_likely=false
candidate_direction_coverage_likely=false
```

The strict `r4_label_noise_likely` flag asks whether more replicas uncover
reliable strong action contrasts that R4 missed. They do not. R4 nevertheless
has substantial finite-sample noise in the opposite direction: it creates
many apparent strong pairs that disappear with more replicas.

The actionable diagnosis is therefore:

> The frozen one-step PPO stochastic candidate set has very little stable
> same-state effect on H96 fall risk. R4 exaggerated the amount of useful
> supervision; neither a shorter horizon nor measured action-direction
> coverage recovers a strong, reproducible action-ranking signal.

## Frozen diagnostic cohort

The cohort was selected outcome-blind from development data and preserved the
registered allocation:

| Split | Boundary | Medium | Normal | Total |
|---|---:|---:|---:|---:|
| Train | 160 | 80 | 80 | 320 |
| Calibration | 40 | 20 | 20 | 80 |
| Total | 200 | 100 | 100 | 400 |

Every split/stratum cell is divided equally between PPO seed137 and seed138.
The original 16 frozen physical candidates were reused exactly. Replicas R5
through R16 added 76,800 branches; the combined R16 analysis contains 102,400
branches. Candidate generation was not rerun or altered.

## Replica diagnosis

| Replicas | Mean within-state range | States with strong pair | Split-half ordering agreement | Independent oracle reduction |
|---:|---:|---:|---:|---:|
| R4 | 20.56 pp | 18.0% | 0.449 | 0.375 pp |
| R8 | 17.25 pp | 6.0% | 0.445 | 0.688 pp |
| R16 | 13.91 pp | 3.0% | 0.462 | 0.563 pp |

From R4 to R16, strong-pair coverage falls by 15.0 pp (two-sided 95% CI
`[-18.5, -11.5]` pp). Ordering agreement improves by 0.155 among comparable
states (95% CI `[0.063, 0.245]`), but the final R16 agreement is only 0.462 and
its 95% interval includes 0.5. More replicas therefore remove false strong
pairs; they do not reveal a reliable ordering hidden by R4.

The R16 independent oracle reduction is 0.563 pp, with one-sided 95% LCB
0.031 pp. This is statistically just above zero under the preregistered
one-sided summary but far smaller than the earlier R4 development-oracle
estimate and too sparse to satisfy the supervision gate.

## Horizon diagnosis

All horizon values are computed from the same R16 `first_fall_step` outcomes.

| Horizon | Mean candidate fall risk | Mean within-state range | Strong-pair coverage | Independent oracle reduction |
|---:|---:|---:|---:|---:|
| H16 | 0.22% | 0.22 pp | 0.0% | 0.000 pp |
| H32 | 1.17% | 2.83 pp | 1.0% | 0.188 pp |
| H64 | 3.55% | 8.88 pp | 2.75% | 0.656 pp |
| H96 | 6.17% | 13.91 pp | 3.0% | 0.563 pp |

H16 and H32 contain less action contrast than H96, not more. Their strong-pair
coverage differences relative to H96 are negative, and neither has positive
independent-oracle LCB. H96 credit dilution is therefore not supported.

At H96, useful headroom is concentrated in Boundary states:

| Stratum | Mean range | Strong-pair coverage | Independent oracle reduction |
|---|---:|---:|---:|
| Boundary | 16.47 pp | 5.5% | 1.438 pp |
| Medium | 11.69 pp | 0.0% | 0.375 pp |
| Normal | 11.00 pp | 1.0% | -1.000 pp |

This does not authorize retraining on Boundary alone; it identifies where a
future, separately preregistered candidate design should be tested.

## Collector and action-coverage diagnosis

At R16, seed137 and seed138 candidate distances are close: mean normalized
physical distance is 1.055 and 1.036 respectively. Seed137 has a 4.5 pp larger
mean within-state risk range, but the action-distribution model predicts only
0.625 pp of that gap (13.9%) and has negative grouped cross-validation R².

No joint/direction combination meets the preregistered structured-effect rule
of an absolute 5 pp risk effect with a confidence interval excluding zero.
Near, medium, and far candidates have similar absolute risk deltas. Per-joint
reach differences exist, but they are not predictively related to risk.

The evidence rules out narrower PPO actions or a missing single-joint
direction as the main explanation. The simple state-distribution model also
has negative grouped cross-validation R², so the remaining seed difference is
not positively attributed to a particular measured state feature. It is most
consistent with sparse fall outcomes plus cohort composition.

## Consequence for the next cycle

The current protocol remains stopped before critic training. A new cycle
should not merely increase R, shorten H, or resample more actions from the same
one-step PPO distribution. The next candidate-space design should be
preregistered around Boundary admission and should create causally stronger
but still deployable actions, such as state-conditioned support/step actions
or short closed-loop options. It must first demonstrate independent oracle
headroom on development states before any critic is trained.

The existing protected roster remains identity-only. It was not consumed and
cannot be reused to tune the next candidate design.

## Reproduction

```bash
python3 scripts/plan_counterfactual_signal_diagnostic.py \
  --development-dataset saved/qsafe_counterfactual_v2/development-r4-h96.npz \
  --output saved/qsafe_counterfactual_v2/signal-diagnostic-400-roster.npz

/home/xyz/micromamba/envs/safesac/bin/python \
  scripts/extend_counterfactual_replicas.py \
  --diagnostic-roster saved/qsafe_counterfactual_v2/signal-diagnostic-400-roster.npz \
  --development-dataset saved/qsafe_counterfactual_v2/development-r4-h96.npz \
  --development-roster saved/qsafe_counterfactual_v2/development-roster.npz \
  --checkpoint137 saved/qsafe_development/natural_ppo/production-30m-seed137-v1/model_19.pt \
  --checkpoint138 saved/qsafe_ppo_sqrl_v1/ppo-seed138-30m/model_19.pt \
  --model-binary saved/qsafe_counterfactual_v2/frozen-states-seed137-64m/model.mjb \
  --round-one-denylist config/qsafe_round_one_protected_denylist.json \
  --workers 16 \
  --output saved/qsafe_counterfactual_v2/signal-diagnostic-r5-r16.npz

python3 scripts/analyze_counterfactual_signal_diagnostic.py \
  --development-dataset saved/qsafe_counterfactual_v2/development-r4-h96.npz \
  --diagnostic-roster saved/qsafe_counterfactual_v2/signal-diagnostic-400-roster.npz \
  --replica-extension saved/qsafe_counterfactual_v2/signal-diagnostic-r5-r16.npz \
  --output saved/qsafe_counterfactual_v2/signal-diagnostic-report-v3.json
```

The detailed machine-readable analysis is in
`saved/qsafe_counterfactual_v2/signal-diagnostic-report-v3.json`; the compact
tracked decision is in
`reports/ppo_counterfactual_signal_diagnostic_v1_decision.json`.
