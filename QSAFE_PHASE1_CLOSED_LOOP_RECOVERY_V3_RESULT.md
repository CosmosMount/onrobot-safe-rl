# Q_safe Phase 1 closed-loop recovery triage v3 result

Status: **primary failed; inconclusive, no model training authorized**

Decision date: 2026-08-09

Protocol: `objective1_closed_loop_recovery_triage_v3`

Generator/analyzer commit: `76818866896bbf5662d3c5714ab8002caed8386f`

## Executive decision

The preregistered global fixed-backup hypothesis failed on the one-shot audit.
Discovery selected `mature_actor_L10`, but its equal-six-seed audit fall-risk
reduction was only `3.105 percentage points` and its one-sided 95% lower
confidence bound was `-2.161 pp`. The effect was negative at the 25k policy age
and for source seeds `7801`, `7802`, and `7811`. Consequently all four primary
checks failed and the locked decision is
`report_inconclusive_no_model_training`.

The no-headroom rule did not fire. In particular, the discovery-locked
per-state rule had a large audit diagnostic (`36.437 pp` reduction; one-sided
95% LCB `33.772 pp`; pair agreement `0.81185`). However, the protocol specified
a hierarchical test: the state-dependent conditional hypothesis was tested
only if the fixed global primary passed. Its recorded status is therefore
`tested=false`, `pass=null`, and it is exploratory rather than confirmatory.
It cannot authorize Q_safe training or an Objective-1 claim.

Consequences:

- `model_training_authorized=false` and `selector_calibration_authorized=false`;
- paired closed-loop and online SAC evaluations remain unauthorized;
- Objective 1 remains incomplete and Phase 2 remains blocked;
- no fixed K9 behavior is promoted as a global backup;
- the exploratory state-dependent signal may motivate a new protocol only on
  wholly fresh train/calibration/test states and fresh confirmation seeds;
- the consumed v3 discovery/audit outcomes cannot be reused as evidence in
  that next protocol.

## Immutable design and provenance

The run used three locked early SAC actor ages (`25,438`, `50,030`, and
`100,359` training steps), two source seeds per age (`7801/7802`, `7811/7812`,
and `7821/7822`), and a mature recovery actor. Each source contributed exactly
64 admission-positive states. The combined design therefore contained 384
independent trajectory states, K=9 behaviors, 64 discovery replicas and 64
audit replicas per candidate, and H=96 policy steps per branch. Discovery and
audit RNG domains were assigned before any candidate outcome.

The failure label was evaluated at the post-action 50 Hz boundary after ten
500 Hz simulator substeps. It used base-body origin height `<0.18 m` or
absolute roll/pitch `>=0.523599 rad`. Candidate action application, PD gains,
policy fingerprints, recursive MJCF dependencies, recovery-program manifest,
and the six source-seed namespaces were bound before the attempt markers.

| Artifact | SHA-256 |
|---|---|
| protocol file | `b43cfc658df400dc07f425c50bd96db4e07751e7918caecd1cdc99d23d19875c` |
| canonical parsed protocol | `07f530c582df38a1ff685fa0f8c0546f01eebb8cb9ec9573911e6f6076a59c3b` |
| admission merge report | `ca79533cd5a2af3ae2e16ad3592f98e23811a9b15ceb8a07b47d729dfec92129` |
| discovery merge report | `859e211c2978b32ed8416ff43036d31e02d58f78c6f86e6504c8d2659a6e31e1` |
| merged discovery file | `e6a29d403bdb18745c3ce50b65201f8cc04c53d6b0209fad2aa5ca2bc34734ad` |
| merged discovery content | `3374df7b87c663ad655369f702684bb9adea3feb582a1fde2675f72899493641` |
| selection lock | `2e277d586a1c96d24d24fa8f5543c6e5c4a0c67b070e78aa3c8497079f88f2af` |
| audit-consumed marker | `f732245ae546f8c7f9a498c53a9747b9b4f01ad08cdf6718be69010d88fc7d85` |
| one-shot triage report | `2fef270a97fa302070f5eb94e617a36635dc3123a1094f2e2b7266f5cd4e8ea6` |

All six collection commands exited zero and published their report last.
Admission merged 384 accepted states from 8,218 proposals without opening any
candidate outcome. The discovery data gate then passed all locked checks:
G384, six seeds with 64 groups each, 384 unique trajectory clusters, K9, R64,
H96, exact behavior names/durations, exact preassigned audit seed shapes,
unique audit seeds, and disjoint discovery/audit seed domains. The merged
discovery data had 3,456 valid group-candidates and mixed-outcome fraction
`1.0`.

Discovery informativeness passed before the audit was opened. The equal-seed
nominal discovery risk was `0.469808`; policy-age risks were `0.390137`,
`0.518066`, and `0.501221`, all within their preregistered inclusive ranges.
The permanent selection lock chose `mature_actor_L10`, whose discovery risk was
`0.438517`.

## One-shot audit result

Positive effect means nominal audit fall risk minus candidate audit fall risk.
The nominal equal-seed audit risk was `0.470907` with one-sided 95% LCB
`0.447510`. Confidence bounds used the locked 50,000-replicate hierarchical
bootstrap with PCG64 seed `20260809`; no bootstrap override was used.

### Primary global fixed backup

| Quantity | Result | Gate |
|---|---:|---|
| absolute fall-risk reduction | `+3.105 pp` | fail: required `>=5 pp` |
| one-sided 95% LCB | `-2.161 pp` | fail: required `>=3 pp` |
| all policy ages positive | no | fail |
| all six source seeds positive | no | fail |

Policy-age effects were `-7.458 pp` at 25k, `+1.648 pp` at 50k, and
`+15.125 pp` at 100k. Source-seed effects were:

| Seed | 7801 | 7802 | 7811 | 7812 | 7821 | 7822 |
|---|---:|---:|---:|---:|---:|---:|
| reduction | `-3.125 pp` | `-11.792 pp` | `-6.421 pp` | `+9.717 pp` | `+17.358 pp` | `+12.891 pp` |

The observed age/seed heterogeneity is inconsistent with the preregistered
robustness requirement: the option was not positive in every locked stratum.
The selected discovery winner therefore does not generalize uniformly across
this locked early-policy cohort. No general monotone age trend was tested.

### Fixed K9 diagnostics

| Candidate | Audit reduction | One-sided 95% LCB |
|---|---:|---:|
| `mature_actor_L10` | `+3.105 pp` | `-2.161 pp` |
| `mature_actor_L25` | `-0.028 pp` | `-4.789 pp` |
| `mature_actor_L50` | `-15.181 pp` | `-20.349 pp` |
| `joint_brake_L10` | `-2.323 pp` | `-7.137 pp` |
| `halfway_neutral_L10` | `-6.498 pp` | `-11.581 pp` |
| `halfway_neutral_L25` | `-13.729 pp` | `-17.647 pp` |
| `ramp_neutral_L25` | `-7.161 pp` | `-11.727 pp` |
| `ramp_crouch_L25` | `+0.537 pp` | `-3.577 pp` |

These fixed-option diagnostics do not justify choosing an audit runner-up; the
protocol explicitly forbids that selection.

### Hierarchy-blocked state-dependent diagnostic

The discovery-locked per-state minimizer rule produced:

- audit reduction `+36.437 pp`, one-sided 95% LCB `+33.772 pp`;
- incremental reduction over the selected global option `+33.332 pp`, LCB
  `+28.951 pp`;
- discovery/audit pair agreement `0.81185`, LCB `0.80205` across 13,824 pair
  comparisons;
- positive incremental effects at all three ages and all six source seeds.

The descriptive incremental effects were positive at all three ages
(`35.980`, `35.977`, and `28.040 pp` for 25k, 50k, and 100k) and all six seeds
(`33.875`, `38.086`, `44.075`, `27.879`, `29.150`, and `26.929 pp`). These
diagnostics are a strong exploratory signal consistent with state-dependent
ordering headroom inside K9, but do not establish it confirmatorily. They do
**not** pass the conditional gate: the preregistered primary-first hierarchy
sets `tested=false`, `checks=null`, and `pass=null` after the global primary
fails. Any subsequent state-dependent model must be trained and evaluated on
wholly fresh data with a newly committed protocol, and its confirmation
analysis must not tune on this audit.

## No-headroom rule

The nominal-opportunity check passed because its audit-risk LCB was `44.751%`.
The simultaneous-UCB check failed: not every one of the eight fixed effects and
the locked per-state rule had UCB strictly below `3 pp`. The common critical
value was `6.799 pp`.

| Effect | Simultaneous one-sided 95% UCB |
|---|---:|
| `mature_actor_L10` | `+9.904 pp` |
| `mature_actor_L25` | `+6.771 pp` |
| `mature_actor_L50` | `-8.382 pp` |
| `joint_brake_L10` | `+4.476 pp` |
| `halfway_neutral_L10` | `+0.301 pp` |
| `halfway_neutral_L25` | `-6.930 pp` |
| `ramp_neutral_L25` | `-0.362 pp` |
| `ramp_crouch_L25` | `+7.336 pp` |
| locked per-state rule | `+43.236 pp` |

Therefore `no_headroom.fires=false`. The protocol is inconclusive and cannot
declare no useful headroom. The exploratory per-state signal motivates fresh
confirmation but does not prove remaining headroom.

## Outcome-orthogonal execution deviations

Two implementation deviations occurred and did not change the locked data,
estimands, gates, or candidate choice:

1. Six initial direct-path collection invocations failed during Python import
   (`ModuleNotFoundError`) before loading policies or the simulator, creating
   the artifact root, publishing a cohort/attempt marker, or generating any
   outcome. The same commands then ran once through their module entry point.
2. The first discovery-merge invocation loaded the fixed discovery outcomes and
   completed validation, but failed before canonical publication, selection
   lock creation, or any audit access because `np.all(mask)` made
   `candidates_exact` a `numpy.bool_`, which the standard JSON encoder cannot
   serialize. Temporary staging files were removed by the existing `finally`
   block. With the same clean HEAD and inputs, the merge was rerun through a
   narrow in-memory wrapper that converted only gate-check boolean scalars to
   built-in `bool`. No value, threshold, data row, hash computation, selection
   rule, or RNG operation was changed. The source fix and a JSON-serialization
   regression test are committed with this report, after the one-shot audit was
   complete.

Neither event inspected an audit outcome before `audit-consumed.json`; neither
is an optional-stopping or resampling event. No selection-bias channel was
identified, but the second event remains a disclosed runtime reproducibility
deviation rather than a claim of perfectly deviation-free execution.

## Required next iteration

The next protocol must be committed before collecting any new outcome. It may
use the v3 result only to formulate hypotheses and architecture; it must not
reuse v3 discovery or audit rows for fitting, calibration, selection, early
stopping, or confirmation. At minimum it must:

1. target state-dependent K9 ranking rather than promote a fixed global backup;
2. allocate fresh source seeds and trajectory states to disjoint train,
   calibration, development-test, paired closed-loop, and online-confirmation
   roles before outcomes;
3. lock a deployable state/action feature contract with no privileged inputs;
4. require independent-replica ranking/calibration gates before any online use;
5. require paired same-state closed-loop benefit before SAC training;
6. require a fixed-exposure, paired-seed online SAC A/B confirmation for either
   zero-to-train at 0.30 m/s or the preregistered small-shift route;
7. keep Phase 2 locked until Objective 1 passes its complete confirmation gate.
