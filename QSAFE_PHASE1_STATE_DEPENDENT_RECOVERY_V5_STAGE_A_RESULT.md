# Q_safe V5 Stage-A state-dependent recovery result

Status: **PASS — Stage B model fitting authorized only**

Decision date: `2026-08-10` (Asia/Shanghai)

Protocol: `objective1_state_dependent_recovery_qsafe_v5`

Generator and analysis commit:
`1452a112c35d94eed87bafb0c3f2bf73ab907324`

This is the canonical tracked disposition of the V5 Stage-A causal-headroom
experiment. It establishes that the frozen K9 recovery library contains a
state-dependent selection signal on fresh, independent audit replicas for the
fixed seed-42 actor at three training ages. It does **not** establish that a
learned Q_safe can recover that signal, reduce falls in paired closed loop, or
reduce falls during SAC-from-zero training.

## 1. Frozen identity and execution order

The experiment ran from the exact clean preregistration commit above. The
protocol file SHA-256 was
`f4c3e796004d124574df3d35ef344f6a4a766d9099acb5792c1d78b8361b49b0`;
its canonical contract SHA-256 was
`1e8667aa17ab361c323771d5deb51258644cce37bd26bed00599ca08d7545ea5`.
The shared cohort-lock SHA-256 was
`495360ceece0573ba168cb10676441771bc6a7ab0208e1583f5b679e95f9fd12`.

Six fresh collectors ran exactly once, with no outcome summary before all six
report-last artifacts existed:

| Policy age | Source seed | Accepted groups | Proposals | Source steps | Trajectories |
|---:|---:|---:|---:|---:|---:|
| 25,438 | 8901 | 64 | 1,708 | 9,956 | 321 |
| 25,438 | 8902 | 64 | 1,533 | 8,965 | 299 |
| 50,030 | 8911 | 64 | 1,413 | 8,091 | 266 |
| 50,030 | 8912 | 64 | 1,565 | 8,967 | 285 |
| 100,359 | 8921 | 64 | 1,568 | 9,522 | 354 |
| 100,359 | 8922 | 64 | 1,159 | 7,195 | 293 |
| **Total** | — | **384** | **8,946** | **52,696** | **1,818** |

Every collection report bound the same clean commit, protocol hashes, and
cohort lock and recorded `candidate_outcomes_summarized=false`,
`audit_opened_for_analysis=false`, `selection_lock_created=false`,
`model_training_authorized=false`, and `phase2_authorized=false`.

The only outcome-bearing operations then ran in the frozen order:

1. admission merge;
2. discovery merge;
3. no-clobber selection lock;
4. one-shot audit, only after the lock emitted `audit_authorized=true`.

The dedicated auditor published the irreversible consumption marker before
first audit access. No audit shard was manually inspected or reused.

## 2. Data and discovery gates

Admission passed with 384 accepted states and 8,946 proposals. Discovery then
passed every structural check with G384, K9, R64, H96, six source seeds, 384
unique trajectory clusters, nine valid candidates per group, and
`mixed_outcome_fraction=1.0`.

The preregistered discovery-informativeness gate passed:

| Quantity | Locked interval | Observed | Pass |
|---|---:|---:|:---:|
| Overall equal-seed nominal risk | [0.15, 0.75] | 0.469116 | yes |
| Age 25,438 nominal risk | [0.10, 0.90] | 0.409302 | yes |
| Age 50,030 nominal risk | [0.10, 0.90] | 0.481445 | yes |
| Age 100,359 nominal risk | [0.10, 0.90] | 0.516602 | yes |

The selection rule was frozen as
`per_state_all_exact_discovery_minima_uniform_expectation`: all exact empirical
discovery minima for each state receive uniform weight. This prevents candidate
column order or arbitrary tie breaking from determining the audit result.

## 3. One-shot audit result

All six preregistered primary checks passed:

| Gate | Threshold | Observed | Pass |
|---|---:|---:|:---:|
| Audit absolute fall-risk reduction | >= 0.10 | 0.344670 | yes |
| Hierarchical one-sided 95% LCB | >= 0.07 | 0.313538 | yes |
| Discovery-to-audit pair agreement | >= 0.70 | 0.792028 | yes |
| Pair-agreement one-sided 95% LCB | >= 0.65 | 0.781684 | yes |
| Effect at each of three policy ages | strictly positive | all positive | yes |
| Effect at all six source seeds | strictly positive | all positive | yes |

The equal-seed audit nominal risk was `0.4615885417`. The primary bootstrap
used 50,000 hierarchical policy-age/source-seed/trajectory-group draws with
PCG64 seed `20260810`, a linear one-sided 95% quantile, and no resampling of
candidates or replicas.

Policy-age effects were:

| Policy age | Absolute fall-risk reduction |
|---:|---:|
| 25,438 | 0.330966 |
| 50,030 | 0.320528 |
| 100,359 | 0.382515 |

Source-seed effects were:

| Source seed | Absolute fall-risk reduction |
|---:|---:|
| 8901 | 0.352191 |
| 8902 | 0.309741 |
| 8911 | 0.268953 |
| 8912 | 0.372103 |
| 8921 | 0.383436 |
| 8922 | 0.381595 |

The discovery minimizer tie fraction was `0.59375`, with a mean of
`2.861979` exact minimizers per state. The audit therefore validates the
predeclared uniform-over-all-minima estimand rather than an opportunistically
chosen single candidate.

## 4. Hash chain

| Artifact | SHA-256 |
|---|---|
| Admission merge report | `de8cd418b65041643e5fb7dcbcc37349bcd94e6f5d77142c8f4ce611f716feb8` |
| Discovery merge report | `937297e31c2409aafeeddf9ea0fcf90a34728e570a8a842a12a41e1c034828c4` |
| Selection lock | `47335e921aed608d5f47384877fb664403c9efae9e73424be4c7206fdafe9375` |
| Audit-consumed marker | `772ac1e71f44bfebbbd8b64ada82c8d9ec7d5f552554eeddb69c0b6ae3da4019` |
| Canonical Stage-A report | `e7ea56546bf8006cfc4d8ade4f5b2c26dbfcbc132e0568e054a98c2be3174b2e` |

The audit identifier was
`8e815ff2af2b48b47b8f3cf844a2ccf8631a5287cfea052f61b8dd5b046fe25a`.
The complete leaf-report and merged-output hash ledger is preserved outside
the Git worktree in the execution ledger bound to this task.

## 5. Decision and remaining claim ladder

The canonical decision is `authorize_stage_B_only`:

- Stage B option-aware Selective Advantage Q_safe fitting, calibration,
  selector freezing, placebo construction, and one-shot model test are now
  authorized under the already frozen Stage-B science.
- Stage C paired closed-loop evaluation remains unauthorized until Stage B
  passes every model gate.
- Stage D SAC-from-zero confirmation remains unauthorized until Stage C passes.
- `objective1_pass=false` and `phase2_authorized=false`; no speed-range
  expansion may begin.
- The current claim scope is conditional on the fixed seed-42 actor and
  admission-positive states. No cross-actor generalization claim is made.

The practical conclusion is narrow but important: earlier learning failures
cannot now be attributed to an absence of reusable state-dependent recovery
headroom in this K9/H96 mechanism. Stage B must determine whether the frozen
deployable 5x46D history plus 82D option descriptor can learn enough of that
signal to select recoveries on fresh actors, sources, and replicas.
