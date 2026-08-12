# Natural-PPO Q_safe Objective 1 model-test result

> **证据状态（2026-08-12）：已作废为 Objective 1 证据。** 本报告只保留为
> state-only danger detection + unified fixed recovery 的失败 baseline。它不
> 评估动作条件 `Q_safe(s,a)`，不得用于 Objective 1 通过判定。替代协议见
> `config/qsafe_action_conditioned_objective1_v1.yaml`。

Date: 2026-08-12 (Asia/Shanghai)

## Frozen inputs

- PPO exposure: 30,000,000 policy-environment steps at +0.30 m/s, without push events.
- Natural PPO falls: 3,268 independent terminal episodes.
- Direct state-risk training pairs: 22,640 fall/normal pairs.
- Q_safe representation: five-member state-risk ensemble trained only on the PPO fit role.
- Probability and uncertainty calibration: actor-disjoint natural SAC roles (seeds 47 and 48).
- Fixed-response selection: natural SAC seed 49 only.
- Protected model test: actors 43--46, sources 9501--9533, no PPO states, no external force.
- Protected units: 1,200 H96-nonoverlapping native MuJoCo snapshots.

## Calibration result

On the selector-calibration actor, joint brake for 10 policy steps reduced H96
fall probability from 47.95% to 40.18% on the frozen trigger set:

- absolute reduction: 7.76 percentage points;
- paired bootstrap 95% CI: [1.83, 13.70] percentage points;
- intervention-rate estimate: 3.65%;
- frozen risk threshold: 0.166588;
- maximum ensemble standard deviation: 0.20.

This passed the preregistered calibration gate but did not authorize an
Objective 1 claim.

## Protected model-test result

The protected one-shot test rejected the fixed-response method:

- natural-state AUROC: 0.9639;
- natural-state ECE: 0.0871 (gate: <= 0.08, failed);
- intervention rate: 10.17% (gate: <= 35%, passed);
- nominal fall rate: 2.00%;
- Q_safe + joint-brake fall rate: 2.25%;
- matched-placebo fall rate: 2.25%;
- overall Q_safe reduction: -0.25 percentage points;
- paired bootstrap 95% CI: [-0.583, 0.000] percentage points;
- triggered-state nominal fall rate: 18.03%;
- triggered-state recovery fall rate: 20.49%;
- triggered-state response reduction: -2.46 percentage points;
- triggered-state 95% CI: [-5.74, 0.00] percentage points.

The state detector transferred strongly, but the selected fixed response did
not transfer across SAC actors. Therefore `model_test_pass=false`,
`objective1_claim_eligible=false`, and `phase2_authorized=false`.

## Post-failure development diagnosis

The already-consumed cohort was used only for development diagnosis after the
failure was recorded. None of the following response families produced a
positive cross-actor effect on the frozen trigger set:

- all five registered fixed non-policy responses;
- mature SAC actor overrides of 10, 25, or 50 steps;
- mature SAC actor overrides of 1, 2, 3, 5, or 10 steps;
- final natural-PPO actor overrides of 1, 2, 3, 5, or 10 steps.

The failure is therefore classified as
`fixed_nonpolicy_response_not_cross_seed_generalizable`, not as failure to
detect risky states. Objective 2 remains forbidden. A subsequent method must
learn or otherwise validate a state-dependent response on new calibration
actors and use entirely fresh actors for its next protected test.

One attempted seed-50 actor-bank run was stopped because its external runtime
was receiving a static already-fallen simulator state. Its partial output was
moved to the explicit abandoned directory
`abandoned-invalid-static-runtime-seed-50-20260812`; it is ineligible for all
training and evidence.
