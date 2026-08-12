# PPO Same-State Counterfactual Q_safe protocol

Date: 2026-08-12 (Asia/Shanghai)

This protocol replaces the failed one-action-per-state PPO safety-data study.
The only first-round question is whether explicit same-state, multi-action H96
supervision prevents action collapse and enables a frozen critic to choose an
action with lower independently measured fall risk.

The machine-readable contract is
`config/qsafe_ppo_counterfactual_v2.yaml`. It fixes the two PPO boundary
collectors, 2,400 episode-disjoint states, 64-to-15 stochastic candidate
selection, paired-CRN H96 branching, model architecture, losses, calibration,
one-time protected analysis, selectors, gates, and stopping rules.

The previous 200-state protected outcomes are historical evidence only. Their
identities are denied to every development tool and their outcomes are moved to
a separate protected path. The previous PPO-1M critic remains frozen and is
used only as a historical baseline.

The evidence order is mandatory:

1. Candidate actions must produce informative within-state H96 risk spread.
2. A model must learn action ranking on calibration data.
3. Frozen ranking and calibration must reduce fall risk on a fresh protected
   cohort.
4. Only then may the critic be transferred to fresh SAC checkpoints.
5. Fresh-SAC online evidence is required before formal Objective 1.

No failed gate may be bypassed by changing the protected threshold, reusing the
same protected cohort, increasing ordinary transitions, changing candidate
distribution, increasing network size, or beginning SAC experiments.
